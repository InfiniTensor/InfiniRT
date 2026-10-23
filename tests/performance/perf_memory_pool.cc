#include <infini/rt.h>
#include <infini/rt/arena_memory_pool.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "perf_common.h"

namespace {

namespace perf = infini::rt::perf;
namespace runtime = infini::rt::runtime;

struct DispatchUpstream {
  using Error = runtime::Error;
  static constexpr Error kSuccess = runtime::kSuccess;

  static Error Malloc(void** ptr, std::size_t size) {
    return runtime::Malloc(ptr, size);
  }

  static Error Free(void* ptr) { return runtime::Free(ptr); }
};

struct PerfArenaConfig {
  static constexpr std::size_t kInitialCapacity = 8ull << 20;
  static constexpr std::size_t kMaxCapacity = 64ull << 20;
  static constexpr std::size_t kSmallThreshold = 1ull << 20;
  static constexpr std::size_t kMinSliceAlignment = 512;
  static constexpr std::size_t kMinSplitRemainder = 512;
  static constexpr std::size_t kShrinkThreshold = 16;
  static constexpr std::uint32_t kEmptyScansToDestroy = 2;
};

using Arena = infini::rt::ArenaMemoryPool<DispatchUpstream, PerfArenaConfig>;

bool Success(runtime::Error status) { return status == runtime::kSuccess; }

bool PrepareRuntime() {
  int device_count = 0;
  if (!Success(runtime::GetDeviceCount(&device_count)) || device_count <= 0) {
    std::cerr << "perf_memory_pool skipped: no available device." << std::endl;
    return false;
  }
  if (!Success(runtime::SetDevice(0))) {
    std::cerr << "perf_memory_pool skipped: device 0 is not available."
              << std::endl;
    return false;
  }
  return true;
}

std::vector<std::size_t> TestSizes() {
  std::vector<std::size_t> sizes{
      4 * 1024,
      64 * 1024,
      1024 * 1024,
      16 * 1024 * 1024,
  };

  if (std::getenv("INFINI_RT_PERF_ENABLE_LARGE") != nullptr) {
    sizes.push_back(256ULL * 1024ULL * 1024ULL);
  }

  return sizes;
}

std::size_t IterationsForSize(std::size_t size) {
  if (size <= 4 * 1024) {
    return 2000;
  }
  if (size <= 64 * 1024) {
    return 1000;
  }
  if (size <= 1024 * 1024) {
    return 500;
  }
  if (size <= 16 * 1024 * 1024) {
    return 100;
  }
  return 20;
}

std::vector<perf::Param> SizeParams(std::size_t size, const char* allocator) {
  return {perf::NumberParam("size_bytes", static_cast<std::uint64_t>(size)),
          perf::StringParam("allocator", allocator)};
}

std::string DescribeSize(std::size_t size) {
  if (size >= 1024 * 1024) {
    return std::to_string(size / (1024 * 1024)) + " MiB";
  }
  if (size >= 1024) {
    return std::to_string(size / 1024) + " KiB";
  }
  return std::to_string(size) + " B";
}

struct Comparison {
  std::string workload;
  std::string params;
  std::string unit;
  double direct_median = 0.0;
  double arena_median = 0.0;

  bool has_direct = true;
  bool valid = false;
};

std::vector<Comparison> g_comparisons;

void Record(const std::string& workload, const std::string& params,
            const std::string& unit, const perf::Measurement& direct,
            const perf::Measurement& arena) {
  g_comparisons.push_back(Comparison{workload, params, unit, direct.median,
                                     arena.median, true, true});
}

bool CanHoldLive(std::size_t size, std::size_t count) {
  std::vector<void*> blocks;
  blocks.reserve(count);
  bool ok = true;

  for (std::size_t i = 0; i < count; ++i) {
    void* ptr = nullptr;
    if (!Success(runtime::Malloc(&ptr, size)) || ptr == nullptr) {
      ok = false;
      break;
    }
    blocks.push_back(ptr);
  }

  for (void* ptr : blocks) {
    runtime::Free(ptr);
  }
  return ok;
}

void CompareSingleBlock(std::size_t size) {
  const auto iterations = IterationsForSize(size);

  if (!CanHoldLive(size, 1)) {
    perf::SkipBenchmark("perf_memory_pool.SingleBlock",
                        "device allocation failed during probe");
    return;
  }

  const auto direct = perf::RunBenchmarkMeasured(
      "perf_memory_pool.SingleBlock", SizeParams(size, "direct"), iterations,
      "us", [size] {
        void* ptr = nullptr;
        auto status = runtime::Malloc(&ptr, size);
        perf::DoNotOptimize(status);
        if (Success(status)) {
          status = runtime::Free(ptr);
          perf::DoNotOptimize(status);
        }
      });

  Arena arena;
  const auto arenaed = perf::RunBenchmarkMeasured(
      "perf_memory_pool.SingleBlock", SizeParams(size, "arena"), iterations,
      "us", [&arena, size] {
        void* ptr = nullptr;
        auto status = arena.Allocate(&ptr, size);
        perf::DoNotOptimize(status);
        if (Success(status)) {
          status = arena.Deallocate(ptr);
          perf::DoNotOptimize(status);
        }
      });

  Record("SingleBlock", DescribeSize(size), "us", direct, arenaed);
}

void CompareWorkingSetChurn(std::size_t size) {
  constexpr std::size_t kLiveBlocks = 8;
  const auto iterations = IterationsForSize(size);

  if (!CanHoldLive(size, kLiveBlocks + 1)) {
    perf::SkipBenchmark("perf_memory_pool.WorkingSetChurn",
                        "device cannot hold the working set");
    return;
  }

  auto make_params = [size](const char* allocator) {
    return std::vector<perf::Param>{
        perf::NumberParam("size_bytes", static_cast<std::uint64_t>(size)),
        perf::NumberParam("live_blocks", kLiveBlocks),
        perf::StringParam("allocator", allocator)};
  };

  std::vector<void*> direct_blocks(kLiveBlocks, nullptr);
  for (auto& block : direct_blocks) {
    if (!Success(runtime::Malloc(&block, size))) {
      for (void* held : direct_blocks) {
        if (held != nullptr) {
          runtime::Free(held);
        }
      }
      perf::SkipBenchmark("perf_memory_pool.WorkingSetChurn",
                          "prefill of the direct arm failed");
      return;
    }
  }

  std::size_t direct_cursor = 0;
  const auto direct = perf::RunBenchmarkMeasured(
      "perf_memory_pool.WorkingSetChurn", make_params("direct"), iterations,
      "us", [&direct_blocks, &direct_cursor, size] {
        void*& slot = direct_blocks[direct_cursor];
        direct_cursor = (direct_cursor + 1) % kLiveBlocks;
        auto status = runtime::Free(slot);
        slot = nullptr;
        if (Success(status)) {
          status = runtime::Malloc(&slot, size);
        }
        perf::DoNotOptimize(status);
      });

  for (void* block : direct_blocks) {
    if (block != nullptr) {
      runtime::Free(block);
    }
  }

  Arena arena;
  std::vector<void*> arena_blocks(kLiveBlocks, nullptr);
  for (auto& block : arena_blocks) {
    if (!Success(arena.Allocate(&block, size))) {
      perf::SkipBenchmark("perf_memory_pool.WorkingSetChurn",
                          "prefill of the arena arm failed");
      return;
    }
  }

  std::size_t arena_cursor = 0;
  const auto arenaed = perf::RunBenchmarkMeasured(
      "perf_memory_pool.WorkingSetChurn", make_params("arena"), iterations, "us",
      [&arena, &arena_blocks, &arena_cursor, size] {
        void*& slot = arena_blocks[arena_cursor];
        arena_cursor = (arena_cursor + 1) % kLiveBlocks;
        auto status = arena.Deallocate(slot);
        slot = nullptr;
        if (Success(status)) {
          status = arena.Allocate(&slot, size);
        }
        perf::DoNotOptimize(status);
      });

  for (void* block : arena_blocks) {
    if (block != nullptr) {
      arena.Deallocate(block);
    }
  }

  Record("WorkingSetChurn",
         DescribeSize(size) + " x" + std::to_string(kLiveBlocks) + " live",
         "us", direct, arenaed);
}

void CompareMixedSizeClasses() {
  constexpr std::size_t kClasses = 32;
  constexpr std::size_t kStride = 512;
  constexpr std::size_t kIterations = 2000;
  const std::string params = std::to_string(kClasses) + " classes";

  auto size_for = [](std::size_t index) { return (index + 1) * kStride; };

  if (!CanHoldLive(size_for(kClasses - 1), 1)) {
    perf::SkipBenchmark("perf_memory_pool.MixedSizeClasses",
                        "device allocation failed during probe");
    return;
  }

  std::vector<perf::Param> direct_params{
      perf::NumberParam("size_classes", kClasses),
      perf::NumberParam("stride_bytes", kStride),
      perf::StringParam("allocator", "direct")};
  std::vector<perf::Param> arena_params{
      perf::NumberParam("size_classes", kClasses),
      perf::NumberParam("stride_bytes", kStride),
      perf::StringParam("allocator", "arena")};

  std::size_t direct_index = 0;
  const auto direct = perf::RunBenchmarkMeasured(
      "perf_memory_pool.MixedSizeClasses", direct_params, kIterations, "us",
      [&direct_index, &size_for] {
        const auto size = size_for(direct_index);
        direct_index = (direct_index + 1) % kClasses;
        void* ptr = nullptr;
        auto status = runtime::Malloc(&ptr, size);
        perf::DoNotOptimize(status);
        if (Success(status)) {
          status = runtime::Free(ptr);
          perf::DoNotOptimize(status);
        }
      });

  Arena arena;
  std::size_t arena_index = 0;
  const auto arenaed = perf::RunBenchmarkMeasured(
      "perf_memory_pool.MixedSizeClasses", arena_params, kIterations, "us",
      [&arena, &arena_index, &size_for] {
        const auto size = size_for(arena_index);
        arena_index = (arena_index + 1) % kClasses;
        void* ptr = nullptr;
        auto status = arena.Allocate(&ptr, size);
        perf::DoNotOptimize(status);
        if (Success(status)) {
          status = arena.Deallocate(ptr);
          perf::DoNotOptimize(status);
        }
      });

  Record("MixedSizeClasses", params, "us", direct, arenaed);

  perf::PrintResult("perf_memory_pool.MixedSizeClassesUpstreamCalls",
                    arena_params, kIterations, "count",
                    static_cast<double>(arena.GetStats().upstream_alloc_count),
                    static_cast<double>(arena.GetStats().upstream_alloc_count));

  perf::PrintResult("perf_memory_pool.MixedSizeClassesBytesReserved",
                    arena_params, kIterations, "bytes",
                    static_cast<double>(arena.GetStats().bytes_reserved),
                    static_cast<double>(arena.GetStats().bytes_reserved));

  g_comparisons.push_back(Comparison{
      "MixedSizeClasses/upstream", params, "calls", 0.0,
      static_cast<double>(arena.GetStats().upstream_alloc_count), false, true});
}

constexpr double kTargetSampleMs = 40.0;

template <typename Op>
std::size_t CalibrateOps(std::size_t threads, Op&& op) {
  constexpr std::size_t kProbeOps = 64;
  constexpr std::size_t kMinOps = 50;
  constexpr std::size_t kMaxOps = 200000;

  for (std::size_t i = 0; i < 8; ++i) {
    op(0, i);
  }

  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < kProbeOps; ++i) {
    op(0, i);
  }
  const auto elapsed = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - start)
                           .count();

  if (elapsed <= 0.0) {
    return kMaxOps;
  }

  const double per_op_ms = elapsed / static_cast<double>(kProbeOps);
  const double wanted =
      kTargetSampleMs / (per_op_ms * static_cast<double>(threads));
  if (wanted < static_cast<double>(kMinOps)) {
    return kMinOps;
  }
  if (wanted > static_cast<double>(kMaxOps)) {
    return kMaxOps;
  }
  return static_cast<std::size_t>(wanted);
}

template <typename Op>
perf::Measurement RunThreaded(const std::string& benchmark,
                              const std::vector<perf::Param>& params,
                              std::size_t threads, std::size_t ops_per_thread,
                              Op&& op) {
  constexpr std::size_t kSamples = 9;
  const auto total_ops = threads * ops_per_thread;
  std::vector<double> samples;
  samples.reserve(kSamples);

  for (std::size_t sample = 0; sample < kSamples + 1; ++sample) {
    std::atomic<bool> go{false};
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (std::size_t t = 0; t < threads; ++t) {
      workers.emplace_back([&go, &op, ops_per_thread, t] {
        while (!go.load(std::memory_order_acquire)) {
        }
        for (std::size_t i = 0; i < ops_per_thread; ++i) {
          op(t, i);
        }
      });
    }

    const auto start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto& worker : workers) {
      worker.join();
    }
    const auto end = std::chrono::steady_clock::now();

    if (sample == 0) {
      continue;
    }
    const auto elapsed_ns =
        std::chrono::duration<double, std::nano>(end - start).count();
    samples.push_back(elapsed_ns / static_cast<double>(total_ops));
  }

  const perf::Measurement measurement{perf::Mean(samples),
                                      perf::Median(samples)};
  perf::PrintResult(benchmark, params, total_ops, "ns", measurement.mean,
                    measurement.median);
  return measurement;
}

void CompareThreadScaling(std::size_t threads) {
  constexpr std::size_t kSize = 4 * 1024;
  const std::string params = std::to_string(threads) + " threads";

  if (!CanHoldLive(kSize, threads)) {
    perf::SkipBenchmark("perf_memory_pool.ThreadScaling",
                        "device allocation failed during probe");
    return;
  }

  Arena arena;

  const auto kOpsPerThread =
      CalibrateOps(threads, [&arena](std::size_t, std::size_t) {
        void* ptr = nullptr;
        if (Success(arena.Allocate(&ptr, kSize))) {
          arena.Deallocate(ptr);
        }
      });

  std::vector<perf::Param> direct_params{
      perf::NumberParam("threads", threads),
      perf::NumberParam("size_bytes", kSize),
      perf::NumberParam("ops_per_thread", kOpsPerThread),
      perf::StringParam("allocator", "direct")};
  std::vector<perf::Param> arena_params{
      perf::NumberParam("threads", threads),
      perf::NumberParam("size_bytes", kSize),
      perf::NumberParam("ops_per_thread", kOpsPerThread),
      perf::StringParam("allocator", "arena")};

  const auto direct =
      RunThreaded("perf_memory_pool.ThreadScaling", direct_params, threads,
                  kOpsPerThread, [](std::size_t, std::size_t) {
                    void* ptr = nullptr;
                    auto status = runtime::Malloc(&ptr, kSize);
                    if (Success(status)) {
                      status = runtime::Free(ptr);
                    }
                    perf::DoNotOptimize(status);
                  });

  const auto arenaed =
      RunThreaded("perf_memory_pool.ThreadScaling", arena_params, threads,
                  kOpsPerThread, [&arena](std::size_t, std::size_t) {
                    void* ptr = nullptr;
                    auto status = arena.Allocate(&ptr, kSize);
                    if (Success(status)) {
                      status = arena.Deallocate(ptr);
                    }
                    perf::DoNotOptimize(status);
                  });

  Record("ThreadScaling", params, "ns", direct, arenaed);
}

void CompareConcurrentMixedSizes(std::size_t threads) {
  constexpr std::size_t kClasses = 16;
  constexpr std::size_t kStride = 512;
  const std::string params = std::to_string(threads) + "T x16cls";

  auto size_for = [](std::size_t thread, std::size_t op) {
    return ((thread + op) % kClasses + 1) * kStride;
  };

  if (!CanHoldLive(kClasses * kStride, threads)) {
    perf::SkipBenchmark("perf_memory_pool.ConcurrentMixedSizes",
                        "device allocation failed during probe");
    return;
  }

  Arena arena;
  const auto kOpsPerThread =
      CalibrateOps(threads, [&arena, &size_for](std::size_t t, std::size_t op) {
        void* ptr = nullptr;
        if (Success(arena.Allocate(&ptr, size_for(t, op)))) {
          arena.Deallocate(ptr);
        }
      });

  std::vector<perf::Param> direct_params{
      perf::NumberParam("threads", threads),
      perf::NumberParam("size_classes", kClasses),
      perf::NumberParam("ops_per_thread", kOpsPerThread),
      perf::StringParam("allocator", "direct")};
  std::vector<perf::Param> arena_params{
      perf::NumberParam("threads", threads),
      perf::NumberParam("size_classes", kClasses),
      perf::NumberParam("ops_per_thread", kOpsPerThread),
      perf::StringParam("allocator", "arena")};

  const auto direct = RunThreaded(
      "perf_memory_pool.ConcurrentMixedSizes", direct_params, threads,
      kOpsPerThread, [&size_for](std::size_t thread, std::size_t op) {
        void* ptr = nullptr;
        auto status = runtime::Malloc(&ptr, size_for(thread, op));
        if (Success(status)) {
          status = runtime::Free(ptr);
        }
        perf::DoNotOptimize(status);
      });

  const auto arenaed = RunThreaded(
      "perf_memory_pool.ConcurrentMixedSizes", arena_params, threads,
      kOpsPerThread, [&arena, &size_for](std::size_t thread, std::size_t op) {
        void* ptr = nullptr;
        auto status = arena.Allocate(&ptr, size_for(thread, op));
        if (Success(status)) {
          status = arena.Deallocate(ptr);
        }
        perf::DoNotOptimize(status);
      });

  Record("ConcurrentMixedSizes", params, "ns", direct, arenaed);
}

void MeasureMissPath(std::size_t size) {
  const auto iterations = IterationsForSize(size);

  if (!CanHoldLive(size, 1)) {
    perf::SkipBenchmark("perf_memory_pool.MissPath",
                        "device allocation failed during probe");
    return;
  }

  Arena arena;
  perf::RunBenchmark("perf_memory_pool.MissPath",
                     SizeParams(size, "arena"), iterations, "us",
                     [&arena, size] {
                       void* ptr = nullptr;
                       auto status = arena.Allocate(&ptr, size);
                       if (Success(status)) {
                         status = arena.Deallocate(ptr);
                       }
                       perf::DoNotOptimize(status);
                       arena.ReleaseCached();
                     });
}

void CompareFirstTouchGrowth() {
  constexpr std::size_t kBlocks = 2000;
  constexpr std::size_t kStride = 512;
  constexpr std::size_t kClasses = 64;
  const std::string params = std::to_string(kBlocks) + " blocks";

  auto size_for = [](std::size_t index) {
    return (index % kClasses + 1) * kStride;
  };

  if (!CanHoldLive(kClasses * kStride, 16)) {
    perf::SkipBenchmark("perf_memory_pool.FirstTouchGrowth",
                        "device allocation failed during probe");
    return;
  }

  std::vector<void*> blocks;
  blocks.reserve(kBlocks);

  const auto direct = perf::RunBenchmarkMeasured(
      "perf_memory_pool.FirstTouchGrowth",
      {perf::NumberParam("blocks", kBlocks),
       perf::StringParam("allocator", "direct")},
      20, "us", [&blocks, &size_for] {
        for (std::size_t i = 0; i < kBlocks; ++i) {
          void* ptr = nullptr;
          if (Success(runtime::Malloc(&ptr, size_for(i)))) {
            blocks.push_back(ptr);
          }
        }
        for (void* ptr : blocks) {
          runtime::Free(ptr);
        }
        blocks.clear();
      });

  std::size_t arena_upstream = 0;
  const auto arenaed = perf::RunBenchmarkMeasured(
      "perf_memory_pool.FirstTouchGrowth",
      {perf::NumberParam("blocks", kBlocks),
       perf::StringParam("allocator", "arena")},
      20, "us", [&blocks, &size_for, &arena_upstream] {
        Arena arena;
        for (std::size_t i = 0; i < kBlocks; ++i) {
          void* ptr = nullptr;
          if (Success(arena.Allocate(&ptr, size_for(i)))) {
            blocks.push_back(ptr);
          }
        }
        arena_upstream = arena.GetStats().upstream_alloc_count;
        for (void* ptr : blocks) {
          arena.Deallocate(ptr);
        }
        blocks.clear();
      });

  Record("FirstTouchGrowth", params, "us", direct, arenaed);

  perf::PrintResult("perf_memory_pool.FirstTouchGrowthUpstreamCalls",
                    {perf::NumberParam("blocks", kBlocks),
                     perf::StringParam("allocator", "arena")},
                    kBlocks, "count", static_cast<double>(arena_upstream),
                    static_cast<double>(arena_upstream));

  g_comparisons.push_back(Comparison{"FirstTouchGrowth/upstream", params,
                                     "calls", static_cast<double>(kBlocks),
                                     static_cast<double>(arena_upstream), true,
                                     true});
}

void MeasureConcurrentMissPath(std::size_t threads) {
  constexpr std::size_t kSize = 4 * 1024;

  if (!CanHoldLive(kSize, threads)) {
    perf::SkipBenchmark("perf_memory_pool.ConcurrentMissPath",
                        "device allocation failed during probe");
    return;
  }

  Arena arena;
  auto miss_op = [&arena](std::size_t, std::size_t) {
    void* ptr = nullptr;
    const auto status = arena.Allocate(&ptr, kSize);
    if (Success(status)) {
      arena.Deallocate(ptr);
    }
    perf::DoNotOptimize(status);
    arena.ReleaseCached();
  };
  const auto kOpsPerThread = CalibrateOps(threads, miss_op);

  RunThreaded("perf_memory_pool.ConcurrentMissPath",
              {perf::NumberParam("threads", threads),
               perf::NumberParam("size_bytes", kSize),
               perf::NumberParam("ops_per_thread", kOpsPerThread),
               perf::StringParam("allocator", "arena")},
              threads, kOpsPerThread, [&arena](std::size_t, std::size_t) {
                void* ptr = nullptr;
                auto status = arena.Allocate(&ptr, kSize);
                if (Success(status)) {
                  status = arena.Deallocate(ptr);
                }
                perf::DoNotOptimize(status);
                arena.ReleaseCached();
              });
}

void MeasureAlignedHit() {
  constexpr std::size_t kSize = 4 * 1024;
  constexpr std::size_t kAlignment = 256;
  constexpr std::size_t kIterations = 200000;

  Arena arena;
  void* probe = nullptr;
  if (!Success(arena.Allocate(&probe, kSize, kAlignment)) || probe == nullptr) {
    perf::SkipBenchmark("perf_memory_pool.AlignedHit",
                        "aligned allocation failed during probe");
    return;
  }
  arena.Deallocate(probe);

  perf::RunBenchmark("perf_memory_pool.AlignedHit",
                     {perf::NumberParam("size_bytes", kSize),
                      perf::NumberParam("alignment", kAlignment),
                      perf::StringParam("allocator", "arena")},
                     kIterations, "ns", [&arena] {
                       void* ptr = nullptr;
                       auto status = arena.Allocate(&ptr, kSize, kAlignment);
                       if (Success(status)) {
                         status = arena.Deallocate(ptr);
                       }
                       perf::DoNotOptimize(status);
                     });
}

void MeasureReleaseCached() {
  constexpr std::size_t kSize = 4 * 1024;
  constexpr std::size_t kCachedBlocks = 64;
  constexpr std::size_t kIterations = 200;

  if (!CanHoldLive(kSize, kCachedBlocks)) {
    perf::SkipBenchmark("perf_memory_pool.ReleaseCached",
                        "device allocation failed during probe");
    return;
  }

  Arena arena;
  std::vector<void*> blocks(kCachedBlocks, nullptr);
  perf::RunBenchmark("perf_memory_pool.ReleaseCached",
                     {perf::NumberParam("cached_blocks", kCachedBlocks),
                      perf::NumberParam("size_bytes", kSize),
                      perf::StringParam("allocator", "arena")},
                     kIterations, "us", [&arena, &blocks] {
                       for (void*& block : blocks) {
                         arena.Allocate(&block, kSize);
                       }
                       for (void*& block : blocks) {
                         arena.Deallocate(block);
                         block = nullptr;
                       }
                       arena.ReleaseCached();
                     });

  const auto before = arena.GetStats().upstream_free_count;
  for (void*& block : blocks) {
    arena.Allocate(&block, kSize);
  }
  for (void*& block : blocks) {
    arena.Deallocate(block);
    block = nullptr;
  }
  arena.ReleaseCached();
  const auto arena_frees =
      static_cast<double>(arena.GetStats().upstream_free_count - before);

  perf::PrintResult("perf_memory_pool.ReleaseCachedUpstreamFrees",
                    {perf::NumberParam("cached_blocks", kCachedBlocks),
                     perf::StringParam("allocator", "arena")},
                    1, "count", arena_frees, arena_frees);

  g_comparisons.push_back(Comparison{
      "ReleaseCached/frees", std::to_string(kCachedBlocks) + " cached", "calls",
      static_cast<double>(kCachedBlocks), arena_frees, true, true});
}

void MeasureBookkeeping() {
  constexpr std::size_t kIterations = 1000000;

  Arena arena;

  std::vector<void*> live;
  for (std::size_t i = 0; i < 64; ++i) {
    void* ptr = nullptr;
    if (Success(arena.Allocate(&ptr, (i % 16 + 1) * 512))) {
      live.push_back(ptr);
    }
  }

  perf::RunBenchmark("perf_memory_pool.GetStats",
                     {perf::StringParam("allocator", "arena")}, kIterations,
                     "ns", [&arena] {
                       const auto stats = arena.GetStats();
                       perf::DoNotOptimize(stats.bytes_in_use);
                     });

  perf::RunBenchmark("perf_memory_pool.AllocateZeroBytes",
                     {perf::StringParam("allocator", "arena")}, kIterations,
                     "ns", [&arena] {
                       void* ptr = nullptr;
                       const auto status = arena.Allocate(&ptr, 0);
                       perf::DoNotOptimize(status);
                       perf::DoNotOptimize(ptr);
                     });

  perf::RunBenchmark("perf_memory_pool.DeallocateNullptr",
                     {perf::StringParam("allocator", "arena")}, kIterations,
                     "ns", [&arena] {
                       const auto status = arena.Deallocate(nullptr);
                       perf::DoNotOptimize(status);
                     });

  for (void* ptr : live) {
    arena.Deallocate(ptr);
  }
}

void PrintComparisonTable() {
  if (g_comparisons.empty()) {
    return;
  }

  std::cerr << "\n=== " << INFINI_RT_PERF_BACKEND_NAME
            << ": no pool vs ArenaMemoryPool "
               "(median, lower is better) ===\n";
  std::cerr << std::left << std::setw(28) << "workload" << std::setw(16)
            << "params" << std::right << std::setw(14) << "no pool"
            << std::setw(14) << "arena" << std::setw(11) << "arena win"
            << "\n";

  for (const Comparison& row : g_comparisons) {
    if (!row.valid) {
      continue;
    }
    std::cerr << std::left << std::setw(28) << row.workload << std::setw(16)
              << row.params << std::right << std::fixed << std::setprecision(3);
    if (row.has_direct) {
      std::cerr << std::setw(11) << row.direct_median << " " << row.unit;
    } else {
      std::cerr << std::setw(14) << "n/a";
    }
    std::cerr << std::setw(11) << row.arena_median << " " << row.unit;

    if (row.has_direct && row.arena_median > 0.0 && row.direct_median > 0.0) {
      std::cerr << std::setw(10) << std::setprecision(2)
                << (row.direct_median / row.arena_median) << "x";
    } else {
      std::cerr << std::setw(11) << "n/a";
    }
    std::cerr << "\n";
  }

  std::cerr << "\n`arena win` > 1 means the arena is faster than going straight "
               "to the backend.\n`calls` rows are exact counts, not timings. "
               "`n/a` in the `no pool` column marks a\nrow with no direct "
               "counterpart -- a trim has no meaning without a cache.\n";
  std::cerr << std::endl;
}

}

int main() {
  if (!PrepareRuntime()) {
    return 0;
  }

  for (const auto size : TestSizes()) {
    CompareSingleBlock(size);
    CompareWorkingSetChurn(size);
    MeasureMissPath(size);
  }

  CompareMixedSizeClasses();
  CompareFirstTouchGrowth();

  const auto hardware_threads = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::thread::hardware_concurrency()));
  for (const std::size_t threads : {std::size_t{1}, std::size_t{2},
                                    std::size_t{4}, std::size_t{8},
                                    std::size_t{16}, std::size_t{32},
                                    std::size_t{64}}) {
    if (threads > hardware_threads) {
      break;
    }
    CompareThreadScaling(threads);
    CompareConcurrentMixedSizes(threads);
    MeasureConcurrentMissPath(threads);
  }

  MeasureAlignedHit();
  MeasureReleaseCached();
  MeasureBookkeeping();

  PrintComparisonTable();
  return 0;
}
