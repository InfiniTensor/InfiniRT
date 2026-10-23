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
#include <utility>
#include <vector>

#include "perf_common.h"

namespace {

namespace perf = infini::rt::perf;
namespace runtime = infini::rt::runtime;

bool Success(runtime::Error status) { return status == runtime::kSuccess; }

#if defined(INFINI_RT_PERF_LARGE_ARENA_CONFIG)
struct MatrixArenaConfig {
  static constexpr std::size_t kInitialCapacity = 64ull << 20;
  static constexpr std::size_t kMaxCapacity = 512ull << 20;
  static constexpr std::size_t kSmallThreshold = 1ull << 20;
  static constexpr std::size_t kMinSliceAlignment = 512;
  static constexpr std::size_t kMinSplitRemainder = 512;
  static constexpr std::size_t kShrinkThreshold = 16;
  static constexpr std::uint32_t kEmptyScansToDestroy = 2;
};
constexpr const char* kConfigName = "production";
#else
struct MatrixArenaConfig {
  static constexpr std::size_t kInitialCapacity = 8ull << 20;
  static constexpr std::size_t kMaxCapacity = 64ull << 20;
  static constexpr std::size_t kSmallThreshold = 1ull << 20;
  static constexpr std::size_t kMinSliceAlignment = 512;
  static constexpr std::size_t kMinSplitRemainder = 512;
  static constexpr std::size_t kShrinkThreshold = 16;
  static constexpr std::uint32_t kEmptyScansToDestroy = 2;
};
constexpr const char* kConfigName = "reduced";
#endif

#if defined(INFINI_RT_PERF_LARGE_ARENA_CONFIG)
constexpr std::size_t kFootprintDivisor = 1;
constexpr bool kDeviceBackend = true;
#else
constexpr std::size_t kFootprintDivisor = 16;
constexpr bool kDeviceBackend = false;
#endif

struct DispatchUpstream {
  using Error = runtime::Error;
  static constexpr Error kSuccess = runtime::kSuccess;

  static Error Malloc(void** ptr, std::size_t size) {
    return runtime::Malloc(ptr, size);
  }
  static Error Free(void* ptr) { return runtime::Free(ptr); }
};

class DirectArm {
 public:
  static constexpr const char* kName = "direct";
  static constexpr bool kStreamOrdered = false;

  static constexpr bool kHasCache = false;

  static bool Available() { return true; }

  runtime::Error Allocate(void** ptr, std::size_t size) {
    ++upstream_allocs_;
    return runtime::Malloc(ptr, size);
  }

  runtime::Error Deallocate(void* ptr) {
    if (ptr == nullptr) {
      return runtime::kSuccess;
    }
    ++upstream_frees_;
    return runtime::Free(ptr);
  }

  void ReleaseCached() {}
  void Sync() {}

  std::size_t UpstreamAllocs() const { return upstream_allocs_; }
  std::size_t UpstreamFrees() const { return upstream_frees_; }

  bool TracksBytes() const { return false; }
  std::size_t BytesReserved() const { return 0; }

 private:
  std::size_t upstream_allocs_ = 0;
  std::size_t upstream_frees_ = 0;
};

class ArenaArm {
 public:
  static constexpr const char* kName = "arena";
  static constexpr bool kStreamOrdered = false;
  static constexpr bool kHasCache = true;

  static bool Available() { return true; }

  runtime::Error Allocate(void** ptr, std::size_t size) {
    return pool_.Allocate(ptr, size);
  }
  runtime::Error Deallocate(void* ptr) { return pool_.Deallocate(ptr); }
  void ReleaseCached() { pool_.ReleaseCached(); }
  void Sync() {}

  std::size_t UpstreamAllocs() const {
    return pool_.GetStats().upstream_alloc_count;
  }
  std::size_t UpstreamFrees() const {
    return pool_.GetStats().upstream_free_count;
  }
  bool TracksBytes() const { return true; }
  std::size_t BytesReserved() const { return pool_.GetStats().bytes_reserved; }

 private:
  infini::rt::ArenaMemoryPool<DispatchUpstream, MatrixArenaConfig> pool_;
};

class CudaAsyncArm {
 public:
  static constexpr const char* kName = "cuda_async";
  static constexpr bool kStreamOrdered = true;

  static constexpr bool kHasCache = false;

  static bool Available() {
    runtime::Stream stream{};
    if (!Success(runtime::StreamCreate(&stream))) {
      return false;
    }
    void* ptr = nullptr;
    const bool ok = Success(runtime::MallocAsync(&ptr, 4096, stream)) &&
                    Success(runtime::StreamSynchronize(stream));
    if (ok) {
      runtime::FreeAsync(ptr, stream);
      runtime::StreamSynchronize(stream);
    }
    runtime::StreamDestroy(stream);
    return ok;
  }

  CudaAsyncArm() { runtime::StreamCreate(&stream_); }
  ~CudaAsyncArm() {
    if (stream_ != runtime::Stream{}) {
      runtime::StreamSynchronize(stream_);
      runtime::StreamDestroy(stream_);
    }
  }

  CudaAsyncArm(const CudaAsyncArm&) = delete;
  CudaAsyncArm& operator=(const CudaAsyncArm&) = delete;

  runtime::Error Allocate(void** ptr, std::size_t size) {
    ++upstream_allocs_;
    return runtime::MallocAsync(ptr, size, stream_);
  }

  runtime::Error Deallocate(void* ptr) {
    if (ptr == nullptr) {
      return runtime::kSuccess;
    }
    ++upstream_frees_;
    return runtime::FreeAsync(ptr, stream_);
  }

  void ReleaseCached() {}
  void Sync() { runtime::StreamSynchronize(stream_); }

  std::size_t UpstreamAllocs() const { return upstream_allocs_; }
  std::size_t UpstreamFrees() const { return upstream_frees_; }
  bool TracksBytes() const { return false; }
  std::size_t BytesReserved() const { return 0; }

 private:
  runtime::Stream stream_{};
  std::size_t upstream_allocs_ = 0;
  std::size_t upstream_frees_ = 0;
};

struct Cell {
  std::string workload;
  std::string params;
  std::string arm;
  std::string unit;
  double value = 0.0;
  bool present = true;
};

std::vector<Cell> g_cells;
std::vector<std::string> g_arm_order;

void RecordCell(const std::string& workload, const std::string& params,
                const std::string& arm, const std::string& unit, double value,
                bool present = true) {
  g_cells.push_back(Cell{workload, params, arm, unit, value, present});
  if (std::find(g_arm_order.begin(), g_arm_order.end(), arm) ==
      g_arm_order.end()) {
    g_arm_order.push_back(arm);
  }
}

std::vector<perf::Param> WithArm(std::vector<perf::Param> params,
                                 const char* arm) {
  params.push_back(perf::StringParam("allocator", arm));
  params.push_back(perf::StringParam("arena_config", kConfigName));
  return params;
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

template <typename Arm>
void BenchCrossThresholdRotation(std::size_t iterations) {
  const std::size_t sizes[] = {1024, 2ull << 20, 4096};
  constexpr std::size_t kCount = 3;

  Arm arm;
  std::size_t index = 0;
  const auto measurement = perf::RunBenchmarkMeasured(
      "allocator_matrix.CrossThresholdRotation",
      WithArm({perf::NumberParam("size_count", kCount)}, Arm::kName), iterations,
      "us", [&arm, &index, &sizes] {
        const std::size_t size = sizes[index];
        index = (index + 1) % kCount;
        void* ptr = nullptr;
        auto status = arm.Allocate(&ptr, size);
        perf::DoNotOptimize(status);
        if (Success(status)) {
          status = arm.Deallocate(ptr);
          perf::DoNotOptimize(status);
        }
      });
  arm.Sync();

  RecordCell("CrossThreshold", "1K/2M/4K", Arm::kName, "us",
             measurement.median);
  RecordCell("CrossThreshold/upstream", "1K/2M/4K", Arm::kName, "calls",
             static_cast<double>(arm.UpstreamAllocs()));
  RecordCell("CrossThreshold/reserved", "1K/2M/4K", Arm::kName, "B",
             static_cast<double>(arm.BytesReserved()), arm.TracksBytes());

  perf::PrintResult("allocator_matrix.CrossThresholdRotationUpstreamCalls",
                    WithArm({perf::NumberParam("size_count", kCount)},
                            Arm::kName),
                    iterations, "count",
                    static_cast<double>(arm.UpstreamAllocs()),
                    static_cast<double>(arm.UpstreamAllocs()));
}

template <typename Arm>
void BenchLargeBlockRecycle(std::size_t iterations) {
  const std::size_t sizes[] = {9ull << 20, 11ull << 20, 13ull << 20};
  constexpr std::size_t kCount = 3;

  {
    std::vector<void*> probe;
    bool ok = true;
    for (std::size_t i = 0; i < kCount && ok; ++i) {
      void* ptr = nullptr;
      ok = Success(runtime::Malloc(&ptr, sizes[i])) && ptr != nullptr;
      if (ok) {
        probe.push_back(ptr);
      }
    }
    for (void* ptr : probe) {
      runtime::Free(ptr);
    }
    if (!ok) {
      perf::SkipBenchmark("allocator_matrix.LargeBlockRecycle",
                          "device cannot hold the working set");
      return;
    }
  }

  Arm arm;
  std::size_t index = 0;
  const auto measurement = perf::RunBenchmarkMeasured(
      "allocator_matrix.LargeBlockRecycle",
      WithArm({perf::NumberParam("size_count", kCount)}, Arm::kName), iterations,
      "us", [&arm, &index, &sizes] {
        const std::size_t size = sizes[index];
        index = (index + 1) % kCount;
        void* ptr = nullptr;
        auto status = arm.Allocate(&ptr, size);
        perf::DoNotOptimize(status);
        if (Success(status)) {
          status = arm.Deallocate(ptr);
          perf::DoNotOptimize(status);
        }
      });
  arm.Sync();

  const std::size_t demand = sizes[kCount - 1];
  RecordCell("LargeRecycle", "9/11/13 MiB", Arm::kName, "us",
             measurement.median);
  RecordCell("LargeRecycle/reserved", "9/11/13 MiB", Arm::kName, "B",
             static_cast<double>(arm.BytesReserved()), arm.TracksBytes());
  RecordCell("LargeRecycle/amplif", "9/11/13 MiB", Arm::kName, "x",
             static_cast<double>(arm.BytesReserved()) /
                 static_cast<double>(demand),
             arm.TracksBytes());
  RecordCell("LargeRecycle/upstream", "9/11/13 MiB", Arm::kName, "calls",
             static_cast<double>(arm.UpstreamAllocs()));
}

template <typename Arm>
void BenchHighWaterGrowth() {
  const std::size_t target = (1ull << 30) / kFootprintDivisor;
  constexpr std::size_t kBlock = 1ull << 20;
  const std::size_t blocks = target / kBlock;

  Arm arm;
  std::vector<void*> live;
  live.reserve(blocks);

  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < blocks; ++i) {
    void* ptr = nullptr;
    if (!Success(arm.Allocate(&ptr, kBlock))) {
      break;
    }
    live.push_back(ptr);
  }
  arm.Sync();
  const auto end = std::chrono::steady_clock::now();

  const auto elapsed_us =
      std::chrono::duration<double, std::micro>(end - start).count();
  const auto reserved = arm.BytesReserved();
  const auto upstream = arm.UpstreamAllocs();
  const bool tracks = arm.TracksBytes();

  for (void* ptr : live) {
    arm.Deallocate(ptr);
  }
  arm.Sync();

  const std::string params = DescribeSize(live.size() * kBlock) + " live";
  RecordCell("HighWater", params, Arm::kName, "us", elapsed_us);
  RecordCell("HighWater/upstream", params, Arm::kName, "calls",
             static_cast<double>(upstream));
  RecordCell("HighWater/reserved", params, Arm::kName, "B",
             static_cast<double>(reserved), tracks);

  perf::PrintResult("allocator_matrix.HighWaterGrowth",
                    WithArm({perf::NumberParam(
                                "live_bytes", static_cast<std::uint64_t>(
                                                  live.size() * kBlock))},
                            Arm::kName),
                    live.size(), "us", elapsed_us, elapsed_us);
  perf::PrintResult("allocator_matrix.HighWaterGrowthUpstreamCalls",
                    WithArm({perf::NumberParam(
                                "live_bytes", static_cast<std::uint64_t>(
                                                  live.size() * kBlock))},
                            Arm::kName),
                    live.size(), "count", static_cast<double>(upstream),
                    static_cast<double>(upstream));
}

template <typename Arm>
void BenchTrimCost(std::size_t cached_blocks) {
  constexpr std::size_t kSize = 64 * 1024;
  constexpr std::size_t kIterations = 100;

  if (!Arm::kHasCache) {
    perf::SkipBenchmark("allocator_matrix.TrimCost",
                        std::string(Arm::kName) + " has no releasable cache");
    return;
  }

  Arm arm;
  std::vector<void*> blocks(cached_blocks, nullptr);

  auto fill_and_trim = [&arm, &blocks] {
    for (void*& block : blocks) {
      arm.Allocate(&block, kSize);
    }
    for (void*& block : blocks) {
      arm.Deallocate(block);
      block = nullptr;
    }
    arm.ReleaseCached();
  };

  const auto measurement = perf::RunBenchmarkMeasured(
      "allocator_matrix.TrimCost",
      WithArm({perf::NumberParam("cached_blocks", cached_blocks),
               perf::NumberParam("size_bytes", kSize)},
              Arm::kName),
      kIterations, "us", fill_and_trim);
  arm.Sync();

  const auto before_frees = arm.UpstreamFrees();
  fill_and_trim();
  arm.Sync();
  const auto frees_per_trim =
      static_cast<double>(arm.UpstreamFrees() - before_frees);

  const std::string params = std::to_string(cached_blocks) + " cached";
  RecordCell("TrimCost", params, Arm::kName, "us", measurement.median);
  RecordCell("TrimCost/frees", params, Arm::kName, "calls", frees_per_trim);
}

template <typename Op>
perf::Measurement RunThreaded(const std::string& benchmark,
                              const std::vector<perf::Param>& params,
                              std::size_t threads, std::size_t ops_per_thread,
                              Op&& op) {
  constexpr std::size_t kSamples = 7;
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
    samples.push_back(
        std::chrono::duration<double, std::nano>(end - start).count() /
        static_cast<double>(total_ops));
  }

  const perf::Measurement measurement{perf::Mean(samples),
                                      perf::Median(samples)};
  perf::PrintResult(benchmark, params, total_ops, "ns", measurement.mean,
                    measurement.median);
  return measurement;
}

template <typename Arm>
void BenchThreadScaling(std::size_t threads, std::size_t ops_per_thread) {
  constexpr std::size_t kClasses = 16;
  constexpr std::size_t kStride = 512;

  Arm arm;
  const auto measurement = RunThreaded(
      "allocator_matrix.ThreadScaling",
      WithArm({perf::NumberParam("threads", threads),
               perf::NumberParam("size_classes", kClasses)},
              Arm::kName),
      threads, ops_per_thread, [&arm](std::size_t thread, std::size_t op) {
        const std::size_t size = ((thread + op) % kClasses + 1) * kStride;
        void* ptr = nullptr;
        auto status = arm.Allocate(&ptr, size);
        if (Success(status)) {
          status = arm.Deallocate(ptr);
        }
        perf::DoNotOptimize(status);
      });
  arm.Sync();

  RecordCell("ThreadScaling", std::to_string(threads) + "T", Arm::kName, "ns",
             measurement.median);
}

template <typename Arm>
void BenchImplicitSyncCost() {
  constexpr std::size_t kSize = 64 * 1024;
  constexpr std::size_t kCachedBlocks = 64;

  constexpr std::size_t kBusyOps = 200;
  const std::size_t busy_bytes = (32ull << 20) / kFootprintDivisor;

  if (!Arm::kHasCache) {
    perf::SkipBenchmark("allocator_matrix.ImplicitSyncCost",
                        std::string(Arm::kName) + " has no releasable cache");
    return;
  }

  runtime::Stream stream{};
  if (!Success(runtime::StreamCreate(&stream))) {
    perf::SkipBenchmark("allocator_matrix.ImplicitSyncCost",
                        "stream creation failed");
    return;
  }

  void* busy_buffer = nullptr;
  if (!Success(runtime::Malloc(&busy_buffer, busy_bytes))) {
    runtime::StreamDestroy(stream);
    perf::SkipBenchmark("allocator_matrix.ImplicitSyncCost",
                        "could not reserve the interference buffer");
    return;
  }

  Arm arm;
  std::vector<void*> blocks(kCachedBlocks, nullptr);

  auto fill_cache = [&arm, &blocks] {
    for (void*& block : blocks) {
      arm.Allocate(&block, kSize);
    }
    for (void*& block : blocks) {
      arm.Deallocate(block);
      block = nullptr;
    }
  };

  auto time_trim = [&](bool with_interference) {
    constexpr std::size_t kSamples = 5;
    std::vector<double> samples;
    samples.reserve(kSamples);

    for (std::size_t sample = 0; sample < kSamples + 1; ++sample) {
      fill_cache();
      if (with_interference) {
        for (std::size_t i = 0; i < kBusyOps; ++i) {
          runtime::MemsetAsync(busy_buffer, static_cast<int>(i & 0xff),
                               busy_bytes, stream);
        }
      }

      const auto start = std::chrono::steady_clock::now();
      arm.ReleaseCached();
      arm.Sync();
      const auto end = std::chrono::steady_clock::now();

      runtime::StreamSynchronize(stream);
      if (sample == 0) {
        continue;
      }
      samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }
    return perf::Median(samples);
  };

  const double idle = time_trim(false);
  const double loaded = time_trim(true);

  runtime::Free(busy_buffer);
  runtime::StreamDestroy(stream);

  const auto params = WithArm(
      {perf::NumberParam("cached_blocks", kCachedBlocks),
       perf::NumberParam("queued_ops", kBusyOps)},
      Arm::kName);
  perf::PrintResult("allocator_matrix.TrimIdleStream", params, kCachedBlocks,
                    "us", idle, idle);
  perf::PrintResult("allocator_matrix.TrimBusyStream", params, kCachedBlocks,
                    "us", loaded, loaded);

  RecordCell("Trim/idle stream", "64 cached", Arm::kName, "us", idle);
  RecordCell("Trim/busy stream", "64 cached", Arm::kName, "us", loaded);

  RecordCell("Trim/sync stall", "64 cached", Arm::kName, "us",
             std::max(0.0, loaded - idle));
}

template <typename Arm>
void MeasureLedgerAccuracy() {
  constexpr std::size_t kBlock = 1ull << 20;
  const std::size_t blocks = (256ull << 20) / kFootprintDivisor / kBlock;

  if (!kDeviceBackend) {
    perf::SkipBenchmark("allocator_matrix.LedgerAccuracy",
                        "MemGetInfo is system-wide on the host backend");
    return;
  }

  std::size_t free_before = 0;
  std::size_t total = 0;
  if (!Success(runtime::MemGetInfo(&free_before, &total)) || free_before == 0) {
    perf::SkipBenchmark("allocator_matrix.LedgerAccuracy",
                        "the backend does not report device memory");
    return;
  }

  Arm arm;
  if (!arm.TracksBytes()) {
    return;
  }

  std::vector<void*> live;
  live.reserve(blocks);
  for (std::size_t i = 0; i < blocks; ++i) {
    void* ptr = nullptr;
    if (!Success(arm.Allocate(&ptr, kBlock))) {
      break;
    }

    runtime::Memset(ptr, 0, kBlock);
    live.push_back(ptr);
  }
  arm.Sync();
  runtime::DeviceSynchronize();

  std::size_t free_after = 0;
  runtime::MemGetInfo(&free_after, &total);
  const double device_consumed =
      free_before > free_after ? static_cast<double>(free_before - free_after)
                               : 0.0;
  const double reported = static_cast<double>(arm.BytesReserved());

  for (void* ptr : live) {
    arm.Deallocate(ptr);
  }
  arm.Sync();

  const std::string params = DescribeSize(live.size() * kBlock) + " live";
  RecordCell("Ledger/reported", params, Arm::kName, "B", reported);
  RecordCell("Ledger/device", params, Arm::kName, "B", device_consumed);
  RecordCell("Ledger/ratio", params, Arm::kName, "x",
             reported == 0.0 ? 0.0 : device_consumed / reported);

  perf::PrintResult(
      "allocator_matrix.LedgerAccuracy",
      WithArm({perf::NumberParam(
                  "live_bytes",
                  static_cast<std::uint64_t>(live.size() * kBlock))},
              Arm::kName),
      live.size(), "bytes", device_consumed, device_consumed);
}

template <typename Arm>
void BenchMultiStream(std::size_t streams, std::size_t ops_per_stream) {
  constexpr std::size_t kSize = 256 * 1024;

  std::vector<runtime::Stream> handles(streams, runtime::Stream{});
  for (auto& stream : handles) {
    if (!Success(runtime::StreamCreate(&stream))) {
      for (auto& created : handles) {
        if (created != runtime::Stream{}) {
          runtime::StreamDestroy(created);
        }
      }
      perf::SkipBenchmark("allocator_matrix.MultiStream",
                          "stream creation failed");
      return;
    }
  }

  Arm arm;
  const auto measurement = RunThreaded(
      "allocator_matrix.MultiStream",
      WithArm({perf::NumberParam("streams", streams),
               perf::NumberParam("size_bytes", kSize)},
              Arm::kName),
      streams, ops_per_stream,
      [&arm, &handles](std::size_t index, std::size_t) {
        void* ptr = nullptr;
        if (!Success(arm.Allocate(&ptr, kSize))) {
          return;
        }

        runtime::MemsetAsync(ptr, 0, kSize, handles[index]);
        runtime::StreamSynchronize(handles[index]);
        const auto status = arm.Deallocate(ptr);
        perf::DoNotOptimize(status);
      });
  arm.Sync();

  for (auto& stream : handles) {
    runtime::StreamDestroy(stream);
  }

  RecordCell("MultiStream", std::to_string(streams) + " streams", Arm::kName,
             "ns", measurement.median);
}

template <typename Arm>
void BenchLayerwiseInference(std::size_t decode_steps) {
  constexpr std::size_t kLayers = 32;
  const std::size_t weight_bytes = (4ull << 20) / kFootprintDivisor;
  const std::size_t activation_bytes = (2ull << 20) / kFootprintDivisor;
  const std::size_t kv_step_bytes = (128ull << 10) / kFootprintDivisor;
  const std::size_t decode_bytes = (64ull << 10) / kFootprintDivisor;

  Arm arm;

  std::vector<void*> weights;
  weights.reserve(kLayers);
  for (std::size_t layer = 0; layer < kLayers; ++layer) {
    void* ptr = nullptr;
    if (!Success(arm.Allocate(&ptr, weight_bytes))) {
      for (void* held : weights) {
        arm.Deallocate(held);
      }
      perf::SkipBenchmark("allocator_matrix.LayerwiseInference",
                          "device cannot hold the model weights");
      return;
    }
    weights.push_back(ptr);
  }
  arm.Sync();

  std::vector<void*> kv_cache;
  kv_cache.reserve(kLayers * (decode_steps + 1));

  const auto before_upstream = arm.UpstreamAllocs();
  const auto start = std::chrono::steady_clock::now();

  void* previous = nullptr;
  for (std::size_t layer = 0; layer < kLayers; ++layer) {
    void* activation = nullptr;
    if (Success(arm.Allocate(&activation, activation_bytes))) {
      if (previous != nullptr) {
        arm.Deallocate(previous);
      }
      previous = activation;
    }
    void* kv = nullptr;

    if (Success(arm.Allocate(&kv, kv_step_bytes + layer * 512))) {
      kv_cache.push_back(kv);
    }
  }
  if (previous != nullptr) {
    arm.Deallocate(previous);
    previous = nullptr;
  }

  for (std::size_t step = 0; step < decode_steps; ++step) {
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
      void* activation = nullptr;
      if (Success(arm.Allocate(&activation, decode_bytes))) {
        arm.Deallocate(activation);
      }
      void* kv = nullptr;
      if (Success(arm.Allocate(&kv, kv_step_bytes + step * 256))) {
        kv_cache.push_back(kv);
      }
    }
  }
  arm.Sync();

  const auto end = std::chrono::steady_clock::now();
  const auto elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  const auto upstream = arm.UpstreamAllocs() - before_upstream;
  const auto reserved = arm.BytesReserved();
  const bool tracks = arm.TracksBytes();

  for (void* ptr : kv_cache) {
    arm.Deallocate(ptr);
  }
  for (void* ptr : weights) {
    arm.Deallocate(ptr);
  }
  arm.Sync();

  const std::string params = std::to_string(decode_steps) + " steps";
  RecordCell("Inference", params, Arm::kName, "ms", elapsed_ms);
  RecordCell("Inference/upstream", params, Arm::kName, "calls",
             static_cast<double>(upstream));
  RecordCell("Inference/reserved", params, Arm::kName, "B",
             static_cast<double>(reserved), tracks);

  const auto json_params =
      WithArm({perf::NumberParam("layers", kLayers),
               perf::NumberParam("decode_steps", decode_steps)},
              Arm::kName);
  perf::PrintResult("allocator_matrix.LayerwiseInference", json_params, 1, "ms",
                    elapsed_ms, elapsed_ms);
  perf::PrintResult("allocator_matrix.LayerwiseInferenceUpstreamCalls",
                    json_params, 1, "count", static_cast<double>(upstream),
                    static_cast<double>(upstream));
}

struct ConcurrentInferenceResult {
  double p50 = 0.0;
  double p99 = 0.0;
  double max = 0.0;
  double throughput = 0.0;
  std::size_t upstream_allocs = 0;
  std::size_t bytes_reserved = 0;
  bool tracks_bytes = false;
};

template <typename Arm>
ConcurrentInferenceResult BenchConcurrentInference(std::size_t threads,
                                                   std::size_t sequences,
                                                   std::size_t decode_steps) {
  constexpr std::size_t kLayers = 8;
  const std::size_t weight_bytes = (4ull << 20) / kFootprintDivisor;
  const std::size_t activation_bytes = (2ull << 20) / kFootprintDivisor;
  const std::size_t kv_step_bytes = (128ull << 10) / kFootprintDivisor;
  const std::size_t decode_bytes = (64ull << 10) / kFootprintDivisor;

  std::vector<runtime::Stream> handles(threads, runtime::Stream{});
  for (auto& stream : handles) {
    if (!Success(runtime::StreamCreate(&stream))) {
      for (auto& created : handles) {
        if (created != runtime::Stream{}) {
          runtime::StreamDestroy(created);
        }
      }
      perf::SkipBenchmark("allocator_matrix.ConcurrentInference",
                          "stream creation failed");
      return {};
    }
  }

  Arm arm;

  std::vector<std::vector<void*>> weights(threads);
  bool weights_ok = true;
  for (std::size_t t = 0; t < threads && weights_ok; ++t) {
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
      void* ptr = nullptr;
      if (!Success(arm.Allocate(&ptr, weight_bytes))) {
        weights_ok = false;
        break;
      }
      weights[t].push_back(ptr);
    }
  }
  if (!weights_ok) {
    for (const auto& held : weights) {
      for (void* ptr : held) {
        arm.Deallocate(ptr);
      }
    }
    for (auto& stream : handles) {
      runtime::StreamDestroy(stream);
    }
    perf::SkipBenchmark("allocator_matrix.ConcurrentInference",
                        "device cannot hold one model per thread");
    return {};
  }
  arm.Sync();

  std::vector<std::vector<double>> samples(threads);
  for (auto& per_thread : samples) {
    per_thread.reserve(sequences);
  }

  const auto before_upstream = arm.UpstreamAllocs();
  std::atomic<bool> go{false};
  std::vector<std::thread> workers;
  workers.reserve(threads);

  for (std::size_t t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      const runtime::Stream stream = handles[t];
      std::vector<void*> kv_cache;
      kv_cache.reserve(kLayers * (decode_steps + 1));

      while (!go.load(std::memory_order_acquire)) {
      }

      for (std::size_t sequence = 0; sequence < sequences; ++sequence) {
        const auto start = std::chrono::steady_clock::now();

        void* previous = nullptr;
        for (std::size_t layer = 0; layer < kLayers; ++layer) {
          void* activation = nullptr;
          if (Success(arm.Allocate(&activation, activation_bytes))) {
            runtime::MemsetAsync(activation, 0, activation_bytes, stream);
            if (previous != nullptr) {
              runtime::StreamSynchronize(stream);
              arm.Deallocate(previous);
            }
            previous = activation;
          }
          void* kv = nullptr;
          if (Success(arm.Allocate(&kv, kv_step_bytes + layer * 512))) {
            kv_cache.push_back(kv);
          }
        }
        if (previous != nullptr) {
          runtime::StreamSynchronize(stream);
          arm.Deallocate(previous);
        }

        for (std::size_t step = 0; step < decode_steps; ++step) {
          for (std::size_t layer = 0; layer < kLayers; ++layer) {
            void* activation = nullptr;
            if (Success(arm.Allocate(&activation, decode_bytes))) {
              runtime::MemsetAsync(activation, 0, decode_bytes, stream);
              runtime::StreamSynchronize(stream);
              arm.Deallocate(activation);
            }
            void* kv = nullptr;
            if (Success(arm.Allocate(&kv, kv_step_bytes + step * 256))) {
              kv_cache.push_back(kv);
            }
          }
        }

        runtime::StreamSynchronize(stream);
        for (void* ptr : kv_cache) {
          arm.Deallocate(ptr);
        }
        kv_cache.clear();

        const auto end = std::chrono::steady_clock::now();
        samples[t].push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
      }
    });
  }

  const auto wall_start = std::chrono::steady_clock::now();
  go.store(true, std::memory_order_release);
  for (auto& worker : workers) {
    worker.join();
  }
  const auto wall_end = std::chrono::steady_clock::now();
  arm.Sync();

  ConcurrentInferenceResult result;
  result.upstream_allocs = arm.UpstreamAllocs() - before_upstream;
  result.bytes_reserved = arm.BytesReserved();
  result.tracks_bytes = arm.TracksBytes();

  const double wall_seconds =
      std::chrono::duration<double>(wall_end - wall_start).count();
  result.throughput = wall_seconds > 0.0
                          ? static_cast<double>(threads * sequences) /
                                wall_seconds
                          : 0.0;

  std::vector<double> all;
  all.reserve(threads * sequences);
  for (const auto& per_thread : samples) {
    all.insert(all.end(), per_thread.begin(), per_thread.end());
  }
  if (!all.empty()) {
    std::sort(all.begin(), all.end());
    result.p50 = all[all.size() / 2];
    const auto p99_index =
        std::min(all.size() - 1, static_cast<std::size_t>(
                                     static_cast<double>(all.size()) * 0.99));
    result.p99 = all[p99_index];
    result.max = all.back();
  }

  for (const auto& held : weights) {
    for (void* ptr : held) {
      arm.Deallocate(ptr);
    }
  }
  arm.Sync();
  for (auto& stream : handles) {
    runtime::StreamDestroy(stream);
  }

  const auto params =
      WithArm({perf::NumberParam("threads", threads),
               perf::NumberParam("layers", kLayers),
               perf::NumberParam("decode_steps", decode_steps)},
              Arm::kName);
  const std::size_t total = threads * sequences;
  perf::PrintResult("allocator_matrix.ConcurrentInferenceP50", params, total,
                    "ms", result.p50, result.p50);
  perf::PrintResult("allocator_matrix.ConcurrentInferenceP99", params, total,
                    "ms", result.p99, result.p99);
  perf::PrintResult("allocator_matrix.ConcurrentInferenceThroughput", params,
                    total, "seq_per_s", result.throughput, result.throughput);
  perf::PrintResult("allocator_matrix.ConcurrentInferenceUpstreamCalls", params,
                    total, "count",
                    static_cast<double>(result.upstream_allocs),
                    static_cast<double>(result.upstream_allocs));

  const std::string label = std::to_string(threads) + "T";
  RecordCell("ConcInfer/p50", label, Arm::kName, "ms", result.p50);
  RecordCell("ConcInfer/p99", label, Arm::kName, "ms", result.p99);
  RecordCell("ConcInfer/max", label, Arm::kName, "ms", result.max);
  RecordCell("ConcInfer/upstream", label, Arm::kName, "calls",
             static_cast<double>(result.upstream_allocs));
  RecordCell("ConcInfer/reserved", label, Arm::kName, "B",
             static_cast<double>(result.bytes_reserved), result.tracks_bytes);
  return result;
}

template <typename Arm>
void BenchFragmentation(std::size_t operations) {
  constexpr std::size_t kSlots = 256;

  const std::size_t probe_bytes = (32ull << 20) / kFootprintDivisor;
  constexpr std::size_t kProbes = 8;

  struct Slot {
    void* ptr = nullptr;
    std::size_t expires_at = 0;
  };

  Arm arm;
  std::vector<Slot> slots(kSlots);

  std::uint64_t state = 0x2545f4914f6cdd1dull;
  auto next = [&state] {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<std::uint32_t>(state >> 33);
  };

  auto random_size = [&next] {
    const std::uint32_t decade = next() % 4;
    const std::size_t base = 1024ull << (4 * decade);
    return base + (next() % base);
  };

  std::size_t failures = 0;
  for (std::size_t op = 0; op < operations; ++op) {
    Slot& slot = slots[next() % kSlots];
    if (slot.ptr != nullptr) {
      if (slot.expires_at > op) {
        continue;
      }
      arm.Deallocate(slot.ptr);
      slot.ptr = nullptr;
    }
    void* ptr = nullptr;
    if (Success(arm.Allocate(&ptr, random_size()))) {
      slot.ptr = ptr;

      slot.expires_at = op + 1 + (next() % 4096);
    } else {
      ++failures;
    }
  }
  arm.Sync();

  const auto churn_reserved = arm.BytesReserved();
  const auto churn_upstream = arm.UpstreamAllocs();

  std::vector<void*> probes;
  probes.reserve(kProbes);
  for (std::size_t i = 0; i < kProbes; ++i) {
    void* ptr = nullptr;
    if (!Success(arm.Allocate(&ptr, probe_bytes)) || ptr == nullptr) {
      break;
    }
    probes.push_back(ptr);
  }
  arm.Sync();
  const double probe_rate =
      static_cast<double>(probes.size()) / static_cast<double>(kProbes);
  const auto probe_upstream = arm.UpstreamAllocs() - churn_upstream;

  for (void* ptr : probes) {
    arm.Deallocate(ptr);
  }
  for (Slot& slot : slots) {
    if (slot.ptr != nullptr) {
      arm.Deallocate(slot.ptr);
      slot.ptr = nullptr;
    }
  }
  arm.Sync();

  const std::string label = std::to_string(operations / 1000) + "k ops";
  RecordCell("Fragment/reserved", label, Arm::kName, "B",
             static_cast<double>(churn_reserved), arm.TracksBytes());
  RecordCell("Fragment/upstream", label, Arm::kName, "calls",
             static_cast<double>(churn_upstream));
  RecordCell("Fragment/fail", label, Arm::kName, "calls",
             static_cast<double>(failures));
  RecordCell("Fragment/probe ok", label, Arm::kName, "x", probe_rate);
  RecordCell("Fragment/probe up", label, Arm::kName, "calls",
             static_cast<double>(probe_upstream));

  const auto params =
      WithArm({perf::NumberParam("operations", operations),
               perf::NumberParam("probe_bytes", probe_bytes)},
              Arm::kName);
  perf::PrintResult("allocator_matrix.FragmentationReserved", params,
                    operations, "bytes", static_cast<double>(churn_reserved),
                    static_cast<double>(churn_reserved));
  perf::PrintResult("allocator_matrix.FragmentationProbeSuccess", params,
                    kProbes, "x", probe_rate, probe_rate);
  perf::PrintResult("allocator_matrix.FragmentationProbeUpstreamCalls", params,
                    kProbes, "count", static_cast<double>(probe_upstream),
                    static_cast<double>(probe_upstream));
}

template <typename Arm>
void BenchLatencyTail(std::size_t size, std::size_t operations) {
  Arm arm;
  std::vector<double> samples;
  samples.reserve(operations);

  for (std::size_t i = 0; i < 64; ++i) {
    void* ptr = nullptr;
    if (Success(arm.Allocate(&ptr, size))) {
      arm.Deallocate(ptr);
    }
  }
  arm.Sync();

  for (std::size_t op = 0; op < operations; ++op) {
    void* ptr = nullptr;
    const auto start = std::chrono::steady_clock::now();
    const auto status = arm.Allocate(&ptr, size);
    const auto end = std::chrono::steady_clock::now();
    if (Success(status)) {
      arm.Deallocate(ptr);
    }
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }
  arm.Sync();

  if (samples.empty()) {
    return;
  }
  std::sort(samples.begin(), samples.end());
  auto quantile = [&samples](double q) {
    const auto index = std::min(
        samples.size() - 1,
        static_cast<std::size_t>(static_cast<double>(samples.size()) * q));
    return samples[index];
  };

  const std::string label = DescribeSize(size);
  RecordCell("Latency/p50", label, Arm::kName, "us", quantile(0.50));
  RecordCell("Latency/p99", label, Arm::kName, "us", quantile(0.99));
  RecordCell("Latency/p999", label, Arm::kName, "us", quantile(0.999));
  RecordCell("Latency/max", label, Arm::kName, "us", samples.back());

  const auto params = WithArm(
      {perf::NumberParam("size_bytes", static_cast<std::uint64_t>(size))},
      Arm::kName);
  perf::PrintResult("allocator_matrix.LatencyP50", params, operations, "us",
                    quantile(0.50), quantile(0.50));
  perf::PrintResult("allocator_matrix.LatencyP99", params, operations, "us",
                    quantile(0.99), quantile(0.99));
  perf::PrintResult("allocator_matrix.LatencyP999", params, operations, "us",
                    quantile(0.999), quantile(0.999));
  perf::PrintResult("allocator_matrix.LatencyMax", params, operations, "us",
                    samples.back(), samples.back());
}

template <typename Arm>
void BenchBandwidth() {
  const std::size_t size = (64ull << 20) / kFootprintDivisor;
  constexpr std::size_t kIterations = 32;

  runtime::Stream stream{};
  if (!Success(runtime::StreamCreate(&stream))) {
    perf::SkipBenchmark("allocator_matrix.Bandwidth", "stream creation failed");
    return;
  }

  Arm arm;

  void* leading = nullptr;
  arm.Allocate(&leading, 4096);

  void* buffer = nullptr;
  if (!Success(arm.Allocate(&buffer, size)) || buffer == nullptr) {
    if (leading != nullptr) {
      arm.Deallocate(leading);
    }
    runtime::StreamDestroy(stream);
    perf::SkipBenchmark("allocator_matrix.Bandwidth",
                        "device cannot hold the bandwidth buffer");
    return;
  }

  runtime::MemsetAsync(buffer, 0, size, stream);
  runtime::StreamSynchronize(stream);

  std::vector<double> samples;
  samples.reserve(kIterations);
  for (std::size_t i = 0; i < kIterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    runtime::MemsetAsync(buffer, static_cast<int>(i & 0xff), size, stream);
    runtime::StreamSynchronize(stream);
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    if (seconds > 0.0) {
      samples.push_back(static_cast<double>(size) / seconds /
                        (1024.0 * 1024.0 * 1024.0));
    }
  }

  arm.Deallocate(buffer);
  if (leading != nullptr) {
    arm.Deallocate(leading);
  }
  arm.Sync();
  runtime::StreamDestroy(stream);

  if (samples.empty()) {
    return;
  }
  const double median = perf::Median(samples);
  RecordCell("Bandwidth", DescribeSize(size), Arm::kName, "GiB/s", median);
  perf::PrintResult(
      "allocator_matrix.Bandwidth",
      WithArm({perf::NumberParam("size_bytes",
                                 static_cast<std::uint64_t>(size))},
              Arm::kName),
      kIterations, "GiB/s", perf::Mean(samples), median);
}

void PrintMatrix() {
  if (g_cells.empty()) {
    return;
  }

  std::vector<std::pair<std::string, std::string>> rows;
  for (const Cell& cell : g_cells) {
    const auto key = std::make_pair(cell.workload, cell.params);
    if (std::find(rows.begin(), rows.end(), key) == rows.end()) {
      rows.push_back(key);
    }
  }

  std::cerr << "\n=== " << INFINI_RT_PERF_BACKEND_NAME
            << " allocator matrix (median; arena config: " << kConfigName
            << ") ===\n";
  std::cerr << std::left << std::setw(24) << "workload" << std::setw(14)
            << "params";
  for (const std::string& arm : g_arm_order) {
    std::cerr << std::right << std::setw(16) << arm;
  }
  std::cerr << std::right << std::setw(12) << "unit" << "\n";

  for (const auto& [workload, params] : rows) {
    std::cerr << std::left << std::setw(24) << workload << std::setw(14)
              << params;
    std::string unit;
    for (const std::string& arm : g_arm_order) {
      const auto found = std::find_if(
          g_cells.begin(), g_cells.end(), [&](const Cell& cell) {
            return cell.workload == workload && cell.params == params &&
                   cell.arm == arm;
          });
      if (found == g_cells.end()) {
        std::cerr << std::right << std::setw(16) << "-";
      } else if (!found->present) {
        std::cerr << std::right << std::setw(16) << "n/a";
        unit = found->unit;
      } else {
        std::cerr << std::right << std::setw(16) << std::fixed
                  << std::setprecision(found->unit == "calls" ? 0 : 2)
                  << found->value;
        unit = found->unit;
      }
    }
    std::cerr << std::right << std::setw(12) << unit << "\n";
  }

  std::cerr
      << "\nLower is better except `x` ratio rows. `calls` rows are exact\n"
         "counts, not timings, and are the cause behind the timing above them.\n"
         "`n/a` means the metric does not apply to that arm; `-` means the arm\n"
         "did not run that workload.\n"
         "cuda_async is stream-ordered: its `Deallocate` does not wait for\n"
         "pending device work, so it offers a weaker guarantee than the other\n"
         "three arms and its timings are not a drop-in speedup.\n";
  std::cerr << std::endl;
}

bool PrepareRuntime() {
  int device_count = 0;
  if (!Success(runtime::GetDeviceCount(&device_count)) || device_count <= 0) {
    std::cerr << "perf_allocator_matrix skipped: no available device."
              << std::endl;
    return false;
  }
  if (!Success(runtime::SetDevice(0))) {
    std::cerr << "perf_allocator_matrix skipped: device 0 is not available."
              << std::endl;
    return false;
  }
  return true;
}

template <template <typename> class Body, typename... Args>
void ForEachArm(bool with_async, Args&&... args) {
  Body<DirectArm>{}(args...);
  Body<ArenaArm>{}(args...);
  if (with_async) {
    Body<CudaAsyncArm>{}(args...);
  }
}

template <typename Arm>
struct RunCrossThreshold {
  void operator()(std::size_t iterations) const {
    BenchCrossThresholdRotation<Arm>(iterations);
  }
};

template <typename Arm>
struct RunLargeRecycle {
  void operator()(std::size_t iterations) const {
    BenchLargeBlockRecycle<Arm>(iterations);
  }
};

template <typename Arm>
struct RunHighWater {
  void operator()() const { BenchHighWaterGrowth<Arm>(); }
};

template <typename Arm>
struct RunTrimCost {
  void operator()(std::size_t cached) const { BenchTrimCost<Arm>(cached); }
};

template <typename Arm>
struct RunThreadScaling {
  void operator()(std::size_t threads, std::size_t ops) const {
    BenchThreadScaling<Arm>(threads, ops);
  }
};

template <typename Arm>
struct RunImplicitSync {
  void operator()() const { BenchImplicitSyncCost<Arm>(); }
};

template <typename Arm>
struct RunLedger {
  void operator()() const { MeasureLedgerAccuracy<Arm>(); }
};

template <typename Arm>
struct RunMultiStream {
  void operator()(std::size_t streams, std::size_t ops) const {
    BenchMultiStream<Arm>(streams, ops);
  }
};

template <typename Arm>
struct RunInference {
  void operator()(std::size_t steps) const {
    BenchLayerwiseInference<Arm>(steps);
  }
};

template <typename Arm>
struct RunConcurrentInference {
  void operator()(std::size_t threads, std::size_t sequences,
                  std::size_t steps) const {
    BenchConcurrentInference<Arm>(threads, sequences, steps);
  }
};

template <typename Arm>
struct RunFragmentation {
  void operator()(std::size_t operations) const {
    BenchFragmentation<Arm>(operations);
  }
};

template <typename Arm>
struct RunLatencyTail {
  void operator()(std::size_t size, std::size_t operations) const {
    BenchLatencyTail<Arm>(size, operations);
  }
};

template <typename Arm>
struct RunBandwidth {
  void operator()() const { BenchBandwidth<Arm>(); }
};

}

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--quick") {
      quick = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cerr << "usage: perf_allocator_matrix [--quick]\n"
                   "  --quick  fewer iterations and thread counts\n";
      return 0;
    }
  }

  if (!PrepareRuntime()) {
    return 0;
  }

  const bool with_async = CudaAsyncArm::Available();
  if (!with_async) {
    std::cerr << "cuda_async arm skipped: the backend does not provide a "
                 "stream-ordered allocator.\n";
  }

  const std::size_t rotations = quick ? 200u : 1000u;
  const std::size_t large_rotations = quick ? 60u : 200u;
  const std::size_t thread_ops = quick ? 5000u : 20000u;
  const std::size_t stream_ops = quick ? 200u : 1000u;
  const std::size_t decode_steps = quick ? 8u : 32u;

  ForEachArm<RunCrossThreshold>(with_async, rotations);
  ForEachArm<RunLargeRecycle>(with_async, large_rotations);
  ForEachArm<RunHighWater>(with_async);
  ForEachArm<RunTrimCost>(with_async, std::size_t{64});

  const std::vector<std::size_t> thread_counts =
      quick ? std::vector<std::size_t>{1, 8, 32}
            : std::vector<std::size_t>{1, 2, 4, 8, 16, 32, 64};
  for (const std::size_t threads : thread_counts) {
    ForEachArm<RunThreadScaling>(with_async, threads, thread_ops);
  }

  ForEachArm<RunImplicitSync>(with_async);
  ForEachArm<RunLedger>(with_async);

  const std::vector<std::size_t> stream_counts =
      quick ? std::vector<std::size_t>{2, 8}
            : std::vector<std::size_t>{2, 4, 8};
  for (const std::size_t streams : stream_counts) {
    ForEachArm<RunMultiStream>(with_async, streams, stream_ops);
  }

  ForEachArm<RunInference>(with_async, decode_steps);

  const std::size_t latency_ops = quick ? 20000u : 200000u;
  for (const std::size_t size : {64ull << 10, 256ull << 10}) {
    ForEachArm<RunLatencyTail>(with_async, static_cast<std::size_t>(size),
                               latency_ops);
  }

  ForEachArm<RunBandwidth>(with_async);

  const std::size_t sequences = quick ? 4u : 16u;
  const std::size_t conc_steps = quick ? 4u : 16u;
  const std::vector<std::size_t> inference_threads =
      quick ? std::vector<std::size_t>{1, 8}
            : std::vector<std::size_t>{1, 2, 4, 8, 16, 32};
  for (const std::size_t threads : inference_threads) {
    ForEachArm<RunConcurrentInference>(with_async, threads, sequences,
                                       conc_steps);
  }

  ForEachArm<RunFragmentation>(with_async, quick ? 50000u : 500000u);

  PrintMatrix();
  return 0;
}
