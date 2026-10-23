#include <infini/rt/arena_memory_pool.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "test_helper.h"

namespace {

using infini::rt::ArenaMemoryPool;

struct MockUpstream {
  using Error = int;
  static constexpr Error kSuccess = 0;
  static constexpr Error kFailure = 7;

  static std::atomic<std::size_t> malloc_calls;
  static std::atomic<std::size_t> free_calls;

  static std::size_t capacity_limit;

  static void Reset() {
    malloc_calls.store(0);
    free_calls.store(0);
    capacity_limit = 0;
  }

  static Error Malloc(void** ptr, std::size_t size) {
    malloc_calls.fetch_add(1, std::memory_order_relaxed);
    if (capacity_limit != 0 && size > capacity_limit) {
      *ptr = nullptr;
      return kFailure;
    }

    void* base = std::aligned_alloc(256, RoundUp(size, 256));
    if (base == nullptr) {
      *ptr = nullptr;
      return kFailure;
    }
    *ptr = base;
    return kSuccess;
  }

  static Error Free(void* ptr) {
    free_calls.fetch_add(1, std::memory_order_relaxed);
    std::free(ptr);
    return kSuccess;
  }

  static std::size_t RoundUp(std::size_t size, std::size_t granularity) {
    return (size + granularity - 1) / granularity * granularity;
  }
};

std::atomic<std::size_t> MockUpstream::malloc_calls{0};
std::atomic<std::size_t> MockUpstream::free_calls{0};
std::size_t MockUpstream::capacity_limit = 0;

struct TinyConfig {
  static constexpr std::size_t kInitialCapacity = 8 * 1024;
  static constexpr std::size_t kMaxCapacity = 64 * 1024;
  static constexpr std::size_t kSmallThreshold = 1024;
  static constexpr std::size_t kMinSliceAlignment = 64;
  static constexpr std::size_t kMinSplitRemainder = 64;
  static constexpr std::size_t kShrinkThreshold = 4;
  static constexpr std::uint32_t kEmptyScansToDestroy = 2;
};

using Pool = ArenaMemoryPool<MockUpstream, TinyConfig>;

struct NoShrinkConfig : TinyConfig {
  static constexpr std::size_t kShrinkThreshold =
      static_cast<std::size_t>(-1) / 2;
};

using StablePool = ArenaMemoryPool<MockUpstream, NoShrinkConfig>;

template <typename P>
void* Alloc(infini::rt::test::TestContext* context, P* pool, std::size_t size,
            std::size_t alignment = 0) {
  void* ptr = nullptr;
  context->ExpectEqual(pool->Allocate(&ptr, size, alignment),
                       MockUpstream::kSuccess, "allocate should succeed");
  context->Expect(ptr != nullptr, "allocate should produce a pointer");
  return ptr;
}

void Fill(void* ptr, std::size_t size, std::uint8_t seed) {
  auto* bytes = static_cast<std::uint8_t*>(ptr);
  for (std::size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<std::uint8_t>(seed + (i & 0x3f));
  }
}

bool Verify(const void* ptr, std::size_t size, std::uint8_t seed) {
  const auto* bytes = static_cast<const std::uint8_t*>(ptr);
  for (std::size_t i = 0; i < size; ++i) {
    if (bytes[i] != static_cast<std::uint8_t>(seed + (i & 0x3f))) {
      return false;
    }
  }
  return true;
}

void TestFirstAllocation(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  Pool pool;

  void* ptr = Alloc(context, &pool, 256);
  Fill(ptr, 256, 0x11);
  context->Expect(Verify(ptr, 256, 0x11), "slice should be writable memory");

  const Pool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.upstream_alloc_count, std::size_t{1},
                       "first allocation creates one backing store");
  context->ExpectEqual(stats.backing_count, std::size_t{1},
                       "one backing store live");
  context->Expect(stats.bytes_reserved >= TinyConfig::kInitialCapacity,
                  "reserved bytes cover the initial capacity");
  context->ExpectEqual(stats.bytes_in_use, std::size_t{256},
                       "one 256 B slice in use");
  context->ExpectEqual(stats.cache_miss_count, std::size_t{1},
                       "the first allocation misses");
  pool.Deallocate(ptr);
}

void TestDegenerateRequests(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  Pool pool;

  void* ptr = reinterpret_cast<void*>(std::uintptr_t{0xdeadbeef});
  context->ExpectEqual(pool.Allocate(&ptr, 0), MockUpstream::kSuccess,
                       "zero-byte allocate should succeed");
  context->ExpectEqual(ptr, static_cast<void*>(nullptr),
                       "zero-byte allocate yields nullptr");
  context->ExpectEqual(pool.Deallocate(nullptr), MockUpstream::kSuccess,
                       "deallocating nullptr is a no-op");
  context->Expect(pool.Allocate(nullptr, 64) != MockUpstream::kSuccess,
                  "a null out-pointer is an invalid value");
  context->ExpectEqual(MockUpstream::malloc_calls.load(), std::size_t{0},
                       "degenerate requests touch no upstream memory");
}

void TestForeignPointer(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  Pool pool;

  int stack_object = 0;
  context->Expect(pool.Deallocate(&stack_object) != MockUpstream::kSuccess,
                  "a foreign pointer must be rejected");

  void* ptr = Alloc(context, &pool, 128);
  context->ExpectEqual(pool.Deallocate(ptr), MockUpstream::kSuccess,
                       "first free should succeed");
  context->Expect(pool.Deallocate(ptr) != MockUpstream::kSuccess,
                  "a double free must be rejected");
}

void TestSlicingAvoidsUpstream(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  StablePool pool;

  std::vector<void*> blocks;
  for (std::size_t i = 0; i < 16; ++i) {
    void* ptr = Alloc(context, &pool, 256);
    Fill(ptr, 256, static_cast<std::uint8_t>(i + 1));
    blocks.push_back(ptr);
  }

  bool distinct = true;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    distinct = distinct && Verify(blocks[i], 256,
                                 static_cast<std::uint8_t>(i + 1));
  }
  context->Expect(distinct, "concurrent slices must not overlap");

  const StablePool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.upstream_alloc_count, std::size_t{1},
                       "16 slices of 256 B fit in one 8 KB backing store");
  context->ExpectEqual(stats.cache_hit_count, std::size_t{15},
                       "only the first allocation creates a backing");

  for (void* ptr : blocks) {
    pool.Deallocate(ptr);
  }
}

void TestAlignment(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  StablePool pool;

  void* filler = Alloc(context, &pool, 64);
  void* aligned = Alloc(context, &pool, 256, 1024);
  context->ExpectEqual(reinterpret_cast<std::uintptr_t>(aligned) % 1024,
                       std::uintptr_t{0},
                       "the returned pointer must honor the alignment");

  const StablePool::Stats before = pool.GetStats();
  pool.Deallocate(aligned);
  pool.Deallocate(filler);
  const StablePool::Stats after = pool.GetStats();

  context->ExpectEqual(after.bytes_in_use, std::size_t{0},
                       "everything has been returned");

  context->ExpectEqual(after.largest_free_chunk,
                       before.bytes_reserved - before.bytes_unusable,
                       "the whole backing coalesces back into one chunk");
}

void TestNaturalAlignment(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  StablePool pool;

  bool all_aligned = true;
  std::vector<void*> blocks;
  for (std::size_t size : {std::size_t{1}, std::size_t{7}, std::size_t{65},
                           std::size_t{130}, std::size_t{999}}) {
    void* ptr = Alloc(context, &pool, size);
    all_aligned = all_aligned &&
                  reinterpret_cast<std::uintptr_t>(ptr) %
                          TinyConfig::kMinSliceAlignment ==
                      0;
    blocks.push_back(ptr);
  }
  context->Expect(all_aligned,
                  "every slice honors the minimum slice alignment");
  for (void* ptr : blocks) {
    pool.Deallocate(ptr);
  }
}

void TestCoalescingRestoresContiguity(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  StablePool pool;

  std::vector<void*> blocks;
  for (std::size_t i = 0; i < 8; ++i) {
    blocks.push_back(Alloc(context, &pool, 512));
  }
  const std::size_t reserved = pool.GetStats().bytes_reserved;
  const std::size_t unusable = pool.GetStats().bytes_unusable;

  for (std::size_t i : {std::size_t{3}, std::size_t{0}, std::size_t{7},
                        std::size_t{1}, std::size_t{5}, std::size_t{2},
                        std::size_t{6}, std::size_t{4}}) {
    pool.Deallocate(blocks[i]);
  }

  const StablePool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.bytes_in_use, std::size_t{0},
                       "all blocks returned");
  context->ExpectEqual(stats.largest_free_chunk, reserved - unusable,
                       "the backing coalesces back into a single extent");

  void* whole = nullptr;
  context->ExpectEqual(pool.Allocate(&whole, reserved - unusable),
                       MockUpstream::kSuccess,
                       "the coalesced extent serves a full-span request");
  context->ExpectEqual(pool.GetStats().upstream_alloc_count, std::size_t{1},
                       "serving it needed no new backing store");
  pool.Deallocate(whole);
}

void TestReuseAcrossSizes(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  StablePool pool;

  void* big = Alloc(context, &pool, 2048);
  pool.Deallocate(big);

  std::vector<void*> blocks;
  for (std::size_t i = 0; i < 4; ++i) {
    blocks.push_back(Alloc(context, &pool, 512));
  }
  context->ExpectEqual(pool.GetStats().upstream_alloc_count, std::size_t{1},
                       "a freed 2 KB block serves four 512 B requests");
  for (void* ptr : blocks) {
    pool.Deallocate(ptr);
  }
}

void TestGrowthRamp(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  StablePool pool;

  std::vector<void*> blocks;
  std::vector<std::size_t> reserved_after;

  for (std::size_t i = 0; i < 64; ++i) {
    blocks.push_back(Alloc(context, &pool, 2048));
    reserved_after.push_back(pool.GetStats().bytes_reserved);
  }

  const StablePool::Stats stats = pool.GetStats();
  context->Expect(stats.backing_count > 1, "growth added backing stores");
  context->ExpectEqual(stats.upstream_alloc_count, stats.backing_count,
                       "one upstream call per backing store");

  context->Expect(stats.upstream_alloc_count <= 8,
                  "the doubling ramp keeps upstream calls sublinear");

  bool capped = true;
  for (std::size_t i = 1; i < reserved_after.size(); ++i) {
    const std::size_t step = reserved_after[i] - reserved_after[i - 1];
    capped = capped && step <= NoShrinkConfig::kMaxCapacity;
  }
  context->Expect(capped, "no backing store exceeds the capacity cap");

  for (void* ptr : blocks) {
    pool.Deallocate(ptr);
  }
}

void TestOversizeRequest(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  StablePool pool;

  constexpr std::size_t kHuge = NoShrinkConfig::kMaxCapacity * 3;
  void* ptr = Alloc(context, &pool, kHuge);
  Fill(ptr, kHuge, 0x5a);
  context->Expect(Verify(ptr, kHuge, 0x5a),
                  "an oversize slice is fully writable");

  const StablePool::Stats stats = pool.GetStats();
  context->Expect(stats.bytes_reserved >= kHuge,
                  "the oversize backing covers the request");
  context->ExpectEqual(stats.bytes_in_use, kHuge, "the whole request is live");

  pool.Deallocate(ptr);
  const std::size_t before = pool.GetStats().bytes_reserved;
  std::vector<void*> blocks;
  for (std::size_t i = 0; i < 200; ++i) {
    blocks.push_back(Alloc(context, &pool, 512));
  }
  const std::size_t grew = pool.GetStats().bytes_reserved - before;
  context->Expect(grew <= NoShrinkConfig::kMaxCapacity,
                  "an oversize backing must not advance the growth ramp");
  for (void* ptr2 : blocks) {
    pool.Deallocate(ptr2);
  }
}

void TestShrinkAfterBurst(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  Pool pool;

  void* resident_block = Alloc(context, &pool, 128);

  std::vector<void*> burst;
  for (std::size_t i = 0; i < 24; ++i) {
    burst.push_back(Alloc(context, &pool, 4096));
  }
  const Pool::Stats peak = pool.GetStats();
  context->Expect(peak.backing_count > 1, "the burst added backing stores");

  for (void* ptr : burst) {
    pool.Deallocate(ptr);
  }
  context->ExpectEqual(pool.GetStats().backing_count, peak.backing_count,
                       "freeing alone does not release backings upstream");

  std::vector<void*> small;
  for (std::size_t i = 0;
       i < TinyConfig::kShrinkThreshold * (TinyConfig::kEmptyScansToDestroy + 2);
       ++i) {
    small.push_back(Alloc(context, &pool, 64));
  }

  const Pool::Stats after = pool.GetStats();
  context->ExpectEqual(after.backing_count, std::size_t{1},
                       "idle backings are destroyed automatically");
  context->Expect(after.shrink_count > 0, "shrink was recorded");
  context->ExpectEqual(after.upstream_free_count, after.shrink_count,
                       "every shrink is one upstream free");
  context->Expect(after.bytes_reserved <= TinyConfig::kInitialCapacity,
                  "reserved memory falls back to the resident backing");

  context->ExpectEqual(pool.Deallocate(resident_block), MockUpstream::kSuccess,
                       "the resident backing survived the shrink");
  for (void* ptr : small) {
    pool.Deallocate(ptr);
  }
}

void TestShrinkSpareLiveBackings(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  Pool pool;

  void* pinned = nullptr;
  std::vector<void*> burst;
  for (std::size_t i = 0; i < 24; ++i) {
    void* ptr = Alloc(context, &pool, 4096);
    if (pool.GetStats().backing_count > 1 && pinned == nullptr) {
      pinned = ptr;
      Fill(pinned, 4096, 0x33);
    } else {
      burst.push_back(ptr);
    }
  }
  context->Expect(pinned != nullptr, "the burst reached a second backing");
  for (void* ptr : burst) {
    pool.Deallocate(ptr);
  }

  std::vector<void*> small;
  for (std::size_t i = 0; i < TinyConfig::kShrinkThreshold * 8; ++i) {
    small.push_back(Alloc(context, &pool, 64));
  }

  context->Expect(pool.GetStats().backing_count >= 2,
                  "a backing holding a live block is never destroyed");
  context->Expect(Verify(pinned, 4096, 0x33),
                  "the live block's contents are intact");

  pool.Deallocate(pinned);
  for (void* ptr : small) {
    pool.Deallocate(ptr);
  }
}

void TestLargeRequestResetsShrinkRun(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  Pool pool;

  std::vector<void*> live;
  live.push_back(Alloc(context, &pool, 64));
  for (std::size_t i = 0; i < 24; ++i) {
    live.push_back(Alloc(context, &pool, 4096));
  }
  const std::size_t peak_backings = pool.GetStats().backing_count;

  for (std::size_t round = 0; round < 12; ++round) {
    for (std::size_t i = 0; i < TinyConfig::kShrinkThreshold - 1; ++i) {
      void* ptr = Alloc(context, &pool, 64);
      pool.Deallocate(ptr);
    }
    void* big = Alloc(context, &pool, 4096);
    pool.Deallocate(big);
  }

  context->ExpectEqual(pool.GetStats().shrink_count, std::size_t{0},
                       "an interleaved large request keeps the arena warm");
  context->ExpectEqual(pool.GetStats().backing_count, peak_backings,
                       "no backing was destroyed mid-burst");
  for (void* ptr : live) {
    pool.Deallocate(ptr);
  }
}

void TestShrinkHysteresis(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  Pool pool;

  void* anchor = Alloc(context, &pool, 64);
  std::vector<void*> burst;
  for (std::size_t i = 0; i < 24; ++i) {
    burst.push_back(Alloc(context, &pool, 4096));
  }
  for (void* ptr : burst) {
    pool.Deallocate(ptr);
  }

  constexpr std::size_t kRounds = 40;
  for (std::size_t round = 0; round < kRounds; ++round) {
    void* big = Alloc(context, &pool, 4096);
    for (std::size_t i = 0; i < 6; ++i) {
      void* ptr = Alloc(context, &pool, 64);
      pool.Deallocate(ptr);
    }
    pool.Deallocate(big);
  }

  const Pool::Stats stats = pool.GetStats();

  context->Expect(stats.shrink_count < kRounds / 2,
                  "hysteresis prevents per-iteration backing thrash");
  context->Expect(stats.upstream_alloc_count < kRounds,
                  "upstream calls do not grow linearly with iterations");
  pool.Deallocate(anchor);
}

void TestOversizeShrinksPromptly(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  Pool pool;

  void* anchor = Alloc(context, &pool, 64);
  void* huge = Alloc(context, &pool, TinyConfig::kMaxCapacity * 3);
  const std::size_t peak_reserved = pool.GetStats().bytes_reserved;
  pool.Deallocate(huge);

  std::vector<void*> small;
  for (std::size_t i = 0; i < TinyConfig::kShrinkThreshold; ++i) {
    small.push_back(Alloc(context, &pool, 64));
  }

  const Pool::Stats stats = pool.GetStats();
  context->Expect(stats.bytes_reserved < peak_reserved,
                  "the oversize backing goes back on the first idle scan");
  context->Expect(stats.shrink_count >= 1, "shrink was recorded");

  pool.Deallocate(anchor);
  for (void* ptr : small) {
    pool.Deallocate(ptr);
  }
}

void TestReleaseCached(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  StablePool pool;

  std::vector<void*> blocks;
  for (std::size_t i = 0; i < 24; ++i) {
    blocks.push_back(Alloc(context, &pool, 4096));
  }
  void* keep = blocks.front();
  for (std::size_t i = 1; i < blocks.size(); ++i) {
    pool.Deallocate(blocks[i]);
  }

  pool.ReleaseCached();
  StablePool::Stats stats = pool.GetStats();
  context->Expect(stats.backing_count >= 1,
                  "the backing holding a live block is retained");
  context->Expect(stats.upstream_free_count > 0,
                  "drained backings were released");

  pool.Deallocate(keep);
  pool.ReleaseCached();
  stats = pool.GetStats();
  context->ExpectEqual(stats.backing_count, std::size_t{0},
                       "an empty pool releases everything");
  context->ExpectEqual(stats.bytes_reserved, std::size_t{0},
                       "reserved bytes drop to zero");
  context->ExpectEqual(stats.upstream_free_count, stats.upstream_alloc_count,
                       "every backing store was freed exactly once");

  void* fresh = Alloc(context, &pool, 128);
  context->ExpectEqual(pool.GetStats().bytes_reserved,
                       std::size_t{NoShrinkConfig::kInitialCapacity},
                       "the growth ramp restarts after a full release");
  pool.Deallocate(fresh);
}

void TestStatsBalance(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  StablePool pool;

  std::mt19937 rng(1234);
  std::uniform_int_distribution<std::size_t> size_dist(1, 4096);
  std::vector<void*> live;
  bool balanced = true;
  bool waste_bounded = true;

  for (std::size_t step = 0; step < 2000; ++step) {
    if (live.empty() || (rng() & 1) != 0) {
      live.push_back(Alloc(context, &pool, size_dist(rng)));
    } else {
      const std::size_t index = rng() % live.size();
      pool.Deallocate(live[index]);
      live.erase(live.begin() + static_cast<std::ptrdiff_t>(index));
    }

    const StablePool::Stats stats = pool.GetStats();
    balanced = balanced &&
               stats.bytes_reserved == stats.bytes_in_use +
                                           stats.bytes_free_in_backings +
                                           stats.bytes_unusable;
    waste_bounded = waste_bounded && stats.bytes_internal_waste <=
                                         stats.bytes_in_use;
  }

  context->Expect(balanced,
                  "reserved == in_use + free_in_backings + unusable, always");
  context->Expect(waste_bounded,
                  "internal waste is a subset of the bytes in use");

  for (void* ptr : live) {
    pool.Deallocate(ptr);
  }
  const StablePool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.bytes_in_use, std::size_t{0},
                       "everything returned");
  context->ExpectEqual(stats.bytes_internal_waste, std::size_t{0},
                       "internal waste clears with the last live chunk");
  context->ExpectEqual(stats.alloc_count, stats.free_count,
                       "every allocation was matched by a free");
}

void TestFragmentationUnderChurn(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  StablePool pool;

  std::mt19937 rng(98765);
  std::uniform_int_distribution<std::size_t> size_dist(64, 8192);
  std::vector<void*> live;
  constexpr std::size_t kSteps = 20000;

  for (std::size_t step = 0; step < kSteps; ++step) {
    if (live.size() < 48 && ((rng() & 3) != 0 || live.empty())) {
      void* ptr = nullptr;
      if (pool.Allocate(&ptr, size_dist(rng)) != MockUpstream::kSuccess) {
        context->Expect(false, "churn must not fail to allocate");
        break;
      }
      live.push_back(ptr);
    } else {
      const std::size_t index = rng() % live.size();
      pool.Deallocate(live[index]);
      live.erase(live.begin() + static_cast<std::ptrdiff_t>(index));
    }
  }

  const StablePool::Stats stats = pool.GetStats();

  context->Expect(stats.upstream_alloc_count < kSteps / 100,
                  "churn does not drive an upstream call per allocation");

  context->Expect(stats.bytes_reserved < 4 * 1024 * 1024,
                  "memory amplification stays bounded under churn");

  for (void* ptr : live) {
    pool.Deallocate(ptr);
  }

  pool.ReleaseCached();
  const StablePool::Stats drained = pool.GetStats();
  context->ExpectEqual(drained.bytes_reserved, std::size_t{0},
                       "a fully drained pool releases every backing");
  context->ExpectEqual(drained.upstream_free_count,
                       drained.upstream_alloc_count,
                       "no backing store leaked across the churn");
}

void TestOomFallbackShrinksRequest(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();

  MockUpstream::capacity_limit = 4096;

  StablePool pool;
  void* ptr = nullptr;
  context->ExpectEqual(pool.Allocate(&ptr, 1024), MockUpstream::kSuccess,
                       "the pool falls back to a capacity upstream can serve");
  context->Expect(ptr != nullptr, "fallback still yields a pointer");
  context->Expect(pool.GetStats().bytes_reserved <= 4096,
                  "the fallback backing fits within the upstream limit");
  pool.Deallocate(ptr);
  MockUpstream::capacity_limit = 0;
}

void TestOomFailurePropagates(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  MockUpstream::capacity_limit = 1024;

  StablePool pool;
  void* ptr = reinterpret_cast<void*>(std::uintptr_t{0xabcd});
  context->Expect(pool.Allocate(&ptr, 64 * 1024) != MockUpstream::kSuccess,
                  "an unservable request returns the upstream error");
  context->ExpectEqual(ptr, static_cast<void*>(nullptr),
                       "a failed allocation yields nullptr");

  MockUpstream::capacity_limit = 0;
  void* recovered = Alloc(context, &pool, 512);
  context->Expect(recovered != nullptr,
                  "the pool still works after an upstream failure");
  pool.Deallocate(recovered);
}

void TestConcurrentTraffic(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  StablePool pool;

  constexpr std::size_t kThreads = 8;
  constexpr std::size_t kOpsPerThread = 4000;
  std::atomic<std::size_t> corruption{0};
  std::atomic<std::size_t> failures{0};

  std::vector<std::thread> threads;
  for (std::size_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool, &corruption, &failures, t] {
      std::mt19937 rng(static_cast<std::uint32_t>(t * 7919 + 13));
      std::uniform_int_distribution<std::size_t> size_dist(64, 4096);
      const auto seed = static_cast<std::uint8_t>(t * 31 + 1);
      std::vector<std::pair<void*, std::size_t>> live;

      for (std::size_t op = 0; op < kOpsPerThread; ++op) {
        if (live.size() < 12 && ((rng() & 3) != 0 || live.empty())) {
          const std::size_t size = size_dist(rng);
          void* ptr = nullptr;
          if (pool.Allocate(&ptr, size) != MockUpstream::kSuccess ||
              ptr == nullptr) {
            failures.fetch_add(1, std::memory_order_relaxed);
            continue;
          }
          Fill(ptr, size, seed);
          live.emplace_back(ptr, size);
        } else {
          const std::size_t index = rng() % live.size();
          const auto [ptr, size] = live[index];
          if (!Verify(ptr, size, seed)) {
            corruption.fetch_add(1, std::memory_order_relaxed);
          }
          pool.Deallocate(ptr);
          live.erase(live.begin() + static_cast<std::ptrdiff_t>(index));
        }
      }

      for (const auto& [ptr, size] : live) {
        if (!Verify(ptr, size, seed)) {
          corruption.fetch_add(1, std::memory_order_relaxed);
        }
        pool.Deallocate(ptr);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  context->ExpectEqual(corruption.load(), std::size_t{0},
                       "concurrent slices must never alias");
  context->ExpectEqual(failures.load(), std::size_t{0},
                       "concurrent allocation must not fail");

  const StablePool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.bytes_in_use, std::size_t{0},
                       "all threads returned their blocks");
  context->ExpectEqual(stats.alloc_count, stats.free_count,
                       "allocation and free counts agree across threads");
  pool.ReleaseCached();
  context->ExpectEqual(pool.GetStats().upstream_free_count,
                       pool.GetStats().upstream_alloc_count,
                       "no backing store leaked under concurrency");
}

void TestDestructorFreesBackingsOnce(infini::rt::test::TestContext* context) {
  MockUpstream::Reset();
  std::size_t allocs = 0;
  {
    StablePool pool;
    std::vector<void*> blocks;
    for (std::size_t i = 0; i < 32; ++i) {
      blocks.push_back(Alloc(context, &pool, 2048));
    }

    for (std::size_t i = 0; i < blocks.size(); i += 2) {
      pool.Deallocate(blocks[i]);
    }
    allocs = pool.GetStats().upstream_alloc_count;
    context->Expect(allocs > 1, "the test spans several backing stores");
  }

  context->ExpectEqual(MockUpstream::free_calls.load(), allocs,
                       "destruction frees each backing store exactly once");
  context->ExpectEqual(MockUpstream::malloc_calls.load(),
                       MockUpstream::free_calls.load(),
                       "no upstream allocation leaked");
}

}

int main() {
  infini::rt::test::TestContext context;

  TestFirstAllocation(&context);
  TestDegenerateRequests(&context);
  TestForeignPointer(&context);
  TestSlicingAvoidsUpstream(&context);
  TestAlignment(&context);
  TestNaturalAlignment(&context);

  TestCoalescingRestoresContiguity(&context);
  TestReuseAcrossSizes(&context);

  TestGrowthRamp(&context);
  TestOversizeRequest(&context);

  TestShrinkAfterBurst(&context);
  TestShrinkSpareLiveBackings(&context);
  TestLargeRequestResetsShrinkRun(&context);
  TestShrinkHysteresis(&context);
  TestOversizeShrinksPromptly(&context);

  TestReleaseCached(&context);
  TestStatsBalance(&context);
  TestFragmentationUnderChurn(&context);
  TestOomFallbackShrinksRequest(&context);
  TestOomFailurePropagates(&context);
  TestConcurrentTraffic(&context);
  TestDestructorFreesBackingsOnce(&context);

  return context.ExitCode();
}
