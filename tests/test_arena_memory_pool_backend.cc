#include <infini/rt.h>
#include <infini/rt/arena_memory_pool.h>
#include INFINI_RT_TEST_RUNTIME_HEADER

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "test_helper.h"

namespace {

using Runtime = infini::rt::runtime::Runtime<INFINI_RT_TEST_DEVICE_TYPE>;

struct BackendConfig {
  static constexpr std::size_t kInitialCapacity = 4ull << 20;
  static constexpr std::size_t kMaxCapacity = 32ull << 20;
  static constexpr std::size_t kSmallThreshold = 64ull << 10;
  static constexpr std::size_t kMinSliceAlignment = 512;
  static constexpr std::size_t kMinSplitRemainder = 512;
  static constexpr std::size_t kShrinkThreshold = 8;
  static constexpr std::uint32_t kEmptyScansToDestroy = 2;
};

using Pool = infini::rt::ArenaMemoryPool<Runtime, BackendConfig>;

constexpr const char* kBackend = INFINI_RT_TEST_BACKEND_NAME;

bool SelectDevice() {
  int device_count = 0;
  if (Runtime::GetDeviceCount(&device_count) != Runtime::kSuccess ||
      device_count <= 0) {
    std::cout << kBackend << " arena memory pool skipped: no available device."
              << std::endl;
    return false;
  }
  if (Runtime::SetDevice(0) != Runtime::kSuccess) {
    std::cout << kBackend << " arena memory pool skipped: device 0 unavailable."
              << std::endl;
    return false;
  }
  return true;
}

template <std::size_t N>
void ExpectUsable(infini::rt::test::TestContext* context, void* ptr,
                  const std::array<std::uint8_t, N>& input,
                  const char* message) {
  if (!context->Expect(ptr != nullptr, message)) {
    return;
  }
  std::array<std::uint8_t, N> output{};
  context->ExpectEqual(
      Runtime::Memcpy(ptr, input.data(), N, Runtime::kMemcpyHostToDevice),
      Runtime::kSuccess, "memcpy host-to-device should succeed");
  context->ExpectEqual(
      Runtime::Memcpy(output.data(), ptr, N, Runtime::kMemcpyDeviceToHost),
      Runtime::kSuccess, "memcpy device-to-host should succeed");
  context->ExpectEqual(output, input,
                       "pool-allocated memory should round-trip bytes");
}

bool FillDevice(void* ptr, std::size_t size, std::uint8_t seed) {
  std::vector<std::uint8_t> host(size);
  for (std::size_t i = 0; i < size; ++i) {
    host[i] = static_cast<std::uint8_t>(seed + (i & 0x3f));
  }
  return Runtime::Memcpy(ptr, host.data(), size,
                         Runtime::kMemcpyHostToDevice) == Runtime::kSuccess;
}

bool VerifyDevice(void* ptr, std::size_t size, std::uint8_t seed) {
  std::vector<std::uint8_t> host(size, 0);
  if (Runtime::Memcpy(host.data(), ptr, size, Runtime::kMemcpyDeviceToHost) !=
      Runtime::kSuccess) {
    return false;
  }
  for (std::size_t i = 0; i < size; ++i) {
    if (host[i] != static_cast<std::uint8_t>(seed + (i & 0x3f))) {
      return false;
    }
  }
  return true;
}

void TestSliceIsUsableDeviceMemory(infini::rt::test::TestContext* context) {
  Pool pool;
  void* ptr = nullptr;
  context->ExpectEqual(pool.Allocate(&ptr, 256), Runtime::kSuccess,
                       "allocate should succeed on a real backend");
  const std::array<std::uint8_t, 8> input{0, 1, 2, 3, 4, 5, 6, 7};
  ExpectUsable(context, ptr, input, "allocation should produce a pointer");

  const Pool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.upstream_alloc_count, std::size_t{1},
                       "one device allocation backs the slice");
  context->Expect(stats.bytes_reserved >= BackendConfig::kInitialCapacity,
                  "the backing covers the configured initial capacity");
  context->ExpectEqual(pool.Deallocate(ptr), Runtime::kSuccess,
                       "deallocate should succeed");
}

void TestSlicesAreDisjointOnDevice(infini::rt::test::TestContext* context) {
  Pool pool;
  constexpr std::size_t kSlices = 24;
  constexpr std::size_t kSize = 8192;

  std::vector<void*> blocks;
  for (std::size_t i = 0; i < kSlices; ++i) {
    void* ptr = nullptr;
    if (pool.Allocate(&ptr, kSize) != Runtime::kSuccess || ptr == nullptr) {
      context->Expect(false, "slicing allocation should succeed");
      break;
    }
    if (!FillDevice(ptr, kSize, static_cast<std::uint8_t>(i * 7 + 1))) {
      context->Expect(false, "writing a slice should succeed");
    }
    blocks.push_back(ptr);
  }

  bool disjoint = true;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    disjoint = disjoint &&
               VerifyDevice(blocks[i], kSize,
                            static_cast<std::uint8_t>(i * 7 + 1));
  }
  context->Expect(disjoint, "slices out of one backing must not overlap");

  const Pool::Stats stats = pool.GetStats();

  context->ExpectEqual(stats.upstream_alloc_count, std::size_t{1},
                       "24 slices need only one device allocation");
  context->ExpectEqual(stats.cache_hit_count, kSlices - 1,
                       "only the first allocation creates a backing");

  for (void* ptr : blocks) {
    pool.Deallocate(ptr);
  }
}

void TestCoalescedSpanIsUsable(infini::rt::test::TestContext* context) {
  Pool pool;
  std::vector<void*> blocks;
  for (std::size_t i = 0; i < 8; ++i) {
    void* ptr = nullptr;
    pool.Allocate(&ptr, 64 * 1024);
    blocks.push_back(ptr);
  }
  for (void* ptr : blocks) {
    pool.Deallocate(ptr);
  }

  const std::size_t span = pool.GetStats().largest_free_chunk /
                           BackendConfig::kMinSliceAlignment *
                           BackendConfig::kMinSliceAlignment;
  context->Expect(span >= 8 * 64 * 1024,
                  "the released slices coalesce into one extent");

  void* whole = nullptr;
  context->ExpectEqual(pool.Allocate(&whole, span), Runtime::kSuccess,
                       "the coalesced extent serves a full-span request");
  context->ExpectEqual(pool.GetStats().upstream_alloc_count, std::size_t{1},
                       "serving it needed no new device allocation");

  const std::array<std::uint8_t, 8> input{2, 4, 6, 8, 10, 12, 14, 16};
  ExpectUsable(context, whole, input, "the span's head is usable");
  ExpectUsable(context, static_cast<char*>(whole) + span - 8, input,
               "the span's tail is usable");
  pool.Deallocate(whole);
}

void TestAlignment(infini::rt::test::TestContext* context) {
  Pool pool;
  constexpr std::size_t kAlignment = 4096;

  void* filler = nullptr;
  pool.Allocate(&filler, 512);

  void* ptr = nullptr;
  context->ExpectEqual(pool.Allocate(&ptr, 1024, kAlignment), Runtime::kSuccess,
                       "aligned allocate should succeed");
  context->ExpectEqual(reinterpret_cast<std::uintptr_t>(ptr) % kAlignment,
                       std::uintptr_t{0},
                       "returned pointer should honor the alignment");
  const std::array<std::uint8_t, 8> input{1, 1, 2, 3, 5, 8, 13, 21};
  ExpectUsable(context, ptr, input, "aligned slice should be usable");
  pool.Deallocate(ptr);
  pool.Deallocate(filler);
}

void TestOversizeRequest(infini::rt::test::TestContext* context) {
  Pool pool;
  constexpr std::size_t kHuge = BackendConfig::kMaxCapacity + (4ull << 20);
  void* ptr = nullptr;
  if (pool.Allocate(&ptr, kHuge) != Runtime::kSuccess || ptr == nullptr) {
    std::cout << kBackend
              << " oversize case skipped: device cannot serve the request."
              << std::endl;
    return;
  }

  const Pool::Stats stats = pool.GetStats();
  context->Expect(stats.bytes_reserved >= kHuge,
                  "the oversize backing covers the request");

  context->Expect(stats.bytes_in_use >= kHuge,
                  "the whole request is live");
  context->Expect(
      stats.bytes_in_use - kHuge < BackendConfig::kMinSplitRemainder +
                                       BackendConfig::kMinSliceAlignment,
      "an oversize backing wastes at most a sliver on the served chunk");
  const std::array<std::uint8_t, 8> input{9, 9, 8, 8, 7, 7, 6, 6};
  ExpectUsable(context, ptr, input, "the oversize slice's head is usable");
  ExpectUsable(context, static_cast<char*>(ptr) + kHuge - 8, input,
               "the oversize slice's tail is usable");
  pool.Deallocate(ptr);
}

void TestShrinkAfterBurst(infini::rt::test::TestContext* context) {
  Pool pool;
  void* resident = nullptr;
  pool.Allocate(&resident, 1024);

  std::vector<void*> burst;
  for (std::size_t i = 0; i < 24; ++i) {
    void* ptr = nullptr;
    if (pool.Allocate(&ptr, 1ull << 20) != Runtime::kSuccess) {
      break;
    }
    burst.push_back(ptr);
  }
  const std::size_t peak_backings = pool.GetStats().backing_count;
  if (peak_backings <= 1) {
    std::cout << kBackend
              << " shrink case skipped: the burst stayed in one backing."
              << std::endl;
    pool.Deallocate(resident);
    for (void* ptr : burst) {
      pool.Deallocate(ptr);
    }
    return;
  }

  for (void* ptr : burst) {
    pool.Deallocate(ptr);
  }
  context->ExpectEqual(pool.GetStats().backing_count, peak_backings,
                       "freeing alone does not release device memory");

  std::vector<void*> small;
  for (std::size_t i = 0;
       i < BackendConfig::kShrinkThreshold *
               (BackendConfig::kEmptyScansToDestroy + 2);
       ++i) {
    void* ptr = nullptr;
    pool.Allocate(&ptr, 1024);
    small.push_back(ptr);
  }

  const Pool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.backing_count, std::size_t{1},
                       "idle backings are returned to the device");
  context->Expect(stats.shrink_count > 0, "shrink was recorded");
  context->ExpectEqual(stats.upstream_free_count, stats.shrink_count,
                       "every shrink is one device free");

  const std::array<std::uint8_t, 8> input{3, 1, 4, 1, 5, 9, 2, 6};
  ExpectUsable(context, resident, input,
               "the resident slice outlived the shrink");
  pool.Deallocate(resident);
  for (void* ptr : small) {
    pool.Deallocate(ptr);
  }
}

void TestReleaseCached(infini::rt::test::TestContext* context) {
  Pool pool;
  std::vector<void*> blocks;
  for (std::size_t i = 0; i < 8; ++i) {
    void* ptr = nullptr;
    pool.Allocate(&ptr, 128 * 1024);
    blocks.push_back(ptr);
  }
  for (void* ptr : blocks) {
    pool.Deallocate(ptr);
  }

  pool.ReleaseCached();
  Pool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.backing_count, std::size_t{0},
                       "an empty pool releases every backing");
  context->ExpectEqual(stats.bytes_reserved, std::size_t{0},
                       "reserved bytes drop to zero");
  context->ExpectEqual(stats.upstream_free_count, stats.upstream_alloc_count,
                       "every device allocation was freed exactly once");

  void* fresh = nullptr;
  context->ExpectEqual(pool.Allocate(&fresh, 4096), Runtime::kSuccess,
                       "the pool still works after a full release");
  const std::array<std::uint8_t, 4> input{7, 7, 7, 7};
  ExpectUsable(context, fresh, input, "a fresh slice is usable");
  pool.Deallocate(fresh);
}

void TestDefaultConfigInstantiates(infini::rt::test::TestContext* context) {
  infini::rt::ArenaMemoryPool<Runtime> pool;
  void* ptr = nullptr;
  if (pool.Allocate(&ptr, 4096) != Runtime::kSuccess || ptr == nullptr) {
    std::cout << kBackend
              << " default-config case skipped: device cannot reserve "
              << (infini::rt::DefaultArenaConfig::kInitialCapacity >> 20)
              << " MB." << std::endl;
    return;
  }
  const std::array<std::uint8_t, 8> input{1, 2, 3, 4, 5, 6, 7, 8};
  ExpectUsable(context, ptr, input, "a default-config slice is usable");
  context->ExpectEqual(pool.Deallocate(ptr), Runtime::kSuccess,
                       "deallocate should succeed");
}

}

int main() {
  infini::rt::test::TestContext context;

  if (!SelectDevice()) {
    return context.ExitCode();
  }

  TestSliceIsUsableDeviceMemory(&context);
  TestSlicesAreDisjointOnDevice(&context);
  TestCoalescedSpanIsUsable(&context);
  TestAlignment(&context);
  TestOversizeRequest(&context);
  TestShrinkAfterBurst(&context);
  TestReleaseCached(&context);
  TestDefaultConfigInstantiates(&context);

  return context.ExitCode();
}
