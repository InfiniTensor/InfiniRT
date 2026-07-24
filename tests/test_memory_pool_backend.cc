// Exercises `MemoryPool` over a *real* runtime backend (CPU, NVIDIA, ...).
//
// `test_memory_pool.cc` already covers the pool's bookkeeping against a mock
// upstream. This test instead instantiates the pool over the backend's actual
// `runtime::Runtime` specialization, so it proves the two compose correctly and
// that pool-handed pointers are genuine device memory. Device pointers cannot
// be dereferenced from the host, so usability is checked through `Memcpy`
// round-trips. The whole suite is skipped when no device is present.
#include <infini/rt.h>
#include <infini/rt/memory_pool.h>
#include INFINI_RT_TEST_RUNTIME_HEADER

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include "test_helper.h"

namespace {

using Runtime = infini::rt::runtime::Runtime<INFINI_RT_TEST_DEVICE_TYPE>;
using Pool = infini::rt::MemoryPool<Runtime>;

constexpr const char* kBackend = INFINI_RT_TEST_BACKEND_NAME;

bool SelectDevice() {
  int device_count = 0;
  if (Runtime::GetDeviceCount(&device_count) != Runtime::kSuccess ||
      device_count <= 0) {
    std::cout << kBackend << " memory pool skipped: no available device."
              << std::endl;
    return false;
  }
  if (Runtime::SetDevice(0) != Runtime::kSuccess) {
    std::cout << kBackend << " memory pool skipped: device 0 unavailable."
              << std::endl;
    return false;
  }
  return true;
}

// Writes `input` into device memory `ptr` and reads it back, asserting the
// bytes survive the round trip. This is the only host-safe way to confirm a
// device pointer is real and usable.
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

// A block returned by the pool must be real, usable device memory.
void TestAllocationIsUsable(infini::rt::test::TestContext* context) {
  Pool pool;
  void* ptr = nullptr;
  context->ExpectEqual(pool.Allocate(&ptr, 256), Runtime::kSuccess,
                       "allocate should succeed on a real backend");
  const std::array<std::uint8_t, 8> input{0, 1, 2, 3, 4, 5, 6, 7};
  ExpectUsable(context, ptr, input, "allocation should produce a pointer");
  context->ExpectEqual(pool.Deallocate(ptr), Runtime::kSuccess,
                       "deallocate should succeed");
}

// Freeing then re-requesting the same size class reuses the cached block
// without touching the upstream device allocator.
void TestCacheReuse(infini::rt::test::TestContext* context) {
  Pool pool;
  void* first = nullptr;
  context->ExpectEqual(pool.Allocate(&first, 4096), Runtime::kSuccess,
                       "first allocate should succeed");
  context->ExpectEqual(pool.Deallocate(first), Runtime::kSuccess,
                       "deallocate should cache the block");

  void* second = nullptr;
  context->ExpectEqual(pool.Allocate(&second, 4096), Runtime::kSuccess,
                       "second allocate should succeed");
  context->ExpectEqual(second, first, "same size class should reuse the block");

  const Pool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.cache_hit_count, std::size_t{1},
                       "one cache hit expected");
  context->ExpectEqual(stats.cache_miss_count, std::size_t{1},
                       "only the first allocation misses");
  context->ExpectEqual(stats.upstream_alloc_count, std::size_t{1},
                       "reuse must not call the device allocator again");
  pool.Deallocate(second);
}

// Two sizes that round to the same class share a block; a distinct class does
// not, and each remains independently usable.
void TestSizeClasses(infini::rt::test::TestContext* context) {
  Pool pool;
  void* a = nullptr;
  pool.Allocate(&a, 100);  // rounds to the 512 B class
  pool.Deallocate(a);
  void* b = nullptr;
  pool.Allocate(&b, 500);  // same 512 B class
  context->ExpectEqual(b, a, "100 and 500 share a size class");

  void* c = nullptr;
  pool.Allocate(&c, 8192);  // a different class
  context->Expect(c != b, "a distinct size class must not reuse the block");

  const std::array<std::uint8_t, 4> input{9, 8, 7, 6};
  ExpectUsable(context, b, input, "reused block should be usable");
  ExpectUsable(context, c, input, "fresh block should be usable");
  pool.Deallocate(b);
  pool.Deallocate(c);
}

// A requested power-of-two alignment must be honored by the returned pointer,
// which must still be usable device memory.
void TestAlignment(infini::rt::test::TestContext* context) {
  Pool pool;
  constexpr std::size_t kAlignment = 256;
  void* ptr = nullptr;
  context->ExpectEqual(pool.Allocate(&ptr, 100, kAlignment), Runtime::kSuccess,
                       "aligned allocate should succeed");
  context->ExpectEqual(reinterpret_cast<std::uintptr_t>(ptr) % kAlignment,
                       std::uintptr_t{0},
                       "returned pointer should honor the alignment");
  const std::array<std::uint8_t, 8> input{1, 1, 2, 3, 5, 8, 13, 21};
  ExpectUsable(context, ptr, input, "aligned allocation should be usable");
  pool.Deallocate(ptr);
}

// Statistics track live/reserved bytes, peaks, and call counts across the
// allocate/deallocate cycle.
void TestStats(infini::rt::test::TestContext* context) {
  Pool pool;
  void* a = nullptr;
  void* b = nullptr;
  pool.Allocate(&a, 1024);  // rounds to 1024
  pool.Allocate(&b, 2048);  // rounds to 2048

  Pool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.bytes_in_use, std::size_t{1024 + 2048},
                       "bytes_in_use tracks rounded sizes");
  context->ExpectEqual(stats.peak_bytes_in_use, std::size_t{1024 + 2048},
                       "peak matches the high-water mark");
  context->ExpectEqual(stats.alloc_count, std::size_t{2},
                       "two allocations counted");

  pool.Deallocate(a);
  stats = pool.GetStats();
  context->ExpectEqual(stats.bytes_in_use, std::size_t{2048},
                       "bytes_in_use drops on free");
  context->ExpectEqual(stats.peak_bytes_in_use, std::size_t{1024 + 2048},
                       "peak stays at the high-water mark");
  context->Expect(stats.bytes_reserved >= 1024 + 2048,
                  "reserved memory retained while cached");
  pool.Deallocate(b);
}

// ReleaseCached hands cached blocks back to the device; live blocks are
// untouched. Verified through stats and the upstream free counter.
void TestReleaseCached(infini::rt::test::TestContext* context) {
  Pool pool;
  void* a = nullptr;
  void* b = nullptr;
  pool.Allocate(&a, 1024);
  pool.Allocate(&b, 4096);
  pool.Deallocate(a);
  pool.Deallocate(b);

  Pool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.upstream_free_count, std::size_t{0},
                       "cached blocks are not yet freed upstream");

  pool.ReleaseCached();
  stats = pool.GetStats();
  context->ExpectEqual(stats.upstream_free_count, std::size_t{2},
                       "release should free both cached blocks upstream");
  context->ExpectEqual(stats.bytes_reserved, std::size_t{0},
                       "reserved bytes drop to zero after release");
}

// Concurrently live blocks must be distinct and independently usable.
void TestDistinctLiveBlocks(infini::rt::test::TestContext* context) {
  Pool pool;
  void* a = nullptr;
  void* b = nullptr;
  pool.Allocate(&a, 512);
  pool.Allocate(&b, 512);
  context->Expect(a != b, "two live blocks must not alias");
  const std::array<std::uint8_t, 4> first{1, 2, 3, 4};
  const std::array<std::uint8_t, 4> second{5, 6, 7, 8};
  ExpectUsable(context, a, first, "first live block should be usable");
  ExpectUsable(context, b, second, "second live block should be usable");
  pool.Deallocate(a);
  pool.Deallocate(b);
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  if (!SelectDevice()) {
    return context.ExitCode();
  }

  TestAllocationIsUsable(&context);
  TestCacheReuse(&context);
  TestSizeClasses(&context);
  TestAlignment(&context);
  TestStats(&context);
  TestReleaseCached(&context);
  TestDistinctLiveBlocks(&context);

  return context.ExitCode();
}
