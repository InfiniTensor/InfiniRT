#ifndef INFINI_RT_ASCEND_RUNTIME__H_
#define INFINI_RT_ASCEND_RUNTIME__H_

#include <cassert>
#include <cstddef>
#include <cstdint>

// clang-format off
#include "acl/acl.h"
// clang-format on

#include "native/ascend/device_.h"
#include "runtime.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kAscend>
    : DeviceRuntime<Runtime<Device::Type::kAscend>> {
  using Error = aclError;

  using Stream = aclrtStream;

  using Event = void*;

  static constexpr Device::Type kDeviceType = Device::Type::kAscend;

  static constexpr Error kSuccess = ACL_SUCCESS;

  static constexpr auto SetDevice = aclrtSetDevice;

  static constexpr auto GetDevice = aclrtGetDevice;

  static auto GetDeviceCount(int* count) {
    assert(count != nullptr);
    std::uint32_t device_count = 0;
    auto status = aclrtGetDeviceCount(&device_count);
    *count = static_cast<int>(device_count);
    return status;
  }

  static constexpr auto DeviceSynchronize = aclrtSynchronizeDevice;

  static constexpr auto Malloc = [](void** ptr, size_t size) {
    return aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  };

  static Error MallocHost(void**, std::size_t) { return Unsupported(); }

  static Error MallocAsync(void**, std::size_t, Stream) {
    return Unsupported();
  }

  static constexpr auto Free = aclrtFree;

  static Error FreeHost(void*) { return Unsupported(); }

  static Error FreeAsync(void*, Stream) { return Unsupported(); }

  static Error MemGetInfo(std::size_t*, std::size_t*) { return Unsupported(); }

  static constexpr auto Memcpy = [](void* dst, const void* src, size_t count,
                                    aclrtMemcpyKind kind) {
    return aclrtMemcpy(dst, count, src, count, kind);
  };

  static constexpr auto MemcpyAsync = [](void* dst, const void* src,
                                         size_t count, aclrtMemcpyKind kind,
                                         Stream stream) {
    return aclrtMemcpyAsync(dst, count, src, count, kind, stream);
  };

  static constexpr auto kMemcpyHostToHost = ACL_MEMCPY_HOST_TO_HOST;

  static constexpr auto kMemcpyHostToDevice = ACL_MEMCPY_HOST_TO_DEVICE;

  static constexpr auto kMemcpyDeviceToHost = ACL_MEMCPY_DEVICE_TO_HOST;

  static constexpr auto kMemcpyDeviceToDevice = ACL_MEMCPY_DEVICE_TO_DEVICE;

  static constexpr auto Memset = [](void* ptr, int value, size_t count) {
    return aclrtMemset(ptr, count, value, count);
  };

  static Error MemsetAsync(void*, int, std::size_t, Stream) {
    return Unsupported();
  }

  static constexpr auto StreamCreate = aclrtCreateStream;

  static constexpr auto StreamDestroy = aclrtDestroyStream;

  static constexpr auto StreamSynchronize = aclrtSynchronizeStream;

  static Error StreamWaitEvent(Stream, Event, unsigned int) {
    return Unsupported();
  }

  static Error EventCreate(Event*) { return Unsupported(); }

  static Error EventCreateWithFlags(Event*, unsigned int) {
    return Unsupported();
  }

  static Error EventRecord(Event, Stream) { return Unsupported(); }

  static Error EventQuery(Event) { return Unsupported(); }

  static Error EventSynchronize(Event) { return Unsupported(); }

  static Error EventDestroy(Event) { return Unsupported(); }

  static Error EventElapsedTime(float*, Event, Event) { return Unsupported(); }

 private:
  static Error Unsupported() { return static_cast<Error>(1); }
};

static_assert(Runtime<Device::Type::kAscend>::Validate());

}  // namespace infini::rt::runtime

#endif
