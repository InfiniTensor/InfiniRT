#ifndef INFINI_RT_ASCEND_RUNTIME__H_
#define INFINI_RT_ASCEND_RUNTIME__H_

#include <cassert>
#include <cstdint>

// clang-format off
#include "acl/acl.h"
// clang-format on

#include "native/ascend/device_.h"
#include "runtime.h"

namespace infini::rt {

template <>
struct Runtime<Device::Type::kAscend>
    : DeviceRuntime<Runtime<Device::Type::kAscend>> {
  using Stream = aclrtStream;

  static constexpr Device::Type kDeviceType = Device::Type::kAscend;

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

  static constexpr auto Free = aclrtFree;

  static constexpr auto Memcpy = [](void* dst, const void* src, size_t count,
                                    aclrtMemcpyKind kind) {
    return aclrtMemcpy(dst, count, src, count, kind);
  };

  static constexpr auto MemcpyHostToHost = ACL_MEMCPY_HOST_TO_HOST;

  static constexpr auto MemcpyHostToDevice = ACL_MEMCPY_HOST_TO_DEVICE;

  static constexpr auto MemcpyDeviceToHost = ACL_MEMCPY_DEVICE_TO_HOST;

  static constexpr auto MemcpyDeviceToDevice = ACL_MEMCPY_DEVICE_TO_DEVICE;

  static constexpr auto Memset = [](void* ptr, int value, size_t count) {
    return aclrtMemset(ptr, count, value, count);
  };
};

static_assert(Runtime<Device::Type::kAscend>::Validate());

}  // namespace infini::rt

#endif
