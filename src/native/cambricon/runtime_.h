#ifndef INFINI_RT_CAMBRICON_RUNTIME__H_
#define INFINI_RT_CAMBRICON_RUNTIME__H_

#include <cnrt.h>

#include <cassert>
#include <cstddef>

#include "native/cambricon/device_.h"
#include "runtime.h"

namespace infini::rt {

template <>
struct Runtime<Device::Type::kCambricon>
    : DeviceRuntime<Runtime<Device::Type::kCambricon>> {
  using Stream = cnrtQueue_t;

  static constexpr Device::Type kDeviceType = Device::Type::kCambricon;

  static constexpr auto SetDevice = cnrtSetDevice;

  static constexpr auto GetDevice = cnrtGetDevice;

  static auto GetDeviceCount(int* count) {
    assert(count != nullptr);
    unsigned int device_count = 0;
    auto status = cnrtGetDeviceCount(&device_count);
    *count = static_cast<int>(device_count);
    return status;
  }

  static constexpr auto DeviceSynchronize = cnrtSyncDevice;

  static constexpr auto Malloc = cnrtMalloc;

  static constexpr auto Free = cnrtFree;

  static constexpr auto Memcpy = [](void* dst, const void* src,
                                    std::size_t size, auto kind) {
    return cnrtMemcpy(dst, const_cast<void*>(src), size, kind);
  };

  static constexpr auto MemcpyHostToHost = cnrtMemcpyHostToHost;

  static constexpr auto MemcpyHostToDevice = cnrtMemcpyHostToDev;

  static constexpr auto MemcpyDeviceToHost = cnrtMemcpyDevToHost;

  static constexpr auto MemcpyDeviceToDevice = cnrtMemcpyDevToDev;

  static constexpr auto Memset = cnrtMemset;
};

static_assert(Runtime<Device::Type::kCambricon>::Validate());

}  // namespace infini::rt

#endif
