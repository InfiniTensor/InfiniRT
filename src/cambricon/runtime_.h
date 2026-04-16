#ifndef INFINI_RT_CAMBRICON_RUNTIME__H_
#define INFINI_RT_CAMBRICON_RUNTIME__H_

#include <cnrt.h>

#include "cambricon/device_.h"
#include "runtime.h"

namespace infini::rt {

template <>
struct Runtime<Device::Type::kCambricon>
    : DeviceRuntime<Runtime<Device::Type::kCambricon>> {
  using Stream = cnrtQueue_t;

  static constexpr Device::Type kDeviceType = Device::Type::kCambricon;

  static constexpr auto Malloc = cnrtMalloc;

  static constexpr auto Free = cnrtFree;

  static constexpr auto Memcpy = cnrtMemcpy;

  static constexpr auto MemcpyHostToDevice = cnrtMemcpyHostToDev;

  static constexpr auto MemcpyDeviceToHost = cnrtMemcpyDevToHost;

  static constexpr auto Memset = cnrtMemset;
};

static_assert(Runtime<Device::Type::kCambricon>::Validate());

}  // namespace infini::rt

#endif
