#ifndef INFINI_RT_METAX_RUNTIME__H_
#define INFINI_RT_METAX_RUNTIME__H_

#include <mcr/mc_runtime.h>

#include "native/cuda/metax/device_.h"
#include "native/cuda/runtime_.h"

namespace infini::rt {

template <>
struct Runtime<Device::Type::kMetax>
    : CudaRuntime<Runtime<Device::Type::kMetax>> {
  using Stream = mcStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  static constexpr auto SetDevice = mcSetDevice;

  static constexpr auto GetDevice = mcGetDevice;

  static constexpr auto GetDeviceCount = mcGetDeviceCount;

  static constexpr auto DeviceSynchronize = mcDeviceSynchronize;

  static constexpr auto Malloc = mcMalloc;

  static constexpr auto Memcpy = mcMemcpy;

  static constexpr auto Free = mcFree;

  static constexpr auto MemcpyHostToHost = mcMemcpyHostToHost;

  static constexpr auto MemcpyHostToDevice = mcMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = mcMemcpyDeviceToHost;

  static constexpr auto MemcpyDeviceToDevice = mcMemcpyDeviceToDevice;

  static constexpr auto Memset = mcMemset;
};

static_assert(Runtime<Device::Type::kMetax>::Validate());

}  // namespace infini::rt

#endif
