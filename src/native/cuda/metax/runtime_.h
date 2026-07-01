#ifndef INFINI_RT_METAX_RUNTIME__H_
#define INFINI_RT_METAX_RUNTIME__H_

#include <mcr/mc_runtime.h>

#include <utility>

#include "native/cuda/metax/device_.h"
#include "native/cuda/runtime_.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kMetax>
    : CudaRuntime<Runtime<Device::Type::kMetax>> {
  using Error = mcError_t;

  using Stream = mcStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  static constexpr Error kSuccess = mcSuccess;

  static constexpr auto SetDevice = mcSetDevice;

  static constexpr auto GetDevice = mcGetDevice;

  static constexpr auto GetDeviceCount = mcGetDeviceCount;

  static constexpr auto DeviceSynchronize = mcDeviceSynchronize;

  static constexpr auto Malloc = [](auto&&... args) {
    return mcMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = [](auto&&... args) {
    return mcMemcpy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyAsync = [](auto&&... args) {
    return mcMemcpyAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Free = [](auto&&... args) {
    return mcFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto kMemcpyHostToHost = mcMemcpyHostToHost;

  static constexpr auto kMemcpyHostToDevice = mcMemcpyHostToDevice;

  static constexpr auto kMemcpyDeviceToHost = mcMemcpyDeviceToHost;

  static constexpr auto kMemcpyDeviceToDevice = mcMemcpyDeviceToDevice;

  static constexpr auto Memset = mcMemset;
};

static_assert(Runtime<Device::Type::kMetax>::Validate());

}  // namespace infini::rt::runtime

#endif
