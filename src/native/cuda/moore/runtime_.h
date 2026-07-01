#ifndef INFINI_RT_MOORE_RUNTIME__H_
#define INFINI_RT_MOORE_RUNTIME__H_

#include <musa_runtime.h>

#include <utility>

#include "native/cuda/moore/device_.h"
#include "native/cuda/runtime_.h"

namespace infini::rt {

template <>
struct Runtime<Device::Type::kMoore>
    : CudaRuntime<Runtime<Device::Type::kMoore>> {
  using Error = musaError_t;

  using Stream = musaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMoore;

  static constexpr Error kSuccess = musaSuccess;

  static constexpr auto SetDevice = musaSetDevice;

  static constexpr auto GetDevice = [](auto&&... args) {
    return musaGetDevice(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GetDeviceCount = [](auto&&... args) {
    return musaGetDeviceCount(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto DeviceSynchronize = [](auto&&... args) {
    return musaDeviceSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Malloc = [](auto&&... args) {
    return musaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = [](auto&&... args) {
    return musaMemcpy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyAsync = [](auto&&... args) {
    return musaMemcpyAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Free = [](auto&&... args) {
    return musaFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto kMemcpyHostToHost = musaMemcpyHostToHost;

  static constexpr auto kMemcpyHostToDevice = musaMemcpyHostToDevice;

  static constexpr auto kMemcpyDeviceToHost = musaMemcpyDeviceToHost;

  static constexpr auto kMemcpyDeviceToDevice = musaMemcpyDeviceToDevice;

  static constexpr auto Memset = musaMemset;
};

static_assert(Runtime<Device::Type::kMoore>::Validate());

}  // namespace infini::rt

#endif
