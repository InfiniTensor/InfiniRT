#ifndef INFINI_RT_MOORE_RUNTIME__H_
#define INFINI_RT_MOORE_RUNTIME__H_

#include <musa_runtime.h>

#include <utility>

#include "native/cuda/runtime_.h"
#include "native/cuda/moore/device_.h"

namespace infini::rt {

template <>
struct Runtime<Device::Type::kMoore>
    : CudaRuntime<Runtime<Device::Type::kMoore>> {
  using Stream = musaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMoore;

  static constexpr auto Malloc = [](auto&&... args) {
    return musaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = [](auto&&... args) {
    return musaMemcpy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Free = [](auto&&... args) {
    return musaFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyHostToDevice = musaMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = musaMemcpyDeviceToHost;

  static constexpr auto Memset = musaMemset;
};

static_assert(Runtime<Device::Type::kMoore>::Validate());

}  // namespace infini::rt

#endif
