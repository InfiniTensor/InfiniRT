#ifndef INFINI_RT_NVIDIA_RUNTIME__H_
#define INFINI_RT_NVIDIA_RUNTIME__H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "cuda/runtime.h"
#include "nvidia/device_.h"
#include "nvidia/runtime_utils.h"

namespace infini::rt {

template <>
struct Runtime<Device::Type::kNvidia>
    : CudaRuntime<Runtime<Device::Type::kNvidia>> {
  using Stream = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kNvidia;

  static constexpr auto Malloc = [](auto&&... args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = cudaMemcpy;

  static constexpr auto Free = cudaFree;

  static constexpr auto MemcpyHostToDevice = cudaMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = cudaMemcpyDeviceToHost;

  static constexpr auto Memset = cudaMemset;
};

static_assert(Runtime<Device::Type::kNvidia>::Validate());

}  // namespace infini::rt

#endif
