#ifndef INFINI_RT_ILUVATAR_RUNTIME__H_
#define INFINI_RT_ILUVATAR_RUNTIME__H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "native/cuda/iluvatar/device_.h"
#include "native/cuda/runtime_.h"

namespace infini::rt {

template <>
struct Runtime<Device::Type::kIluvatar>
    : CudaRuntime<Runtime<Device::Type::kIluvatar>> {
  using Stream = cudaStream_t;

  using Graph = cudaGraph_t;

  using GraphExec = cudaGraphExec_t;

  static constexpr Device::Type kDeviceType = Device::Type::kIluvatar;

  static constexpr auto SetDevice = cudaSetDevice;

  static constexpr auto GetDevice = cudaGetDevice;

  static constexpr auto GetDeviceCount = cudaGetDeviceCount;

  static constexpr auto DeviceSynchronize = cudaDeviceSynchronize;

  static constexpr auto Malloc = [](auto&&... args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = cudaMemcpy;

  static constexpr auto Free = cudaFree;

  static constexpr auto MemcpyHostToHost = cudaMemcpyHostToHost;

  static constexpr auto MemcpyHostToDevice = cudaMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = cudaMemcpyDeviceToHost;

  static constexpr auto MemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;

  static constexpr auto Memset = cudaMemset;

  static constexpr auto StreamCreate = [](auto&&... args) {
    return cudaStreamCreateWithFlags(std::forward<decltype(args)>(args)...,
                                     cudaStreamNonBlocking);
  };

  static constexpr auto StreamDestroy = [](auto&&... args) {
    return cudaStreamDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamSynchronize = [](auto&&... args) {
    return cudaStreamSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyAsync = [](auto&&... args) {
    return cudaMemcpyAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamCaptureModeGlobal = cudaStreamCaptureModeGlobal;

  static constexpr auto StreamCaptureModeThreadLocal =
      cudaStreamCaptureModeThreadLocal;

  static constexpr auto StreamCaptureModeRelaxed = cudaStreamCaptureModeRelaxed;

  static constexpr auto StreamBeginCapture = [](auto&&... args) {
    return cudaStreamBeginCapture(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamEndCapture = [](auto&&... args) {
    return cudaStreamEndCapture(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphDestroy = [](auto&&... args) {
    return cudaGraphDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphInstantiate = [](auto&&... args) {
    return cudaGraphInstantiate(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphExecDestroy = [](auto&&... args) {
    return cudaGraphExecDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphLaunch = [](auto&&... args) {
    return cudaGraphLaunch(std::forward<decltype(args)>(args)...);
  };
};

static_assert(Runtime<Device::Type::kIluvatar>::Validate());

}  // namespace infini::rt

#endif
