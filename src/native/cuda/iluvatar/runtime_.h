#ifndef INFINI_RT_ILUVATAR_RUNTIME__H_
#define INFINI_RT_ILUVATAR_RUNTIME__H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "native/cuda/iluvatar/device_.h"
#include "native/cuda/runtime_.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kIluvatar>
    : CudaRuntime<Runtime<Device::Type::kIluvatar>> {
  using Error = cudaError_t;

  using Stream = cudaStream_t;

  using Graph = cudaGraph_t;

  using GraphExec = cudaGraphExec_t;

  static constexpr Device::Type kDeviceType = Device::Type::kIluvatar;

  static constexpr Error kSuccess = cudaSuccess;

  static constexpr auto SetDevice = cudaSetDevice;

  static constexpr auto GetDevice = cudaGetDevice;

  static constexpr auto GetDeviceCount = cudaGetDeviceCount;

  static constexpr auto DeviceSynchronize = cudaDeviceSynchronize;

  static constexpr auto Malloc = [](auto&&... args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = cudaMemcpy;

  static constexpr auto MemcpyAsync = cudaMemcpyAsync;

  static constexpr auto Free = cudaFree;

  static constexpr auto kMemcpyHostToHost = cudaMemcpyHostToHost;

  static constexpr auto kMemcpyHostToDevice = cudaMemcpyHostToDevice;

  static constexpr auto kMemcpyDeviceToHost = cudaMemcpyDeviceToHost;

  static constexpr auto kMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;

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

  static constexpr auto kStreamCaptureModeGlobal = cudaStreamCaptureModeGlobal;

  static constexpr auto kStreamCaptureModeThreadLocal =
      cudaStreamCaptureModeThreadLocal;

  static constexpr auto kStreamCaptureModeRelaxed =
      cudaStreamCaptureModeRelaxed;

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

}  // namespace infini::rt::runtime

#endif
