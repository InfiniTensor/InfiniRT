#ifndef INFINI_RT_QY_RUNTIME__H_
#define INFINI_RT_QY_RUNTIME__H_

#include <cstddef>
#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "native/cuda/qy/device_.h"
#include "native/cuda/runtime_.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kQy>
    : GraphRuntime<Runtime<Device::Type::kQy>,
                   CudaRuntime<Runtime<Device::Type::kQy>>> {
  using Error = cudaError_t;

  using Stream = cudaStream_t;

  using Graph = cudaGraph_t;

  using GraphExec = cudaGraphExec_t;

  using Event = cudaEvent_t;

  using StreamCaptureMode = cudaStreamCaptureMode;

  static constexpr Device::Type kDeviceType = Device::Type::kQy;

  static constexpr Error kSuccess = cudaSuccess;

  // Denglin headers overload several cuda* entry points; wrap them in
  // forwarding lambdas so Clang CUDA can resolve a single candidate.
  static constexpr auto SetDevice = [](auto&&... args) {
    return cudaSetDevice(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GetDevice = [](auto&&... args) {
    return cudaGetDevice(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GetDeviceCount = [](auto&&... args) {
    return cudaGetDeviceCount(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto DeviceSynchronize = [](auto&&... args) {
    return cudaDeviceSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Malloc = [](auto&&... args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MallocHost = [](auto&&... args) {
    return cudaMallocHost(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MallocAsync = [](auto&&... args) {
    return cudaMallocAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Free = [](auto&&... args) {
    return cudaFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto FreeHost = [](auto&&... args) {
    return cudaFreeHost(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto FreeAsync = [](auto&&... args) {
    return cudaFreeAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemGetInfo = [](auto&&... args) {
    return cudaMemGetInfo(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = [](auto&&... args) {
    return cudaMemcpy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyAsync = [](auto&&... args) {
    return cudaMemcpyAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto kMemcpyHostToHost = cudaMemcpyHostToHost;

  static constexpr auto kMemcpyHostToDevice = cudaMemcpyHostToDevice;

  static constexpr auto kMemcpyDeviceToHost = cudaMemcpyDeviceToHost;

  static constexpr auto kMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;

  static constexpr auto Memset = [](auto&&... args) {
    return cudaMemset(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemsetAsync = [](auto&&... args) {
    return cudaMemsetAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamCreate = [](auto&&... args) {
    return cudaStreamCreate(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamDestroy = [](auto&&... args) {
    return cudaStreamDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamSynchronize = [](auto&&... args) {
    return cudaStreamSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamWaitEvent = [](auto&&... args) {
    return cudaStreamWaitEvent(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventCreate = [](auto&&... args) {
    return cudaEventCreate(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventCreateWithFlags = [](auto&&... args) {
    return cudaEventCreateWithFlags(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventRecord = [](auto&&... args) {
    return cudaEventRecord(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventQuery = [](auto&&... args) {
    return cudaEventQuery(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventSynchronize = [](auto&&... args) {
    return cudaEventSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventDestroy = [](auto&&... args) {
    return cudaEventDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventElapsedTime = [](auto&&... args) {
    return cudaEventElapsedTime(std::forward<decltype(args)>(args)...);
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

  // Denglin SDK exposes the older 5-argument cudaGraphInstantiate ABI.
  static Error GraphInstantiate(GraphExec* graph_exec, Graph graph) {
    return cudaGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0);
  }

  static constexpr auto GraphExecDestroy = [](auto&&... args) {
    return cudaGraphExecDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphLaunch = [](auto&&... args) {
    return cudaGraphLaunch(std::forward<decltype(args)>(args)...);
  };
};

static_assert(Runtime<Device::Type::kQy>::Validate());

}  // namespace infini::rt::runtime

#endif
