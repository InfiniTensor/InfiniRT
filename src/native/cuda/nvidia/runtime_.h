#ifndef INFINI_RT_NVIDIA_RUNTIME__H_
#define INFINI_RT_NVIDIA_RUNTIME__H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "native/cuda/nvidia/device_.h"
#include "native/cuda/runtime_.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kNvidia>
    : CudaRuntime<Runtime<Device::Type::kNvidia>> {
  using Error = cudaError_t;

  using Stream = cudaStream_t;

  using Graph = cudaGraph_t;

  using GraphExec = cudaGraphExec_t;

  using Event = cudaEvent_t;

  static constexpr Device::Type kDeviceType = Device::Type::kNvidia;

  static constexpr Error kSuccess = cudaSuccess;

  static constexpr auto SetDevice = cudaSetDevice;

  static constexpr auto GetDevice = cudaGetDevice;

  static constexpr auto GetDeviceCount = cudaGetDeviceCount;

  static constexpr auto DeviceSynchronize = cudaDeviceSynchronize;

  static constexpr auto Malloc = [](auto&&... args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MallocHost = [](auto&&... args) {
    return cudaMallocHost(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MallocAsync = [](auto&&... args) {
    return cudaMallocAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Free = cudaFree;

  static constexpr auto FreeHost = [](auto&&... args) {
    return cudaFreeHost(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto FreeAsync = [](auto&&... args) {
    return cudaFreeAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemGetInfo = [](auto&&... args) {
    return cudaMemGetInfo(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = cudaMemcpy;

  static constexpr auto MemcpyAsync = cudaMemcpyAsync;

  static constexpr auto kMemcpyHostToHost = cudaMemcpyHostToHost;

  static constexpr auto kMemcpyHostToDevice = cudaMemcpyHostToDevice;

  static constexpr auto kMemcpyDeviceToHost = cudaMemcpyDeviceToHost;

  static constexpr auto kMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;

  static constexpr auto Memset = cudaMemset;

  static constexpr auto MemsetAsync = [](auto&&... args) {
    return cudaMemsetAsync(std::forward<decltype(args)>(args)...);
  };

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

  static constexpr auto GraphInstantiate = [](auto&&... args) {
    return cudaGraphInstantiate(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphExecDestroy = [](auto&&... args) {
    return cudaGraphExecDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphLaunch = [](auto&&... args) {
    return cudaGraphLaunch(std::forward<decltype(args)>(args)...);
  };

  static constexpr bool Validate() {
    CudaRuntime<Runtime<Device::Type::kNvidia>>::Validate();
    static_assert(sizeof(Graph) > 0,
                  "`Runtime` must define a `Graph` type alias.");
    static_assert(sizeof(GraphExec) > 0,
                  "`Runtime` must define a `GraphExec` type alias.");
    static_assert(std::is_invocable_v<decltype(StreamCreate), Stream*>,
                  "`Runtime::StreamCreate` must be callable with `(Stream*)`.");
    static_assert(std::is_invocable_v<decltype(StreamDestroy), Stream>,
                  "`Runtime::StreamDestroy` must be callable with `(Stream)`.");
    static_assert(
        std::is_invocable_v<decltype(StreamSynchronize), Stream>,
        "`Runtime::StreamSynchronize` must be callable with `(Stream)`.");
    static_assert(std::is_invocable_v<decltype(MemcpyAsync), void*, const void*,
                                      std::size_t, cudaMemcpyKind, Stream>,
                  "`Runtime::MemcpyAsync` must be callable with "
                  "`(void*, const void*, size_t, cudaMemcpyKind, Stream)`.");
    static_assert(std::is_invocable_v<decltype(StreamBeginCapture), Stream,
                                      cudaStreamCaptureMode>,
                  "`Runtime::StreamBeginCapture` must be callable with "
                  "`(Stream, cudaStreamCaptureMode)`.");
    static_assert(
        std::is_invocable_v<decltype(StreamEndCapture), Stream, Graph*>,
        "`Runtime::StreamEndCapture` must be callable with "
        "`(Stream, Graph*)`.");
    static_assert(std::is_invocable_v<decltype(GraphDestroy), Graph>,
                  "`Runtime::GraphDestroy` must be callable with `(Graph)`.");
    static_assert(
        std::is_invocable_v<decltype(GraphInstantiate), GraphExec*, Graph>,
        "`Runtime::GraphInstantiate` must be callable with "
        "`(GraphExec*, Graph)`.");
    static_assert(
        std::is_invocable_v<decltype(GraphExecDestroy), GraphExec>,
        "`Runtime::GraphExecDestroy` must be callable with `(GraphExec)`.");
    static_assert(
        std::is_invocable_v<decltype(GraphLaunch), GraphExec, Stream>,
        "`Runtime::GraphLaunch` must be callable with `(GraphExec, Stream)`.");
    return true;
  }
};

static_assert(Runtime<Device::Type::kNvidia>::Validate());

}  // namespace infini::rt::runtime

#endif
