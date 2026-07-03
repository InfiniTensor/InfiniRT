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
};

static_assert(Runtime<Device::Type::kNvidia>::Validate());

}  // namespace infini::rt::runtime

#endif
