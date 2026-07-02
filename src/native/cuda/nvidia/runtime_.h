#ifndef INFINI_RT_NVIDIA_RUNTIME__H_
#define INFINI_RT_NVIDIA_RUNTIME__H_

#include <cstddef>

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

  static constexpr Device::Type kDeviceType = Device::Type::kNvidia;

  static constexpr Error kSuccess = cudaSuccess;

  static Error SetDevice(int device) { return cudaSetDevice(device); }

  static Error GetDevice(int* device) { return cudaGetDevice(device); }

  static Error GetDeviceCount(int* count) { return cudaGetDeviceCount(count); }

  static Error DeviceSynchronize() { return cudaDeviceSynchronize(); }

  static Error Malloc(void** ptr, std::size_t size) {
    return cudaMalloc(ptr, size);
  }

  static Error Memcpy(void* dst, const void* src, std::size_t count,
                      cudaMemcpyKind kind) {
    return cudaMemcpy(dst, src, count, kind);
  }

  static Error MemcpyAsync(void* dst, const void* src, std::size_t count,
                           cudaMemcpyKind kind, Stream stream) {
    return cudaMemcpyAsync(dst, src, count, kind, stream);
  }

  static Error Free(void* ptr) { return cudaFree(ptr); }

  static constexpr auto kMemcpyHostToHost = cudaMemcpyHostToHost;

  static constexpr auto kMemcpyHostToDevice = cudaMemcpyHostToDevice;

  static constexpr auto kMemcpyDeviceToHost = cudaMemcpyDeviceToHost;

  static constexpr auto kMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;

  static Error Memset(void* ptr, int value, std::size_t count) {
    return cudaMemset(ptr, value, count);
  }
};

static_assert(Runtime<Device::Type::kNvidia>::Validate());

}  // namespace infini::rt::runtime

#endif
