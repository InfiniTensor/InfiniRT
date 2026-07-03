#ifndef INFINI_RT_MOORE_RUNTIME__H_
#define INFINI_RT_MOORE_RUNTIME__H_

#include <musa_runtime.h>

#include <cstddef>
#include <utility>

#include "native/cuda/moore/device_.h"
#include "native/cuda/runtime_.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kMoore>
    : CudaRuntime<Runtime<Device::Type::kMoore>> {
  using Error = musaError_t;

  using Stream = musaStream_t;

  using Graph = void*;

  using GraphExec = void*;

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

  static int StreamCreate(Stream*) { return 1; }

  static int StreamDestroy(Stream) { return 1; }

  static int StreamSynchronize(Stream) { return 1; }

  static constexpr int kStreamCaptureModeGlobal = 0;

  static constexpr int kStreamCaptureModeThreadLocal = 1;

  static constexpr int kStreamCaptureModeRelaxed = 2;

  static int StreamBeginCapture(Stream, int) { return 1; }

  static int StreamEndCapture(Stream, Graph*) { return 1; }

  static int GraphDestroy(Graph) { return 1; }

  static int GraphInstantiate(GraphExec*, Graph) { return 1; }

  static int GraphExecDestroy(GraphExec) { return 1; }

  static int GraphLaunch(GraphExec, Stream) { return 1; }
};

static_assert(Runtime<Device::Type::kMoore>::Validate());

}  // namespace infini::rt::runtime

#endif
