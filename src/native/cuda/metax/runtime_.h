#ifndef INFINI_RT_METAX_RUNTIME__H_
#define INFINI_RT_METAX_RUNTIME__H_

#include <mcr/mc_runtime.h>

#include <cstddef>
#include <utility>

#include "native/cuda/metax/device_.h"
#include "native/cuda/runtime_.h"

namespace infini::rt {

template <>
struct Runtime<Device::Type::kMetax>
    : CudaRuntime<Runtime<Device::Type::kMetax>> {
  using Stream = mcStream_t;

  using Graph = void*;

  using GraphExec = void*;

  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  static constexpr auto SetDevice = mcSetDevice;

  static constexpr auto GetDevice = mcGetDevice;

  static constexpr auto GetDeviceCount = mcGetDeviceCount;

  static constexpr auto DeviceSynchronize = mcDeviceSynchronize;

  static constexpr auto Malloc = [](auto&&... args) {
    return mcMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = [](auto&&... args) {
    return mcMemcpy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Free = [](auto&&... args) {
    return mcFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyHostToHost = mcMemcpyHostToHost;

  static constexpr auto MemcpyHostToDevice = mcMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = mcMemcpyDeviceToHost;

  static constexpr auto MemcpyDeviceToDevice = mcMemcpyDeviceToDevice;

  static constexpr auto Memset = mcMemset;

  static int StreamCreate(Stream*) { return 1; }

  static int StreamDestroy(Stream) { return 1; }

  static int StreamSynchronize(Stream) { return 1; }

  static int MemcpyAsync(void*, const void*, std::size_t,
                         decltype(MemcpyHostToDevice), Stream) {
    return 1;
  }

  static constexpr int StreamCaptureModeGlobal = 0;

  static constexpr int StreamCaptureModeThreadLocal = 1;

  static constexpr int StreamCaptureModeRelaxed = 2;

  static int StreamBeginCapture(Stream, int) { return 1; }

  static int StreamEndCapture(Stream, Graph*) { return 1; }

  static int GraphDestroy(Graph) { return 1; }

  static int GraphInstantiate(GraphExec*, Graph) { return 1; }

  static int GraphExecDestroy(GraphExec) { return 1; }

  static int GraphLaunch(GraphExec, Stream) { return 1; }
};

static_assert(Runtime<Device::Type::kMetax>::Validate());

}  // namespace infini::rt

#endif
