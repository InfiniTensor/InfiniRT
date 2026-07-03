#ifndef INFINI_RT_CAMBRICON_RUNTIME__H_
#define INFINI_RT_CAMBRICON_RUNTIME__H_

#include <cnrt.h>

#include <cassert>
#include <cstddef>

#include "native/cambricon/device_.h"
#include "runtime.h"

namespace infini::rt {

template <>
struct Runtime<Device::Type::kCambricon>
    : DeviceRuntime<Runtime<Device::Type::kCambricon>> {
  using Stream = cnrtQueue_t;

  // CNRT's native graph equivalent is TaskTopo, not a cnrtGraph* API.
  using Graph = cnrtTaskTopo_t;

  using GraphExec = cnrtTaskTopoEntity_t;

  static constexpr Device::Type kDeviceType = Device::Type::kCambricon;

  static constexpr auto SetDevice = cnrtSetDevice;

  static constexpr auto GetDevice = cnrtGetDevice;

  static auto GetDeviceCount(int* count) {
    assert(count != nullptr);
    unsigned int device_count = 0;
    auto status = cnrtGetDeviceCount(&device_count);
    *count = static_cast<int>(device_count);
    return status;
  }

  static constexpr auto DeviceSynchronize = cnrtSyncDevice;

  static constexpr auto Malloc = cnrtMalloc;

  static constexpr auto Free = cnrtFree;

  static constexpr auto Memcpy = [](void* dst, const void* src,
                                    std::size_t size, auto kind) {
    return cnrtMemcpy(dst, const_cast<void*>(src), size, kind);
  };

  static auto MemcpyAsync(void* dst, const void* src, std::size_t size,
                          cnrtMemTransDir_t kind, Stream stream) {
    return cnrtMemcpyAsync_V2(dst, const_cast<void*>(src), size, stream, kind);
  }

  static constexpr auto MemcpyHostToHost = cnrtMemcpyHostToHost;

  static constexpr auto MemcpyHostToDevice = cnrtMemcpyHostToDev;

  static constexpr auto MemcpyDeviceToHost = cnrtMemcpyDevToHost;

  static constexpr auto MemcpyDeviceToDevice = cnrtMemcpyDevToDev;

  static constexpr auto Memset = cnrtMemset;

  static constexpr auto StreamCreate = cnrtQueueCreate;

  static constexpr auto StreamDestroy = cnrtQueueDestroy;

  static constexpr auto StreamSynchronize = cnrtQueueSync;

  static constexpr auto StreamCaptureModeGlobal = cnrtQueueCaptureModeGlobal;

  static constexpr auto StreamCaptureModeThreadLocal =
      cnrtQueueCaptureModeThreadLocal;

  static constexpr auto StreamCaptureModeRelaxed = cnrtQueueCaptureModeRelaxed;

  static auto StreamBeginCapture(Stream stream, cnrtQueueCaptureMode_t mode) {
    return cnrtQueueBeginCapture(stream, mode);
  }

  static auto StreamEndCapture(Stream stream, Graph* graph) {
    assert(graph != nullptr);
    return cnrtQueueEndCapture(stream, graph);
  }

  static auto GraphDestroy(Graph graph) {
    if (graph == nullptr) {
      return cnrtSuccess;
    }
    return cnrtTaskTopoDestroy(graph);
  }

  static auto GraphInstantiate(GraphExec* graph_exec, Graph graph) {
    assert(graph_exec != nullptr);
    return cnrtTaskTopoInstantiate(graph_exec, graph, nullptr, nullptr, 0);
  }

  static auto GraphExecDestroy(GraphExec graph_exec) {
    if (graph_exec == nullptr) {
      return cnrtSuccess;
    }
    return cnrtTaskTopoEntityDestroy(graph_exec);
  }

  static auto GraphLaunch(GraphExec graph_exec, Stream stream) {
    return cnrtTaskTopoEntityInvoke(graph_exec, stream);
  }
};

static_assert(Runtime<Device::Type::kCambricon>::Validate());

}  // namespace infini::rt

#endif
