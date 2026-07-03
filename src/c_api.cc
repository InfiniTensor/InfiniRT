#include <infini/rt/c_api.h>

#include <new>
#include <type_traits>
#include <utility>

#include "runtime.h"

#if defined(WITH_NVIDIA)
#include "native/cuda/nvidia/runtime_.h"
#endif

namespace {

using infini::rt::Device;
using infini::rt::runtime::Runtime;

struct CStream {
  Device::Type device_type;
  void* raw;
};

struct CGraph {
  Device::Type device_type;
  void* raw;
};

struct CGraphExec {
  Device::Type device_type;
  void* raw;
};

template <typename Func>
infiniRtStatus_t Guard(Func&& func) {
  return std::forward<Func>(func)();
}

template <typename Func>
infiniRtStatus_t CheckBackendCall(Func&& func) {
  using ReturnType = decltype(std::forward<Func>(func)());
  if constexpr (std::is_void_v<ReturnType>) {
    std::forward<Func>(func)();
    return INFINI_RT_STATUS_SUCCESS;
  } else {
    return std::forward<Func>(func)() == ReturnType{}
               ? INFINI_RT_STATUS_SUCCESS
               : INFINI_RT_STATUS_RUNTIME_ERROR;
  }
}

Device::Type ToCppDeviceType(infiniRtDeviceType_t type) {
  switch (type) {
    case INFINI_RT_DEVICE_CPU:
      return Device::Type::kCpu;
    case INFINI_RT_DEVICE_NVIDIA:
      return Device::Type::kNvidia;
    case INFINI_RT_DEVICE_CAMBRICON:
      return Device::Type::kCambricon;
    case INFINI_RT_DEVICE_ASCEND:
      return Device::Type::kAscend;
    case INFINI_RT_DEVICE_METAX:
      return Device::Type::kMetax;
    case INFINI_RT_DEVICE_MOORE:
      return Device::Type::kMoore;
    case INFINI_RT_DEVICE_ILUVATAR:
      return Device::Type::kIluvatar;
    case INFINI_RT_DEVICE_KUNLUN:
      return Device::Type::kKunlun;
    case INFINI_RT_DEVICE_HYGON:
      return Device::Type::kHygon;
    case INFINI_RT_DEVICE_QY:
      return Device::Type::kQy;
  }
  return Device::Type::kCount;
}

#if defined(WITH_NVIDIA)
auto ToNvidiaCaptureMode(infiniRtStreamCaptureMode_t mode) {
  switch (mode) {
    case INFINI_RT_STREAM_CAPTURE_MODE_GLOBAL:
      return Runtime<Device::Type::kNvidia>::kStreamCaptureModeGlobal;
    case INFINI_RT_STREAM_CAPTURE_MODE_THREAD_LOCAL:
      return Runtime<Device::Type::kNvidia>::kStreamCaptureModeThreadLocal;
    case INFINI_RT_STREAM_CAPTURE_MODE_RELAXED:
      return Runtime<Device::Type::kNvidia>::kStreamCaptureModeRelaxed;
  }
  return Runtime<Device::Type::kNvidia>::kStreamCaptureModeRelaxed;
}

auto RawNvidiaStream(const CStream* stream) {
  return static_cast<typename Runtime<Device::Type::kNvidia>::Stream>(
      stream->raw);
}

auto RawNvidiaGraph(const CGraph* graph) {
  return static_cast<typename Runtime<Device::Type::kNvidia>::Graph>(
      graph->raw);
}

auto RawNvidiaGraphExec(const CGraphExec* graph_exec) {
  return static_cast<typename Runtime<Device::Type::kNvidia>::GraphExec>(
      graph_exec->raw);
}
#endif

CStream* AsStream(infiniRtStream_t stream) {
  return static_cast<CStream*>(stream);
}

CGraph* AsGraph(infiniRtGraph_t graph) { return static_cast<CGraph*>(graph); }

CGraphExec* AsGraphExec(infiniRtGraphExec_t graph_exec) {
  return static_cast<CGraphExec*>(graph_exec);
}

infiniRtStatus_t Unsupported() { return INFINI_RT_STATUS_UNSUPPORTED_DEVICE; }

}  // namespace

infiniRtStatus_t infiniRtStreamWrap(infiniRtDevice_t device,
                                    void* native_stream,
                                    infiniRtStream_t* stream) {
  if (native_stream == nullptr || stream == nullptr) {
    return INFINI_RT_STATUS_INVALID_ARGUMENT;
  }
  return Guard([&] {
    const auto device_type = ToCppDeviceType(device.type);
    if (device_type == Device::Type::kCount) {
      return INFINI_RT_STATUS_UNSUPPORTED_DEVICE;
    }
    auto* wrapped = new (std::nothrow) CStream{device_type, native_stream};
    if (wrapped == nullptr) {
      return INFINI_RT_STATUS_RUNTIME_ERROR;
    }
    *stream = wrapped;

    return INFINI_RT_STATUS_SUCCESS;
  });
}

infiniRtStatus_t infiniRtStreamDestroy(infiniRtStream_t stream) {
  if (stream == nullptr) {
    return INFINI_RT_STATUS_INVALID_ARGUMENT;
  }
  delete AsStream(stream);
  return INFINI_RT_STATUS_SUCCESS;
}

infiniRtStatus_t infiniRtStreamBeginCapture(infiniRtStream_t stream,
                                            infiniRtStreamCaptureMode_t mode) {
  if (stream == nullptr) {
    return INFINI_RT_STATUS_INVALID_ARGUMENT;
  }
  return Guard([&] {
    auto* wrapped = AsStream(stream);
    switch (wrapped->device_type) {
#if defined(WITH_NVIDIA)
      case Device::Type::kNvidia:
        return CheckBackendCall([&] {
          return Runtime<Device::Type::kNvidia>::StreamBeginCapture(
              RawNvidiaStream(wrapped), ToNvidiaCaptureMode(mode));
        });
#endif
      default:
        return Unsupported();
    }
  });
}

infiniRtStatus_t infiniRtStreamEndCapture(infiniRtStream_t stream,
                                          infiniRtGraph_t* graph) {
  if (stream == nullptr || graph == nullptr) {
    return INFINI_RT_STATUS_INVALID_ARGUMENT;
  }
  return Guard([&] {
    auto* wrapped = AsStream(stream);
    switch (wrapped->device_type) {
#if defined(WITH_NVIDIA)
      case Device::Type::kNvidia: {
        typename Runtime<Device::Type::kNvidia>::Graph raw_graph = {};
        const auto status = CheckBackendCall([&] {
          return Runtime<Device::Type::kNvidia>::StreamEndCapture(
              RawNvidiaStream(wrapped), &raw_graph);
        });
        if (status != INFINI_RT_STATUS_SUCCESS) {
          return status;
        }
        auto* wrapped_graph = new (std::nothrow)
            CGraph{Device::Type::kNvidia, static_cast<void*>(raw_graph)};
        if (wrapped_graph == nullptr) {
          return INFINI_RT_STATUS_RUNTIME_ERROR;
        }
        *graph = wrapped_graph;

        return INFINI_RT_STATUS_SUCCESS;
      }
#endif
      default:
        return Unsupported();
    }
  });
}

infiniRtStatus_t infiniRtGraphDestroy(infiniRtGraph_t graph) {
  if (graph == nullptr) {
    return INFINI_RT_STATUS_INVALID_ARGUMENT;
  }
  return Guard([&] {
    auto* wrapped = AsGraph(graph);
    switch (wrapped->device_type) {
#if defined(WITH_NVIDIA)
      case Device::Type::kNvidia: {
        const auto status = CheckBackendCall([&] {
          return Runtime<Device::Type::kNvidia>::GraphDestroy(
              RawNvidiaGraph(wrapped));
        });
        // The C wrapper owns only the wrapper object. The backend destroy call
        // above owns the native graph handle.
        delete wrapped;
        return status;
      }
#endif
      default:
        delete wrapped;
        return Unsupported();
    }
  });
}

infiniRtStatus_t infiniRtGraphInstantiate(infiniRtGraphExec_t* graph_exec,
                                          infiniRtGraph_t graph) {
  if (graph_exec == nullptr || graph == nullptr) {
    return INFINI_RT_STATUS_INVALID_ARGUMENT;
  }
  return Guard([&] {
    auto* wrapped = AsGraph(graph);
    switch (wrapped->device_type) {
#if defined(WITH_NVIDIA)
      case Device::Type::kNvidia: {
        typename Runtime<Device::Type::kNvidia>::GraphExec raw_exec = {};
        const auto status = CheckBackendCall([&] {
          return Runtime<Device::Type::kNvidia>::GraphInstantiate(
              &raw_exec, RawNvidiaGraph(wrapped));
        });
        if (status != INFINI_RT_STATUS_SUCCESS) {
          return status;
        }
        auto* wrapped_exec = new (std::nothrow)
            CGraphExec{Device::Type::kNvidia, static_cast<void*>(raw_exec)};
        if (wrapped_exec == nullptr) {
          return INFINI_RT_STATUS_RUNTIME_ERROR;
        }
        *graph_exec = wrapped_exec;

        return INFINI_RT_STATUS_SUCCESS;
      }
#endif
      default:
        return Unsupported();
    }
  });
}

infiniRtStatus_t infiniRtGraphExecDestroy(infiniRtGraphExec_t graph_exec) {
  if (graph_exec == nullptr) {
    return INFINI_RT_STATUS_INVALID_ARGUMENT;
  }
  return Guard([&] {
    auto* wrapped = AsGraphExec(graph_exec);
    switch (wrapped->device_type) {
#if defined(WITH_NVIDIA)
      case Device::Type::kNvidia: {
        const auto status = CheckBackendCall([&] {
          return Runtime<Device::Type::kNvidia>::GraphExecDestroy(
              RawNvidiaGraphExec(wrapped));
        });
        // The C wrapper owns only the wrapper object. The backend destroy call
        // above owns the native executable graph handle.
        delete wrapped;
        return status;
      }
#endif
      default:
        delete wrapped;
        return Unsupported();
    }
  });
}

infiniRtStatus_t infiniRtGraphLaunch(infiniRtGraphExec_t graph_exec,
                                     infiniRtStream_t stream) {
  if (graph_exec == nullptr || stream == nullptr) {
    return INFINI_RT_STATUS_INVALID_ARGUMENT;
  }
  return Guard([&] {
    auto* exec = AsGraphExec(graph_exec);
    auto* wrapped_stream = AsStream(stream);
    if (exec->device_type != wrapped_stream->device_type) {
      return INFINI_RT_STATUS_INVALID_ARGUMENT;
    }
    switch (exec->device_type) {
#if defined(WITH_NVIDIA)
      case Device::Type::kNvidia:
        return CheckBackendCall([&] {
          return Runtime<Device::Type::kNvidia>::GraphLaunch(
              RawNvidiaGraphExec(exec), RawNvidiaStream(wrapped_stream));
        });
#endif
      default:
        return Unsupported();
    }
  });
}
