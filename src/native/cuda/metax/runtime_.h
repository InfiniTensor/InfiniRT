#ifndef INFINI_RT_METAX_RUNTIME__H_
#define INFINI_RT_METAX_RUNTIME__H_

#if defined(INFINIRT_METAX_USE_HPCC) || defined(USE_HPCC)
#include <hcr/hc_runtime.h>
#define INFINIRT_METAX_RUNTIME_API_(prefix, name) prefix##name
#define INFINIRT_METAX_RUNTIME_API(prefix, name) \
  INFINIRT_METAX_RUNTIME_API_(prefix, name)
#define INFINIRT_METAX_API(name) INFINIRT_METAX_RUNTIME_API(hc, name)
#else
#include <mcr/mc_runtime.h>
#define INFINIRT_METAX_RUNTIME_API_(prefix, name) prefix##name
#define INFINIRT_METAX_RUNTIME_API(prefix, name) \
  INFINIRT_METAX_RUNTIME_API_(prefix, name)
#define INFINIRT_METAX_API(name) INFINIRT_METAX_RUNTIME_API(mc, name)
#endif

#include <utility>

#include "native/cuda/metax/device_.h"
#include "native/cuda/runtime_.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kMetax>
    : GraphRuntime<Runtime<Device::Type::kMetax>,
                   CudaRuntime<Runtime<Device::Type::kMetax>>> {
  using Error = INFINIRT_METAX_API(Error_t);

  using Stream = INFINIRT_METAX_API(Stream_t);

  using Graph = INFINIRT_METAX_API(Graph_t);

  using GraphExec = INFINIRT_METAX_API(GraphExec_t);

  using Event = INFINIRT_METAX_API(Event_t);

  using StreamCaptureMode = INFINIRT_METAX_API(StreamCaptureMode);

  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  static constexpr Error kSuccess = INFINIRT_METAX_API(Success);

  static constexpr auto SetDevice = INFINIRT_METAX_API(SetDevice);

  static constexpr auto GetDevice = INFINIRT_METAX_API(GetDevice);

  static constexpr auto GetDeviceCount = INFINIRT_METAX_API(GetDeviceCount);

  static constexpr auto DeviceSynchronize =
      INFINIRT_METAX_API(DeviceSynchronize);

  static constexpr auto Malloc = [](auto&&... args) {
    return INFINIRT_METAX_API(Malloc)(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MallocHost = [](auto&&... args) {
    return INFINIRT_METAX_API(MallocHost)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MallocAsync = [](auto&&... args) {
    return INFINIRT_METAX_API(MallocAsync)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Free = [](auto&&... args) {
    return INFINIRT_METAX_API(Free)(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto FreeHost = [](auto&&... args) {
    return INFINIRT_METAX_API(FreeHost)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto FreeAsync = [](auto&&... args) {
    return INFINIRT_METAX_API(FreeAsync)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemGetInfo = [](auto&&... args) {
    return INFINIRT_METAX_API(MemGetInfo)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = [](auto&&... args) {
    return INFINIRT_METAX_API(Memcpy)(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyAsync = [](auto&&... args) {
    return INFINIRT_METAX_API(MemcpyAsync)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto kMemcpyHostToHost =
      INFINIRT_METAX_API(MemcpyHostToHost);

  static constexpr auto kMemcpyHostToDevice =
      INFINIRT_METAX_API(MemcpyHostToDevice);

  static constexpr auto kMemcpyDeviceToHost =
      INFINIRT_METAX_API(MemcpyDeviceToHost);

  static constexpr auto kMemcpyDeviceToDevice =
      INFINIRT_METAX_API(MemcpyDeviceToDevice);

  static constexpr auto Memset = INFINIRT_METAX_API(Memset);

  static constexpr auto MemsetAsync = [](auto&&... args) {
    return INFINIRT_METAX_API(MemsetAsync)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamCreate = [](auto&&... args) {
    return INFINIRT_METAX_API(StreamCreate)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamDestroy = [](auto&&... args) {
    return INFINIRT_METAX_API(StreamDestroy)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamSynchronize = [](auto&&... args) {
    return INFINIRT_METAX_API(StreamSynchronize)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamWaitEvent = [](auto&&... args) {
    return INFINIRT_METAX_API(StreamWaitEvent)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventCreate = [](auto&&... args) {
    return INFINIRT_METAX_API(EventCreate)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventCreateWithFlags = [](auto&&... args) {
    return INFINIRT_METAX_API(EventCreateWithFlags)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventRecord = [](auto&&... args) {
    return INFINIRT_METAX_API(EventRecord)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventQuery = [](auto&&... args) {
    return INFINIRT_METAX_API(EventQuery)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventSynchronize = [](auto&&... args) {
    return INFINIRT_METAX_API(EventSynchronize)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventDestroy = [](auto&&... args) {
    return INFINIRT_METAX_API(EventDestroy)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventElapsedTime = [](auto&&... args) {
    return INFINIRT_METAX_API(EventElapsedTime)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto kStreamCaptureModeGlobal =
      INFINIRT_METAX_API(StreamCaptureModeGlobal);

  static constexpr auto kStreamCaptureModeThreadLocal =
      INFINIRT_METAX_API(StreamCaptureModeThreadLocal);

  static constexpr auto kStreamCaptureModeRelaxed =
      INFINIRT_METAX_API(StreamCaptureModeRelaxed);

  static constexpr auto StreamBeginCapture = [](auto&&... args) {
    return INFINIRT_METAX_API(StreamBeginCapture)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamEndCapture = [](auto&&... args) {
    return INFINIRT_METAX_API(StreamEndCapture)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphDestroy = [](auto&&... args) {
    return INFINIRT_METAX_API(GraphDestroy)(
        std::forward<decltype(args)>(args)...);
  };

  static Error GraphInstantiate(GraphExec* graph_exec, Graph graph) {
    return INFINIRT_METAX_API(GraphInstantiate)(
        graph_exec, graph, nullptr, nullptr, 0);
  }

  static constexpr auto GraphExecDestroy = [](auto&&... args) {
    return INFINIRT_METAX_API(GraphExecDestroy)(
        std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphLaunch = [](auto&&... args) {
    return INFINIRT_METAX_API(GraphLaunch)(
        std::forward<decltype(args)>(args)...);
  };
};

static_assert(Runtime<Device::Type::kMetax>::Validate());

}  // namespace infini::rt::runtime

#undef INFINIRT_METAX_API
#undef INFINIRT_METAX_RUNTIME_API
#undef INFINIRT_METAX_RUNTIME_API_

#endif
