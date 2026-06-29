#ifndef INFINI_RT_ASCEND_RUNTIME__H_
#define INFINI_RT_ASCEND_RUNTIME__H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>

// clang-format off
#include "acl/acl.h"
// clang-format on

#include "native/ascend/device_.h"
#include "runtime.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kAscend>
    : DeviceRuntime<Runtime<Device::Type::kAscend>> {
  using Error = aclError;

  using Stream = aclrtStream;

  using Graph = void*;

  using GraphExec = void*;

  using Event = void*;

  static constexpr Device::Type kDeviceType = Device::Type::kAscend;

  static constexpr Error kSuccess = ACL_SUCCESS;

  static constexpr auto SetDevice = aclrtSetDevice;

  static constexpr auto GetDevice = aclrtGetDevice;

  static auto GetDeviceCount(int* count) {
    assert(count != nullptr);
    std::uint32_t device_count = 0;
    auto status = aclrtGetDeviceCount(&device_count);
    *count = static_cast<int>(device_count);
    return status;
  }

  static constexpr auto DeviceSynchronize = aclrtSynchronizeDevice;

  static constexpr auto Malloc = [](void** ptr, size_t size) {
    return aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  };

  static Error MallocHost(void**, std::size_t) { return Unsupported(); }

  static Error MallocAsync(void**, std::size_t, Stream) {
    return Unsupported();
  }

  static constexpr auto Free = aclrtFree;

  static Error FreeHost(void*) { return Unsupported(); }

  static Error FreeAsync(void*, Stream) { return Unsupported(); }

  static Error MemGetInfo(std::size_t*, std::size_t*) { return Unsupported(); }

  static constexpr auto Memcpy = [](void* dst, const void* src, size_t count,
                                    aclrtMemcpyKind kind) {
    return aclrtMemcpy(dst, count, src, count, kind);
  };

  static constexpr auto MemcpyAsync = [](void* dst, const void* src,
                                         size_t count, aclrtMemcpyKind kind,
                                         Stream stream) {
    return aclrtMemcpyAsync(dst, count, src, count, kind, stream);
  };

  static constexpr auto kMemcpyHostToHost = ACL_MEMCPY_HOST_TO_HOST;

  static constexpr auto kMemcpyHostToDevice = ACL_MEMCPY_HOST_TO_DEVICE;

  static constexpr auto kMemcpyDeviceToHost = ACL_MEMCPY_DEVICE_TO_HOST;

  static constexpr auto kMemcpyDeviceToDevice = ACL_MEMCPY_DEVICE_TO_DEVICE;

  static constexpr auto Memset = [](void* ptr, int value, size_t count) {
    return aclrtMemset(ptr, count, value, count);
  };

  static Error MemsetAsync(void*, int, std::size_t, Stream) {
    return Unsupported();
  }

  static constexpr auto StreamCreate = aclrtCreateStream;

  static constexpr auto StreamDestroy = aclrtDestroyStream;

  static constexpr auto StreamSynchronize = aclrtSynchronizeStream;

  static Error StreamWaitEvent(Stream, Event, unsigned int) {
    return Unsupported();
  }

  static Error EventCreate(Event*) { return Unsupported(); }

  static Error EventCreateWithFlags(Event*, unsigned int) {
    return Unsupported();
  }

  static Error EventRecord(Event, Stream) { return Unsupported(); }

  static Error EventQuery(Event) { return Unsupported(); }

  static Error EventSynchronize(Event) { return Unsupported(); }

  static Error EventDestroy(Event) { return Unsupported(); }

  static Error EventElapsedTime(float*, Event, Event) { return Unsupported(); }

  static constexpr int kStreamCaptureModeGlobal = 0;

  static constexpr int kStreamCaptureModeThreadLocal = 1;

  static constexpr int kStreamCaptureModeRelaxed = 2;

  struct RIApi {
    using CaptureBeginFn = aclError (*)(aclrtStream, int);
    using CaptureGetInfoFn = aclError (*)(aclrtStream, int*, void**);
    using CaptureEndFn = aclError (*)(aclrtStream, void**);
    using RIExecuteAsyncFn = aclError (*)(void*, aclrtStream);
    using RIDestroyFn = aclError (*)(void*);

    CaptureBeginFn capture_begin = nullptr;
    CaptureGetInfoFn capture_get_info = nullptr;
    CaptureEndFn capture_end = nullptr;
    RIExecuteAsyncFn execute_async = nullptr;
    RIDestroyFn destroy = nullptr;

    bool available() const {
      return capture_begin != nullptr && capture_get_info != nullptr &&
             capture_end != nullptr && execute_async != nullptr;
    }
  };

  static const RIApi& GraphApi() {
    static const RIApi api = [] {
      RIApi loaded{};
      auto load_symbols = [](void* lib, RIApi* api) {
        api->capture_begin = reinterpret_cast<RIApi::CaptureBeginFn>(
            dlsym(lib, "aclmdlRICaptureBegin"));
        api->capture_get_info = reinterpret_cast<RIApi::CaptureGetInfoFn>(
            dlsym(lib, "aclmdlRICaptureGetInfo"));
        api->capture_end = reinterpret_cast<RIApi::CaptureEndFn>(
            dlsym(lib, "aclmdlRICaptureEnd"));
        api->execute_async = reinterpret_cast<RIApi::RIExecuteAsyncFn>(
            dlsym(lib, "aclmdlRIExecuteAsync"));
        api->destroy =
            reinterpret_cast<RIApi::RIDestroyFn>(dlsym(lib, "aclmdlRIDestroy"));
      };

      load_symbols(RTLD_DEFAULT, &loaded);
      if (loaded.available()) {
        return loaded;
      }

      void* lib = dlopen("libascendcl.so", RTLD_NOW | RTLD_NOLOAD);
      if (lib == nullptr) {
        lib = dlopen("libascendcl.so", RTLD_NOW);
      }
      if (lib == nullptr) {
        return loaded;
      }

      load_symbols(lib, &loaded);
      return loaded;
    }();
    return api;
  }

  static aclError UnsupportedGraphApi() { return static_cast<aclError>(1); }

  static aclError StreamBeginCapture(Stream stream, int mode) {
    const auto& api = GraphApi();
    if (!api.available()) {
      return UnsupportedGraphApi();
    }
    return api.capture_begin(stream, mode);
  }

  static aclError StreamEndCapture(Stream stream, Graph* graph) {
    assert(graph != nullptr);
    const auto& api = GraphApi();
    if (!api.available()) {
      return UnsupportedGraphApi();
    }
    int capture_status = 0;
    void* capturing_model_ri = nullptr;
    const auto info_status =
        api.capture_get_info(stream, &capture_status, &capturing_model_ri);
    if (info_status != ACL_SUCCESS) {
      return info_status;
    }
    void* model_ri = nullptr;
    const auto status = api.capture_end(stream, &model_ri);
    if (status != ACL_SUCCESS) {
      return status;
    }
    *graph = model_ri;
    return ACL_SUCCESS;
  }

  static aclError GraphDestroy(Graph graph) {
    const auto& api = GraphApi();
    if (api.destroy == nullptr || graph == nullptr) {
      return ACL_SUCCESS;
    }
    return api.destroy(graph);
  }

  static aclError GraphInstantiate(GraphExec* graph_exec, Graph graph) {
    assert(graph_exec != nullptr);
    *graph_exec = graph;
    return ACL_SUCCESS;
  }

  static aclError GraphExecDestroy(GraphExec) { return ACL_SUCCESS; }

  static aclError GraphLaunch(GraphExec graph_exec, Stream stream) {
    const auto& api = GraphApi();
    if (!api.available()) {
      return UnsupportedGraphApi();
    }
    return api.execute_async(graph_exec, stream);
  }

 private:
  static Error Unsupported() { return static_cast<Error>(1); }
};

static_assert(Runtime<Device::Type::kAscend>::Validate());

}  // namespace infini::rt::runtime

#endif
