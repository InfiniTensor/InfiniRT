#ifndef INFINI_RT_KUNLUN_RUNTIME__H_
#define INFINI_RT_KUNLUN_RUNTIME__H_

#include <xpu/runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#include "native/kunlun/device_.h"
#include "runtime.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kKunlun>
    : DeviceRuntime<Runtime<Device::Type::kKunlun>> {
  using Error = int;
  using Stream = XPUStream;
  using Event = XPUEvent;
  using Graph = void*;
  using GraphExec = void*;
  using StreamCaptureMode = int;

  static constexpr Device::Type kDeviceType = Device::Type::kKunlun;
  static constexpr Error kSuccess = XPU_SUCCESS;
  static constexpr auto SetDevice = xpu_set_device;
  static constexpr auto GetDevice = xpu_current_device;
  static constexpr auto GetDeviceCount = xpu_device_count;

  static Error DeviceSynchronize() {
    int device;
    auto status = GetDevice(&device);
    if (status != kSuccess) return status;
    std::lock_guard<std::mutex> lock(stream_mutex_);
    status = xpu_wait(nullptr);
    if (status != kSuccess) return status;
    // XRE only synchronizes individual streams. Include all streams created
    // by this runtime on the current device, not just the default stream.
    for (const auto& entry : streams_) {
      if (entry.first == device) {
        status = xpu_wait(entry.second);
        if (status != kSuccess) return status;
      }
    }
    return kSuccess;
  }

  static Error Malloc(void** ptr, std::size_t size) {
    return xpu_malloc(ptr, size, XPU_MEM_MAIN);
  }
  static Error MallocHost(void** ptr, std::size_t size) {
    return xpu_host_alloc(ptr, size, 0);
  }
  static Error MallocAsync(void**, std::size_t, Stream) {
    return XPUERR_NOSUPPORT;
  }
  static constexpr auto Free = xpu_free;
  static constexpr auto FreeHost = xpu_host_free;
  static Error FreeAsync(void*, Stream) { return XPUERR_NOSUPPORT; }
  static Error MemGetInfo(std::size_t*, std::size_t*) {
    return XPUERR_NOSUPPORT;
  }

  // XRE has no host-to-host enumerator.
  static constexpr int kMemcpyHostToHost = 3;
  static constexpr int kMemcpyHostToDevice = XPU_HOST_TO_DEVICE;
  static constexpr int kMemcpyDeviceToHost = XPU_DEVICE_TO_HOST;
  static constexpr int kMemcpyDeviceToDevice = XPU_DEVICE_TO_DEVICE;

  static Error Memcpy(void* dst, const void* src, std::size_t size, int kind) {
    if (kind == kMemcpyHostToHost) {
      std::memcpy(dst, src, size);
      return kSuccess;
    }
    return xpu_memcpy(dst, src, size, static_cast<XPUMemcpyKind>(kind));
  }
  static Error MemcpyAsync(void* dst, const void* src, std::size_t size,
                           int kind, Stream stream) {
    if (kind == kMemcpyHostToHost) {
      const auto status = xpu_wait(stream);
      if (status != kSuccess) return status;
      return Memcpy(dst, src, size, kind);
    }
    return xpu_memcpy_async(dst, src, size, static_cast<XPUMemcpyKind>(kind),
                            stream);
  }

  static Error Memset(void* ptr, int value, std::size_t size) {
    // XRE has no memset primitive. Bound the synchronous staging memory.
    std::array<unsigned char, 4096> staging;
    staging.fill(static_cast<unsigned char>(value));
    auto* dst = static_cast<unsigned char*>(ptr);
    for (std::size_t done = 0; done < size;) {
      const auto bytes = std::min(staging.size(), size - done);
      const auto status =
          xpu_memcpy(dst + done, staging.data(), bytes, XPU_HOST_TO_DEVICE);
      if (status != kSuccess) return status;
      done += bytes;
    }
    return kSuccess;
  }
  static Error MemsetAsync(void*, int, std::size_t, Stream) {
    return XPUERR_NOSUPPORT;
  }

  static Error StreamCreate(Stream* stream) {
    int device;
    auto status = GetDevice(&device);
    if (status != kSuccess) return status;
    std::lock_guard<std::mutex> lock(stream_mutex_);
    status = xpu_stream_create(stream);
    if (status == kSuccess) streams_.emplace_back(device, *stream);
    return status;
  }
  static Error StreamDestroy(Stream stream) {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    const auto status = xpu_stream_destroy(stream);
    if (status == kSuccess) {
      streams_.erase(std::remove_if(streams_.begin(), streams_.end(),
                                    [stream](const auto& entry) {
                                      return entry.second == stream;
                                    }),
                     streams_.end());
    }
    return status;
  }
  static constexpr auto StreamSynchronize = xpu_wait;
  static Error StreamWaitEvent(Stream stream, Event event, unsigned int flags) {
    return flags == 0 ? xpu_stream_wait_event(stream, event) : XPUERR_NOSUPPORT;
  }

  static constexpr auto EventCreate = xpu_event_create;
  static Error EventCreateWithFlags(Event* event, unsigned int flags) {
    return flags == 0 ? EventCreate(event) : XPUERR_NOSUPPORT;
  }
  static constexpr auto EventRecord = xpu_event_record;
  static constexpr auto EventQuery = xpu_event_query;
  static constexpr auto EventSynchronize = xpu_event_wait;
  static constexpr auto EventDestroy = xpu_event_destroy;
  static Error EventElapsedTime(float*, Event, Event) {
    return XPUERR_NOSUPPORT;
  }

  static constexpr StreamCaptureMode kStreamCaptureModeGlobal = 0;
  static constexpr StreamCaptureMode kStreamCaptureModeThreadLocal = 1;
  static constexpr StreamCaptureMode kStreamCaptureModeRelaxed = 2;
  static Error StreamBeginCapture(Stream, StreamCaptureMode) {
    return XPUERR_NOSUPPORT;
  }
  static Error StreamEndCapture(Stream, Graph*) { return XPUERR_NOSUPPORT; }
  static Error GraphDestroy(Graph) { return XPUERR_NOSUPPORT; }
  static Error GraphInstantiate(GraphExec*, Graph) { return XPUERR_NOSUPPORT; }
  static Error GraphExecDestroy(GraphExec) { return XPUERR_NOSUPPORT; }
  static Error GraphLaunch(GraphExec, Stream) { return XPUERR_NOSUPPORT; }

 private:
  inline static std::mutex stream_mutex_;
  inline static std::vector<std::pair<int, Stream>> streams_;
};

static_assert(Runtime<Device::Type::kKunlun>::Validate());

}  // namespace infini::rt::runtime

#endif
