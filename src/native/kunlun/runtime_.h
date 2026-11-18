#ifndef INFINI_RT_KUNLUN_RUNTIME__H_
#define INFINI_RT_KUNLUN_RUNTIME__H_

#include <algorithm>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

// clang-format off
#include <xpu/runtime.h>
#include <xpu/runtime_ex.h>
// clang-format on

#include "native/kunlun/device_.h"
#include "runtime.h"

namespace infini::rt::runtime {

/// Kunlun XPU runtime (XRE `libxpurt`).
///
/// Wraps the `xpu_*` C API in the CRTP `Runtime` interface. XRE exposes no
/// graph capture, no event query/elapsed-time, and no memset: memset is
/// emulated with chunked H2D copies from a host staging buffer, and the
/// unsupported graph entry points are simply absent from this specialization
/// so the generated dispatch omits them for this device.
template <>
struct Runtime<Device::Type::kKunlun>
    : DeviceRuntime<Runtime<Device::Type::kKunlun>> {
  using Error = int;

  using Stream = XPUStream;

  using Event = XPUEvent;

  static constexpr Device::Type kDeviceType = Device::Type::kKunlun;

  static constexpr Error kSuccess = XPU_SUCCESS;

  static constexpr auto SetDevice = xpu_set_device;

  static constexpr auto GetDevice = xpu_current_device;

  static constexpr auto GetDeviceCount = xpu_device_count;

  static Error DeviceSynchronize() {
    // XRE has no whole-device synchronize; `xpu_wait()` waits on the
    // default stream, which matches the legacy kunlun backend behavior.
    return xpu_wait();
  }

  static constexpr auto Malloc = [](void** ptr, std::size_t size) -> Error {
    return xpu_malloc(ptr, static_cast<std::uint64_t>(size));
  };

  static constexpr auto MallocHost = [](void** ptr, std::size_t size) -> Error {
    return xpu_host_alloc(ptr, static_cast<std::uint64_t>(size), 0);
  };

  // XRE has no stream-ordered allocation; fall back to sync allocation,
  // mirroring the legacy backend.
  static constexpr auto MallocAsync = [](void** ptr, std::size_t size,
                                         Stream) -> Error {
    return xpu_malloc(ptr, static_cast<std::uint64_t>(size));
  };

  static constexpr auto Free = xpu_free;

  static constexpr auto FreeHost = xpu_host_free;

  // XRE has no stream-ordered free; fall back to sync free.
  static constexpr auto FreeAsync = [](void* ptr, Stream) -> Error {
    return xpu_free(ptr);
  };

  static Error MemGetInfo(std::size_t* free, std::size_t* total) {
    (void)free;
    (void)total;
    return XPUERR_NOSUPPORT;
  }

  static Error Memcpy(void* dst, const void* src, std::size_t count, int kind) {
    switch (kind) {
      case kMemcpyHostToHost:
        std::memcpy(dst, src, count);
        return kSuccess;
      case kMemcpyHostToDevice:
        return xpu_memcpy(dst, src, static_cast<std::uint64_t>(count),
                          XPU_HOST_TO_DEVICE);
      case kMemcpyDeviceToHost:
        return xpu_memcpy(dst, src, static_cast<std::uint64_t>(count),
                          XPU_DEVICE_TO_HOST);
      case kMemcpyDeviceToDevice:
        return xpu_memcpy(dst, src, static_cast<std::uint64_t>(count),
                          XPU_DEVICE_TO_DEVICE);
      default:
        return XPUERR_INVALID_PARAM;
    }
  }

  static Error MemcpyAsync(void* dst, const void* src, std::size_t count,
                           int kind, Stream stream) {
    switch (kind) {
      case kMemcpyHostToHost:
        std::memcpy(dst, src, count);
        return kSuccess;
      case kMemcpyHostToDevice:
        return xpu_memcpy_async(dst, src, static_cast<std::uint64_t>(count),
                                XPU_HOST_TO_DEVICE, stream);
      case kMemcpyDeviceToHost:
        return xpu_memcpy_async(dst, src, static_cast<std::uint64_t>(count),
                                XPU_DEVICE_TO_HOST, stream);
      case kMemcpyDeviceToDevice:
        return xpu_memcpy_async(dst, src, static_cast<std::uint64_t>(count),
                                XPU_DEVICE_TO_DEVICE, stream);
      default:
        return XPUERR_INVALID_PARAM;
    }
  }

  /// XRE exposes no memset API: emulate it with chunked H2D copies from a
  /// host staging buffer filled with the requested byte value. The staging
  /// buffer stays bounded (16 MiB) regardless of the fill size, matching the
  /// legacy backend implementation.
  static Error Memset(void* ptr, int value, std::size_t count) {
    if (ptr == nullptr) {
      return XPUERR_INVALID_PARAM;
    }
    if (count == 0) {
      return kSuccess;
    }

    static std::mutex staging_mutex;
    static std::vector<std::uint8_t> staging_buffer;
    constexpr std::size_t kStagingBytes = 16ULL * 1024 * 1024;

    std::lock_guard<std::mutex> lock(staging_mutex);

    const std::size_t chunk = std::min(count, kStagingBytes);
    if (staging_buffer.size() < chunk) {
      staging_buffer.resize(chunk);
    }
    std::memset(staging_buffer.data(), value, chunk);

    auto* dst = static_cast<std::uint8_t*>(ptr);
    for (std::size_t done = 0; done < count;) {
      const std::size_t n = std::min(chunk, count - done);
      const Error status =
          xpu_memcpy(dst + done, staging_buffer.data(),
                     static_cast<std::uint64_t>(n), XPU_HOST_TO_DEVICE);
      if (status != kSuccess) {
        return status;
      }
      done += n;
    }
    return kSuccess;
  }

  // XRE has no async memset; fall back to the (host-blocking) sync fill.
  static Error MemsetAsync(void* ptr, int value, std::size_t count, Stream) {
    return Memset(ptr, value, count);
  }

  static constexpr auto StreamCreate = [](Stream* stream) -> Error {
    return xpu_stream_create(stream);
  };

  static constexpr auto StreamDestroy = [](Stream stream) -> Error {
    return xpu_stream_destroy(stream);
  };

  static constexpr auto StreamSynchronize = [](Stream stream) -> Error {
    return xpu_wait(stream);
  };

  // The generated dispatch bridges `StreamWaitEvent(stream, event, flags)`;
  // XRE has no flags parameter, so it is ignored.
  static Error StreamWaitEvent(Stream stream, Event event, unsigned int) {
    return xpu_stream_wait_event(stream, event);
  }

  static constexpr auto EventCreate = [](Event* event) -> Error {
    return xpu_event_create(event);
  };

  static Error EventCreateWithFlags(Event*, unsigned int) {
    return XPUERR_NOSUPPORT;
  }

  static constexpr auto EventRecord = [](Event event, Stream stream) -> Error {
    return xpu_event_record(event, stream);
  };

  static Error EventQuery(Event) { return XPUERR_NOSUPPORT; }

  static constexpr auto EventSynchronize = [](Event event) -> Error {
    return xpu_event_wait(event);
  };

  static constexpr auto EventDestroy = [](Event event) -> Error {
    return xpu_event_destroy(event);
  };

  static Error EventElapsedTime(float*, Event, Event) {
    return XPUERR_NOSUPPORT;
  }

  // Follow the CPU/backend-neutral convention (0..3); the generated dispatch
  // maps the public MemcpyKind enum onto these via RuntimeMemcpyKind.
  static constexpr int kMemcpyHostToHost = 0;

  static constexpr int kMemcpyHostToDevice = 1;

  static constexpr int kMemcpyDeviceToHost = 2;

  static constexpr int kMemcpyDeviceToDevice = 3;
};

static_assert(Runtime<Device::Type::kKunlun>::Validate());

}  // namespace infini::rt::runtime

#endif
