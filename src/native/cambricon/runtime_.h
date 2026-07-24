#ifndef INFINI_RT_CAMBRICON_RUNTIME__H_
#define INFINI_RT_CAMBRICON_RUNTIME__H_

#include <cnrt.h>

#include <cassert>
#include <cstddef>

#include "native/cambricon/device_.h"
#include "runtime.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kCambricon>
    : DeviceRuntime<Runtime<Device::Type::kCambricon>> {
  using Error = cnrtRet_t;

  using Stream = cnrtQueue_t;

  using Event = void*;

  static constexpr Device::Type kDeviceType = Device::Type::kCambricon;

#ifdef CNRT_RET_SUCCESS
  static constexpr Error kSuccess = CNRT_RET_SUCCESS;
#else
  static constexpr Error kSuccess = cnrtSuccess;
#endif

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

  static Error MallocHost(void**, std::size_t) { return Unsupported(); }

  static Error MallocAsync(void**, std::size_t, Stream) {
    return Unsupported();
  }

  static constexpr auto Free = cnrtFree;

  static Error FreeHost(void*) { return Unsupported(); }

  static Error FreeAsync(void*, Stream) { return Unsupported(); }

  static Error MemGetInfo(std::size_t*, std::size_t*) { return Unsupported(); }

  static constexpr auto Memcpy = [](void* dst, const void* src,
                                    std::size_t size, auto kind) {
    return cnrtMemcpy(dst, const_cast<void*>(src), size, kind);
  };

  static constexpr auto MemcpyAsync = [](void* dst, const void* src,
                                         std::size_t size, auto kind,
                                         Stream stream) {
    return cnrtMemcpyAsync_V2(dst, const_cast<void*>(src), size, stream, kind);
  };

  static constexpr auto kMemcpyHostToHost = cnrtMemcpyHostToHost;

  static constexpr auto kMemcpyHostToDevice = cnrtMemcpyHostToDev;

  static constexpr auto kMemcpyDeviceToHost = cnrtMemcpyDevToHost;

  static constexpr auto kMemcpyDeviceToDevice = cnrtMemcpyDevToDev;

  static constexpr auto Memset = cnrtMemset;

  static Error MemsetAsync(void*, int, std::size_t, Stream) {
    return Unsupported();
  }

  static constexpr auto StreamCreate = cnrtQueueCreate;

  static constexpr auto StreamDestroy = cnrtQueueDestroy;

  static constexpr auto StreamSynchronize = cnrtQueueSync;

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

 private:
  static Error Unsupported() { return static_cast<Error>(1); }
};

static_assert(Runtime<Device::Type::kCambricon>::Validate());

}  // namespace infini::rt::runtime

#endif
