#ifndef INFINI_RT_CPU_RUNTIME__H_
#define INFINI_RT_CPU_RUNTIME__H_

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "runtime.h"

namespace infini::rt {

template <>
struct Runtime<Device::Type::kCpu> : RuntimeBase<Runtime<Device::Type::kCpu>> {
  static constexpr Device::Type kDeviceType = Device::Type::kCpu;

  static infini::rt::Error SetDevice(int index) {
    return index == 0 ? infini::rt::kSuccess : infini::rt::kUnSuccess;
  }

  static infini::rt::Error GetDevice(int* index) {
    if (index == nullptr) {
      return infini::rt::kUnSuccess;
    }
    *index = 0;
    return infini::rt::kSuccess;
  }

  static infini::rt::Error GetDeviceCount(int* count) {
    if (count == nullptr) {
      return infini::rt::kUnSuccess;
    }
    *count = 1;
    return infini::rt::kSuccess;
  }

  static infini::rt::Error DeviceSynchronize() { return infini::rt::kSuccess; }

  static infini::rt::Error Malloc(void** ptr, std::size_t size) {
    if (ptr == nullptr) {
      return infini::rt::kUnSuccess;
    }
    *ptr = std::malloc(size);
    return size != 0 && *ptr == nullptr ? infini::rt::kUnSuccess
                                        : infini::rt::kSuccess;
  }

  static infini::rt::Error Free(void* ptr) {
    std::free(ptr);
    return infini::rt::kSuccess;
  }

  static infini::rt::Error Memcpy(void* dst, const void* src, std::size_t size,
                                  infini::rt::MemcpyKind) {
    if ((dst == nullptr || src == nullptr) && size != 0) {
      return infini::rt::kUnSuccess;
    }
    std::memcpy(dst, src, size);
    return infini::rt::kSuccess;
  }

  static infini::rt::Error Memset(void* ptr, int value, std::size_t count) {
    if (ptr == nullptr && count != 0) {
      return infini::rt::kUnSuccess;
    }
    std::memset(ptr, value, count);
    return infini::rt::kSuccess;
  }

  static infini::rt::Error MemGetInfo(std::size_t* free_bytes,
                                      std::size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
      return infini::rt::kUnSuccess;
    }
    *free_bytes = 0;
    *total_bytes = 0;

#ifndef _WIN32
    FILE* fp = std::fopen("/proc/meminfo", "r");
    if (fp == nullptr) {
      return infini::rt::kUnSuccess;
    }

    char label[64];
    std::size_t value = 0;
    while (std::fscanf(fp, "%63s %zu %*s", label, &value) == 2) {
      if (std::strcmp(label, "MemTotal:") == 0) {
        *total_bytes = value * 1024;
      } else if (std::strcmp(label, "MemAvailable:") == 0) {
        *free_bytes = value * 1024;
      }
    }
    std::fclose(fp);
#endif
    if (*total_bytes == 0) {
      return infini::rt::kUnSuccess;
    }
    return infini::rt::kSuccess;
  }

  static infini::rt::Error StreamCreate(infini::rt::Stream* stream) {
    if (stream == nullptr) {
      return infini::rt::kUnSuccess;
    }
    *stream = nullptr;
    return infini::rt::kSuccess;
  }

  static infini::rt::Error StreamDestroy(infini::rt::Stream) {
    return infini::rt::kSuccess;
  }

  static infini::rt::Error StreamSynchronize(infini::rt::Stream) {
    return infini::rt::kSuccess;
  }

  static infini::rt::Error StreamWaitEvent(infini::rt::Stream,
                                           infini::rt::Event, std::uint32_t) {
    return infini::rt::kSuccess;
  }

  using CpuEvent = std::chrono::steady_clock::time_point;

  static infini::rt::Error EventCreate(infini::rt::Event* event) {
    if (event == nullptr) {
      return infini::rt::kUnSuccess;
    }
    *event = new (std::nothrow) CpuEvent(std::chrono::steady_clock::now());
    return *event == nullptr ? infini::rt::kUnSuccess : infini::rt::kSuccess;
  }

  static infini::rt::Error EventCreateWithFlags(infini::rt::Event* event,
                                                std::uint32_t) {
    return EventCreate(event);
  }

  static infini::rt::Error EventRecord(infini::rt::Event event,
                                       infini::rt::Stream) {
    if (event == nullptr) {
      return infini::rt::kUnSuccess;
    }
    *static_cast<CpuEvent*>(event) = std::chrono::steady_clock::now();
    return infini::rt::kSuccess;
  }

  static infini::rt::Error EventQuery(infini::rt::Event event) {
    return event == nullptr ? infini::rt::kUnSuccess : infini::rt::kSuccess;
  }

  static infini::rt::Error EventSynchronize(infini::rt::Event event) {
    return event == nullptr ? infini::rt::kUnSuccess : infini::rt::kSuccess;
  }

  static infini::rt::Error EventDestroy(infini::rt::Event event) {
    delete static_cast<CpuEvent*>(event);
    return infini::rt::kSuccess;
  }

  static infini::rt::Error EventElapsedTime(float* ms, infini::rt::Event start,
                                            infini::rt::Event end) {
    if (ms == nullptr || start == nullptr || end == nullptr) {
      return infini::rt::kUnSuccess;
    }
    const auto* start_time = static_cast<const CpuEvent*>(start);
    const auto* end_time = static_cast<const CpuEvent*>(end);
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        *end_time - *start_time);
    *ms = static_cast<float>(duration.count()) / 1000.0f;
    return infini::rt::kSuccess;
  }

  static infini::rt::Error MallocHost(void** ptr, std::size_t size) {
    return Malloc(ptr, size);
  }

  static infini::rt::Error FreeHost(void* ptr) { return Free(ptr); }

  static infini::rt::Error MemcpyAsync(void* dst, const void* src,
                                       std::size_t size,
                                       infini::rt::MemcpyKind kind,
                                       infini::rt::Stream) {
    return Memcpy(dst, src, size, kind);
  }

  static infini::rt::Error MallocAsync(void** ptr, std::size_t size,
                                       infini::rt::Stream) {
    return Malloc(ptr, size);
  }

  static infini::rt::Error FreeAsync(void* ptr, infini::rt::Stream) {
    return Free(ptr);
  }

  static infini::rt::Error MemsetAsync(void* ptr, int value, std::size_t count,
                                       infini::rt::Stream) {
    return Memset(ptr, value, count);
  }

  static constexpr auto kMemcpyHostToHost =
      infini::rt::MemcpyKind::kMemcpyHostToHost;

  static constexpr auto kMemcpyHostToDevice =
      infini::rt::MemcpyKind::kMemcpyHostToDevice;

  static constexpr auto kMemcpyDeviceToHost =
      infini::rt::MemcpyKind::kMemcpyDeviceToHost;

  static constexpr auto kMemcpyDeviceToDevice =
      infini::rt::MemcpyKind::kMemcpyDeviceToDevice;
};

static_assert(Runtime<Device::Type::kCpu>::Validate());

}  // namespace infini::rt

#endif
