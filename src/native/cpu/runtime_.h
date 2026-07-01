#ifndef INFINI_RT_CPU_RUNTIME__H_
#define INFINI_RT_CPU_RUNTIME__H_

#include <cstdlib>
#include <cstring>

#include "runtime.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kCpu> : RuntimeBase<Runtime<Device::Type::kCpu>> {
  static constexpr Device::Type kDeviceType = Device::Type::kCpu;

  using Error = int;

  using Stream = void*;

  static constexpr Error kSuccess = 0;

  static constexpr Error kErrorInvalidValue = 1;

  static constexpr Error kErrorMemoryAllocation = 2;

  static Error SetDevice(int device) {
    if (device != 0) {
      return kErrorInvalidValue;
    }

    return kSuccess;
  }

  static Error GetDevice(int* device) {
    if (device == nullptr) {
      return kErrorInvalidValue;
    }

    *device = 0;

    return kSuccess;
  }

  static Error GetDeviceCount(int* count) {
    if (count == nullptr) {
      return kErrorInvalidValue;
    }

    *count = 1;

    return kSuccess;
  }

  static Error DeviceSynchronize() { return kSuccess; }

  static Error Malloc(void** ptr, std::size_t size) {
    if (ptr == nullptr) {
      return kErrorInvalidValue;
    }

    *ptr = std::malloc(size);

    if (size != 0 && *ptr == nullptr) {
      return kErrorMemoryAllocation;
    }

    return kSuccess;
  }

  static Error Free(void* ptr) {
    std::free(ptr);

    return kSuccess;
  }

  static Error Memcpy(void* dst, const void* src, std::size_t size, int) {
    if ((dst == nullptr || src == nullptr) && size != 0) {
      return kErrorInvalidValue;
    }

    std::memcpy(dst, src, size);

    return kSuccess;
  }

  static Error Memset(void* ptr, int value, std::size_t count) {
    if (ptr == nullptr && count != 0) {
      return kErrorInvalidValue;
    }

    std::memset(ptr, value, count);

    return kSuccess;
  }

  static Error MemcpyAsync(void* dst, const void* src, std::size_t size,
                           int kind, Stream) {
    return kErrorInvalidValue;
  }

  static constexpr int kMemcpyHostToHost = 0;

  static constexpr int kMemcpyHostToDevice = 1;

  static constexpr int kMemcpyDeviceToHost = 2;

  static constexpr int kMemcpyDeviceToDevice = 3;
};

static_assert(Runtime<Device::Type::kCpu>::Validate());

}  // namespace infini::rt::runtime

#endif
