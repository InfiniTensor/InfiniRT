#ifndef INFINI_RT_CPU_RUNTIME__H_
#define INFINI_RT_CPU_RUNTIME__H_

#include <cassert>
#include <cstdlib>
#include <cstring>

#include "runtime.h"

namespace infini::rt {

template <>
struct Runtime<Device::Type::kCpu> : RuntimeBase<Runtime<Device::Type::kCpu>> {
  using Stream = void*;

  using Graph = void*;

  using GraphExec = void*;

  static constexpr Device::Type kDeviceType = Device::Type::kCpu;

  static void SetDevice(int index) {
    if (index != 0) {
      assert(false && "CPU device index must be 0");
      std::abort();
    }
  }

  static void GetDevice(int* index) {
    assert(index != nullptr);
    *index = 0;
  }

  static void GetDeviceCount(int* count) {
    assert(count != nullptr);
    *count = 1;
  }

  static void DeviceSynchronize() {}

  static void Malloc(void** ptr, std::size_t size) { *ptr = std::malloc(size); }

  static void Free(void* ptr) { std::free(ptr); }

  static void Memcpy(void* dst, const void* src, std::size_t size, int) {
    std::memcpy(dst, src, size);
  }

  static int MemcpyAsync(void*, const void*, std::size_t, int, Stream) {
    return 1;
  }

  static constexpr int MemcpyHostToHost = 0;

  static constexpr int MemcpyHostToDevice = 0;

  static constexpr int MemcpyDeviceToHost = 1;

  static constexpr int MemcpyDeviceToDevice = 0;

  static void Memset(void* ptr, int value, std::size_t count) {
    std::memset(ptr, value, count);
  }

  static int StreamCreate(Stream*) { return 1; }

  static int StreamDestroy(Stream) { return 1; }

  static int StreamSynchronize(Stream) { return 1; }

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

static_assert(Runtime<Device::Type::kCpu>::Validate());

}  // namespace infini::rt

#endif
