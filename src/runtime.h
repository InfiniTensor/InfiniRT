#ifndef INFINI_RT_RUNTIME_H_
#define INFINI_RT_RUNTIME_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "device.h"

namespace infini::rt {

template <Device::Type device_type>
struct Runtime;

/// ## Interface enforcement via CRTP.
///
/// Inherit from the appropriate base to declare which interface level a
/// `Runtime` specialization implements. After the struct is fully defined, call
/// `static_assert(Runtime<...>::Validate())`. The chained `Validate()` checks
/// every required member's existence and signature at compile time, analogous
/// to how `override` catches signature mismatches for virtual functions.
///
/// - `RuntimeBase`: `kDeviceType` only (e.g. CPU).
/// - `DeviceRuntime`: adds `Stream`, `Malloc`, and `Free` (e.g. Cambricon).

/// Every Runtime must provide `static constexpr Device::Type kDeviceType`.
template <typename Derived>
struct RuntimeBase {
  static constexpr bool Validate() {
    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Derived::kDeviceType)>,
                       Device::Type>,
        "`Runtime` must define `static constexpr Device::Type kDeviceType`.");
    return true;
  }
};

/// Runtimes with device memory must additionally provide `Stream`, `Malloc`,
/// and `Free`.
template <typename Derived>
struct DeviceRuntime : RuntimeBase<Derived> {
  static constexpr bool Validate() {
    RuntimeBase<Derived>::Validate();
    static_assert(sizeof(typename Derived::Stream) > 0,
                  "`Runtime` must define a `Stream` type alias.");
    static_assert(
        std::is_invocable_v<decltype(Derived::Malloc), void**, size_t>,
        "`Runtime::Malloc` must be callable with `(void**, size_t)`.");
    static_assert(std::is_invocable_v<decltype(Derived::Free), void*>,
                  "`Runtime::Free` must be callable with `(void*)`.");
    return true;
  }
};

enum class Error {
  kSuccess = 0,
  kUnSuccess = -1,
};

inline constexpr infini::rt::Error kSuccess = infini::rt::Error::kSuccess;
inline constexpr infini::rt::Error kUnSuccess = infini::rt::Error::kUnSuccess;

enum class MemcpyKind {
  kMemcpyHostToHost,
  kMemcpyHostToDevice,
  kMemcpyDeviceToHost,
  kMemcpyDeviceToDevice,
};

using Stream = void*;
using Event = void*;

infini::rt::Error SetDevice(int device);

infini::rt::Error GetDevice(int* device);

infini::rt::Error GetDeviceCount(int* count);

infini::rt::Error DeviceSynchronize();

infini::rt::Error Malloc(void** ptr, std::size_t size);

infini::rt::Error Free(void* ptr);

infini::rt::Error Memset(void* ptr, int value, std::size_t count);

infini::rt::Error Memcpy(void* dst, const void* src, std::size_t count,
                         infini::rt::MemcpyKind kind);

infini::rt::Error MallocHost(void** ptr, std::size_t size);

infini::rt::Error FreeHost(void* ptr);

infini::rt::Error MemcpyAsync(void* dst, const void* src, std::size_t count,
                              infini::rt::MemcpyKind kind,
                              infini::rt::Stream stream);

infini::rt::Error MallocAsync(void** ptr, std::size_t size,
                              infini::rt::Stream stream);

infini::rt::Error FreeAsync(void* ptr, infini::rt::Stream stream);

infini::rt::Error MemsetAsync(void* ptr, int value, std::size_t count,
                              infini::rt::Stream stream);

infini::rt::Error MemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes);

infini::rt::Error StreamCreate(infini::rt::Stream* stream);

infini::rt::Error StreamDestroy(infini::rt::Stream stream);

infini::rt::Error StreamSynchronize(infini::rt::Stream stream);

infini::rt::Error StreamWaitEvent(infini::rt::Stream stream,
                                  infini::rt::Event event, std::uint32_t flags);

infini::rt::Error EventCreate(infini::rt::Event* event);

infini::rt::Error EventCreateWithFlags(infini::rt::Event* event,
                                       std::uint32_t flags);

infini::rt::Error EventRecord(infini::rt::Event event,
                              infini::rt::Stream stream);

infini::rt::Error EventQuery(infini::rt::Event event);

infini::rt::Error EventSynchronize(infini::rt::Event event);

infini::rt::Error EventDestroy(infini::rt::Event event);

infini::rt::Error EventElapsedTime(float* ms, infini::rt::Event start,
                                   infini::rt::Event end);

}  // namespace infini::rt

#endif
