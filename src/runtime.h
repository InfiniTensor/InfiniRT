#ifndef INFINI_RT_RUNTIME_H_
#define INFINI_RT_RUNTIME_H_

#include <cstddef>
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

enum class MemcpyKind {
  kHostToHost,
  kHostToDevice,
  kDeviceToHost,
  kDeviceToDevice,
};

void SetDevice(Device device);

void GetDevice(Device* device);

void GetDeviceCount(int* count, Device::Type type);

void DeviceSynchronize();

void Malloc(void** ptr, std::size_t size);

void Free(void* ptr);

void Memset(void* ptr, int value, std::size_t count);

void Memcpy(void* dst, const void* src, std::size_t count, MemcpyKind kind);

}  // namespace infini::rt

#endif
