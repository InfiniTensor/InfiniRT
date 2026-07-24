#ifndef INFINI_RT_HPCC_RUNTIME_H_
#define INFINI_RT_HPCC_RUNTIME_H_

#include <type_traits>

#include "runtime.h"

namespace infini::rt::runtime {

template <typename Derived>
struct HpccRuntime : DeviceRuntime<Derived> {
  static constexpr bool Validate() {
    DeviceRuntime<Derived>::Validate();
    static_assert(
        std::is_invocable_v<decltype(Derived::Memcpy), void*, const void*,
                            size_t, decltype(Derived::kMemcpyHostToDevice)>,
        "`Runtime::Memcpy` must accept the HPCC memcpy kind.");
    static_assert(
        std::is_invocable_v<decltype(Derived::MemcpyAsync), void*, const void*,
                            size_t, decltype(Derived::kMemcpyHostToDevice),
                            typename Derived::Stream>,
        "`Runtime::MemcpyAsync` must accept the HPCC memcpy kind and stream.");
    return true;
  }
};

}  // namespace infini::rt::runtime

#endif
