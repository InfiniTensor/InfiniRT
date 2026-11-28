#include <cstdint>
#include <cstring>

#include "data_type.h"
#include "native/kunlun/device_.h"

// The generated public header now pulls this file into host compilation
// units (upstream per-device data_type includes). XTDK device headers
// (`xpu/kernel/xtdk_*.h`) only compile under the XTDK device compiler
// (`__xpu__`), so host TUs get plain 16-bit payload views while device
// compilation keeps using the native XTDK types.
#ifndef __xpu__
namespace infini::rt {

// Host-side 16-bit payload views (bit-identical layout to the device types).
struct kunlun_half {
  std::uint16_t bits;
};

struct kunlun_bfloat16 {
  std::uint16_t bits;
};

template <>
struct TypeMap<Device::Type::kKunlun, DataType::kFloat16> {
  using type = kunlun_half;
};

template <>
struct TypeMap<Device::Type::kKunlun, DataType::kBFloat16> {
  using type = kunlun_bfloat16;
};

}  // namespace infini::rt
#else
namespace infini::rt {

// Kunlun XPU3 (XTDK) provides host+device `half` (`__half`, from
// xtdk_fp16.h) and `bfloat16_t` (`__xpu_bfloat16`, from xtdk_bf16.hpp).
using kunlun_half = half;
using kunlun_bfloat16 = bfloat16_t;

template <>
struct TypeMap<Device::Type::kKunlun, DataType::kFloat16> {
  using type = kunlun_half;
};

template <>
struct TypeMap<Device::Type::kKunlun, DataType::kBFloat16> {
  using type = kunlun_bfloat16;
};

}  // namespace infini::rt
#endif  // __xpu__

