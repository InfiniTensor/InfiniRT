#ifndef INFINI_RT_KUNLUN_DATA_TYPE__H_
#define INFINI_RT_KUNLUN_DATA_TYPE__H_

#include "data_type.h"
#include "native/kunlun/device_.h"

namespace infini::rt {

// Host storage types; device kernels use the corresponding XTDK intrinsics.
template <>
struct TypeMap<Device::Type::kKunlun, DataType::kFloat16> {
  using type = Float16;
};

template <>
struct TypeMap<Device::Type::kKunlun, DataType::kBFloat16> {
  using type = BFloat16;
};

}  // namespace infini::rt

#endif
