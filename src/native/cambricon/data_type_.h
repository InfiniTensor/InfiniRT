#ifndef INFINI_RT_CAMBRICON_DATA_TYPE__H_
#define INFINI_RT_CAMBRICON_DATA_TYPE__H_

#include "bang_bf16.h"
#include "bang_fp16.h"
#include "native/cambricon/device_.h"
#include "data_type.h"

namespace infini::rt {

template <>
struct TypeMap<Device::Type::kCambricon, DataType::kFloat16> {
  using type = __half;
};

template <>
struct TypeMap<Device::Type::kCambricon, DataType::kBFloat16> {
  using type = __bang_bfloat16;
};

}  // namespace infini::rt

#endif
