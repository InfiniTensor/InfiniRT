#ifndef INFINI_RT_MARS_DATA_TYPE__H_
#define INFINI_RT_MARS_DATA_TYPE__H_

#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>
#include <hcr/hc_runtime.h>

#include "data_type.h"
#include "native/hpcc/mars/device_.h"

namespace infini::rt {

template <>
struct TypeMap<Device::Type::kMars, DataType::kFloat16> {
  using type = __half;
};

template <>
struct TypeMap<Device::Type::kMars, DataType::kBFloat16> {
  using type = __maca_bfloat16;
};

}  // namespace infini::rt

#endif
