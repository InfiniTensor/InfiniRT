#ifndef INFINI_RT_CPU_DATA_TYPE__H_
#define INFINI_RT_CPU_DATA_TYPE__H_

#include "cpu/device_.h"
#include "data_type.h"

namespace infini::rt {

template <>
struct TypeMap<Device::Type::kCpu, DataType::kFloat16> {
  using type = Float16;
};

template <>
struct TypeMap<Device::Type::kCpu, DataType::kBFloat16> {
  using type = BFloat16;
};

}  // namespace infini::rt

#endif
