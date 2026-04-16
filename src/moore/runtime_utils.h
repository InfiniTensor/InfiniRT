#ifndef INFINI_RT_MOORE_RUNTIME_UTILS_H_
#define INFINI_RT_MOORE_RUNTIME_UTILS_H_

#include "cuda/runtime_utils.h"
#include "moore/device_property.h"

namespace infini::rt {

template <>
struct RuntimeUtils<Device::Type::kMoore>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::rt

#endif
