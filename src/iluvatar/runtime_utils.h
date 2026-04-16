#ifndef INFINI_RT_ILUVATAR_RUNTIME_UTILS_H_
#define INFINI_RT_ILUVATAR_RUNTIME_UTILS_H_

#include "cuda/runtime_utils.h"
#include "iluvatar/device_property.h"

namespace infini::rt {

template <>
struct RuntimeUtils<Device::Type::kIluvatar>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::rt

#endif
