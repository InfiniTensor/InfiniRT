#ifndef INFINI_RT_METAX_RUNTIME_UTILS_H_
#define INFINI_RT_METAX_RUNTIME_UTILS_H_

#include "cuda/runtime_utils.h"
#include "metax/device_property.h"

namespace infini::rt {

template <>
struct RuntimeUtils<Device::Type::kMetax>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::rt

#endif
