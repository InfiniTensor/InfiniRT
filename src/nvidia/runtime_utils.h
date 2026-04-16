#ifndef INFINI_RT_NVIDIA_RUNTIME_UTILS_H_
#define INFINI_RT_NVIDIA_RUNTIME_UTILS_H_

#include "cuda/runtime_utils.h"
#include "nvidia/device_property.h"

namespace infini::rt {

template <>
struct RuntimeUtils<Device::Type::kNvidia>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::rt

#endif
