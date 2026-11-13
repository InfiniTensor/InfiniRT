#ifndef INFINI_RT_QY_DEVICE__H_
#define INFINI_RT_QY_DEVICE__H_

#include "device.h"

namespace infini::rt {

template <>
struct DeviceEnabled<Device::Type::kQy> : std::true_type {};

}  // namespace infini::rt

#endif
