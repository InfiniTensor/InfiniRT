#ifndef INFINI_RT_ALI_DEVICE__H_
#define INFINI_RT_ALI_DEVICE__H_

#include "device.h"

namespace infini::rt {

template <>
struct DeviceEnabled<Device::Type::kAli> : std::true_type {};

}  // namespace infini::rt

#endif
