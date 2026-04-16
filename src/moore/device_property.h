#ifndef INFINI_RT_MOORE_DEVICE_PROPERTY_H_
#define INFINI_RT_MOORE_DEVICE_PROPERTY_H_

#include <musa_runtime.h>

namespace infini::rt {

inline int QueryMaxThreadsPerBlock() {
  int device = 0;
  musaGetDevice(&device);
  musaDeviceProp prop;
  musaGetDeviceProperties(&prop, device);
  return prop.maxThreadsPerBlock;
}

}  // namespace infini::rt

#endif
