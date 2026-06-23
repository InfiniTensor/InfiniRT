#ifndef INFINI_RT_GRAPH_H_
#define INFINI_RT_GRAPH_H_

#include <cstdint>

#include "device.h"

namespace infini::rt {

enum class StreamCaptureMode {
  kGlobal = 0,
  kThreadLocal = 1,
  kRelaxed = 2,
};

// Public dispatch wrappers keep backend handles opaque while preserving
// enough device identity for cross-device graph dispatch.
class Stream {
 public:
  Stream() = default;

  Stream(Device::Type device_type, void* raw)
      : device_type_{device_type}, raw_{raw} {}

  Device::Type device_type() const { return device_type_; }

  void* raw() const { return raw_; }

  explicit operator bool() const { return raw_ != nullptr; }

 private:
  Device::Type device_type_{Device::Type::kCpu};
  void* raw_{nullptr};
};

class Graph {
 public:
  Graph() = default;

  Graph(Device::Type device_type, void* raw)
      : device_type_{device_type}, raw_{raw} {}

  Device::Type device_type() const { return device_type_; }

  void* raw() const { return raw_; }

  explicit operator bool() const { return raw_ != nullptr; }

 private:
  Device::Type device_type_{Device::Type::kCpu};
  void* raw_{nullptr};
};

class GraphExec {
 public:
  GraphExec() = default;

  GraphExec(Device::Type device_type, void* raw)
      : device_type_{device_type}, raw_{raw} {}

  Device::Type device_type() const { return device_type_; }

  void* raw() const { return raw_; }

  explicit operator bool() const { return raw_ != nullptr; }

 private:
  Device::Type device_type_{Device::Type::kCpu};
  void* raw_{nullptr};
};

}  // namespace infini::rt

#endif
