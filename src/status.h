#ifndef INFINI_RT_STATUS_H_
#define INFINI_RT_STATUS_H_

#include <absl/status/status.h>
#include <absl/status/statusor.h>

namespace infini::rt {

using Status = absl::Status;

using StatusCode = absl::StatusCode;

using absl::InvalidArgumentError;
using absl::OkStatus;

template <typename T>
using StatusOr = absl::StatusOr<T>;

}  // namespace infini::rt

#endif
