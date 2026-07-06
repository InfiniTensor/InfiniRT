# Compatibility

This page defines the intended documentation and compatibility boundary for
InfiniRT consumers.

## Stable User Entry

Use:

```cpp
#include <infini/rt.h>
```

The stable user-facing surface is:

- `infini::rt::Device`
- `infini::rt::DataType`
- `infini::rt::TensorView`
- `infini::rt::set_runtime_device_type`
- `infini::rt::runtime_device_type`
- `infini::rt::runtime` dispatching functions and constants documented in
  [Runtime API](api/runtime.md)

## Implementation-Facing Headers

Installed headers under `infini/rt/detail/*` are generated to support the public
entry header. They are not intended as direct user includes.

Backend wrapper headers such as `infini/rt/cpu/runtime_.h` expose
backend-specific `runtime::Runtime<Device::Type::...>` specializations. They are
useful for backend tests and advanced integrations, but ordinary consumers
should prefer `<infini/rt.h>`.

## Generated Headers

Generated public headers depend on the backend options used during CMake
configuration. A consumer should compile against the installed prefix produced
by the same configured build.

## ABI Notes

InfiniRT currently exposes a C++ API. Consumers should treat the installed
headers and `libinfinirt.so` as a matching pair from the same build or release.

