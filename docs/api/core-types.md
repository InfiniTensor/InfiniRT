# Core Types

Core types are available through:

```cpp
#include <infini/rt.h>
```

## Device

`infini::rt::Device` identifies a backend type and an index.

```cpp
infini::rt::Device device{infini::rt::Device::Type::kCpu, 0};
auto type = device.type();
auto index = device.index();
```

Known device types include:

- `Device::Type::kCpu`
- `Device::Type::kNvidia`
- `Device::Type::kIluvatar`
- `Device::Type::kMetax`
- `Device::Type::kMoore`
- `Device::Type::kHygon`
- `Device::Type::kCambricon`
- `Device::Type::kAscend`

## DataType

`infini::rt::DataType` describes tensor element types.

Supported values include signed integers, unsigned integers, reduced precision
floating point, and standard floating point types:

- `kInt8`, `kInt16`, `kInt32`, `kInt64`
- `kUInt8`, `kUInt16`, `kUInt32`, `kUInt64`
- `kFloat16`, `kBFloat16`, `kFloat32`, `kFloat64`

## TensorView

`infini::rt::TensorView` is a non-owning description of tensor memory.

```cpp
std::vector<float> data(16);

infini::rt::TensorView tensor{
    data.data(),
    std::vector<std::size_t>{4, 4},
    infini::rt::DataType::kFloat32,
    infini::rt::Device{infini::rt::Device::Type::kCpu, 0},
    std::vector<std::ptrdiff_t>{4, 1}};

auto elements = tensor.numel();
auto bytes_per_element = tensor.element_size();
auto contiguous = tensor.IsContiguous();
```

`TensorView` stores:

- data pointer
- shape
- data type
- device
- strides

It does not own the memory it references.
