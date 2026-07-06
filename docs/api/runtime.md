# Runtime API

The runtime API is exposed through:

```cpp
#include <infini/rt.h>
```

Runtime functions live in `infini::rt::runtime`.

## Backend Selection

The top-level functions select which enabled backend receives
`infini::rt::runtime` calls:

```cpp
infini::rt::set_runtime_device_type(infini::rt::Device::Type::kCpu);
auto current = infini::rt::runtime_device_type();
```

The selected backend is thread-local. A GPU backend is selected initially when
one is enabled; otherwise CPU is selected.

Use the dispatching API for ordinary code:

```cpp
namespace rt = infini::rt::runtime;

infini::rt::set_runtime_device_type(infini::rt::Device::Type::kCpu);

void* ptr = nullptr;
rt::Malloc(&ptr, 1024);
rt::Free(ptr);
```

## CUDA Runtime API Alignment

The `infini::rt::runtime` namespace follows CUDA Runtime API naming, call
shapes, return-status conventions, and handle-oriented stream/event usage where
practical. Backend implementations translate those calls to the enabled runtime
backend, while unsupported operations return a non-success status without
changing the API shape. The signatures below are therefore listed without
per-function explanations; for CUDA-like operations, use the corresponding CUDA
Runtime API behavior as the baseline unless a backend support note says
otherwise.

## Runtime Status and Types

```cpp
runtime::Error
runtime::Stream
runtime::Event
runtime::MemcpyKind
runtime::kSuccess
runtime::kMemcpyHostToHost
runtime::kMemcpyHostToDevice
runtime::kMemcpyDeviceToHost
runtime::kMemcpyDeviceToDevice
```

## Device Functions

```cpp
runtime::Error SetDevice(int device);
runtime::Error GetDevice(int* device);
runtime::Error GetDeviceCount(int* count);
runtime::Error DeviceSynchronize();
```

## Memory Functions

```cpp
runtime::Error Malloc(void** ptr, std::size_t size);
runtime::Error MallocHost(void** ptr, std::size_t size);
runtime::Error MallocAsync(void** ptr, std::size_t size, runtime::Stream stream);
runtime::Error Free(void* ptr);
runtime::Error FreeHost(void* ptr);
runtime::Error FreeAsync(void* ptr, runtime::Stream stream);
runtime::Error MemGetInfo(std::size_t* free, std::size_t* total);
runtime::Error Memcpy(void* dst, const void* src, std::size_t count,
                      runtime::MemcpyKind kind);
runtime::Error MemcpyAsync(void* dst, const void* src, std::size_t count,
                           runtime::MemcpyKind kind, runtime::Stream stream);
runtime::Error Memset(void* ptr, int value, std::size_t count);
runtime::Error MemsetAsync(void* ptr, int value, std::size_t count,
                           runtime::Stream stream);
```

Not every backend supports every optional operation. Unsupported operations
return a non-success status. See [Backends](../backends.md).

## Stream and Event Functions

```cpp
runtime::Error StreamCreate(runtime::Stream* stream);
runtime::Error StreamDestroy(runtime::Stream stream);
runtime::Error StreamSynchronize(runtime::Stream stream);
runtime::Error StreamWaitEvent(runtime::Stream stream, runtime::Event event,
                               unsigned int flags);
runtime::Error EventCreate(runtime::Event* event);
runtime::Error EventCreateWithFlags(runtime::Event* event, unsigned int flags);
runtime::Error EventRecord(runtime::Event event, runtime::Stream stream);
runtime::Error EventQuery(runtime::Event event);
runtime::Error EventSynchronize(runtime::Event event);
runtime::Error EventDestroy(runtime::Event event);
runtime::Error EventElapsedTime(float* ms, runtime::Event start,
                                runtime::Event end);
```

## Backend-Specific Runtime Templates

Backend runtime specializations are available as implementation-facing headers,
for example:

```cpp
#include <infini/rt/cpu/runtime_.h>
```

These headers are useful for backend tests and low-level integrations, but
ordinary consumers should prefer the dispatching API from `<infini/rt.h>`.
