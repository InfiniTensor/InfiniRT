# InfiniRT

InfiniRT is the runtime library of the InfiniCore project. It provides a stable runtime API for device management and memory operations across supported backends such as CPU and NVIDIA.

The recommended public entry is:

```cpp
#include <infini/rt.h>
```

## Build and Install

Configure, build, and install InfiniRT with CMake:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/infini-rt-prefix
cmake --build build -j
cmake --install build
```

`infini-rt-prefix` means the installation prefix used by CMake. After installation, headers and libraries are placed under this prefix, for example:

```text
/path/to/infini-rt-prefix/include
/path/to/infini-rt-prefix/lib
```

## Build Options

Common options include:

```bash
-DWITH_CPU=ON
-DWITH_NVIDIA=ON
-DINFINI_RT_BUILD_TESTING=ON
```

Example CPU build:

```bash
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build
```

Example NVIDIA build:

```bash
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=/path/to/infini-rt-prefix \
  -DWITH_NVIDIA=ON \
  -DINFINI_RT_BUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build
```

## Basic Usage

```cpp
#include <cstddef>
#include <infini/rt.h>

int main() {
  infini::rt::runtime::SetDevice(0);

  constexpr std::size_t size = 1024;
  void* ptr = nullptr;

  infini::rt::runtime::Malloc(&ptr, size);
  infini::rt::runtime::Memset(ptr, 0, size);
  infini::rt::runtime::Free(ptr);

  return 0;
}
```

The CUDA Runtime API-aligned layer lives under `infini::rt::runtime`. The
top-level `infini::rt::set_runtime_device_type` and
`infini::rt::runtime_device_type` APIs select which enabled backend receives
those runtime calls. A GPU backend is selected initially when one is enabled;
otherwise CPU is selected.

```cpp
constexpr std::size_t size = 1024;
void* ptr = nullptr;

infini::rt::set_runtime_device_type(infini::rt::Device::Type::kCpu);
infini::rt::runtime::Malloc(&ptr, size);
infini::rt::runtime::Free(ptr);

infini::rt::set_runtime_device_type(infini::rt::Device::Type::kNvidia);
infini::rt::runtime::Malloc(&ptr, size);
infini::rt::runtime::Free(ptr);
```

Use `infini::rt::runtime::Runtime<infini::rt::Device::Type::kCpu>` when CPU
runtime calls are needed explicitly in a build that also enables an accelerator
backend.

## Using Installed InfiniRT From Another Project

Downstream projects should consume the installed headers and libraries instead of depending on the InfiniRT source tree.

Typical include and library paths are:

```text
/path/to/infini-rt-prefix/include
/path/to/infini-rt-prefix/lib
```

A simple compile command may look like:

```bash
c++ main.cc \
  -I/path/to/infini-rt-prefix/include \
  -L/path/to/infini-rt-prefix/lib \
  -linfinirt
```

At runtime, make sure the dynamic linker can find `libinfinirt.so`, for example:

```bash
export LD_LIBRARY_PATH=/path/to/infini-rt-prefix/lib:$LD_LIBRARY_PATH
```

## Tests

Enable C++ tests with:

```bash
-DINFINI_RT_BUILD_TESTING=ON
```

Then run:

```bash
ctest --test-dir build --output-on-failure
```

## Formatting

C++ code should be formatted with `clang-format`.

Python scripts should be checked and formatted with Ruff:

```bash
ruff format --check .
ruff check .
```

## Contributing

Please follow the repository's `CONTRIBUTING.md`.
