# Build and Test

InfiniRT uses CMake and C++17.

## Common Options

```bash
-DWITH_CPU=ON
-DWITH_NVIDIA=ON
-DWITH_ILUVATAR=ON
-DWITH_METAX=ON
-DWITH_MOORE=ON
-DWITH_HYGON=ON
-DWITH_CAMBRICON=ON
-DWITH_ASCEND=ON
-DAUTO_DETECT_DEVICES=ON
-DINFINI_RT_BUILD_TESTING=ON
-DINFINI_RT_BUILD_DOCS=ON
```

Only one GPU backend can be enabled in a build. CPU can be enabled together
with a GPU backend. If no backend option is enabled, CPU is enabled by default.

## CPU Build

```bash
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build
```

The CPU backend requires OpenMP.

## NVIDIA Build

```bash
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON \
  -DWITH_NVIDIA=ON \
  -DINFINI_RT_BUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build
```

The NVIDIA backend requires the CUDA toolkit. The installed public headers may
include CUDA headers, so consumers also need access to the CUDA include
directory when compiling against an NVIDIA build.

## Generated Public Headers

During CMake configuration, InfiniRT generates backend-specific public headers
under `generated/include`. The generated headers match the backend options used
for that build and are installed with the hand-written headers.

The recommended consumer include remains:

```cpp
#include <infini/rt.h>
```

## Tests

Enable tests with:

```bash
-DINFINI_RT_BUILD_TESTING=ON
```

Run them with:

```bash
ctest --test-dir build --output-on-failure
```

The `test_install_consumer` test installs InfiniRT to a temporary prefix,
compiles a small external consumer against the installed prefix, and runs it.

## Documentation

Enable the Doxygen documentation target with:

```bash
cmake -S . -B build -DINFINI_RT_BUILD_DOCS=ON
cmake --build build --target infinirt_docs
```

The generated HTML is written to `build/docs/reference/html` under the source
tree.

To preview the generated structure, serve the HTML directory:

```bash
python3 -m http.server 8000 --directory build/docs/reference/html
```

The Documentation Pages workflow uses the same `infinirt_docs` target. Pull
requests build and upload the generated HTML as a Pages artifact for
validation; pushes to `master` deploy it to GitHub Pages.
