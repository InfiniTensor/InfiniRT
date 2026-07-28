# Backends

InfiniRT can be built with CPU and one accelerator backend. The public runtime
API remains the same, while support for optional operations can differ by
backend.

## Backend Options

| Backend | CMake option | Notes |
| --- | --- | --- |
| CPU | `WITH_CPU` | Enabled by default when no backend option is set. |
| NVIDIA | `WITH_NVIDIA` | Requires CUDA toolkit. |
| Iluvatar | `WITH_ILUVATAR` | CUDA-compatible backend using clang with ivcore flags. |
| MetaX | `WITH_METAX` | Requires `MACA_PATH`. |
| Moore | `WITH_MOORE` | Requires `MUSA_ROOT`, `MUSA_HOME`, or `MUSA_PATH`. |
| Hygon | `WITH_HYGON` | Requires DTK and a DTK CUDA toolkit. |
| Cambricon | `WITH_CAMBRICON` | Requires `NEUWARE_HOME`. |
| Ascend | `WITH_ASCEND` | Requires Ascend toolkit. |

Only one accelerator backend can be enabled at a time. CPU can be enabled
together with the selected accelerator backend.

## Runtime API Support

All enabled runtime backends provide the core device and memory entry points
needed by the dispatching runtime API. Optional operations may return an error
status on backends that do not support them.

Current test expectations are:

| Backend | Async memcpy | Host memory | Async memory | Memory info | Async memset | Events | Graph capture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU | No | Yes | No | Yes | No | Yes | No |
| NVIDIA | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Iluvatar | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| MetaX | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Moore | Yes | Yes | No | Yes | Yes | Yes | No |
| Hygon | Yes | Yes | Yes | Yes | Yes | Yes | No |
| Cambricon | Yes | No | No | No | No | No | No |
| Ascend | Yes | No | No | No | No | No | Yes |

Treat a non-`kSuccess` status as the portable way to detect unsupported
operations.

## Header Dependencies

Backend public wrappers may include vendor headers. For example, an NVIDIA
build may require CUDA headers to be available when compiling a downstream
consumer that includes `<infini/rt.h>`.
