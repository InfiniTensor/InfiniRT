import argparse
import dataclasses
import pathlib
import re


_DETAIL_PREFIX = "infini/rt/detail"

_DEVICE_HEADERS = {
    "cpu": (
        ("cpu", "data_type_.h", "native/cpu/data_type_.h"),
        ("cpu", "device_.h", "native/cpu/device_.h"),
        ("cpu", "runtime_.h", "native/cpu/runtime_.h"),
    ),
    "nvidia": (
        ("nvidia", "data_type_.h", "native/cuda/nvidia/data_type_.h"),
        ("nvidia", "device_.h", "native/cuda/nvidia/device_.h"),
        ("nvidia", "runtime_.h", "native/cuda/nvidia/runtime_.h"),
    ),
    "iluvatar": (
        ("iluvatar", "data_type_.h", "native/cuda/iluvatar/data_type_.h"),
        ("iluvatar", "device_.h", "native/cuda/iluvatar/device_.h"),
        ("iluvatar", "runtime_.h", "native/cuda/iluvatar/runtime_.h"),
    ),
    "hygon": (
        ("hygon", "data_type_.h", "native/cuda/hygon/data_type_.h"),
        ("hygon", "device_.h", "native/cuda/hygon/device_.h"),
        ("hygon", "runtime_.h", "native/cuda/hygon/runtime_.h"),
    ),
    "metax": (
        ("metax", "data_type_.h", "native/cuda/metax/data_type_.h"),
        ("metax", "device_.h", "native/cuda/metax/device_.h"),
        ("metax", "runtime_.h", "native/cuda/metax/runtime_.h"),
    ),
    "moore": (
        ("moore", "data_type_.h", "native/cuda/moore/data_type_.h"),
        ("moore", "device_.h", "native/cuda/moore/device_.h"),
        ("moore", "runtime_.h", "native/cuda/moore/runtime_.h"),
    ),
    "cambricon": (
        ("cambricon", "data_type_.h", "native/cambricon/data_type_.h"),
        ("cambricon", "device_.h", "native/cambricon/device_.h"),
        ("cambricon", "runtime_.h", "native/cambricon/runtime_.h"),
    ),
    "ascend": (
        ("ascend", "data_type_.h", "native/ascend/data_type_.h"),
        ("ascend", "device_.h", "native/ascend/device_.h"),
        ("ascend", "runtime_.h", "native/ascend/runtime_.h"),
    ),
}

_DEVICE_TYPES = {
    "cpu": "Device::Type::kCpu",
    "nvidia": "Device::Type::kNvidia",
    "iluvatar": "Device::Type::kIluvatar",
    "hygon": "Device::Type::kHygon",
    "metax": "Device::Type::kMetax",
    "moore": "Device::Type::kMoore",
    "cambricon": "Device::Type::kCambricon",
    "ascend": "Device::Type::kAscend",
}

_DEFAULT_DEVICE_PRIORITY = (
    "nvidia",
    "iluvatar",
    "hygon",
    "metax",
    "moore",
    "cambricon",
    "ascend",
    "cpu",
)


def _guard(path):
    token = "_".join(path.parts).replace(".", "_").upper()

    return f"INFINI_RT_GENERATED_{token}_"


def _write_wrapper(include_root, device, header_name, target):
    path = include_root / "infini" / "rt" / device / header_name
    guard = _guard(path.relative_to(include_root))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#ifndef {guard}
#define {guard}

#include <{_DETAIL_PREFIX}/{target}>

#endif
"""
    )


def _detail_include(path):
    return f"<{_DETAIL_PREFIX}/{path}>"


def _rewrite_detail_include(match):
    target = match.group(1)
    return f"#include {_detail_include(target)}"


_DETAIL_INCLUDE_PATTERN = re.compile(
    r'#include "((?:common|native)/[^"]+|data_type\.h|device\.h|dispatcher\.h|hash\.h|runtime\.h|tensor_view\.h)"'
)


def _detail_header_dependencies(source_root, relative_path):
    source_path = source_root / relative_path
    text = source_path.read_text()

    return {
        target
        for target in _DETAIL_INCLUDE_PATTERN.findall(text)
        if (source_root / target).exists()
    }


def _write_detail_header(include_root, source_root, relative_path):
    source_path = source_root / relative_path
    output_path = include_root / _DETAIL_PREFIX / relative_path
    text = source_path.read_text()
    text = _DETAIL_INCLUDE_PATTERN.sub(_rewrite_detail_include, text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)


def _write_detail_headers(include_root, source_root, devices):
    detail_headers = {
        "common/constexpr_map.h",
        "common/traits.h",
        "data_type.h",
        "device.h",
        "dispatcher.h",
        "hash.h",
        "runtime.h",
        "tensor_view.h",
    }

    for device in devices:
        for _, _, target in _DEVICE_HEADERS[device]:
            detail_headers.add(target)

    pending_headers = list(detail_headers)
    while pending_headers:
        relative_path = pending_headers.pop()
        for dependency in _detail_header_dependencies(source_root, relative_path):
            if dependency not in detail_headers:
                detail_headers.add(dependency)
                pending_headers.append(dependency)

    for relative_path in sorted(detail_headers):
        _write_detail_header(include_root, source_root, relative_path)


def _write_generated_header(include_root, devices):
    default_device = _default_device(devices)
    default_device_type = _DEVICE_TYPES[default_device]
    includes = [
        "#include <cstddef>",
        f"#include {_detail_include('data_type.h')}",
        f"#include {_detail_include('device.h')}",
        f"#include {_detail_include('hash.h')}",
        f"#include {_detail_include('runtime.h')}",
        f"#include {_detail_include('tensor_view.h')}",
    ]

    for device in devices:
        includes.append(f"#include <infini/rt/{device}/device_.h>")

    for device in devices:
        includes.append(f"#include <infini/rt/{device}/runtime_.h>")

    path = include_root / "infini" / "rt" / "generated.h"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#ifndef INFINI_RT_GENERATED_PUBLIC_H_
#define INFINI_RT_GENERATED_PUBLIC_H_

{chr(10).join(includes)}

namespace infini::rt {{
namespace generated_detail {{

using DefaultErrorRuntime = Runtime<{default_device_type}>;

inline constexpr Device::Type kDefaultDeviceType = {default_device_type};

}}  // namespace generated_detail

using Error = typename generated_detail::DefaultErrorRuntime::Error;

using Stream = typename generated_detail::DefaultErrorRuntime::Stream;

inline constexpr Error kSuccess = generated_detail::DefaultErrorRuntime::kSuccess;

enum class MemcpyKind {{
  kMemcpyHostToHost =
      static_cast<int>(generated_detail::DefaultErrorRuntime::kMemcpyHostToHost),
  kMemcpyHostToDevice =
      static_cast<int>(generated_detail::DefaultErrorRuntime::kMemcpyHostToDevice),
  kMemcpyDeviceToHost =
      static_cast<int>(generated_detail::DefaultErrorRuntime::kMemcpyDeviceToHost),
  kMemcpyDeviceToDevice =
      static_cast<int>(generated_detail::DefaultErrorRuntime::kMemcpyDeviceToDevice),
}};

template <>
struct Runtime<Device::Type::kCount>
    : RuntimeBase<Runtime<Device::Type::kCount>> {{
  using Error = infini::rt::Error;

  using Stream = infini::rt::Stream;

  static constexpr Device::Type kDeviceType = Device::Type::kCount;

  static constexpr Error kSuccess = infini::rt::kSuccess;

  static constexpr MemcpyKind kMemcpyHostToHost =
      MemcpyKind::kMemcpyHostToHost;

  static constexpr MemcpyKind kMemcpyHostToDevice =
      MemcpyKind::kMemcpyHostToDevice;

  static constexpr MemcpyKind kMemcpyDeviceToHost =
      MemcpyKind::kMemcpyDeviceToHost;

  static constexpr MemcpyKind kMemcpyDeviceToDevice =
      MemcpyKind::kMemcpyDeviceToDevice;

  static Error SetDeviceType(Device::Type device_type);

  static Device::Type GetDeviceType();

  static Error SetDevice(int device);

  static Error GetDevice(int* device);

  static Error GetDeviceCount(int* count);

  static Error DeviceSynchronize();

  static Error Malloc(void** ptr, std::size_t size);

  static Error Free(void* ptr);

  static Error Memset(void* ptr, int value, std::size_t count);

  static Error Memcpy(void* dst, const void* src, std::size_t count,
                      MemcpyKind kind);

  static Error MemcpyAsync(void* dst, const void* src, std::size_t count,
                           MemcpyKind kind, Stream stream);

 private:
  inline static thread_local Device::Type device_type_ =
      generated_detail::kDefaultDeviceType;
}};

static_assert(Runtime<Device::Type::kCount>::Validate());

Error SetDevice(int device);

Error GetDevice(int* device);

Error GetDeviceCount(int* count);

Error DeviceSynchronize();

Error Malloc(void** ptr, std::size_t size);

Error Free(void* ptr);

Error Memset(void* ptr, int value, std::size_t count);

Error Memcpy(void* dst, const void* src, std::size_t count, MemcpyKind kind);

Error MemcpyAsync(void* dst, const void* src, std::size_t count,
                  MemcpyKind kind, Stream stream);

}}  // namespace infini::rt

#endif
"""
    )


@dataclasses.dataclass(frozen=True)
class _Param:
    type: str
    name: str


@dataclasses.dataclass(frozen=True)
class _Function:
    return_type: str
    name: str
    params: tuple[_Param, ...]

    def signature(self):
        return f"{self.return_type} {self.name}({self.params_decl()})"

    def params_decl(self):
        return ", ".join(f"{param.type} {param.name}" for param in self.params)


_PUBLIC_RUNTIME_FUNCTIONS = (
    _Function("Error", "SetDevice", (_Param("int", "device"),)),
    _Function("Error", "GetDevice", (_Param("int*", "device"),)),
    _Function("Error", "GetDeviceCount", (_Param("int*", "count"),)),
    _Function("Error", "DeviceSynchronize", ()),
    _Function(
        "Error",
        "Malloc",
        (_Param("void**", "ptr"), _Param("std::size_t", "size")),
    ),
    _Function("Error", "Free", (_Param("void*", "ptr"),)),
    _Function(
        "Error",
        "Memset",
        (
            _Param("void*", "ptr"),
            _Param("int", "value"),
            _Param("std::size_t", "count"),
        ),
    ),
    _Function(
        "Error",
        "Memcpy",
        (
            _Param("void*", "dst"),
            _Param("const void*", "src"),
            _Param("std::size_t", "count"),
            _Param("MemcpyKind", "kind"),
        ),
    ),
    _Function(
        "Error",
        "MemcpyAsync",
        (
            _Param("void*", "dst"),
            _Param("const void*", "src"),
            _Param("std::size_t", "count"),
            _Param("MemcpyKind", "kind"),
            _Param("Stream", "stream"),
        ),
    ),
)


def _default_device(devices):
    for device in _DEFAULT_DEVICE_PRIORITY:
        if device in devices:
            return device

    raise ValueError("at least one device is required")


def _runtime_arg(param, device):
    device_type = _DEVICE_TYPES[device]
    if param.type == "MemcpyKind":
        return f"RuntimeMemcpyKind<{device_type}>({param.name})"
    if param.type == "Stream":
        return (
            f"reinterpret_cast<typename Runtime<{device_type}>::Stream>"
            f"({param.name})"
        )

    return param.name


def _runtime_args(function, device):
    args = (_runtime_arg(param, device) for param in function.params)

    return ", ".join(arg for arg in args if arg is not None)


def _runtime_call(function, device):
    device_type = _DEVICE_TYPES[device]
    args = _runtime_args(function, device)
    if args:
        return f"Runtime<{device_type}>::{function.name}({args})"

    return f"Runtime<{device_type}>::{function.name}()"


def _call_args(function):
    return ", ".join(param.name for param in function.params)


def _member_signature(function):
    return (
        f"{function.return_type} Runtime<Device::Type::kCount>::"
        f"{function.name}({function.params_decl()})"
    )


def _dispatch_cases(devices, function):
    return "\n".join(
        f"""    case {_DEVICE_TYPES[device]}:
      return CheckCall([&] {{ return {_runtime_call(function, device)}; }});"""
        for device in devices
    )


def _write_member_dispatch_function(function, devices):
    return f"""{_member_signature(function)} {{
  switch (device_type_) {{
{_dispatch_cases(devices, function)}
  }}

  return InvalidValueError();
}}
"""


def _write_public_dispatch_function(function):
    args = _call_args(function)
    if args:
        call = f"Runtime<Device::Type::kCount>::{function.name}({args})"
    else:
        call = f"Runtime<Device::Type::kCount>::{function.name}()"

    return f"""{function.signature()} {{
  return {call};
}}
"""


def _write_runtime_dispatch(source_path, devices):
    functions = _PUBLIC_RUNTIME_FUNCTIONS
    member_dispatch_functions = "\n".join(
        _write_member_dispatch_function(function, devices=devices)
        for function in functions
    )
    public_dispatch_functions = "\n".join(
        _write_public_dispatch_function(function) for function in functions
    )
    set_device_type_cases = "\n".join(
        f"""    case {_DEVICE_TYPES[device]}:
      device_type_ = device_type;
      return kSuccess;"""
        for device in devices
    )

    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        f"""#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <infini/rt/generated.h>

namespace infini::rt {{
namespace {{

template <typename Func>
Error CheckCall(Func&& func) {{
  using ReturnType = decltype(std::forward<Func>(func)());

  if constexpr (std::is_void_v<ReturnType>) {{
    std::forward<Func>(func)();
    return kSuccess;
  }} else {{
    return static_cast<Error>(std::forward<Func>(func)());
  }}
}}

Error InvalidValueError() {{ return static_cast<Error>(1); }}

template <Device::Type device_type>
auto RuntimeMemcpyKind(MemcpyKind kind) {{
  using DeviceRuntime = Runtime<device_type>;

  switch (kind) {{
    case MemcpyKind::kMemcpyHostToHost:
      return DeviceRuntime::kMemcpyHostToHost;
    case MemcpyKind::kMemcpyHostToDevice:
      return DeviceRuntime::kMemcpyHostToDevice;
    case MemcpyKind::kMemcpyDeviceToHost:
      return DeviceRuntime::kMemcpyDeviceToHost;
    case MemcpyKind::kMemcpyDeviceToDevice:
      return DeviceRuntime::kMemcpyDeviceToDevice;
  }}

  assert(false && "unsupported memcpy kind");
  return DeviceRuntime::kMemcpyHostToHost;
}}

}}  // namespace

Error Runtime<Device::Type::kCount>::SetDeviceType(Device::Type device_type) {{
  switch (device_type) {{
{set_device_type_cases}
  }}

  return InvalidValueError();
}}

Device::Type Runtime<Device::Type::kCount>::GetDeviceType() {{
  return device_type_;
}}

{member_dispatch_functions}
{public_dispatch_functions}
}}  // namespace infini::rt
"""
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="generated/include")
    parser.add_argument("--source-output", default="generated/src/runtime_dispatch.cc")
    parser.add_argument("--runtime-header", default="src/runtime.h")
    parser.add_argument("--devices", nargs="+", required=True)
    args = parser.parse_args()

    devices = []
    for device in args.devices:
        if device not in _DEVICE_HEADERS:
            raise ValueError(f"unsupported device {device!r}")
        if device not in devices:
            devices.append(device)

    include_root = pathlib.Path(args.output_dir)
    source_root = pathlib.Path(args.runtime_header).parent

    _write_detail_headers(include_root, source_root, devices)
    for device in devices:
        for wrapper_device, header_name, target in _DEVICE_HEADERS[device]:
            _write_wrapper(include_root, wrapper_device, header_name, target)

    _write_generated_header(include_root, devices)
    _write_runtime_dispatch(pathlib.Path(args.source_output), devices)


if __name__ == "__main__":
    main()
