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

_NVIDIA_RUNTIME_HEADER = "native/cuda/nvidia/runtime_.h"

_PUBLIC_TYPE_NAMES = {
    "cudaMemcpyKind": "MemcpyKind",
    "size_t": "std::size_t",
}


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


def _write_generated_header(include_root, source_root, devices):
    runtime_api = _parse_nvidia_runtime_api(source_root)
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

    runtime_aliases = "\n\n".join(
        f"using {alias} = typename generated_detail::DefaultRuntime::{alias};"
        for alias in runtime_api.aliases
    )
    memcpy_kind_values = "\n".join(
        f"""  {constant} =
      static_cast<int>(generated_detail::DefaultRuntime::{constant}),"""
        for constant in runtime_api.memcpy_kind_constants
    )
    runtime_declarations = "\n\n".join(
        f"{function.signature()};" for function in runtime_api.functions
    )

    path = include_root / "infini" / "rt" / "generated.h"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#ifndef INFINI_RT_GENERATED_PUBLIC_H_
#define INFINI_RT_GENERATED_PUBLIC_H_

{chr(10).join(includes)}

namespace infini::rt {{

void set_runtime_device_type(Device::Type device_type);

Device::Type runtime_device_type();

namespace runtime {{
namespace generated_detail {{

using DefaultRuntime = Runtime<{default_device_type}>;

inline constexpr Device::Type kDefaultDeviceType = {default_device_type};

}}  // namespace generated_detail

{runtime_aliases}

inline constexpr Error kSuccess = generated_detail::DefaultRuntime::kSuccess;

enum class MemcpyKind {{
{memcpy_kind_values}
}};

{runtime_declarations}

}}  // namespace runtime
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


@dataclasses.dataclass(frozen=True)
class _RuntimeApi:
    aliases: tuple[str, ...]
    memcpy_kind_constants: tuple[str, ...]
    functions: tuple[_Function, ...]


def _runtime_struct_body(source_root):
    text = (source_root / _NVIDIA_RUNTIME_HEADER).read_text()
    match = re.search(
        r"struct Runtime<Device::Type::kNvidia>.*?\{\n(?P<body>.*?)\n\};",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("could not find NVIDIA Runtime specialization")

    return match.group("body")


def _public_type_name(name):
    return _PUBLIC_TYPE_NAMES.get(name, name)


def _parse_param(param):
    param_type, param_name = " ".join(param.strip().split()).rsplit(" ", 1)

    return _Param(_public_type_name(param_type), param_name)


def _parse_function(match):
    params = match.group("params").strip()
    return _Function(
        _public_type_name(match.group("return_type")),
        match.group("name"),
        tuple(_parse_param(param) for param in re.split(r"\s*,\s*", params) if param),
    )


def _parse_nvidia_runtime_api(source_root):
    body = _runtime_struct_body(source_root)
    aliases = tuple(
        alias
        for alias in re.findall(r"^\s+using\s+(\w+)\s*=", body, flags=re.MULTILINE)
        if alias in {"Error", "Stream"}
    )
    memcpy_kind_constants = tuple(
        re.findall(
            r"^\s+static constexpr auto (kMemcpy\w+)\s*=",
            body,
            flags=re.MULTILINE,
        )
    )
    functions = tuple(
        _parse_function(match)
        for match in re.finditer(
            r"^\s+static\s+(?P<return_type>\w+)\s+"
            r"(?P<name>[A-Z]\w*)\((?P<params>[^)]*)\)\s*\{",
            body,
            flags=re.MULTILINE,
        )
    )

    if not aliases or not memcpy_kind_constants or not functions:
        raise ValueError("NVIDIA Runtime specialization has no public API")

    return _RuntimeApi(aliases, memcpy_kind_constants, functions)


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
            f"reinterpret_cast<typename Runtime<{device_type}>::Stream>({param.name})"
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


def _dispatch_cases(devices, function):
    return "\n".join(
        f"""    case {_DEVICE_TYPES[device]}:
      return CheckCall([&] {{ return {_runtime_call(function, device)}; }});"""
        for device in devices
    )


def _write_runtime_dispatch_function(function, devices):
    return f"""{function.signature()} {{
  switch (infini::rt::runtime_device_type()) {{
{_dispatch_cases(devices, function)}
  }}

  assert(false && "unsupported runtime device type");
  return InvalidValueError();
}}
"""


def _write_runtime_dispatch(source_path, source_root, devices):
    functions = _parse_nvidia_runtime_api(source_root).functions
    dispatch_functions = "\n".join(
        _write_runtime_dispatch_function(function, devices=devices)
        for function in functions
    )
    set_device_type_cases = "\n".join(
        f"""    case {_DEVICE_TYPES[device]}:
      runtime_device_type_ = device_type;
      return;"""
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

thread_local Device::Type runtime_device_type_ =
    runtime::generated_detail::kDefaultDeviceType;

}}  // namespace

void set_runtime_device_type(Device::Type device_type) {{
  switch (device_type) {{
{set_device_type_cases}
  }}

  assert(false && "unsupported runtime device type");
}}

Device::Type runtime_device_type() {{
  return runtime_device_type_;
}}

}}  // namespace infini::rt

namespace infini::rt::runtime {{
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

{dispatch_functions}
}}  // namespace infini::rt::runtime
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

    _write_generated_header(include_root, source_root, devices)
    _write_runtime_dispatch(pathlib.Path(args.source_output), source_root, devices)


if __name__ == "__main__":
    main()
