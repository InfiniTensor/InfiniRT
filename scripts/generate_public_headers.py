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
    "metax": "Device::Type::kMetax",
    "moore": "Device::Type::kMoore",
    "cambricon": "Device::Type::kCambricon",
    "ascend": "Device::Type::kAscend",
}

_RUNTIME_HEADERS = {
    "cpu": "native/cpu/runtime_.h",
    "nvidia": "native/cuda/nvidia/runtime_.h",
    "iluvatar": "native/cuda/iluvatar/runtime_.h",
    "metax": "native/cuda/metax/runtime_.h",
    "moore": "native/cuda/moore/runtime_.h",
    "cambricon": "native/cambricon/runtime_.h",
    "ascend": "native/ascend/runtime_.h",
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


def _write_generated_header(include_root, devices):
    includes = [
        f"#include {_detail_include('data_type.h')}",
        f"#include {_detail_include('device.h')}",
        f"#include {_detail_include('hash.h')}",
        f"#include {_detail_include('runtime.h')}",
        f"#include {_detail_include('tensor_view.h')}",
    ]

    for device in devices:
        includes.append(f"#include <infini/rt/{device}/device_.h>")

    path = include_root / "infini" / "rt" / "generated.h"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#ifndef INFINI_RT_GENERATED_PUBLIC_H_
#define INFINI_RT_GENERATED_PUBLIC_H_

{chr(10).join(includes)}

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


def _parse_param(param):
    param_type, param_name = param.strip().rsplit(" ", 1)

    return _Param(param_type, param_name)


def _parse_runtime_functions(runtime_header):
    text = pathlib.Path(runtime_header).read_text()
    return tuple(
        _Function(
            return_type,
            name,
            tuple(
                _parse_param(param)
                for param in re.split(r",\s*", params.strip())
                if param
            ),
        )
        for return_type, name, params in re.findall(
            r"^(infini::rt::Error|Error|void) ([A-Z]\w*)\(([^()]*)\);$",
            text,
            re.MULTILINE,
        )
    )


def _unsupported_statement():
    return "      return infini::rt::kUnSuccess;"


def _dispatch_cases(devices, statements):
    return "\n".join(
        f"""    case {_DEVICE_TYPES[device]}: {{
{statements.replace("__DEVICE_TYPE__", _DEVICE_TYPES[device])}
    }}"""
        for device in devices
    )


def _selector(function):
    for param in function.params:
        if param.type == "Device":
            return f"{param.name}.type()"
        if param.type == "Device::Type":
            return param.name

    return "current_device.type()"


def _runtime_arg(function, param):
    if param.type == "Device":
        if function.name in {"SetDevice", "GetDeviceResourceSnapshot"}:
            return f"{param.name}.index()"
        return None
    if param.type == "Device::Type":
        return None
    if param.type in {"MemcpyKind", "infini::rt::MemcpyKind"}:
        return f"RuntimeMemcpyKind<__DEVICE_TYPE__>({param.name})"

    return param.name


def _runtime_args(function):
    args = (_runtime_arg(function, param) for param in function.params)

    return ", ".join(arg for arg in args if arg is not None)


def _preconditions(function):
    required_pointer_names = {
        "GetDevice": {"device"},
        "GetDeviceCount": {"count"},
    }
    checks = []
    for param in function.params:
        if (
            param.type.endswith("**")
            or param.type.endswith("*")
            or param.name in required_pointer_names.get(function.name, set())
        ):
            checks.append(f"  if ({param.name} == nullptr) {{")
            checks.append("    return infini::rt::kUnSuccess;")
            checks.append("  }")

    return "\n".join(checks)


def _post_dispatch(function):
    if function.name == "SetDevice":
        return "\n      if (rt_status == infini::rt::kSuccess) {\n        current_device = Device{current_device.type(), device};\n      }"

    return ""


def _runtime_call(function):
    args = _runtime_args(function)
    if args:
        return f"Runtime<__DEVICE_TYPE__>::{function.name}({args})"

    return f"Runtime<__DEVICE_TYPE__>::{function.name}()"


def _write_get_device(function, devices):
    device_param = function.params[0].name
    cases = _dispatch_cases(
        devices,
        f"""      infini::rt::Error status = CheckCall([&] {{ return Runtime<__DEVICE_TYPE__>::GetDevice({device_param}); }});
      if (status != infini::rt::kSuccess) {{
        return status;
      }}
      current_device = Device{{current_device.type(), *{device_param}}};
      return infini::rt::kSuccess;""",
    )

    return f"""{function.return_type} GetDevice(int* {device_param}) {{
  if ({device_param} == nullptr) {{
    return infini::rt::kUnSuccess;
  }}

  switch (current_device.type()) {{
{cases}
    default:
{_unsupported_statement()}
  }}
}}
"""


def _write_dispatch_function(function, devices):
    if function.name == "GetDevice":
        return _write_get_device(function, devices)

    cases = _dispatch_cases(
        devices,
        f"""      infini::rt::Error rt_status = CheckCall([&] {{ return {_runtime_call(function)}; }});{_post_dispatch(function)}
      return rt_status;""",
    )
    preconditions = _preconditions(function)
    if preconditions:
        preconditions = f"{preconditions}\n\n"

    return f"""{function.signature()} {{
{preconditions}  switch ({_selector(function)}) {{
{cases}
    default:
{_unsupported_statement()}
  }}
}}
"""


def _runtime_header_for_device(source_root, device):
    return source_root / _RUNTIME_HEADERS[device]


def _devices_for_function(function, devices, source_root):
    enabled = []
    pattern = re.compile(r"\b" + re.escape(function.name) + r"\b")
    for device in devices:
        runtime_header = _runtime_header_for_device(source_root, device)
        if runtime_header.exists() and pattern.search(runtime_header.read_text()):
            enabled.append(device)
    return enabled


def _write_runtime_dispatch(source_path, runtime_header, devices):
    first_device_type = _DEVICE_TYPES[devices[0]]
    includes = ['#include "runtime.h"']
    includes.extend(f'#include "{_RUNTIME_HEADERS[device]}"' for device in devices)
    functions = _parse_runtime_functions(runtime_header)
    source_root = pathlib.Path(runtime_header).parent
    dispatch_functions = "\n".join(
        _write_dispatch_function(
            function, _devices_for_function(function, devices, source_root)
        )
        for function in functions
    )

    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        f"""#include <cassert>
#include <type_traits>
#include <utility>

{chr(10).join(includes)}

namespace infini::rt {{
namespace {{

thread_local Device current_device{{{first_device_type}, 0}};

template <typename Func>
infini::rt::Error CheckCall(Func&& func) {{
  using ReturnType = decltype(std::forward<Func>(func)());

  if constexpr (std::is_void_v<ReturnType>) {{
    std::forward<Func>(func)();
    return infini::rt::kSuccess;
  }} else {{
    ReturnType status = std::forward<Func>(func)();
    if constexpr (std::is_same_v<ReturnType, infini::rt::Error>) {{
      return status == infini::rt::kSuccess ? infini::rt::kSuccess
                                            : infini::rt::kUnSuccess;
    }} else {{
      return status == ReturnType{{}} ? infini::rt::kSuccess
                                      : infini::rt::kUnSuccess;
    }}
  }}
}}

template <Device::Type kDev>
auto RuntimeMemcpyKind(infini::rt::MemcpyKind kind) {{
  switch (kind) {{
    case infini::rt::MemcpyKind::kMemcpyHostToHost:
      return Runtime<kDev>::kMemcpyHostToHost;
    case infini::rt::MemcpyKind::kMemcpyHostToDevice:
      return Runtime<kDev>::kMemcpyHostToDevice;
    case infini::rt::MemcpyKind::kMemcpyDeviceToHost:
      return Runtime<kDev>::kMemcpyDeviceToHost;
    case infini::rt::MemcpyKind::kMemcpyDeviceToDevice:
      return Runtime<kDev>::kMemcpyDeviceToDevice;
  }}

  assert(false && "unsupported memcpy kind");
  return Runtime<kDev>::kMemcpyHostToHost;
}}

}}  // namespace

{dispatch_functions}
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
    _write_runtime_dispatch(
        pathlib.Path(args.source_output), args.runtime_header, devices
    )


if __name__ == "__main__":
    main()
