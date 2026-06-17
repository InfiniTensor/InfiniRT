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
            tuple(_parse_param(param) for param in params.split(", ") if param),
        )
        for return_type, name, params in re.findall(
            r"^(void) ([A-Z]\w*)\(([^()]*)\);$", text, re.MULTILINE
        )
    )


def _abort_statement(message):
    return f"""      assert(false && "{message}");
      std::abort();"""


def _dispatch_cases(devices, statements):
    return "\n".join(
        f"""    case {_DEVICE_TYPES[device]}: {{
{statements.replace("__DEVICE_TYPE__", _DEVICE_TYPES[device])}
      return;
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


def _runtime_arg(param):
    if param.type == "Device":
        return f"{param.name}.index()"
    if param.type == "Device::Type":
        return None
    if param.type == "MemcpyKind":
        return f"RuntimeMemcpyKind<__DEVICE_TYPE__>({param.name})"

    return param.name


def _runtime_args(function):
    args = (_runtime_arg(param) for param in function.params)

    return ", ".join(arg for arg in args if arg is not None)


def _preconditions(function):
    required_pointer_names = {
        "GetDevice": {"device"},
        "GetDeviceCount": {"count"},
    }
    checks = []
    for param in function.params:
        if param.type.endswith("**") or param.name in required_pointer_names.get(
            function.name, set()
        ):
            checks.append(f"  assert({param.name} != nullptr);")

    return "\n".join(checks)


def _post_dispatch(function):
    if function.name == "SetDevice":
        return "\n      current_device = device;"

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
        f"""      int index = current_device.index();
      CheckCall([&] {{ return Runtime<__DEVICE_TYPE__>::GetDevice(&index); }});
      current_device = Device{{current_device.type(), index}};
      *{device_param} = current_device;""",
    )

    return f"""void GetDevice(Device* {device_param}) {{
  assert({device_param} != nullptr);

  switch (current_device.type()) {{
{cases}
    default:
{_abort_statement("runtime device is not enabled")}
  }}
}}
"""


def _write_dispatch_function(function, devices):
    if function.name == "GetDevice":
        return _write_get_device(function, devices)

    cases = _dispatch_cases(
        devices,
        f"""      CheckCall([&] {{ return {_runtime_call(function)}; }});{_post_dispatch(function)}""",
    )
    preconditions = _preconditions(function)
    if preconditions:
        preconditions = f"{preconditions}\n\n"

    return f"""{function.signature()} {{
{preconditions}  switch ({_selector(function)}) {{
{cases}
    default:
{_abort_statement("runtime device is not enabled")}
  }}
}}
"""


def _write_runtime_dispatch(source_path, runtime_header, devices):
    first_device_type = _DEVICE_TYPES[devices[0]]
    includes = ['#include "runtime.h"']
    includes.extend(f'#include "{_RUNTIME_HEADERS[device]}"' for device in devices)
    functions = _parse_runtime_functions(runtime_header)
    dispatch_functions = "\n".join(
        _write_dispatch_function(function, devices) for function in functions
    )

    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        f"""#include <cassert>
#include <cstdlib>
#include <type_traits>
#include <utility>

{chr(10).join(includes)}

namespace infini::rt {{
namespace {{

thread_local Device current_device{{{first_device_type}, 0}};

template <typename Func>
void CheckCall(Func&& func) {{
  using ReturnType = decltype(std::forward<Func>(func)());

  if constexpr (std::is_void_v<ReturnType>) {{
    std::forward<Func>(func)();
  }} else {{
    ReturnType status = std::forward<Func>(func)();
    if (status != ReturnType{{}}) {{
      assert(false && "runtime call failed");
      std::abort();
    }}
  }}
}}

template <Device::Type kDev>
auto RuntimeMemcpyKind(MemcpyKind kind) {{
  switch (kind) {{
    case MemcpyKind::kHostToHost:
      return Runtime<kDev>::MemcpyHostToHost;
    case MemcpyKind::kHostToDevice:
      return Runtime<kDev>::MemcpyHostToDevice;
    case MemcpyKind::kDeviceToHost:
      return Runtime<kDev>::MemcpyDeviceToHost;
    case MemcpyKind::kDeviceToDevice:
      return Runtime<kDev>::MemcpyDeviceToDevice;
  }}

  assert(false && "unsupported memcpy kind");
  std::abort();
  return Runtime<kDev>::MemcpyHostToHost;
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
