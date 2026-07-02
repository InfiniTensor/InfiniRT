#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "test_helper.h"

namespace {

namespace runtime = infini::rt::runtime;

void ExpectSuccess(infini::rt::test::TestContext* context,
                   runtime::Error status, const char* message) {
  context->Expect(status == runtime::kSuccess, message);
}

void ExpectStatusOk(infini::rt::test::TestContext* context,
                    const infini::rt::Status& status, const char* message) {
  context->Expect(status.ok(), message);
}

void TestInvalidDeviceType(infini::rt::test::TestContext* context) {
  const auto before = infini::rt::runtime_device_type();
  context->Expect(before.ok(),
                  "Default runtime should report its initial device type.");

  const auto status =
      infini::rt::set_runtime_device_type(infini::rt::Device::Type::kQy);
  context->Expect(!status.ok(),
                  "Default runtime should reject disabled device types.");

  const auto after = infini::rt::runtime_device_type();
  context->Expect(after.ok(),
                  "Rejected device type should keep runtime state valid.");
  if (before.ok() && after.ok()) {
    context->Expect(*after == *before,
                    "Rejected device type should not change dispatch target.");
  }
}

#if defined(INFINI_RT_TEST_WITH_CPU)
void TestCpuDispatch(infini::rt::test::TestContext* context) {
  ExpectStatusOk(
      context,
      infini::rt::set_runtime_device_type(infini::rt::Device::Type::kCpu),
      "Default runtime should select CPU dispatch.");
  const auto device_type = infini::rt::runtime_device_type();
  context->Expect(
      device_type.ok() && *device_type == infini::rt::Device::Type::kCpu,
      "Default runtime should report CPU dispatch.");

  std::array<std::uint8_t, 4> input{1, 2, 3, 4};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, runtime::SetDevice(0),
                "CPU dispatch should set device 0.");
  ExpectSuccess(context, runtime::Malloc(&ptr, input.size()),
                "CPU dispatch should allocate memory.");
  if (ptr == nullptr) {
    return;
  }

  ExpectSuccess(context,
                runtime::Memcpy(ptr, input.data(), input.size(),
                                runtime::MemcpyKind::kMemcpyHostToDevice),
                "CPU dispatch should copy host data to runtime memory.");
  context->Expect(runtime::MemcpyAsync(ptr, input.data(), input.size(),
                                       runtime::MemcpyKind::kMemcpyHostToDevice,
                                       nullptr) != runtime::kSuccess,
                  "CPU dispatch should not report async memcpy success.");
  ExpectSuccess(context,
                runtime::Memcpy(output.data(), ptr, output.size(),
                                runtime::MemcpyKind::kMemcpyDeviceToHost),
                "CPU dispatch should copy runtime memory to host.");
  ExpectSuccess(context, runtime::Free(ptr),
                "CPU dispatch should free memory.");

  context->ExpectEqual(output, input,
                       "CPU dispatch should preserve copied bytes.");
}
#endif

#if defined(INFINI_RT_TEST_WITH_NVIDIA)
void TestNvidiaDispatch(infini::rt::test::TestContext* context) {
  ExpectStatusOk(
      context,
      infini::rt::set_runtime_device_type(infini::rt::Device::Type::kNvidia),
      "Default runtime should select NVIDIA dispatch.");
  const auto device_type = infini::rt::runtime_device_type();
  context->Expect(
      device_type.ok() && *device_type == infini::rt::Device::Type::kNvidia,
      "Default runtime should report NVIDIA dispatch.");

  std::array<std::uint8_t, 4> input{5, 6, 7, 8};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, runtime::SetDevice(0),
                "NVIDIA dispatch should set device 0.");
  ExpectSuccess(context, runtime::Malloc(&ptr, input.size()),
                "NVIDIA dispatch should allocate memory.");
  if (ptr == nullptr) {
    return;
  }

  ExpectSuccess(
      context,
      runtime::MemcpyAsync(ptr, input.data(), input.size(),
                           runtime::MemcpyKind::kMemcpyHostToDevice, nullptr),
      "NVIDIA dispatch should support async host-to-device copy.");
  ExpectSuccess(context, runtime::DeviceSynchronize(),
                "NVIDIA dispatch should synchronize the device.");
  ExpectSuccess(context,
                runtime::Memcpy(output.data(), ptr, output.size(),
                                runtime::MemcpyKind::kMemcpyDeviceToHost),
                "NVIDIA dispatch should copy device data to host.");
  ExpectSuccess(context, runtime::Free(ptr),
                "NVIDIA dispatch should free memory.");

  context->ExpectEqual(output, input,
                       "NVIDIA dispatch should preserve copied bytes.");
}
#endif

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  TestInvalidDeviceType(&context);

#if defined(INFINI_RT_TEST_WITH_CPU)
  TestCpuDispatch(&context);
#endif

#if defined(INFINI_RT_TEST_WITH_NVIDIA)
  TestNvidiaDispatch(&context);
#endif

  return context.ExitCode();
}
