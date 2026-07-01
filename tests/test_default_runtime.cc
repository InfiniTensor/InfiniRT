#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "test_helper.h"

namespace {

using DefaultRuntime = infini::rt::Runtime<>;

void ExpectSuccess(infini::rt::test::TestContext* context,
                   infini::rt::Error status, const char* message) {
  context->Expect(status == infini::rt::kSuccess, message);
}

void TestInvalidDeviceType(infini::rt::test::TestContext* context) {
  const auto before = DefaultRuntime::GetDeviceType();

  context->Expect(DefaultRuntime::SetDeviceType(infini::rt::Device::Type::kQy) !=
                      infini::rt::kSuccess,
                  "Default runtime should reject disabled device types.");
  context->Expect(DefaultRuntime::GetDeviceType() == before,
                  "Rejected device type should not change dispatch target.");
}

#if defined(INFINI_RT_TEST_WITH_CPU)
void TestCpuDispatch(infini::rt::test::TestContext* context) {
  ExpectSuccess(context,
                DefaultRuntime::SetDeviceType(infini::rt::Device::Type::kCpu),
                "Default runtime should select CPU dispatch.");
  context->Expect(DefaultRuntime::GetDeviceType() ==
                      infini::rt::Device::Type::kCpu,
                  "Default runtime should report CPU dispatch.");

  std::array<std::uint8_t, 4> input{1, 2, 3, 4};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, infini::rt::SetDevice(0),
                "CPU dispatch should set device 0.");
  ExpectSuccess(context, infini::rt::Malloc(&ptr, input.size()),
                "CPU dispatch should allocate memory.");
  if (ptr == nullptr) {
    return;
  }

  ExpectSuccess(context,
                infini::rt::Memcpy(ptr, input.data(), input.size(),
                                   infini::rt::MemcpyKind::kMemcpyHostToDevice),
                "CPU dispatch should copy host data to runtime memory.");
  context->Expect(
      infini::rt::MemcpyAsync(ptr, input.data(), input.size(),
                              infini::rt::MemcpyKind::kMemcpyHostToDevice,
                              nullptr) != infini::rt::kSuccess,
      "CPU dispatch should not report async memcpy success.");
  ExpectSuccess(context,
                infini::rt::Memcpy(output.data(), ptr, output.size(),
                                   infini::rt::MemcpyKind::kMemcpyDeviceToHost),
                "CPU dispatch should copy runtime memory to host.");
  ExpectSuccess(context, infini::rt::Free(ptr),
                "CPU dispatch should free memory.");

  context->ExpectEqual(output, input,
                       "CPU dispatch should preserve copied bytes.");
}
#endif

#if defined(INFINI_RT_TEST_WITH_NVIDIA)
void TestNvidiaDispatch(infini::rt::test::TestContext* context) {
  ExpectSuccess(
      context, DefaultRuntime::SetDeviceType(infini::rt::Device::Type::kNvidia),
      "Default runtime should select NVIDIA dispatch.");
  context->Expect(DefaultRuntime::GetDeviceType() ==
                      infini::rt::Device::Type::kNvidia,
                  "Default runtime should report NVIDIA dispatch.");

  std::array<std::uint8_t, 4> input{5, 6, 7, 8};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, infini::rt::SetDevice(0),
                "NVIDIA dispatch should set device 0.");
  ExpectSuccess(context, infini::rt::Malloc(&ptr, input.size()),
                "NVIDIA dispatch should allocate memory.");
  if (ptr == nullptr) {
    return;
  }

  ExpectSuccess(context,
                infini::rt::MemcpyAsync(
                    ptr, input.data(), input.size(),
                    infini::rt::MemcpyKind::kMemcpyHostToDevice, nullptr),
                "NVIDIA dispatch should support async host-to-device copy.");
  ExpectSuccess(context, infini::rt::DeviceSynchronize(),
                "NVIDIA dispatch should synchronize the device.");
  ExpectSuccess(context,
                infini::rt::Memcpy(output.data(), ptr, output.size(),
                                   infini::rt::MemcpyKind::kMemcpyDeviceToHost),
                "NVIDIA dispatch should copy device data to host.");
  ExpectSuccess(context, infini::rt::Free(ptr),
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
