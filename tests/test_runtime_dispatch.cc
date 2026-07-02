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

#if defined(INFINI_RT_TEST_WITH_CPU)
void TestCpuDispatch(infini::rt::test::TestContext* context) {
  infini::rt::set_runtime_device_type(infini::rt::Device::Type::kCpu);
  context->Expect(
      infini::rt::runtime_device_type() == infini::rt::Device::Type::kCpu,
      "Runtime dispatch should report CPU dispatch.");

  std::array<std::uint8_t, 4> input{1, 2, 3, 4};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, runtime::SetDevice(0),
                "CPU dispatch should set device 0.");
  int current_device = -1;
  ExpectSuccess(context, runtime::GetDevice(&current_device),
                "CPU dispatch should get the current device.");
  context->ExpectEqual(current_device, 0,
                       "CPU dispatch should keep the current device.");

  int device_count = 0;
  ExpectSuccess(context, runtime::GetDeviceCount(&device_count),
                "CPU dispatch should get the device count.");
  context->Expect(device_count > 0,
                  "CPU dispatch should report at least one device.");

  ExpectSuccess(context, runtime::Malloc(&ptr, input.size()),
                "CPU dispatch should allocate memory.");
  if (ptr == nullptr) {
    return;
  }

  ExpectSuccess(context,
                runtime::Memcpy(ptr, input.data(), input.size(),
                                runtime::kMemcpyHostToDevice),
                "CPU dispatch should copy host data to runtime memory.");
  context->Expect(runtime::MemcpyAsync(ptr, input.data(), input.size(),
                                       runtime::kMemcpyHostToDevice,
                                       nullptr) != runtime::kSuccess,
                  "CPU dispatch should not report async memcpy success.");
  ExpectSuccess(context,
                runtime::Memcpy(output.data(), ptr, output.size(),
                                runtime::kMemcpyDeviceToHost),
                "CPU dispatch should copy runtime memory to host.");

  context->ExpectEqual(output, input,
                       "CPU dispatch should preserve copied bytes.");

  ExpectSuccess(context, runtime::Memset(ptr, 0x5A, output.size()),
                "CPU dispatch should fill runtime memory.");
  ExpectSuccess(context,
                runtime::Memcpy(output.data(), ptr, output.size(),
                                runtime::kMemcpyDeviceToHost),
                "CPU dispatch should copy filled memory to host.");
  for (const auto value : output) {
    context->ExpectEqual(value, static_cast<std::uint8_t>(0x5A),
                         "CPU dispatch should preserve filled bytes.");
  }

  ExpectSuccess(context, runtime::Free(ptr),
                "CPU dispatch should free memory.");
}
#endif

#if defined(INFINI_RT_TEST_WITH_NVIDIA)
void TestNvidiaDispatch(infini::rt::test::TestContext* context) {
  infini::rt::set_runtime_device_type(infini::rt::Device::Type::kNvidia);
  context->Expect(
      infini::rt::runtime_device_type() == infini::rt::Device::Type::kNvidia,
      "Runtime dispatch should report NVIDIA dispatch.");

  std::array<std::uint8_t, 4> input{5, 6, 7, 8};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, runtime::SetDevice(0),
                "NVIDIA dispatch should set device 0.");
  int current_device = -1;
  ExpectSuccess(context, runtime::GetDevice(&current_device),
                "NVIDIA dispatch should get the current device.");
  context->ExpectEqual(current_device, 0,
                       "NVIDIA dispatch should keep the current device.");

  int device_count = 0;
  ExpectSuccess(context, runtime::GetDeviceCount(&device_count),
                "NVIDIA dispatch should get the device count.");
  context->Expect(device_count > 0,
                  "NVIDIA dispatch should report at least one device.");

  ExpectSuccess(context, runtime::Malloc(&ptr, input.size()),
                "NVIDIA dispatch should allocate memory.");
  if (ptr == nullptr) {
    return;
  }

  ExpectSuccess(
      context,
      runtime::MemcpyAsync(ptr, input.data(), input.size(),
                           runtime::kMemcpyHostToDevice, nullptr),
      "NVIDIA dispatch should support async host-to-device copy.");
  ExpectSuccess(context, runtime::DeviceSynchronize(),
                "NVIDIA dispatch should synchronize the device.");
  ExpectSuccess(context,
                runtime::Memcpy(output.data(), ptr, output.size(),
                                runtime::kMemcpyDeviceToHost),
                "NVIDIA dispatch should copy device data to host.");

  context->ExpectEqual(output, input,
                       "NVIDIA dispatch should preserve copied bytes.");

  ExpectSuccess(context, runtime::Memset(ptr, 0x5A, output.size()),
                "NVIDIA dispatch should fill runtime memory.");
  ExpectSuccess(context, runtime::DeviceSynchronize(),
                "NVIDIA dispatch should synchronize filled memory.");
  ExpectSuccess(context,
                runtime::Memcpy(output.data(), ptr, output.size(),
                                runtime::kMemcpyDeviceToHost),
                "NVIDIA dispatch should copy filled memory to host.");
  for (const auto value : output) {
    context->ExpectEqual(value, static_cast<std::uint8_t>(0x5A),
                         "NVIDIA dispatch should preserve filled bytes.");
  }

  ExpectSuccess(context, runtime::Free(ptr),
                "NVIDIA dispatch should free memory.");
}
#endif

}  // namespace

int main() {
  infini::rt::test::TestContext context;

#if defined(INFINI_RT_TEST_WITH_CPU)
  TestCpuDispatch(&context);
#endif

#if defined(INFINI_RT_TEST_WITH_NVIDIA)
  TestNvidiaDispatch(&context);
#endif

  return context.ExitCode();
}
