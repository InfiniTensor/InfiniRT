#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

#include "test_helper.h"

namespace {

namespace runtime = infini::rt::runtime;

void ExpectSuccess(infini::rt::test::TestContext* context,
                   runtime::Error status, std::string_view message) {
  context->Expect(status == runtime::kSuccess, message);
}

void ExpectFailure(infini::rt::test::TestContext* context,
                   runtime::Error status, std::string_view message) {
  context->Expect(status != runtime::kSuccess, message);
}

std::string Message(std::string_view backend_name, std::string_view message) {
  return std::string{backend_name} + " dispatch should " +
         std::string{message} + ".";
}

bool SelectDevice(infini::rt::test::TestContext* context,
                  infini::rt::Device::Type device_type,
                  const char* backend_name) {
  infini::rt::set_runtime_device_type(device_type);
  if (!context->Expect(infini::rt::runtime_device_type() == device_type,
                       Message(backend_name, "report the selected backend"))) {
    return false;
  }

  int device_count = 0;
  if (runtime::GetDeviceCount(&device_count) != runtime::kSuccess ||
      device_count <= 0) {
    std::cout << backend_name << " dispatch skipped: no available device."
              << std::endl;
    return false;
  }

  if (runtime::SetDevice(0) != runtime::kSuccess) {
    std::cout << backend_name << " dispatch skipped: device 0 is not available."
              << std::endl;
    return false;
  }

  return true;
}

void TestDispatch(infini::rt::test::TestContext* context,
                  infini::rt::Device::Type device_type,
                  const char* backend_name, bool expect_async_memcpy_success) {
  if (!SelectDevice(context, device_type, backend_name)) {
    return;
  }

  std::array<std::uint8_t, 4> input{1, 2, 3, 4};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  int current_device = -1;
  ExpectSuccess(context, runtime::GetDevice(&current_device),
                Message(backend_name, "get the current device"));
  context->ExpectEqual(current_device, 0,
                       Message(backend_name, "keep the current device"));

  ExpectSuccess(context, runtime::Malloc(&ptr, input.size()),
                Message(backend_name, "allocate memory"));
  if (ptr == nullptr) {
    return;
  }

  ExpectSuccess(context,
                runtime::Memcpy(ptr, input.data(), input.size(),
                                runtime::kMemcpyHostToDevice),
                Message(backend_name, "copy host data to runtime memory"));

  runtime::Stream stream{};
  const auto async_status = runtime::MemcpyAsync(
      ptr, input.data(), input.size(), runtime::kMemcpyHostToDevice, stream);
  if (expect_async_memcpy_success) {
    ExpectSuccess(context, async_status,
                  Message(backend_name, "support async host-to-device copy"));
    ExpectSuccess(context, runtime::DeviceSynchronize(),
                  Message(backend_name, "synchronize async copy"));
  } else {
    ExpectFailure(context, async_status,
                  Message(backend_name, "not report async memcpy success"));
  }

  ExpectSuccess(context,
                runtime::Memcpy(output.data(), ptr, output.size(),
                                runtime::kMemcpyDeviceToHost),
                Message(backend_name, "copy runtime memory to host"));

  context->ExpectEqual(output, input,
                       Message(backend_name, "preserve copied bytes"));

  ExpectSuccess(context, runtime::Memset(ptr, 0x5A, output.size()),
                Message(backend_name, "fill runtime memory"));
  ExpectSuccess(context, runtime::DeviceSynchronize(),
                Message(backend_name, "synchronize filled memory"));
  ExpectSuccess(context,
                runtime::Memcpy(output.data(), ptr, output.size(),
                                runtime::kMemcpyDeviceToHost),
                Message(backend_name, "copy filled memory to host"));
  for (const auto value : output) {
    context->ExpectEqual(value, static_cast<std::uint8_t>(0x5A),
                         Message(backend_name, "preserve filled bytes"));
  }

  ExpectSuccess(context, runtime::Free(ptr),
                Message(backend_name, "free memory"));
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

#if defined(INFINI_RT_TEST_WITH_CPU)
  TestDispatch(&context, infini::rt::Device::Type::kCpu, "CPU", false);
#endif

#if defined(INFINI_RT_TEST_WITH_NVIDIA)
  TestDispatch(&context, infini::rt::Device::Type::kNvidia, "NVIDIA", true);
#endif

#if defined(INFINI_RT_TEST_WITH_ILUVATAR)
  TestDispatch(&context, infini::rt::Device::Type::kIluvatar, "ILUVATAR", true);
#endif

#if defined(INFINI_RT_TEST_WITH_HYGON)
  TestDispatch(&context, infini::rt::Device::Type::kHygon, "HYGON", true);
#endif

#if defined(INFINI_RT_TEST_WITH_METAX)
  TestDispatch(&context, infini::rt::Device::Type::kMetax, "METAX", true);
#endif

#if defined(INFINI_RT_TEST_WITH_MOORE)
  TestDispatch(&context, infini::rt::Device::Type::kMoore, "MOORE", true);
#endif

#if defined(INFINI_RT_TEST_WITH_CAMBRICON)
  TestDispatch(&context, infini::rt::Device::Type::kCambricon, "CAMBRICON",
               true);
#endif

#if defined(INFINI_RT_TEST_WITH_ASCEND)
  TestDispatch(&context, infini::rt::Device::Type::kAscend, "ASCEND", true);
#endif

  return context.ExitCode();
}
