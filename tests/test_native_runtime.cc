#include <infini/rt.h>
#include INFINI_RT_TEST_RUNTIME_HEADER

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include "test_helper.h"

namespace {

using Runtime = infini::rt::runtime::Runtime<INFINI_RT_TEST_DEVICE_TYPE>;

constexpr bool kExpectAsyncMemcpySuccess =
    INFINI_RT_TEST_EXPECT_ASYNC_MEMCPY_SUCCESS != 0;

void ExpectSuccess(infini::rt::test::TestContext* context,
                   typename Runtime::Error status, const char* message) {
  context->Expect(status == Runtime::kSuccess, message);
}

void ExpectFailure(infini::rt::test::TestContext* context,
                   typename Runtime::Error status, const char* message) {
  context->Expect(status != Runtime::kSuccess, message);
}

bool SelectDevice() {
  int device_count = 0;
  if (Runtime::GetDeviceCount(&device_count) != Runtime::kSuccess ||
      device_count <= 0) {
    std::cout << INFINI_RT_TEST_BACKEND_NAME
              << " runtime skipped: no available device." << std::endl;
    return false;
  }

  if (Runtime::SetDevice(0) != Runtime::kSuccess) {
    std::cout << INFINI_RT_TEST_BACKEND_NAME
              << " runtime skipped: device 0 is not available." << std::endl;
    return false;
  }

  return true;
}

void TestDevice(infini::rt::test::TestContext* context) {
  int current_device = -1;
  ExpectSuccess(context, Runtime::GetDevice(&current_device),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should get the current device.");
  context->ExpectEqual(current_device, 0,
                       INFINI_RT_TEST_BACKEND_NAME
                       " runtime should keep the current device.");

  int device_count = 0;
  ExpectSuccess(context, Runtime::GetDeviceCount(&device_count),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should get the device count.");
  context->Expect(device_count > 0, INFINI_RT_TEST_BACKEND_NAME
                  " runtime should report at least one device.");
}

void TestMallocAndFree(infini::rt::test::TestContext* context) {
  void* ptr = nullptr;
  ExpectSuccess(context, Runtime::Malloc(&ptr, 16),
                INFINI_RT_TEST_BACKEND_NAME " runtime should allocate memory.");
  context->Expect(ptr != nullptr, INFINI_RT_TEST_BACKEND_NAME
                  " runtime allocation should produce a pointer.");
  if (ptr != nullptr) {
    ExpectSuccess(context, Runtime::Free(ptr),
                  INFINI_RT_TEST_BACKEND_NAME " runtime should free memory.");
  }
}

void TestMemcpyRoundTrip(infini::rt::test::TestContext* context) {
  std::array<std::uint8_t, 8> input{0, 1, 2, 3, 4, 5, 6, 7};
  std::array<std::uint8_t, 8> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, Runtime::Malloc(&ptr, input.size()),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should allocate copy memory.");
  if (ptr == nullptr) {
    return;
  }

  ExpectSuccess(context,
                Runtime::Memcpy(ptr, input.data(), input.size(),
                                Runtime::kMemcpyHostToDevice),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should copy host data to runtime memory.");
  ExpectSuccess(context,
                Runtime::Memcpy(output.data(), ptr, output.size(),
                                Runtime::kMemcpyDeviceToHost),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should copy runtime memory to host.");
  ExpectSuccess(context, Runtime::Free(ptr),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should free copy memory.");

  context->ExpectEqual(output, input,
                       INFINI_RT_TEST_BACKEND_NAME
                       " runtime should preserve copied bytes.");
}

void TestMemcpyAsync(infini::rt::test::TestContext* context) {
  std::array<std::uint8_t, 4> input{8, 9, 10, 11};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, Runtime::Malloc(&ptr, input.size()),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should allocate async copy memory.");
  if (ptr == nullptr) {
    return;
  }

  typename Runtime::Stream stream{};
  const auto async_status = Runtime::MemcpyAsync(
      ptr, input.data(), input.size(), Runtime::kMemcpyHostToDevice, stream);

  if constexpr (kExpectAsyncMemcpySuccess) {
    ExpectSuccess(context, async_status,
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should support async host-to-device copy.");
    ExpectSuccess(context, Runtime::DeviceSynchronize(),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should synchronize async host-to-device copy.");
    ExpectSuccess(context,
                  Runtime::Memcpy(output.data(), ptr, output.size(),
                                  Runtime::kMemcpyDeviceToHost),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should copy async data back to host.");
    context->ExpectEqual(output, input,
                         INFINI_RT_TEST_BACKEND_NAME
                         " runtime should preserve async copied bytes.");
  } else {
    ExpectFailure(context, async_status,
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should not report async memcpy success.");
  }

  ExpectSuccess(context, Runtime::Free(ptr),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should free async copy memory.");
}

void TestMemset(infini::rt::test::TestContext* context) {
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, Runtime::Malloc(&ptr, output.size()),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should allocate memset memory.");
  if (ptr == nullptr) {
    return;
  }

  ExpectSuccess(context, Runtime::Memset(ptr, 0x5A, output.size()),
                INFINI_RT_TEST_BACKEND_NAME " runtime should fill memory.");
  ExpectSuccess(context,
                Runtime::Memcpy(output.data(), ptr, output.size(),
                                Runtime::kMemcpyDeviceToHost),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should copy filled memory to host.");
  ExpectSuccess(context, Runtime::Free(ptr),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should free memset memory.");

  for (const auto value : output) {
    context->ExpectEqual(value, static_cast<std::uint8_t>(0x5A),
                         INFINI_RT_TEST_BACKEND_NAME
                         " runtime should preserve filled bytes.");
  }
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  if (!SelectDevice()) {
    return context.ExitCode();
  }

  TestDevice(&context);
  TestMallocAndFree(&context);
  TestMemcpyRoundTrip(&context);
  TestMemcpyAsync(&context);
  TestMemset(&context);

  return context.ExitCode();
}
