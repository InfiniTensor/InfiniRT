#include <infini/rt.h>
#include <infini/rt/cpu/runtime_.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "test_helper.h"

namespace {

using CpuRuntime = infini::rt::Runtime<infini::rt::Device::Type::kCpu>;

void TestMallocAndFree(infini::rt::test::TestContext* context) {
  void* ptr = nullptr;
  CpuRuntime::Malloc(&ptr, 16);

  context->Expect(ptr != nullptr, "CPU runtime should allocate memory.");

  CpuRuntime::Free(ptr);
}

void TestMemcpyRoundTrip(infini::rt::test::TestContext* context) {
  std::array<std::uint8_t, 8> input{0, 1, 2, 3, 4, 5, 6, 7};
  std::array<std::uint8_t, 8> output{};
  void* ptr = nullptr;

  CpuRuntime::Malloc(&ptr, input.size());
  context->Expect(ptr != nullptr, "CPU runtime should allocate copy memory.");
  if (ptr == nullptr) {
    return;
  }

  CpuRuntime::Memcpy(ptr, input.data(), input.size(),
                     CpuRuntime::kMemcpyHostToDevice);
  CpuRuntime::Memcpy(output.data(), ptr, output.size(),
                     CpuRuntime::kMemcpyDeviceToHost);
  CpuRuntime::Free(ptr);

  context->ExpectEqual(output, input,
                       "CPU runtime should copy data through runtime memory.");
}

void TestMemcpyAsyncUnsupported(infini::rt::test::TestContext* context) {
  std::array<std::uint8_t, 1> input{1};
  std::array<std::uint8_t, 1> output{};

  context->Expect(CpuRuntime::MemcpyAsync(output.data(), input.data(),
                                          input.size(),
                                          CpuRuntime::kMemcpyHostToHost,
                                          nullptr) != CpuRuntime::kSuccess,
                  "CPU runtime should not report async memcpy success.");
}

void TestMemset(infini::rt::test::TestContext* context) {
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  CpuRuntime::Malloc(&ptr, output.size());
  context->Expect(ptr != nullptr, "CPU runtime should allocate memset memory.");
  if (ptr == nullptr) {
    return;
  }

  CpuRuntime::Memset(ptr, 0x5A, output.size());
  CpuRuntime::Memcpy(output.data(), ptr, output.size(),
                     CpuRuntime::kMemcpyDeviceToHost);
  CpuRuntime::Free(ptr);

  for (const auto value : output) {
    context->ExpectEqual(value, static_cast<std::uint8_t>(0x5A),
                         "CPU runtime should fill memory with memset.");
  }
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  TestMallocAndFree(&context);
  TestMemcpyRoundTrip(&context);
  TestMemcpyAsyncUnsupported(&context);
  TestMemset(&context);

  return context.ExitCode();
}
