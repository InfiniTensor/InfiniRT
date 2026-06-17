#include <infini/rt.h>
#include <infini/rt/nvidia/runtime_.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "test_helper.h"

namespace {

using NvidiaRuntime = infini::rt::Runtime<infini::rt::Device::Type::kNvidia>;

void ExpectCudaSuccess(infini::rt::test::TestContext* context,
                       cudaError_t status, const char* message) {
  context->Expect(status == cudaSuccess, message);
}

void TestMallocAndFree(infini::rt::test::TestContext* context) {
  void* ptr = nullptr;
  ExpectCudaSuccess(context, NvidiaRuntime::Malloc(&ptr, 16),
                    "NVIDIA runtime should allocate device memory.");
  context->Expect(ptr != nullptr,
                  "NVIDIA runtime allocation should produce a pointer.");
  if (ptr != nullptr) {
    ExpectCudaSuccess(context, NvidiaRuntime::Free(ptr),
                      "NVIDIA runtime should free device memory.");
  }
}

void TestMemcpyRoundTrip(infini::rt::test::TestContext* context) {
  std::array<std::uint8_t, 8> input{0, 1, 2, 3, 4, 5, 6, 7};
  std::array<std::uint8_t, 8> output{};
  void* ptr = nullptr;

  ExpectCudaSuccess(context, NvidiaRuntime::Malloc(&ptr, input.size()),
                    "NVIDIA runtime should allocate copy memory.");
  if (ptr == nullptr) {
    return;
  }

  ExpectCudaSuccess(context,
                    NvidiaRuntime::Memcpy(ptr, input.data(), input.size(),
                                          NvidiaRuntime::MemcpyHostToDevice),
                    "NVIDIA runtime should copy host data to device memory.");
  ExpectCudaSuccess(context,
                    NvidiaRuntime::Memcpy(output.data(), ptr, output.size(),
                                          NvidiaRuntime::MemcpyDeviceToHost),
                    "NVIDIA runtime should copy device data to host memory.");
  ExpectCudaSuccess(context, NvidiaRuntime::Free(ptr),
                    "NVIDIA runtime should free copy memory.");

  context->ExpectEqual(
      output, input, "NVIDIA runtime should copy data through device memory.");
}

void TestMemset(infini::rt::test::TestContext* context) {
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  ExpectCudaSuccess(context, NvidiaRuntime::Malloc(&ptr, output.size()),
                    "NVIDIA runtime should allocate memset memory.");
  if (ptr == nullptr) {
    return;
  }

  ExpectCudaSuccess(context, NvidiaRuntime::Memset(ptr, 0x5A, output.size()),
                    "NVIDIA runtime should memset device memory.");
  ExpectCudaSuccess(context,
                    NvidiaRuntime::Memcpy(output.data(), ptr, output.size(),
                                          NvidiaRuntime::MemcpyDeviceToHost),
                    "NVIDIA runtime should copy memset data to host memory.");
  ExpectCudaSuccess(context, NvidiaRuntime::Free(ptr),
                    "NVIDIA runtime should free memset memory.");

  for (const auto value : output) {
    context->ExpectEqual(value, static_cast<std::uint8_t>(0x5A),
                         "NVIDIA runtime should fill memory with memset.");
  }
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  TestMallocAndFree(&context);
  TestMemcpyRoundTrip(&context);
  TestMemset(&context);

  return context.ExitCode();
}
