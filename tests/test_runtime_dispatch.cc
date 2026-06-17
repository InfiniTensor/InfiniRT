#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "test_helper.h"

namespace {

infini::rt::Device RuntimeTestDevice() {
#if defined(WITH_NVIDIA)
  return infini::rt::Device{infini::rt::Device::Type::kNvidia};
#else
  return infini::rt::Device{infini::rt::Device::Type::kCpu};
#endif
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;
  const infini::rt::Device device = RuntimeTestDevice();
  std::array<std::uint8_t, 4> input{1, 2, 3, 4};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  infini::rt::SetDevice(device);

  infini::rt::Device current_device;
  infini::rt::GetDevice(&current_device);
  context.ExpectEqual(current_device, device,
                      "Runtime dispatch should keep the current device.");

  int device_count = 0;
  infini::rt::GetDeviceCount(&device_count, device.type());
  context.Expect(device_count > 0,
                 "Runtime dispatch should report at least one device.");

  infini::rt::Malloc(&ptr, input.size());
  context.Expect(ptr != nullptr, "Runtime dispatch should allocate memory.");
  if (ptr == nullptr) {
    return context.ExitCode();
  }

  infini::rt::Memcpy(ptr, input.data(), input.size(),
                     infini::rt::MemcpyKind::kHostToDevice);
  infini::rt::Memcpy(output.data(), ptr, output.size(),
                     infini::rt::MemcpyKind::kDeviceToHost);
  context.ExpectEqual(output, input,
                      "Runtime dispatch should copy data through memory.");

  infini::rt::Memset(ptr, 0x5A, output.size());
  infini::rt::Memcpy(output.data(), ptr, output.size(),
                     infini::rt::MemcpyKind::kDeviceToHost);
  for (const auto value : output) {
    context.ExpectEqual(value, static_cast<std::uint8_t>(0x5A),
                        "Runtime dispatch should fill memory.");
  }

  infini::rt::DeviceSynchronize();
  infini::rt::Free(ptr);

  return context.ExitCode();
}
