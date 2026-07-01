#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

int main() {
  std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
  const infini::rt::Device device{infini::rt::Device::Type::kCpu};
  const infini::rt::TensorView tensor{data.data(), std::vector<std::size_t>{4},
                                      infini::rt::DataType::kFloat32, device};

  if (device.ToString() != "cpu:0") {
    return 1;
  }

  if (tensor.numel() != 4 || !tensor.IsContiguous()) {
    return 1;
  }

#if defined(INFINI_RT_CONSUMER_BACKEND_CPU) || \
    defined(INFINI_RT_CONSUMER_BACKEND_NVIDIA)
  namespace runtime = infini::rt::runtime;
#if defined(INFINI_RT_CONSUMER_BACKEND_CPU)
  constexpr auto kExpectedDeviceType = infini::rt::Device::Type::kCpu;
#else
  constexpr auto kExpectedDeviceType = infini::rt::Device::Type::kNvidia;
#endif
  infini::rt::set_runtime_device_type(kExpectedDeviceType);
  if (infini::rt::runtime_device_type() != kExpectedDeviceType) {
    return 1;
  }

  std::array<std::uint8_t, 4> input{1, 2, 3, 4};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;
  if (runtime::SetDevice(0) != runtime::kSuccess) {
    return 1;
  }
  int current_device = -1;
  if (runtime::GetDevice(&current_device) != runtime::kSuccess) {
    return 1;
  }
  if (current_device != 0) {
    return 1;
  }
  int device_count = 0;
  if (runtime::GetDeviceCount(&device_count) != runtime::kSuccess) {
    return 1;
  }
  if (device_count <= 0) {
    return 1;
  }
  if (runtime::Malloc(&ptr, input.size()) != runtime::kSuccess) {
    return 1;
  }
  if (ptr == nullptr) {
    return 1;
  }
  if (runtime::Memcpy(ptr, input.data(), input.size(),
                      runtime::MemcpyKind::kMemcpyHostToDevice) !=
      runtime::kSuccess) {
    return 1;
  }
#if defined(INFINI_RT_CONSUMER_BACKEND_CPU)
  if (runtime::MemcpyAsync(ptr, input.data(), input.size(),
                           runtime::MemcpyKind::kMemcpyHostToDevice,
                           nullptr) == runtime::kSuccess) {
    return 1;
  }
#else
  if (runtime::MemcpyAsync(ptr, input.data(), input.size(),
                           runtime::MemcpyKind::kMemcpyHostToDevice,
                           nullptr) != runtime::kSuccess) {
    return 1;
  }
#endif
  if (runtime::DeviceSynchronize() != runtime::kSuccess) {
    return 1;
  }
  if (runtime::Memcpy(output.data(), ptr, output.size(),
                      runtime::MemcpyKind::kMemcpyDeviceToHost) !=
      runtime::kSuccess) {
    return 1;
  }
  if (output != input) {
    return 1;
  }
  if (runtime::Free(ptr) != runtime::kSuccess) {
    return 1;
  }
#endif

  return 0;
}
