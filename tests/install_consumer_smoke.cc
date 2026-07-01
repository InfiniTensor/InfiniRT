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
  std::array<std::uint8_t, 4> input{1, 2, 3, 4};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;
  if (infini::rt::SetDevice(0) != infini::rt::kSuccess) {
    return 1;
  }
  int current_device = -1;
  if (infini::rt::GetDevice(&current_device) != infini::rt::kSuccess) {
    return 1;
  }
  if (current_device != 0) {
    return 1;
  }
  int device_count = 0;
  if (infini::rt::GetDeviceCount(&device_count) != infini::rt::kSuccess) {
    return 1;
  }
  if (device_count <= 0) {
    return 1;
  }
  if (infini::rt::Malloc(&ptr, input.size()) != infini::rt::kSuccess) {
    return 1;
  }
  if (ptr == nullptr) {
    return 1;
  }
  if (infini::rt::Memcpy(ptr, input.data(), input.size(),
                         infini::rt::MemcpyKind::kMemcpyHostToDevice) !=
      infini::rt::kSuccess) {
    return 1;
  }
#if defined(INFINI_RT_CONSUMER_BACKEND_CPU)
  if (infini::rt::MemcpyAsync(ptr, input.data(), input.size(),
                              infini::rt::MemcpyKind::kMemcpyHostToDevice,
                              nullptr) == infini::rt::kSuccess) {
    return 1;
  }
#else
  if (infini::rt::MemcpyAsync(ptr, input.data(), input.size(),
                              infini::rt::MemcpyKind::kMemcpyHostToDevice,
                              nullptr) != infini::rt::kSuccess) {
    return 1;
  }
#endif
  if (infini::rt::DeviceSynchronize() != infini::rt::kSuccess) {
    return 1;
  }
  if (infini::rt::Memcpy(output.data(), ptr, output.size(),
                         infini::rt::MemcpyKind::kMemcpyDeviceToHost) !=
      infini::rt::kSuccess) {
    return 1;
  }
  if (output != input) {
    return 1;
  }
  if (infini::rt::Free(ptr) != infini::rt::kSuccess) {
    return 1;
  }
#endif

  return 0;
}
