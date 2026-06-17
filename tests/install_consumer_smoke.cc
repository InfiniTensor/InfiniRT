#include <infini/rt.h>

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
#if defined(INFINI_RT_CONSUMER_BACKEND_NVIDIA)
  const infini::rt::Device runtime_device{infini::rt::Device::Type::kNvidia};
#else
  const infini::rt::Device runtime_device{infini::rt::Device::Type::kCpu};
#endif

  void* ptr = nullptr;
  infini::rt::SetDevice(runtime_device);
  infini::rt::Malloc(&ptr, sizeof(std::uint32_t));
  if (ptr == nullptr) {
    return 1;
  }
  infini::rt::Free(ptr);
#endif

  return 0;
}
