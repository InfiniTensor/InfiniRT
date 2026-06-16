#include <infini/rt.h>

#include <cstdint>
#include <string>
#include <vector>

#include "test_helper.h"

namespace {

using infini::rt::DataType;
using infini::rt::Device;
using infini::rt::TensorView;

void TestDevice(infini::rt::test::TestContext* context) {
  const Device cpu{Device::Type::kCpu};
  const Device nvidia{Device::Type::kNvidia, 1};

  context->ExpectEqual(cpu.type(), Device::Type::kCpu,
                       "CPU device should keep its type.");
  context->ExpectEqual(cpu.index(), 0, "CPU device should default to index 0.");
  context->ExpectEqual(nvidia.index(), 1,
                       "Device should keep an explicit index.");
  context->ExpectEqual(Device::TypeFromString("cpu"), Device::Type::kCpu,
                       "Device should parse the CPU name.");
  context->ExpectEqual(Device::StringFromType(Device::Type::kNvidia), "nvidia",
                       "Device should stringify the NVIDIA type.");
  context->ExpectEqual(cpu.ToString(), std::string{"cpu:0"},
                       "Device should stringify type and index.");
  context->Expect(cpu != nvidia, "Different devices should not compare equal.");
}

void TestDataType(infini::rt::test::TestContext* context) {
  context->ExpectEqual(infini::rt::kDataTypeToSize.at(DataType::kInt8),
                       std::size_t{1}, "int8 should have one byte.");
  context->ExpectEqual(infini::rt::kDataTypeToSize.at(DataType::kFloat32),
                       std::size_t{4}, "float32 should have four bytes.");
  context->ExpectEqual(infini::rt::kDataTypeToDesc.at(DataType::kFloat64),
                       "float64", "float64 should have a stable name.");
  context->ExpectEqual(infini::rt::kStringToDataType.at("uint16"),
                       DataType::kUInt16, "uint16 should parse by name.");
}

void TestTensorView(infini::rt::test::TestContext* context) {
  std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  TensorView tensor{data.data(), std::vector<std::size_t>{2, 3},
                    DataType::kFloat32, Device{Device::Type::kCpu}};

  context->ExpectEqual(tensor.ndim(), std::size_t{2},
                       "TensorView should keep its rank.");
  context->ExpectEqual(tensor.numel(), std::size_t{6},
                       "TensorView should compute element count.");
  context->ExpectEqual(tensor.element_size(), std::size_t{4},
                       "TensorView should compute element size.");
  context->ExpectEqual(tensor.size(0), std::size_t{2},
                       "TensorView should expose dimension sizes.");
  context->ExpectEqual(tensor.size(-1), std::size_t{3},
                       "TensorView should support negative dimension sizes.");
  context->ExpectEqual(tensor.stride(0), std::ptrdiff_t{3},
                       "TensorView should compute default row-major strides.");
  context->ExpectEqual(tensor.stride(1), std::ptrdiff_t{1},
                       "TensorView should compute default innermost stride.");
  context->Expect(tensor.IsContiguous(),
                  "Default TensorView strides should be contiguous.");

  TensorView transposed = tensor.T();
  context->ExpectEqual(transposed.shape(), TensorView::Shape({3, 2}),
                       "Transposed TensorView should swap shape.");
  context->ExpectEqual(transposed.strides(), TensorView::Strides({1, 3}),
                       "Transposed TensorView should swap strides.");
  context->Expect(!transposed.IsContiguous(),
                  "Transposed TensorView should not be contiguous.");

  TensorView strided{data.data(),
                     std::vector<std::size_t>{2, 3},
                     DataType::kFloat32,
                     Device{Device::Type::kCpu},
                     std::vector<std::ptrdiff_t>{4, 1}};
  context->Expect(!strided.IsContiguous(),
                  "TensorView with row padding should not be contiguous.");
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  TestDevice(&context);
  TestDataType(&context);
  TestTensorView(&context);

  return context.ExitCode();
}
