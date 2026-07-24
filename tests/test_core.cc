#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include "test_helper.h"

namespace {

using infini::rt::DataType;
using infini::rt::Device;
using infini::rt::TensorView;

static_assert(std::is_copy_constructible_v<TensorView>,
              "TensorView should remain copy constructible.");
static_assert(std::is_move_constructible_v<TensorView>,
              "TensorView should remain move constructible.");
static_assert(!std::is_copy_assignable_v<TensorView>,
              "TensorView should not become copy assignable.");
static_assert(!std::is_move_assignable_v<TensorView>,
              "TensorView should not become move assignable.");
static_assert(!std::is_constructible_v<TensorView, std::vector<TensorView>>,
              "TensorView should not treat tensor containers as tensor-like.");

struct VectorTensorLike {
  void* data_value;

  std::vector<std::size_t> shape_value;

  DataType dtype_value;

  Device device_value;

  std::vector<std::ptrdiff_t> strides_value;

  mutable std::size_t data_call_count{0};

  mutable std::size_t shape_call_count{0};

  mutable std::size_t dtype_call_count{0};

  mutable std::size_t device_call_count{0};

  mutable std::size_t strides_call_count{0};

  void* data() const {
    ++data_call_count;
    return data_value;
  }

  std::vector<std::size_t> shape() const {
    ++shape_call_count;
    return shape_value;
  }

  DataType dtype() const {
    ++dtype_call_count;
    return dtype_value;
  }

  Device device() const {
    ++device_call_count;
    return device_value;
  }

  std::vector<std::ptrdiff_t> strides() const {
    ++strides_call_count;
    return strides_value;
  }
};

std::vector<std::size_t> MakeShape(std::size_t rank) {
  return std::vector<std::size_t>(rank, 2);
}

std::vector<std::ptrdiff_t> MakeContiguousStrides(
    const std::vector<std::size_t>& shape) {
  std::vector<std::ptrdiff_t> strides(shape.size());
  std::ptrdiff_t stride = 1;

  for (std::size_t index = shape.size(); index > 0; --index) {
    strides[index - 1] = stride;
    stride *= static_cast<std::ptrdiff_t>(shape[index - 1]);
  }

  return strides;
}

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

void TestTensorViewRanks(infini::rt::test::TestContext* context) {
  std::array<float, 512> data{};
  const Device cpu{Device::Type::kCpu};
  const std::array<std::size_t, 10> ranks{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  for (const std::size_t rank : ranks) {
    const std::vector<std::size_t> shape = MakeShape(rank);
    const std::vector<std::ptrdiff_t> strides =
        MakeContiguousStrides(shape);
    const TensorView tensor{data.data(), shape, DataType::kFloat32, cpu};
    const std::string rank_prefix =
        "Rank " + std::to_string(rank) + ": ";
    std::size_t expected_numel = 1;

    for (const std::size_t size : shape) {
      expected_numel *= size;
    }

    context->ExpectEqual(
        tensor.ndim(), rank,
        rank_prefix + "TensorView should preserve the tested rank.");
    context->ExpectEqual(
        tensor.shape(), shape,
        rank_prefix + "TensorView should preserve the complete shape.");
    context->ExpectEqual(
        tensor.strides(), strides,
        rank_prefix +
            "TensorView should generate complete contiguous strides.");
    context->ExpectEqual(
        tensor.numel(), expected_numel,
        rank_prefix + "TensorView should compute the element count.");
    context->Expect(
        tensor.IsContiguous(),
        rank_prefix + "TensorView should report contiguous metadata.");
  }
}

void TestTensorLikeValueAccessors(
    infini::rt::test::TestContext* context) {
  std::array<double, 6> data{};
  const VectorTensorLike tensor_like{data.data(),
                                     {2, 3},
                                     DataType::kFloat64,
                                     Device{Device::Type::kCpu, 1},
                                     {3, 1}};
  const TensorView tensor{tensor_like};

  context->ExpectEqual(
      tensor.data(), static_cast<const void*>(tensor_like.data_value),
      "TensorView should preserve TensorLike data returned by value.");
  context->ExpectEqual(
      tensor.shape(), tensor_like.shape_value,
      "TensorView should own shape metadata returned by value.");
  context->ExpectEqual(
      tensor.dtype(), tensor_like.dtype_value,
      "TensorView should preserve TensorLike dtype returned by value.");
  context->ExpectEqual(
      tensor.device(), tensor_like.device_value,
      "TensorView should preserve TensorLike device returned by value.");
  context->ExpectEqual(
      tensor.strides(), tensor_like.strides_value,
      "TensorView should own stride metadata returned by value.");
  context->ExpectEqual(tensor.numel(), std::size_t{6},
                       "TensorLike construction should preserve the shape.");
  context->Expect(tensor.IsContiguous(),
                  "TensorLike construction should preserve contiguity.");
  context->ExpectEqual(tensor_like.data_call_count, std::size_t{1},
                       "TensorView should evaluate data exactly once.");
  context->ExpectEqual(tensor_like.shape_call_count, std::size_t{1},
                       "TensorView should evaluate shape exactly once.");
  context->ExpectEqual(tensor_like.dtype_call_count, std::size_t{1},
                       "TensorView should evaluate dtype exactly once.");
  context->ExpectEqual(tensor_like.device_call_count, std::size_t{1},
                       "TensorView should evaluate device exactly once.");
  context->ExpectEqual(tensor_like.strides_call_count, std::size_t{1},
                       "TensorView should evaluate strides exactly once.");
}

void TestTensorViewOperations(infini::rt::test::TestContext* context) {
  std::array<float, 6> data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::array<float, 6> equal_data{};
  const Device cpu{Device::Type::kCpu};
  const std::vector<std::size_t> shape{2, 3};
  const std::vector<std::ptrdiff_t> strides{3, 1};
  const TensorView tensor{data.data(), shape, DataType::kFloat32, cpu};

  context->ExpectEqual(tensor.element_size(), std::size_t{4},
                       "TensorView should compute element size.");
  context->ExpectEqual(tensor.size(0), std::size_t{2},
                       "TensorView should expose dimension sizes.");
  context->ExpectEqual(tensor.size(-1), std::size_t{3},
                       "TensorView should support negative dimension sizes.");
  context->ExpectEqual(tensor.stride(0), std::ptrdiff_t{3},
                       "TensorView should expose dimension strides.");
  context->ExpectEqual(tensor.stride(-1), std::ptrdiff_t{1},
                       "TensorView should support negative dimension strides.");

  const TensorView indexed = tensor[1];
  const TensorView negative_indexed = tensor[-1];
  const std::vector<std::size_t> indexed_shape{3};
  const std::vector<std::ptrdiff_t> indexed_strides{1};
  context->ExpectEqual(indexed.shape(), indexed_shape,
                       "Indexing should remove the leading dimension.");
  context->ExpectEqual(indexed.strides(), indexed_strides,
                       "Indexing should remove the leading stride.");
  context->ExpectEqual(
      indexed.data(), static_cast<const void*>(data.data() + 3),
      "Indexing should offset the data pointer by the leading stride.");
  context->ExpectEqual(
      negative_indexed.data(), indexed.data(),
      "Negative indexing should select the matching leading element.");

  const TensorView transposed = tensor.T();
  const std::vector<std::size_t> transposed_shape{3, 2};
  const std::vector<std::ptrdiff_t> transposed_strides{1, 3};
  context->ExpectEqual(transposed.shape(), transposed_shape,
                       "Transposing should swap the complete shape.");
  context->ExpectEqual(transposed.strides(), transposed_strides,
                       "Transposing should swap the complete strides.");
  context->Expect(!transposed.IsContiguous(),
                  "A transposed matrix should not be contiguous.");

  const TensorView equal_tensor{equal_data.data(), shape, DataType::kFloat32,
                                cpu, strides};
  const TensorView strided{data.data(), shape, DataType::kFloat32, cpu,
                           std::vector<std::ptrdiff_t>{4, 1}};
  context->Expect(std::equal_to<TensorView>{}(tensor, equal_tensor),
                  "Equivalent TensorViews should compare equal.");
  context->Expect(!std::equal_to<TensorView>{}(tensor, strided),
                  "Different strides should make TensorViews unequal.");
  context->ExpectEqual(std::hash<TensorView>{}(tensor),
                       std::hash<TensorView>{}(equal_tensor),
                       "Equivalent TensorViews should have equal hashes.");
  context->Expect(!strided.IsContiguous(),
                  "TensorView with row padding should not be contiguous.");
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  TestDevice(&context);
  TestDataType(&context);
  TestTensorViewRanks(&context);
  TestTensorLikeValueAccessors(&context);
  TestTensorViewOperations(&context);

  return context.ExitCode();
}
