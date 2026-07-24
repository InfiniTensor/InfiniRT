#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>
#include <vector>

#include "test_helper.h"

namespace {

thread_local bool count_allocations = false;
thread_local std::size_t allocation_count = 0;

class AllocationScope {
 public:
  AllocationScope() {
    allocation_count = 0;
    count_allocations = true;
  }

  AllocationScope(const AllocationScope&) = delete;
  AllocationScope& operator=(const AllocationScope&) = delete;

  ~AllocationScope() { count_allocations = false; }

  std::size_t count() const { return allocation_count; }
};

template <typename Function>
std::size_t CountAllocations(Function&& function) {
  AllocationScope scope;
  std::forward<Function>(function)();
  return scope.count();
}

void ExpectAllocationCount(infini::rt::test::TestContext* context,
                           std::size_t actual, std::size_t expected,
                           const char* message) {
  context->ExpectEqual(actual, expected, message);
}

}  // namespace

void* operator new(std::size_t size) {
  if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
    if (count_allocations) {
      ++allocation_count;
    }
    return pointer;
  }
  throw std::bad_alloc{};
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void operator delete(void* pointer) noexcept { std::free(pointer); }

void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }

void operator delete(void* pointer, std::size_t) noexcept {
  ::operator delete(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
  ::operator delete[](pointer);
}

namespace {

using infini::rt::DataType;
using infini::rt::Device;
using infini::rt::TensorView;

struct VectorTensorLike {
  void* data_value;

  std::vector<std::size_t> shape_value;

  DataType dtype_value;

  Device device_value;

  std::vector<std::ptrdiff_t> strides_value;

  void* data() const { return data_value; }

  const std::vector<std::size_t>& shape() const { return shape_value; }

  DataType dtype() const { return dtype_value; }

  Device device() const { return device_value; }

  const std::vector<std::ptrdiff_t>& strides() const {
    return strides_value;
  }
};

void TestConstructionAllocationMatrix(
    infini::rt::test::TestContext* context) {
  std::array<float, 512> data{};
  const Device cpu{Device::Type::kCpu};
  const TensorView::Shape shape8{2, 2, 2, 2, 2, 2, 2, 2};
  const TensorView::Strides strides8{128, 64, 32, 16, 8, 4, 2, 1};
  const TensorView::Shape shape9{2, 2, 2, 2, 2, 2, 2, 2, 2};
  const TensorView::Strides strides9{256, 128, 64, 32, 16, 8, 4, 2, 1};
  const VectorTensorLike tensor_like8{
      data.data(),
      {2, 2, 2, 2, 2, 2, 2, 2},
      DataType::kFloat32,
      cpu,
      {128, 64, 32, 16, 8, 4, 2, 1}};
  const VectorTensorLike tensor_like9{
      data.data(),
      {2, 2, 2, 2, 2, 2, 2, 2, 2},
      DataType::kFloat32,
      cpu,
      {256, 128, 64, 32, 16, 8, 4, 2, 1}};

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), shape8, DataType::kFloat32, cpu,
                          strides8};
        (void)tensor;
      }),
      0, "Rank-8 lvalue metadata should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), shape9, DataType::kFloat32, cpu,
                          strides9};
        (void)tensor;
      }),
      2, "Rank-9 lvalue metadata should allocate two owned arrays.");

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{
            data.data(), TensorView::Shape{2, 2, 2, 2, 2, 2, 2, 2},
            DataType::kFloat32, cpu,
            TensorView::Strides{128, 64, 32, 16, 8, 4, 2, 1}};
        (void)tensor;
      }),
      0, "Rank-8 exact metadata temporaries should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{
            data.data(), TensorView::Shape{2, 2, 2, 2, 2, 2, 2, 2, 2},
            DataType::kFloat32, cpu,
            TensorView::Strides{256, 128, 64, 32, 16, 8, 4, 2, 1}};
        (void)tensor;
      }),
      2, "Rank-9 exact metadata temporaries should allocate twice.");

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(),
                          {2, 2, 2, 2, 2, 2, 2, 2},
                          DataType::kFloat32,
                          cpu,
                          {128, 64, 32, 16, 8, 4, 2, 1}};
        (void)tensor;
      }),
      0, "Rank-8 initializer-list metadata should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(),
                          {2, 2, 2, 2, 2, 2, 2, 2, 2},
                          DataType::kFloat32,
                          cpu,
                          {256, 128, 64, 32, 16, 8, 4, 2, 1}};
        (void)tensor;
      }),
      2, "Rank-9 initializer-list metadata should allocate twice.");

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{tensor_like8};
        (void)tensor;
      }),
      0, "Rank-8 vector-backed TensorLike metadata should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{tensor_like9};
        (void)tensor;
      }),
      2, "Rank-9 vector-backed TensorLike metadata should allocate twice.");

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), shape8, DataType::kFloat32, cpu};
        (void)tensor;
      }),
      0, "Rank-8 ordinary default-stride construction should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), shape9, DataType::kFloat32, cpu};
        (void)tensor;
      }),
      2, "Rank-9 ordinary default-stride construction should allocate twice.");

  TensorView::Shape explicit_move_shape8{2, 2, 2, 2, 2, 2, 2, 2};
  TensorView::Strides explicit_move_strides8{128, 64, 32, 16, 8, 4, 2, 1};
  TensorView::Shape explicit_move_shape9{2, 2, 2, 2, 2, 2, 2, 2, 2};
  TensorView::Strides explicit_move_strides9{256, 128, 64, 32, 16, 8, 4, 2, 1};

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), std::move(explicit_move_shape8),
                          DataType::kFloat32, cpu,
                          std::move(explicit_move_strides8)};
        (void)tensor;
      }),
      0, "Moving Rank-8 explicit metadata should not allocate.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), std::move(explicit_move_shape9),
                          DataType::kFloat32, cpu,
                          std::move(explicit_move_strides9)};
        (void)tensor;
      }),
      0, "Moving Rank-9 explicit metadata should not allocate.");

  TensorView::Shape default_move_shape8{2, 2, 2, 2, 2, 2, 2, 2};
  TensorView::Shape default_move_shape9{2, 2, 2, 2, 2, 2, 2, 2, 2};

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), std::move(default_move_shape8),
                          DataType::kFloat32, cpu};
        (void)tensor;
      }),
      0, "Moving a Rank-8 shape while generating strides should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), std::move(default_move_shape9),
                          DataType::kFloat32, cpu};
        (void)tensor;
      }),
      1, "Moving a Rank-9 shape should allocate only default strides.");
}

void TestValueAndDerivedViewAllocations(
    infini::rt::test::TestContext* context) {
  std::array<float, 512> data{};
  const Device cpu{Device::Type::kCpu};
  const TensorView::Shape shape8{2, 2, 2, 2, 2, 2, 2, 2};
  const TensorView::Strides strides8{128, 64, 32, 16, 8, 4, 2, 1};
  const TensorView::Shape shape9{2, 2, 2, 2, 2, 2, 2, 2, 2};
  const TensorView::Strides strides9{256, 128, 64, 32, 16, 8, 4, 2, 1};
  const TensorView source8{data.data(), shape8, DataType::kFloat32, cpu,
                           strides8};
  const TensorView source9{data.data(), shape9, DataType::kFloat32, cpu,
                           strides9};

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView copied{source8};
        (void)copied;
      }),
      0, "Copying Rank-8 metadata should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView copied{source9};
        (void)copied;
      }),
      2, "Copying Rank-9 metadata should allocate two independent arrays.");

  TensorView move_source8{data.data(), shape8, DataType::kFloat32, cpu,
                          strides8};
  TensorView move_source9{data.data(), shape9, DataType::kFloat32, cpu,
                          strides9};

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView moved{std::move(move_source8)};
        (void)moved;
      }),
      0, "Moving Rank-8 metadata should not allocate.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView moved{std::move(move_source9)};
        (void)moved;
      }),
      0, "Moving Rank-9 metadata should transfer heap storage.");

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView indexed = source8[0];
        (void)indexed;
      }),
      0, "Indexing Rank-8 to Rank-7 should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView indexed = source9[0];
        (void)indexed;
      }),
      0, "Indexing Rank-9 to Rank-8 should stay inline.");

  const TensorView transpose_source{data.data(), TensorView::Shape{2, 2},
                                    DataType::kFloat32, cpu,
                                    TensorView::Strides{2, 1}};
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView transposed = transpose_source.T();
        (void)transposed;
      }),
      0, "Transposing rank-2 metadata should stay inline.");
}

void TestDefaultMetadataAllocations(
    infini::rt::test::TestContext* context) {
  std::array<float, 32> data{};
  const Device cpu{Device::Type::kCpu};
  const Device indexed_cpu{Device::Type::kCpu, 1};
  const TensorView::Shape expected_shape{2, 3};
  const TensorView::Strides expected_strides{3, 1};

  bool shape_only_metadata_is_default = false;
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), TensorView::Shape{2, 3}};
        shape_only_metadata_is_default =
            tensor.shape() == expected_shape &&
            tensor.dtype() == DataType::kFloat32 && tensor.device() == cpu &&
            tensor.strides() == expected_strides;
      }),
      0, "Rank-2 shape-only construction should stay inline.");
  context->Expect(shape_only_metadata_is_default,
                  "Shape-only construction should keep default metadata.");

  bool dtype_only_metadata_is_default = false;
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), TensorView::Shape{2, 3},
                          DataType::kFloat64};
        dtype_only_metadata_is_default =
            tensor.shape() == expected_shape &&
            tensor.dtype() == DataType::kFloat64 && tensor.device() == cpu &&
            tensor.strides() == expected_strides;
      }),
      0, "Rank-2 shape and dtype construction should stay inline.");
  context->Expect(
      dtype_only_metadata_is_default,
      "Shape and dtype construction should keep default device and strides.");

  bool device_only_metadata_is_default = false;
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), TensorView::Shape{2, 3}, indexed_cpu};
        device_only_metadata_is_default =
            tensor.shape() == expected_shape &&
            tensor.dtype() == DataType::kFloat32 &&
            tensor.device() == indexed_cpu &&
            tensor.strides() == expected_strides;
      }),
      0, "Rank-2 shape and device construction should stay inline.");
  context->Expect(
      device_only_metadata_is_default,
      "Shape and device construction should keep default dtype and strides.");
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  TestConstructionAllocationMatrix(&context);
  TestValueAndDerivedViewAllocations(&context);
  TestDefaultMetadataAllocations(&context);

  return context.ExitCode();
}
