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
  std::array<float, 32> data{};
  const Device cpu{Device::Type::kCpu};
  const TensorView::Shape shape4{2, 2, 2, 2};
  const TensorView::Strides strides4{8, 4, 2, 1};
  const TensorView::Shape shape5{2, 2, 2, 2, 2};
  const TensorView::Strides strides5{16, 8, 4, 2, 1};
  const VectorTensorLike tensor_like4{data.data(),
                                      {2, 2, 2, 2},
                                      DataType::kFloat32,
                                      cpu,
                                      {8, 4, 2, 1}};
  const VectorTensorLike tensor_like5{data.data(),
                                      {2, 2, 2, 2, 2},
                                      DataType::kFloat32,
                                      cpu,
                                      {16, 8, 4, 2, 1}};

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), shape4, DataType::kFloat32, cpu,
                          strides4};
        (void)tensor;
      }),
      0, "Rank-4 lvalue metadata should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), shape5, DataType::kFloat32, cpu,
                          strides5};
        (void)tensor;
      }),
      2, "Rank-5 lvalue metadata should allocate two owned arrays.");

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), TensorView::Shape{2, 2, 2, 2},
                          DataType::kFloat32, cpu,
                          TensorView::Strides{8, 4, 2, 1}};
        (void)tensor;
      }),
      0, "Rank-4 exact metadata temporaries should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), TensorView::Shape{2, 2, 2, 2, 2},
                          DataType::kFloat32, cpu,
                          TensorView::Strides{16, 8, 4, 2, 1}};
        (void)tensor;
      }),
      2, "Rank-5 exact metadata temporaries should allocate twice.");

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), {2, 2, 2, 2}, DataType::kFloat32, cpu,
                          {8, 4, 2, 1}};
        (void)tensor;
      }),
      0, "Rank-4 initializer-list metadata should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), {2, 2, 2, 2, 2},
                          DataType::kFloat32, cpu, {16, 8, 4, 2, 1}};
        (void)tensor;
      }),
      2, "Rank-5 initializer-list metadata should allocate twice.");

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{tensor_like4};
        (void)tensor;
      }),
      0, "Rank-4 vector-backed TensorLike metadata should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{tensor_like5};
        (void)tensor;
      }),
      2, "Rank-5 vector-backed TensorLike metadata should allocate twice.");

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), shape4, DataType::kFloat32, cpu};
        (void)tensor;
      }),
      0, "Rank-4 ordinary default-stride construction should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), shape5, DataType::kFloat32, cpu};
        (void)tensor;
      }),
      2, "Rank-5 ordinary default-stride construction should allocate twice.");

  TensorView::Shape explicit_move_shape4{2, 2, 2, 2};
  TensorView::Strides explicit_move_strides4{8, 4, 2, 1};
  TensorView::Shape explicit_move_shape5{2, 2, 2, 2, 2};
  TensorView::Strides explicit_move_strides5{16, 8, 4, 2, 1};

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), std::move(explicit_move_shape4),
                          DataType::kFloat32, cpu,
                          std::move(explicit_move_strides4)};
        (void)tensor;
      }),
      0, "Moving rank-4 explicit metadata should not allocate.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), std::move(explicit_move_shape5),
                          DataType::kFloat32, cpu,
                          std::move(explicit_move_strides5)};
        (void)tensor;
      }),
      0, "Moving rank-5 explicit metadata should not allocate.");

  TensorView::Shape default_move_shape4{2, 2, 2, 2};
  TensorView::Shape default_move_shape5{2, 2, 2, 2, 2};

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), std::move(default_move_shape4),
                          DataType::kFloat32, cpu};
        (void)tensor;
      }),
      0, "Moving a rank-4 shape while generating strides should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView tensor{data.data(), std::move(default_move_shape5),
                          DataType::kFloat32, cpu};
        (void)tensor;
      }),
      1, "Moving a rank-5 shape should allocate only default strides.");
}

void TestValueAndDerivedViewAllocations(
    infini::rt::test::TestContext* context) {
  std::array<float, 32> data{};
  const Device cpu{Device::Type::kCpu};
  const TensorView::Shape shape4{2, 2, 2, 2};
  const TensorView::Strides strides4{8, 4, 2, 1};
  const TensorView::Shape shape5{2, 2, 2, 2, 2};
  const TensorView::Strides strides5{16, 8, 4, 2, 1};
  const TensorView source4{data.data(), shape4, DataType::kFloat32, cpu,
                           strides4};
  const TensorView source5{data.data(), shape5, DataType::kFloat32, cpu,
                           strides5};

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView copied{source4};
        (void)copied;
      }),
      0, "Copying rank-4 metadata should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView copied{source5};
        (void)copied;
      }),
      2, "Copying rank-5 metadata should allocate two independent arrays.");

  TensorView move_source4{data.data(), shape4, DataType::kFloat32, cpu,
                          strides4};
  TensorView move_source5{data.data(), shape5, DataType::kFloat32, cpu,
                          strides5};

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView moved{std::move(move_source4)};
        (void)moved;
      }),
      0, "Moving rank-4 metadata should not allocate.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView moved{std::move(move_source5)};
        (void)moved;
      }),
      0, "Moving rank-5 metadata should transfer heap storage.");

  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView indexed = source4[0];
        (void)indexed;
      }),
      0, "Indexing rank 4 to rank 3 should stay inline.");
  ExpectAllocationCount(
      context,
      CountAllocations([&] {
        TensorView indexed = source5[0];
        (void)indexed;
      }),
      0, "Indexing rank 5 to rank 4 should stay inline.");

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
