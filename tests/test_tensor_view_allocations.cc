#include <infini/rt.h>

#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>

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

void TestTensorViewAllocations(infini::rt::test::TestContext* context) {
  alignas(float) std::byte data[6 * sizeof(float)]{};
  const Device cpu{Device::Type::kCpu};
  const Device indexed_cpu{Device::Type::kCpu, 1};
  const TensorView::Shape shape{2, 3};
  const TensorView::Strides strides{3, 1};

  bool shape_only_metadata_is_default = false;
  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data, TensorView::Shape{2, 3}};
        shape_only_metadata_is_default = tensor.dtype() == DataType::kFloat32 &&
                                         tensor.device() == cpu &&
                                         tensor.strides() == strides;
      }),
      2, "Rvalue shape construction should use default metadata directly.");
  context->Expect(shape_only_metadata_is_default,
                  "Shape-only construction should keep default metadata.");

  bool dtype_only_metadata_is_default = false;
  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data, TensorView::Shape{2, 3}, DataType::kFloat64};
        dtype_only_metadata_is_default = tensor.dtype() == DataType::kFloat64 &&
                                         tensor.device() == cpu &&
                                         tensor.strides() == strides;
      }),
      2, "Rvalue shape and dtype should use default device and strides.");
  context->Expect(
      dtype_only_metadata_is_default,
      "Shape and dtype construction should keep default device and strides.");

  bool device_only_metadata_is_default = false;
  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data, TensorView::Shape{2, 3}, indexed_cpu};
        device_only_metadata_is_default =
            tensor.dtype() == DataType::kFloat32 &&
            tensor.device() == indexed_cpu && tensor.strides() == strides;
      }),
      2, "Rvalue shape and device should use default dtype and strides.");
  context->Expect(
      device_only_metadata_is_default,
      "Shape and device construction should keep default dtype and strides.");

  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data, shape, DataType::kFloat32, cpu, strides};
        (void)tensor;
      }),
      2, "Lvalue shape and strides should allocate only their owned copies.");

  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data, TensorView::Shape{2, 3}, DataType::kFloat32,
                          cpu, TensorView::Strides{3, 1}};
        (void)tensor;
      }),
      2, "Rvalue shape and strides should transfer their allocations.");

  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data, TensorView::Shape{2, 3}, DataType::kFloat32,
                          cpu};
        (void)tensor;
      }),
      2,
      "Rvalue shape construction should allocate shape and default strides.");

  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data, {2, 3}, DataType::kFloat32, cpu, {3, 1}};
        (void)tensor;
      }),
      2, "Initializer lists should construct owned metadata directly.");

  const TensorView source{data, shape, DataType::kFloat32, cpu, strides};

  ExpectAllocationCount(context, CountAllocations([&] {
                          TensorView indexed = source[0];
                          (void)indexed;
                        }),
                        2,
                        "Indexing should allocate only the result metadata.");

  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView transposed = source.T();
        (void)transposed;
      }),
      2, "Transposing should allocate only the result metadata.");

  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView copied{source};
        (void)copied;
      }),
      2, "Copying should allocate one owned shape and one owned stride array.");

  TensorView move_source{data, shape, DataType::kFloat32, cpu, strides};
  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView moved{std::move(move_source)};
        (void)moved;
      }),
      0, "Moving should transfer owned metadata without allocating.");
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  TestTensorViewAllocations(&context);

  return context.ExitCode();
}
