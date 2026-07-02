#include <infini/rt.h>

#include <cstdint>
#include <type_traits>
#include <vector>

#include "test_helper.h"

namespace {

using infini::rt::DataType;
using infini::rt::Device;
using infini::rt::TensorView;

static_assert(!std::is_constructible_v<TensorView, std::vector<TensorView>>,
              "TensorView should not treat tensor containers as tensor-like.");

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

  TensorView strided{data.data(), std::vector<std::size_t>{2, 3},
                     DataType::kFloat32, Device{Device::Type::kCpu},
                     std::vector<std::ptrdiff_t>{4, 1}};
  context->Expect(!strided.IsContiguous(),
                  "TensorView with row padding should not be contiguous.");
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  TestTensorView(&context);

  return context.ExitCode();
}
