#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

#include "perf_common.h"

namespace {

namespace perf = infini::rt::perf;

using infini::rt::DataType;
using infini::rt::Device;
using infini::rt::TensorView;

}  // namespace

int main() {
  constexpr std::size_t kIterations = 200000;
  std::array<float, 32 * 64> data{};
  const TensorView::Shape shape{32, 64};
  const TensorView::Strides contiguous_strides{64, 1};
  const TensorView::Strides transposed_strides{1, 32};
  const Device cpu_device{Device::Type::kCpu};

  perf::RunBenchmark("perf_tensor_view.construct_contiguous",
                     {perf::NumberParam("ndim", 2)}, kIterations, "ns", [&] {
                       TensorView tensor{data.data(), shape, DataType::kFloat32,
                                         cpu_device, contiguous_strides};
                       perf::DoNotOptimize(tensor);
                     });

  perf::RunBenchmark("perf_tensor_view.construct_rvalue_full_metadata",
                     {perf::NumberParam("ndim", 2)}, kIterations, "ns", [&] {
                       TensorView tensor{data.data(), TensorView::Shape{32, 64},
                                         DataType::kFloat32, cpu_device,
                                         TensorView::Strides{64, 1}};
                       perf::DoNotOptimize(tensor);
                     });

  perf::RunBenchmark("perf_tensor_view.construct_rvalue_default_strides",
                     {perf::NumberParam("ndim", 2)}, kIterations, "ns", [&] {
                       TensorView tensor{data.data(), TensorView::Shape{32, 64},
                                         DataType::kFloat32, cpu_device};
                       perf::DoNotOptimize(tensor);
                     });

  perf::RunBenchmark(
      "perf_tensor_view.construct_initializer_list",
      {perf::NumberParam("ndim", 2)}, kIterations, "ns", [&] {
        TensorView tensor{
            data.data(), {32, 64}, DataType::kFloat32, cpu_device, {64, 1}};
        perf::DoNotOptimize(tensor);
      });

  TensorView contiguous{data.data(), shape, DataType::kFloat32, cpu_device,
                        contiguous_strides};
  TensorView transposed{data.data(), shape, DataType::kFloat32, cpu_device,
                        transposed_strides};

  perf::RunBenchmark("perf_tensor_view.transpose",
                     {perf::NumberParam("ndim", 2)}, kIterations, "ns", [&] {
                       const auto result = contiguous.T();
                       perf::DoNotOptimize(result);
                     });

  perf::RunBenchmark("perf_tensor_view.operator_index",
                     {perf::NumberParam("ndim", 2)}, kIterations, "ns", [&] {
                       const auto slice = contiguous[1];
                       perf::DoNotOptimize(slice);
                     });

  perf::RunBenchmark("perf_tensor_view.numel", {perf::NumberParam("ndim", 2)},
                     kIterations, "ns", [&] {
                       const auto count = contiguous.numel();
                       perf::DoNotOptimize(count);
                     });

  perf::RunBenchmark("perf_tensor_view.is_contiguous_true",
                     {perf::NumberParam("ndim", 2)}, kIterations, "ns", [&] {
                       const auto result = contiguous.IsContiguous();
                       perf::DoNotOptimize(result);
                     });

  perf::RunBenchmark("perf_tensor_view.is_contiguous_false",
                     {perf::NumberParam("ndim", 2)}, kIterations, "ns", [&] {
                       const auto result = transposed.IsContiguous();
                       perf::DoNotOptimize(result);
                     });

  perf::RunBenchmark("perf_tensor_view.hash", {perf::NumberParam("ndim", 2)},
                     kIterations, "ns", [&] {
                       const auto value = std::hash<TensorView>{}(contiguous);
                       perf::DoNotOptimize(value);
                     });

  return 0;
}
