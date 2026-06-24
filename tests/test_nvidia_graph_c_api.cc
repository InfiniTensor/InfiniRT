#include <cuda_runtime.h>
#include <infini/rt/c_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "test_helper.h"

namespace {

bool ExpectCudaSuccess(infini::rt::test::TestContext* context,
                       cudaError_t status, std::string_view message) {
  return context->Expect(status == cudaSuccess, message);
}

bool ExpectRtSuccess(infini::rt::test::TestContext* context,
                     infiniRtStatus_t status, std::string_view message) {
  return context->Expect(status == INFINI_RT_STATUS_SUCCESS, message);
}

void FillPattern(std::array<std::uint8_t, 16>* input, std::uint8_t salt) {
  for (std::size_t i = 0; i < input->size(); ++i) {
    (*input)[i] = static_cast<std::uint8_t>(i * 13 + salt);
  }
}

bool CopyDeviceToHostAndValidate(infini::rt::test::TestContext* context,
                                 void* device_ptr,
                                 const std::array<std::uint8_t, 16>& expected,
                                 std::string_view message) {
  std::array<std::uint8_t, 16> output{};
  if (!ExpectCudaSuccess(context,
                         cudaMemcpy(output.data(), device_ptr, output.size(),
                                    cudaMemcpyDeviceToHost),
                         "Failed to copy device output to host.")) {
    return false;
  }
  return context->ExpectEqual(output, expected, message);
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  cudaStream_t native_stream = nullptr;
  void* src = nullptr;
  void* dst = nullptr;
  infiniRtStream_t stream = nullptr;
  infiniRtGraph_t graph = nullptr;
  infiniRtGraphExec_t graph_exec = nullptr;

  const auto device = infiniRtDevice_t{INFINI_RT_DEVICE_NVIDIA, 0};

  ExpectCudaSuccess(&context, cudaSetDevice(device.index),
                    "Failed to set CUDA device.");
  ExpectCudaSuccess(
      &context,
      cudaStreamCreateWithFlags(&native_stream, cudaStreamNonBlocking),
      "Failed to create CUDA stream.");

  std::array<std::uint8_t, 16> capture_input{};
  FillPattern(&capture_input, 7);

  ExpectCudaSuccess(&context, cudaMalloc(&src, capture_input.size()),
                    "Failed to allocate source buffer.");
  ExpectCudaSuccess(&context, cudaMalloc(&dst, capture_input.size()),
                    "Failed to allocate destination buffer.");
  ExpectCudaSuccess(&context,
                    cudaMemcpy(src, capture_input.data(), capture_input.size(),
                               cudaMemcpyHostToDevice),
                    "Failed to initialize source buffer.");
  ExpectCudaSuccess(&context, cudaMemset(dst, 0, capture_input.size()),
                    "Failed to initialize destination buffer.");

  ExpectRtSuccess(&context, infiniRtStreamWrap(device, native_stream, &stream),
                  "Failed to wrap native CUDA stream.");
  ExpectRtSuccess(
      &context,
      infiniRtStreamBeginCapture(stream, INFINI_RT_STREAM_CAPTURE_MODE_RELAXED),
      "Failed to begin graph capture through C API.");
  ExpectCudaSuccess(&context,
                    cudaMemcpyAsync(dst, src, capture_input.size(),
                                    cudaMemcpyDeviceToDevice, native_stream),
                    "Failed to record device-to-device copy.");
  ExpectRtSuccess(&context, infiniRtStreamEndCapture(stream, &graph),
                  "Failed to end graph capture through C API.");
  ExpectRtSuccess(&context, infiniRtGraphInstantiate(&graph_exec, graph),
                  "Failed to instantiate graph through C API.");

  std::array<std::uint8_t, 16> replay_input{};
  FillPattern(&replay_input, 31);

  ExpectCudaSuccess(&context,
                    cudaMemcpy(src, replay_input.data(), replay_input.size(),
                               cudaMemcpyHostToDevice),
                    "Failed to refresh source buffer.");
  ExpectCudaSuccess(&context, cudaMemset(dst, 0, replay_input.size()),
                    "Failed to clear destination buffer.");
  ExpectRtSuccess(&context, infiniRtGraphLaunch(graph_exec, stream),
                  "Failed to launch graph through C API.");
  ExpectCudaSuccess(&context, cudaStreamSynchronize(native_stream),
                    "Failed to synchronize CUDA stream.");
  CopyDeviceToHostAndValidate(&context, dst, replay_input,
                              "C API graph replay should copy D2D data.");

  if (graph_exec != nullptr) {
    ExpectRtSuccess(&context, infiniRtGraphExecDestroy(graph_exec),
                    "Failed to destroy graph exec through C API.");
  }
  if (graph != nullptr) {
    ExpectRtSuccess(&context, infiniRtGraphDestroy(graph),
                    "Failed to destroy graph through C API.");
  }
  if (stream != nullptr) {
    ExpectRtSuccess(&context, infiniRtStreamDestroy(stream),
                    "Failed to destroy wrapped stream through C API.");
  }
  if (dst != nullptr) {
    ExpectCudaSuccess(&context, cudaFree(dst),
                      "Failed to free destination buffer.");
  }
  if (src != nullptr) {
    ExpectCudaSuccess(&context, cudaFree(src), "Failed to free source buffer.");
  }
  if (native_stream != nullptr) {
    ExpectCudaSuccess(&context, cudaStreamDestroy(native_stream),
                      "Failed to destroy CUDA stream.");
  }

  return context.ExitCode();
}
