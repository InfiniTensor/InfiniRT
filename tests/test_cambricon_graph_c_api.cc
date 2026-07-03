#include <cnrt.h>
#include <infini/rt/c_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "test_helper.h"

namespace {

bool ExpectCnrtSuccess(infini::rt::test::TestContext* context,
                       cnrtRet_t status, std::string_view message) {
  return context->Expect(status == cnrtSuccess, message);
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
  if (!ExpectCnrtSuccess(context,
                         cnrtMemcpy(output.data(), device_ptr, output.size(),
                                    cnrtMemcpyDevToHost),
                         "Failed to copy device output to host.")) {
    return false;
  }
  return context->ExpectEqual(output, expected, message);
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  cnrtQueue_t native_stream = nullptr;
  void* src = nullptr;
  void* dst = nullptr;
  infiniRtStream_t stream = nullptr;
  infiniRtGraph_t graph = nullptr;
  infiniRtGraphExec_t graph_exec = nullptr;

  const auto device = infiniRtDevice_t{INFINI_RT_DEVICE_CAMBRICON, 0};

  ExpectCnrtSuccess(&context, cnrtSetDevice(device.index),
                    "Failed to set Cambricon device.");
  ExpectCnrtSuccess(&context, cnrtQueueCreate(&native_stream),
                    "Failed to create CNRT queue.");

  std::array<std::uint8_t, 16> capture_input{};
  FillPattern(&capture_input, 7);

  ExpectCnrtSuccess(&context, cnrtMalloc(&src, capture_input.size()),
                    "Failed to allocate source buffer.");
  ExpectCnrtSuccess(&context, cnrtMalloc(&dst, capture_input.size()),
                    "Failed to allocate destination buffer.");
  ExpectCnrtSuccess(&context,
                    cnrtMemcpy(src, capture_input.data(), capture_input.size(),
                               cnrtMemcpyHostToDev),
                    "Failed to initialize source buffer.");
  ExpectCnrtSuccess(&context, cnrtMemset(dst, 0, capture_input.size()),
                    "Failed to initialize destination buffer.");

  ExpectRtSuccess(&context, infiniRtStreamWrap(device, native_stream, &stream),
                  "Failed to wrap native CNRT queue.");
  ExpectRtSuccess(
      &context,
      infiniRtStreamBeginCapture(stream, INFINI_RT_STREAM_CAPTURE_MODE_RELAXED),
      "Failed to begin graph capture through C API.");
  ExpectCnrtSuccess(
      &context,
      cnrtMemcpyAsync_V2(dst, src, capture_input.size(), native_stream,
                         cnrtMemcpyDevToDev),
      "Failed to record device-to-device copy.");
  ExpectRtSuccess(&context, infiniRtStreamEndCapture(stream, &graph),
                  "Failed to end graph capture through C API.");
  ExpectRtSuccess(&context, infiniRtGraphInstantiate(&graph_exec, graph),
                  "Failed to instantiate graph through C API.");

  std::array<std::uint8_t, 16> replay_input{};
  FillPattern(&replay_input, 31);

  ExpectCnrtSuccess(&context,
                    cnrtMemcpy(src, replay_input.data(), replay_input.size(),
                               cnrtMemcpyHostToDev),
                    "Failed to refresh source buffer.");
  ExpectCnrtSuccess(&context, cnrtMemset(dst, 0, replay_input.size()),
                    "Failed to clear destination buffer.");
  ExpectRtSuccess(&context, infiniRtGraphLaunch(graph_exec, stream),
                  "Failed to launch graph through C API.");
  ExpectCnrtSuccess(&context, cnrtQueueSync(native_stream),
                    "Failed to synchronize CNRT queue.");
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
    ExpectCnrtSuccess(&context, cnrtFree(dst),
                      "Failed to free destination buffer.");
  }
  if (src != nullptr) {
    ExpectCnrtSuccess(&context, cnrtFree(src), "Failed to free source buffer.");
  }
  if (native_stream != nullptr) {
    ExpectCnrtSuccess(&context, cnrtQueueDestroy(native_stream),
                      "Failed to destroy CNRT queue.");
  }

  return context.ExitCode();
}
