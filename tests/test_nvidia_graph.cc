#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "test_helper.h"

namespace {

namespace runtime = infini::rt::runtime;

void ExpectSuccess(infini::rt::test::TestContext* context,
                   runtime::Error status, const char* message) {
  context->Expect(status == runtime::kSuccess, message);
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
  runtime::Memcpy(output.data(), device_ptr, output.size(),
                  runtime::kMemcpyDeviceToHost);
  return context->ExpectEqual(output, expected, message);
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  infini::rt::set_runtime_device_type(infini::rt::Device::Type::kNvidia);
  ExpectSuccess(&context, runtime::SetDevice(0),
                "Failed to set NVIDIA runtime device.");

  void* src = nullptr;
  void* dst = nullptr;
  runtime::Stream stream = nullptr;
  runtime::Graph graph = nullptr;
  runtime::GraphExec graph_exec = nullptr;

  std::array<std::uint8_t, 16> capture_input{};
  FillPattern(&capture_input, 7);

  ExpectSuccess(&context, runtime::Malloc(&src, capture_input.size()),
                "Failed to allocate source buffer.");
  ExpectSuccess(&context, runtime::Malloc(&dst, capture_input.size()),
                "Failed to allocate destination buffer.");
  ExpectSuccess(&context, runtime::StreamCreate(&stream),
                "Failed to create stream.");

  ExpectSuccess(&context,
                runtime::Memcpy(src, capture_input.data(), capture_input.size(),
                                runtime::kMemcpyHostToDevice),
                "Failed to initialize source buffer.");
  ExpectSuccess(&context, runtime::Memset(dst, 0, capture_input.size()),
                "Failed to initialize destination buffer.");

  ExpectSuccess(
      &context,
      runtime::StreamBeginCapture(
          stream, runtime::StreamCaptureMode::kStreamCaptureModeRelaxed),
      "Failed to begin stream capture.");
  ExpectSuccess(&context,
                runtime::MemcpyAsync(dst, src, capture_input.size(),
                                     runtime::kMemcpyDeviceToDevice, stream),
                "Failed to record device-to-device copy.");
  ExpectSuccess(&context, runtime::StreamEndCapture(stream, &graph),
                "Failed to end stream capture.");

  ExpectSuccess(&context, runtime::GraphInstantiate(&graph_exec, graph),
                "Failed to instantiate graph.");

  std::array<std::uint8_t, 16> replay_input_1{};
  std::array<std::uint8_t, 16> replay_input_2{};
  FillPattern(&replay_input_1, 31);
  FillPattern(&replay_input_2, 53);

  ExpectSuccess(
      &context,
      runtime::Memcpy(src, replay_input_1.data(), replay_input_1.size(),
                      runtime::kMemcpyHostToDevice),
      "Failed to refresh first source buffer.");
  ExpectSuccess(&context, runtime::Memset(dst, 0, replay_input_1.size()),
                "Failed to clear first destination buffer.");
  ExpectSuccess(&context, runtime::DeviceSynchronize(),
                "Failed to synchronize first replay inputs.");
  ExpectSuccess(&context, runtime::GraphLaunch(graph_exec, stream),
                "Failed to launch first graph replay.");
  ExpectSuccess(&context, runtime::StreamSynchronize(stream),
                "Failed to synchronize first graph replay.");
  CopyDeviceToHostAndValidate(&context, dst, replay_input_1,
                              "First graph replay should copy D2D data.");

  ExpectSuccess(
      &context,
      runtime::Memcpy(src, replay_input_2.data(), replay_input_2.size(),
                      runtime::kMemcpyHostToDevice),
      "Failed to refresh second source buffer.");
  ExpectSuccess(&context, runtime::Memset(dst, 0, replay_input_2.size()),
                "Failed to clear second destination buffer.");
  ExpectSuccess(&context, runtime::DeviceSynchronize(),
                "Failed to synchronize second replay inputs.");
  ExpectSuccess(&context, runtime::GraphLaunch(graph_exec, stream),
                "Failed to launch second graph replay.");
  ExpectSuccess(&context, runtime::StreamSynchronize(stream),
                "Failed to synchronize second graph replay.");
  CopyDeviceToHostAndValidate(&context, dst, replay_input_2,
                              "Second graph replay should copy D2D data.");

  if (graph_exec != nullptr) {
    ExpectSuccess(&context, runtime::GraphExecDestroy(graph_exec),
                  "Failed to destroy graph exec.");
  }
  if (graph != nullptr) {
    ExpectSuccess(&context, runtime::GraphDestroy(graph),
                  "Failed to destroy graph.");
  }
  if (stream != nullptr) {
    ExpectSuccess(&context, runtime::StreamDestroy(stream),
                  "Failed to destroy stream.");
  }
  if (dst != nullptr) {
    ExpectSuccess(&context, runtime::Free(dst),
                  "Failed to free destination buffer.");
  }
  if (src != nullptr) {
    ExpectSuccess(&context, runtime::Free(src),
                  "Failed to free source buffer.");
  }

  return context.ExitCode();
}
