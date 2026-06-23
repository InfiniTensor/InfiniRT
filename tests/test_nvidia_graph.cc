#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "test_helper.h"

namespace {

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
  infini::rt::Memcpy(output.data(), device_ptr, output.size(),
                     infini::rt::MemcpyKind::kDeviceToHost);
  return context->ExpectEqual(output, expected, message);
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;
  const infini::rt::Device device{infini::rt::Device::Type::kNvidia, 0};

  infini::rt::SetDevice(device);

  void* src = nullptr;
  void* dst = nullptr;
  infini::rt::Stream stream;
  infini::rt::Graph graph;
  infini::rt::GraphExec graph_exec;

  std::array<std::uint8_t, 16> capture_input{};
  FillPattern(&capture_input, 7);

  infini::rt::Malloc(&src, capture_input.size());
  infini::rt::Malloc(&dst, capture_input.size());
  infini::rt::StreamCreate(&stream);

  context.Expect(stream.device_type() == infini::rt::Device::Type::kNvidia,
                 "Stream should remember its NVIDIA device type.");

  infini::rt::Memcpy(src, capture_input.data(), capture_input.size(),
                     infini::rt::MemcpyKind::kHostToDevice);
  infini::rt::Memset(dst, 0, capture_input.size());

  infini::rt::StreamBeginCapture(stream,
                                 infini::rt::StreamCaptureMode::kRelaxed);
  infini::rt::MemcpyAsync(dst, src, capture_input.size(),
                          infini::rt::MemcpyKind::kDeviceToDevice, stream);
  infini::rt::StreamEndCapture(stream, &graph);

  context.Expect(graph.device_type() == infini::rt::Device::Type::kNvidia,
                 "Graph should remember its NVIDIA device type.");

  infini::rt::GraphInstantiate(&graph_exec, graph);
  context.Expect(graph_exec.device_type() == infini::rt::Device::Type::kNvidia,
                 "GraphExec should remember its NVIDIA device type.");

  std::array<std::uint8_t, 16> replay_input_1{};
  std::array<std::uint8_t, 16> replay_input_2{};
  FillPattern(&replay_input_1, 31);
  FillPattern(&replay_input_2, 53);

  infini::rt::Memcpy(src, replay_input_1.data(), replay_input_1.size(),
                     infini::rt::MemcpyKind::kHostToDevice);
  infini::rt::Memset(dst, 0, replay_input_1.size());
  infini::rt::GraphLaunch(graph_exec, stream);
  infini::rt::StreamSynchronize(stream);
  CopyDeviceToHostAndValidate(&context, dst, replay_input_1,
                              "First graph replay should copy D2D data.");

  infini::rt::Memcpy(src, replay_input_2.data(), replay_input_2.size(),
                     infini::rt::MemcpyKind::kHostToDevice);
  infini::rt::Memset(dst, 0, replay_input_2.size());
  infini::rt::GraphLaunch(graph_exec, stream);
  infini::rt::StreamSynchronize(stream);
  CopyDeviceToHostAndValidate(&context, dst, replay_input_2,
                              "Second graph replay should copy D2D data.");

  infini::rt::GraphExecDestroy(graph_exec);
  infini::rt::GraphDestroy(graph);
  infini::rt::StreamDestroy(stream);
  infini::rt::Free(dst);
  infini::rt::Free(src);

  return context.ExitCode();
}
