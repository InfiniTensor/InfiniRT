#include <infini/rt.h>

#include <cstdint>
#include <string>

#include "test_helper.h"

namespace {

using infini::rt::DataType;
using infini::rt::Device;

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

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  TestDevice(&context);
  TestDataType(&context);

  return context.ExitCode();
}
