#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>

#include "model/input_encoder/shared_encoder.hpp"
#include "shared_encoder_fixture.hpp"

namespace {

using neuroevolution::model::input_encoder::EncodedTurnVector;
using neuroevolution::model::input_encoder::ForwardOccupiedTurn;
using neuroevolution::model::input_encoder::ForwardSharedEncoder;
using neuroevolution::model::input_encoder::kEncoderOutputSize;
using neuroevolution::tests::input_encoder::SharedEncoderGoldenFixture;

constexpr float kTolerance = 1.0e-6f;

bool ExpectTrue(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    return false;
  }

  return true;
}

bool ExpectVectorNear(const EncodedTurnVector& actual,
                      const EncodedTurnVector& expected,
                      const std::string_view label) {
  bool ok = true;

  for (std::size_t index = 0; index < kEncoderOutputSize; ++index) {
    const float delta = std::fabs(actual[index] - expected[index]);
    if (delta > kTolerance) {
      std::cerr << "FAIL: " << label << " mismatch at index " << index
                << ", expected " << expected[index]
                << ", got " << actual[index] << '\n';
      ok = false;
    }
  }

  return ok;
}

bool TestSharedEncoderForwardPassGoldenCase() {
  const SharedEncoderGoldenFixture fixture{};
  const auto input_vector = fixture.MaterializedInput();

  const EncodedTurnVector direct_output =
      ForwardSharedEncoder(fixture.parameters, input_vector);
  const EncodedTurnVector occupied_turn_output =
      ForwardOccupiedTurn(fixture.parameters, fixture.turn);

  bool ok = true;
  ok &= ExpectTrue(direct_output[0] == 5.5f, "Expected output neuron 0 to equal 5.5");
  ok &= ExpectTrue(direct_output[1] == 8.0f, "Expected output neuron 1 to equal 8.0");
  ok &= ExpectTrue(direct_output[2] == 1.25f, "Expected output neuron 2 to equal 1.25");
  ok &= ExpectVectorNear(direct_output, fixture.expected_output,
                         "ForwardSharedEncoder");
  ok &= ExpectVectorNear(occupied_turn_output, fixture.expected_output,
                         "ForwardOccupiedTurn");

  return ok;
}

}  // namespace

int main() {
  if (!TestSharedEncoderForwardPassGoldenCase()) {
    return 1;
  }

  std::cout << "PASS: shared_encoder_test\n";
  return 0;
}
