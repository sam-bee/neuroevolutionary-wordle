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
using neuroevolution::model::input_encoder::kEncoderOutputSize;
using neuroevolution::model::input_encoder::detail::ForwardSharedEncoder;
using neuroevolution::tests::input_encoder::SharedEncoderGoldenFixture;

constexpr float kTolerance = 1.0e-6f;

bool ExpectNeuronEquals(const float actual, const float expected, const std::size_t neuron_index) {
    if (actual != expected) {
        std::cerr << "FAIL: output neuron " << neuron_index << " expected " << expected << ", got " << actual << '\n';
        return false;
    }

    return true;
}

bool ExpectVectorNear(const EncodedTurnVector &actual, const EncodedTurnVector &expected,
                      const std::string_view label) {
    bool ok = true;

    for (std::size_t index = 0; index < kEncoderOutputSize; ++index) {
        const float delta = std::fabs(actual[index] - expected[index]);
        if (delta > kTolerance) {
            std::cerr << "FAIL: " << label << " output neuron " << index << " expected " << expected[index] << ", got "
                      << actual[index] << '\n';
            ok = false;
        }
    }

    return ok;
}

bool TestSharedEncoderForwardPassGoldenCase() {
    const SharedEncoderGoldenFixture fixture{};
    const auto input_vector = fixture.MaterializedInput();

    EncodedTurnVector direct_output{};
    ForwardSharedEncoder(fixture.parameters, input_vector, direct_output);
    const EncodedTurnVector occupied_turn_output = ForwardOccupiedTurn(fixture.parameters, fixture.turn);

    bool ok = true;
    ok &= ExpectNeuronEquals(direct_output[0], fixture.expected_output[0], 0);
    ok &= ExpectNeuronEquals(direct_output[1], fixture.expected_output[1], 1);
    ok &= ExpectNeuronEquals(direct_output[2], fixture.expected_output[2], 2);
    ok &= ExpectVectorNear(direct_output, fixture.expected_output, "ForwardSharedEncoder");
    ok &= ExpectVectorNear(occupied_turn_output, fixture.expected_output, "ForwardOccupiedTurn");

    return ok;
}

} // namespace

int main() {
    if (!TestSharedEncoderForwardPassGoldenCase()) {
        return 1;
    }

    std::cout << "PASS: shared_encoder_test\n";
    return 0;
}
