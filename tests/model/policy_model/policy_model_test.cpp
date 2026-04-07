#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>

#include "model/policy_model/policy_model.hpp"
#include "policy_model_fixture.hpp"
#include "wordle/wordle_grid.hpp"

namespace {

using neuroevolution::model::policy_model::ForwardPolicyModel;
using neuroevolution::model::policy_model::PolicyVector;
using neuroevolution::model::policy_model::TryForwardPolicyModel;
using neuroevolution::tests::policy_model::PolicyModelGoldenFixture;
using neuroevolution::wordle::MakeWordleGrid;

constexpr float kTolerance = 1.0e-6f;

bool ExpectVectorNear(const PolicyVector &actual, const PolicyVector &expected, const std::string_view label) {
    bool ok = true;

    for (std::size_t index = 0; index < neuroevolution::model::dense_trunk::kDenseTrunkOutputSize; ++index) {
        const float delta = std::fabs(actual[index] - expected[index]);
        if (delta > kTolerance) {
            std::cerr << "FAIL: " << label << " output neuron " << index << " expected " << expected[index] << ", got "
                      << actual[index] << '\n';
            ok = false;
        }
    }

    return ok;
}

bool TestForwardPolicyModelSingleTurnGoldenCase() {
    const PolicyModelGoldenFixture fixture{};
    const auto grid = fixture.MakeSingleTurnGrid();

    PolicyVector output{};
    const bool forward_ok = TryForwardPolicyModel(fixture.parameters, grid, output);
    const PolicyVector returned_output = ForwardPolicyModel(fixture.parameters, grid);

    bool ok = true;
    ok &= forward_ok;
    ok &= ExpectVectorNear(output, fixture.expected_single_turn_output, "TryForwardPolicyModel(single-turn)");
    ok &= ExpectVectorNear(returned_output, fixture.expected_single_turn_output, "ForwardPolicyModel(single-turn)");
    return ok;
}

bool TestForwardPolicyModelVirginGridUsesVirginFlag() {
    const PolicyModelGoldenFixture fixture{};
    const auto grid = MakeWordleGrid(fixture.solution);

    PolicyVector output{};
    const bool forward_ok = TryForwardPolicyModel(fixture.parameters, grid, output);
    const PolicyVector returned_output = ForwardPolicyModel(fixture.parameters, grid);

    bool ok = true;
    ok &= forward_ok;
    ok &= ExpectVectorNear(output, fixture.expected_virgin_output, "TryForwardPolicyModel(virgin)");
    ok &= ExpectVectorNear(returned_output, fixture.expected_virgin_output, "ForwardPolicyModel(virgin)");
    return ok;
}

} // namespace

int main() {
    if (!TestForwardPolicyModelSingleTurnGoldenCase()) {
        return 1;
    }

    if (!TestForwardPolicyModelVirginGridUsesVirginFlag()) {
        return 1;
    }

    std::cout << "PASS: policy_model_test\n";
    return 0;
}
