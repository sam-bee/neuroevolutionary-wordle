#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>

#include "model/dense_trunk/dense_trunk.hpp"
#include "model/model_input/model_input_spec.hpp"

namespace {

using neuroevolution::model::dense_trunk::DenseTrunkInputVector;
using neuroevolution::model::dense_trunk::DenseTrunkParameters;
using neuroevolution::model::dense_trunk::ForwardDenseTrunk;
using neuroevolution::model::dense_trunk::kDenseTrunkHiddenSize0;
using neuroevolution::model::dense_trunk::kDenseTrunkHiddenSize1;
using neuroevolution::model::dense_trunk::kDenseTrunkInputSize;
using neuroevolution::model::dense_trunk::kDenseTrunkOutputSize;
using neuroevolution::model::dense_trunk::PolicyVector;

constexpr float kTolerance = 1.0e-6f;

bool ExpectVectorNear(const PolicyVector &actual, const PolicyVector &expected, const std::string_view label) {
    bool ok = true;

    for (std::size_t index = 0; index < kDenseTrunkOutputSize; ++index) {
        const float delta = std::fabs(actual[index] - expected[index]);
        if (delta > kTolerance) {
            std::cerr << "FAIL: " << label << " output neuron " << index << " expected " << expected[index] << ", got "
                      << actual[index] << '\n';
            ok = false;
        }
    }

    return ok;
}

DenseTrunkInputVector MakeGoldenInput() {
    DenseTrunkInputVector input{};
    input[0] = 2.0f;
    input[1] = 1.0f;
    input[5] = 1.0f;
    input[64] = 3.0f;
    input[129] = 1.0f;
    return input;
}

DenseTrunkParameters MakeGoldenParameters() {
    DenseTrunkParameters parameters{};

    parameters.input_to_hidden0.biases[0] = -1.0f;
    parameters.input_to_hidden0.weights[(0 * kDenseTrunkInputSize) + 0] = 0.5f;
    parameters.input_to_hidden0.weights[(0 * kDenseTrunkInputSize) + 1] = 1.5f;

    parameters.input_to_hidden0.biases[1] = -0.5f;
    parameters.input_to_hidden0.weights[(1 * kDenseTrunkInputSize) + 64] = 1.0f;
    parameters.input_to_hidden0.weights[(1 * kDenseTrunkInputSize) + 129] = 2.0f;

    parameters.input_to_hidden0.biases[2] = -2.0f;
    parameters.input_to_hidden0.weights[(2 * kDenseTrunkInputSize) + 5] = 1.0f;

    parameters.hidden0_to_hidden1.biases[0] = 0.25f;
    parameters.hidden0_to_hidden1.weights[(0 * kDenseTrunkHiddenSize0) + 0] = 2.0f;
    parameters.hidden0_to_hidden1.weights[(0 * kDenseTrunkHiddenSize0) + 1] = -0.5f;

    parameters.hidden0_to_hidden1.biases[1] = -3.0f;
    parameters.hidden0_to_hidden1.weights[(1 * kDenseTrunkHiddenSize0) + 1] = 0.5f;

    parameters.hidden1_to_output.biases[0] = -1.0f;
    parameters.hidden1_to_output.weights[(0 * kDenseTrunkHiddenSize1) + 0] = 3.0f;

    parameters.hidden1_to_output.biases[1] = 0.5f;
    parameters.hidden1_to_output.weights[(1 * kDenseTrunkHiddenSize1) + 0] = -2.0f;
    parameters.hidden1_to_output.weights[(1 * kDenseTrunkHiddenSize1) + 1] = 4.0f;

    parameters.hidden1_to_output.biases[2] = 1.25f;
    parameters.hidden1_to_output.weights[(2 * kDenseTrunkHiddenSize1) + 1] = 7.0f;

    return parameters;
}

PolicyVector MakeGoldenExpectedOutput() {
    PolicyVector expected{};
    expected[0] = 2.0f;
    expected[1] = -1.5f;
    expected[2] = 1.25f;
    return expected;
}

bool TestDenseTrunkGoldenCase() {
    const DenseTrunkParameters parameters = MakeGoldenParameters();
    const DenseTrunkInputVector input = MakeGoldenInput();
    const PolicyVector expected = MakeGoldenExpectedOutput();

    PolicyVector output{};
    ForwardDenseTrunk(parameters, input, output);
    const PolicyVector returned_output = ForwardDenseTrunk(parameters, input);

    bool ok = true;
    ok &= ExpectVectorNear(output, expected, "ForwardDenseTrunk(in-place)");
    ok &= ExpectVectorNear(returned_output, expected, "ForwardDenseTrunk(returning)");
    return ok;
}

bool TestDenseTrunkAppliesReLUToHiddenLayers() {
    DenseTrunkParameters parameters{};
    DenseTrunkInputVector input{};

    parameters.input_to_hidden0.biases[0] = -2.0f;

    parameters.hidden0_to_hidden1.biases[0] = 3.0f;
    parameters.hidden0_to_hidden1.weights[(0 * kDenseTrunkHiddenSize0) + 0] = 1.0f;

    parameters.hidden0_to_hidden1.biases[1] = -4.0f;

    parameters.hidden1_to_output.biases[0] = 0.5f;
    parameters.hidden1_to_output.weights[(0 * kDenseTrunkHiddenSize1) + 0] = 2.0f;

    parameters.hidden1_to_output.biases[1] = 0.25f;
    parameters.hidden1_to_output.weights[(1 * kDenseTrunkHiddenSize1) + 1] = 3.0f;

    PolicyVector expected{};
    expected[0] = 6.5f;
    expected[1] = 0.25f;

    PolicyVector output{};
    ForwardDenseTrunk(parameters, input, output);

    return ExpectVectorNear(output, expected, "ForwardDenseTrunk(ReLU)");
}

} // namespace

int main() {
    if (!TestDenseTrunkGoldenCase()) {
        return 1;
    }

    if (!TestDenseTrunkAppliesReLUToHiddenLayers()) {
        return 1;
    }

    std::cout << "PASS: dense_trunk_test\n";
    return 0;
}
