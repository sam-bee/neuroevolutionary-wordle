#include <cstddef>
#include <iostream>
#include <string_view>

#include "common/float16.hpp"
#include "model/initialization/parameter_initialization.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::model::initialization::IsValidParameterInitializationConfig;
using neuroevolution::model::initialization::MakeRandomOutputEmbeddingTrainableTails;
using neuroevolution::model::initialization::MakeRandomPolicyModelParameters;
using neuroevolution::model::initialization::ParameterInitializationConfig;
using neuroevolution::model::output_embedding::kTrainableFeatureDimension;
using neuroevolution::model::policy_model::PolicyModelParameters;

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

template <typename Scalar, std::size_t Size>
bool BuffersMatch(const neuroevolution::common::FixedBuffer<Scalar, Size> &lhs,
                  const neuroevolution::common::FixedBuffer<Scalar, Size> &rhs) {
    for (std::size_t index = 0; index < Size; ++index) {
        if (ToFloat(lhs[index]) != ToFloat(rhs[index])) {
            return false;
        }
    }

    return true;
}

template <typename DenseLayer> bool DenseLayerMatches(const DenseLayer &lhs, const DenseLayer &rhs) {
    return BuffersMatch(lhs.weights, rhs.weights) && BuffersMatch(lhs.biases, rhs.biases);
}

bool PolicyModelParametersMatch(const PolicyModelParameters &lhs, const PolicyModelParameters &rhs) {
    return DenseLayerMatches(lhs.input_encoder.input_to_hidden, rhs.input_encoder.input_to_hidden) &&
           DenseLayerMatches(lhs.input_encoder.hidden_to_output, rhs.input_encoder.hidden_to_output) &&
           DenseLayerMatches(lhs.dense_trunk.input_to_hidden0, rhs.dense_trunk.input_to_hidden0) &&
           DenseLayerMatches(lhs.dense_trunk.hidden0_to_hidden1, rhs.dense_trunk.hidden0_to_hidden1) &&
           DenseLayerMatches(lhs.dense_trunk.hidden1_to_output, rhs.dense_trunk.hidden1_to_output);
}

template <typename DenseLayer> bool DenseLayerBiasesAreZero(const DenseLayer &layer) {
    for (std::size_t index = 0; index < sizeof(layer.biases.values) / sizeof(layer.biases.values[0]); ++index) {
        if (ToFloat(layer.biases[index]) != 0.0f) {
            return false;
        }
    }

    return true;
}

bool PolicyModelBiasesAreZero(const PolicyModelParameters &parameters) {
    return DenseLayerBiasesAreZero(parameters.input_encoder.input_to_hidden) &&
           DenseLayerBiasesAreZero(parameters.input_encoder.hidden_to_output) &&
           DenseLayerBiasesAreZero(parameters.dense_trunk.input_to_hidden0) &&
           DenseLayerBiasesAreZero(parameters.dense_trunk.hidden0_to_hidden1) &&
           DenseLayerBiasesAreZero(parameters.dense_trunk.hidden1_to_output);
}

bool PolicyModelHasAnyNonZeroWeight(const PolicyModelParameters &parameters) {
    return ToFloat(parameters.input_encoder.input_to_hidden.weights[0]) != 0.0f ||
           ToFloat(parameters.input_encoder.hidden_to_output.weights[0]) != 0.0f ||
           ToFloat(parameters.dense_trunk.input_to_hidden0.weights[0]) != 0.0f ||
           ToFloat(parameters.dense_trunk.hidden0_to_hidden1.weights[0]) != 0.0f ||
           ToFloat(parameters.dense_trunk.hidden1_to_output.weights[0]) != 0.0f;
}

bool OutputEmbeddingTailsMatch(
    const neuroevolution::common::FixedBuffer<neuroevolution::model::output_embedding::TrainableActionEmbeddingTail, 3>
        &lhs,
    const neuroevolution::common::FixedBuffer<neuroevolution::model::output_embedding::TrainableActionEmbeddingTail, 3>
        &rhs) {
    for (std::size_t action_index = 0; action_index < 3; ++action_index) {
        if (!BuffersMatch(lhs[action_index], rhs[action_index])) {
            return false;
        }
    }

    return true;
}

bool OutputEmbeddingHasAnyNonZeroTailValue(
    const neuroevolution::common::FixedBuffer<neuroevolution::model::output_embedding::TrainableActionEmbeddingTail, 3>
        &trainable_tails) {
    for (std::size_t action_index = 0; action_index < 3; ++action_index) {
        for (std::size_t feature_index = 0; feature_index < kTrainableFeatureDimension; ++feature_index) {
            if (ToFloat(trainable_tails[action_index][feature_index]) != 0.0f) {
                return true;
            }
        }
    }

    return false;
}

bool TestPolicyModelInitializationIsSeededAndZeroBias() {
    const PolicyModelParameters parameters_a = MakeRandomPolicyModelParameters(1234);
    const PolicyModelParameters parameters_b = MakeRandomPolicyModelParameters(1234);
    const PolicyModelParameters parameters_c = MakeRandomPolicyModelParameters(5678);

    bool ok = true;
    ok &= ExpectTrue(PolicyModelParametersMatch(parameters_a, parameters_b),
                     "Expected same seed to reproduce policy-model parameters");
    ok &= ExpectTrue(!PolicyModelParametersMatch(parameters_a, parameters_c),
                     "Expected different seeds to produce different policy-model parameters");
    ok &= ExpectTrue(PolicyModelBiasesAreZero(parameters_a), "Expected dense-layer biases to start at zero");
    ok &= ExpectTrue(PolicyModelHasAnyNonZeroWeight(parameters_a),
                     "Expected random policy-model initialization to create non-zero weights");
    return ok;
}

bool TestOutputEmbeddingTailInitializationIsSeededAndUsesConfiguredStddev() {
    ParameterInitializationConfig config{};
    config.output_embedding_tail_stddev = 0.2f;

    auto tails_a = MakeRandomOutputEmbeddingTrainableTails<3>(42, config);
    auto tails_b = MakeRandomOutputEmbeddingTrainableTails<3>(42, config);
    auto tails_c = MakeRandomOutputEmbeddingTrainableTails<3>(43, config);

    bool ok = true;
    ok &= ExpectTrue(OutputEmbeddingTailsMatch(tails_a, tails_b),
                     "Expected same seed to reproduce output-embedding tails");
    ok &= ExpectTrue(!OutputEmbeddingTailsMatch(tails_a, tails_c),
                     "Expected different seeds to produce different output-embedding tails");
    ok &= ExpectTrue(OutputEmbeddingHasAnyNonZeroTailValue(tails_a),
                     "Expected output-embedding tail initialization to create non-zero values");
    return ok;
}

bool TestInitializationConfigValidation() {
    ParameterInitializationConfig valid_config{};
    ParameterInitializationConfig invalid_gain = valid_config;
    invalid_gain.dense_weight_gain = 0.0f;

    ParameterInitializationConfig invalid_stddev = valid_config;
    invalid_stddev.output_embedding_tail_stddev = -0.25f;

    bool ok = true;
    ok &= ExpectTrue(IsValidParameterInitializationConfig(valid_config), "Expected default init config to be valid");
    ok &= ExpectTrue(!IsValidParameterInitializationConfig(invalid_gain),
                     "Expected non-positive dense gain to be rejected");
    ok &= ExpectTrue(!IsValidParameterInitializationConfig(invalid_stddev),
                     "Expected negative output-embedding stddev to be rejected");
    return ok;
}

} // namespace

int main() {
    if (!TestPolicyModelInitializationIsSeededAndZeroBias()) {
        return 1;
    }

    if (!TestOutputEmbeddingTailInitializationIsSeededAndUsesConfiguredStddev()) {
        return 1;
    }

    if (!TestInitializationConfigValidation()) {
        return 1;
    }

    std::cout << "PASS: parameter_initialization_test\n";
    return 0;
}
