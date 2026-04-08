#include "model/initialization/parameter_initialization.hpp"

#include <stdexcept>

namespace neuroevolution::model::initialization {

void InitializeRandomPolicyModelParameters(policy_model::PolicyModelParameters &parameters, RandomEngine &random_engine,
                                           const ParameterInitializationConfig &config) {
    if (!IsValidParameterInitializationConfig(config)) {
        throw std::invalid_argument("Parameter initialization config must use positive dense gain and non-negative "
                                    "output-embedding tail stddev.");
    }

    detail::InitializeDenseLayer(parameters.input_encoder.input_to_hidden, random_engine,
                                 detail::HeNormalStddev(input_encoder::kTurnFeatureCount, config.dense_weight_gain));
    detail::InitializeDenseLayer(parameters.input_encoder.hidden_to_output, random_engine,
                                 detail::HeNormalStddev(input_encoder::kEncoderHiddenSize, config.dense_weight_gain));
    detail::InitializeDenseLayer(parameters.dense_trunk.input_to_hidden0, random_engine,
                                 detail::HeNormalStddev(dense_trunk::kDenseTrunkInputSize, config.dense_weight_gain));
    detail::InitializeDenseLayer(parameters.dense_trunk.hidden0_to_hidden1, random_engine,
                                 detail::HeNormalStddev(dense_trunk::kDenseTrunkHiddenSize0, config.dense_weight_gain));
    detail::InitializeDenseLayer(parameters.dense_trunk.hidden1_to_output, random_engine,
                                 detail::HeNormalStddev(dense_trunk::kDenseTrunkHiddenSize1, config.dense_weight_gain));
}

policy_model::PolicyModelParameters MakeRandomPolicyModelParameters(const std::uint32_t seed,
                                                                    const ParameterInitializationConfig &config) {
    RandomEngine random_engine(seed);
    policy_model::PolicyModelParameters parameters{};
    InitializeRandomPolicyModelParameters(parameters, random_engine, config);
    return parameters;
}

} // namespace neuroevolution::model::initialization
