#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>

#include "common/cuda_compat.hpp"
#include "common/float16.hpp"
#include "genetic_algorithm/genome.hpp"

namespace neuroevolution::genetic_algorithm {

using MutationRandomEngine = std::mt19937;

struct MutationConfig {
    float mutation_probability = 0.02f;
    float mutation_sigma = 0.05f;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidMutationConfig(const MutationConfig &config) noexcept {
    return (config.mutation_probability >= 0.0f) && (config.mutation_probability <= 1.0f) &&
           (config.mutation_sigma >= 0.0f);
}

namespace detail {

inline bool ShouldMutateParameter(MutationRandomEngine &random_engine, const MutationConfig &config) {
    std::bernoulli_distribution distribution(config.mutation_probability);
    return distribution(random_engine);
}

inline void MutateParameterScalar(common::Float16 &value, MutationRandomEngine &random_engine,
                                  const MutationConfig &config) {
    if (!ShouldMutateParameter(random_engine, config)) {
        return;
    }

    if (config.mutation_sigma == 0.0f) {
        return;
    }

    std::normal_distribution<float> distribution(0.0f, config.mutation_sigma);
    value = common::ToFloat16(common::ToFloat(value) + distribution(random_engine));
}

template <std::size_t Size>
inline void MutateFixedBuffer(common::FixedBuffer<common::Float16, Size> &buffer, MutationRandomEngine &random_engine,
                              const MutationConfig &config) {
    for (std::size_t index = 0; index < Size; ++index) {
        MutateParameterScalar(buffer[index], random_engine, config);
    }
}

template <std::size_t InputSize, std::size_t OutputSize>
inline void MutateDenseLayerParameters(model::input_encoder::DenseLayerParameters<InputSize, OutputSize> &layer,
                                       MutationRandomEngine &random_engine, const MutationConfig &config) {
    MutateFixedBuffer(layer.weights, random_engine, config);
    MutateFixedBuffer(layer.biases, random_engine, config);
}

template <std::size_t InputSize, std::size_t OutputSize>
inline void MutateDenseLayerParameters(model::dense_trunk::DenseLayerParameters<InputSize, OutputSize> &layer,
                                       MutationRandomEngine &random_engine, const MutationConfig &config) {
    MutateFixedBuffer(layer.weights, random_engine, config);
    MutateFixedBuffer(layer.biases, random_engine, config);
}

} // namespace detail

inline bool TryMutatePolicyModelParameters(model::policy_model::PolicyModelParameters &parameters,
                                           MutationRandomEngine &random_engine, const MutationConfig &config = {}) {
    if (!IsValidMutationConfig(config)) {
        return false;
    }

    detail::MutateDenseLayerParameters(parameters.input_encoder.input_to_hidden, random_engine, config);
    detail::MutateDenseLayerParameters(parameters.input_encoder.hidden_to_output, random_engine, config);
    detail::MutateDenseLayerParameters(parameters.dense_trunk.input_to_hidden0, random_engine, config);
    detail::MutateDenseLayerParameters(parameters.dense_trunk.hidden0_to_hidden1, random_engine, config);
    detail::MutateDenseLayerParameters(parameters.dense_trunk.hidden1_to_output, random_engine, config);
    return true;
}

template <std::size_t ActionCapacity>
inline bool TryMutateOutputEmbeddingGenome(OutputEmbeddingGenome<ActionCapacity> &genome,
                                           MutationRandomEngine &random_engine, const MutationConfig &config = {}) {
    if (!IsValidMutationConfig(config) || !IsValidOutputEmbeddingGenome(genome)) {
        return false;
    }

    for (std::size_t action_index = 0; action_index < genome.active_count; ++action_index) {
        detail::MutateFixedBuffer(genome.trainable_tails[action_index], random_engine, config);
    }

    return true;
}

template <std::size_t ActionCapacity>
inline bool TryMutateGenome(ModelGenome<ActionCapacity> &genome, MutationRandomEngine &random_engine,
                            const MutationConfig &config = {}) {
    if (!IsValidMutationConfig(config) || !IsValidModelGenome(genome)) {
        return false;
    }

    return TryMutatePolicyModelParameters(genome.policy_model, random_engine, config) &&
           TryMutateOutputEmbeddingGenome(genome.output_embedding, random_engine, config);
}

template <std::size_t ActionCapacity>
inline void MutateGenome(ModelGenome<ActionCapacity> &genome, MutationRandomEngine &random_engine,
                         const MutationConfig &config = {}) {
    if (!TryMutateGenome(genome, random_engine, config)) {
        throw std::invalid_argument("Mutation requires probability in [0,1] and non-negative sigma.");
    }
}

} // namespace neuroevolution::genetic_algorithm
