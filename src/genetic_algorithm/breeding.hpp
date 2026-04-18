#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "common/float16.hpp"
#include "genetic_algorithm/genome.hpp"
#include "genetic_algorithm/output_tail_ops.hpp"
#include "genetic_algorithm/selection.hpp"

namespace neuroevolution::genetic_algorithm {

using BreedingRandomEngine = std::mt19937;

struct BreedingConfig {
    float first_parent_probability = 0.5f;
    float output_tail_row_arithmetic_recombination_probability =
        output_tail_ops::kDefaultRowArithmeticRecombinationProbability;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidBreedingConfig(const BreedingConfig &config) noexcept {
    return (config.first_parent_probability >= 0.0f) && (config.first_parent_probability <= 1.0f) &&
           (config.output_tail_row_arithmetic_recombination_probability >= 0.0f) &&
           (config.output_tail_row_arithmetic_recombination_probability <= 1.0f);
}

namespace detail {

template <typename Scalar>
inline Scalar SelectParentScalar(const Scalar &first_parent_value, const Scalar &second_parent_value,
                                 BreedingRandomEngine &random_engine, const BreedingConfig &config) {
    std::bernoulli_distribution distribution(config.first_parent_probability);
    return distribution(random_engine) ? first_parent_value : second_parent_value;
}

template <typename Scalar, std::size_t Size>
inline void BreedFixedBuffer(const common::FixedBuffer<Scalar, Size> &first_parent,
                             const common::FixedBuffer<Scalar, Size> &second_parent,
                             common::FixedBuffer<Scalar, Size> &child, BreedingRandomEngine &random_engine,
                             const BreedingConfig &config) {
    for (std::size_t index = 0; index < Size; ++index) {
        child[index] = SelectParentScalar(first_parent[index], second_parent[index], random_engine, config);
    }
}

inline bool ShouldUseOutputTailRowArithmeticRecombination(BreedingRandomEngine &random_engine,
                                                          const BreedingConfig &config) {
    std::bernoulli_distribution distribution(config.output_tail_row_arithmetic_recombination_probability);
    return distribution(random_engine);
}

inline float SampleOutputTailRowBlendLambda(BreedingRandomEngine &random_engine) {
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    return output_tail_ops::MapArcsineUnitSampleToBlendLambda(distribution(random_engine));
}

template <std::size_t Size>
inline void BreedOutputTailRow(const common::FixedBuffer<common::Float16, Size> &first_parent,
                               const common::FixedBuffer<common::Float16, Size> &second_parent,
                               common::FixedBuffer<common::Float16, Size> &child,
                               BreedingRandomEngine &random_engine, const BreedingConfig &config) {
    if (!ShouldUseOutputTailRowArithmeticRecombination(random_engine, config)) {
        std::bernoulli_distribution distribution(config.first_parent_probability);
        child = distribution(random_engine) ? first_parent : second_parent;
        return;
    }

    const float lambda = SampleOutputTailRowBlendLambda(random_engine);
    const float other_lambda = 1.0f - lambda;
    for (std::size_t index = 0; index < Size; ++index) {
        const float blended_value =
            (lambda * common::ToFloat(first_parent[index])) + (other_lambda * common::ToFloat(second_parent[index]));
        child[index] = common::ToFloat16(blended_value);
    }
}

template <std::size_t InputSize, std::size_t OutputSize>
inline void
BreedDenseLayerParameters(const model::input_encoder::DenseLayerParameters<InputSize, OutputSize> &first_parent,
                          const model::input_encoder::DenseLayerParameters<InputSize, OutputSize> &second_parent,
                          model::input_encoder::DenseLayerParameters<InputSize, OutputSize> &child,
                          BreedingRandomEngine &random_engine, const BreedingConfig &config) {
    BreedFixedBuffer(first_parent.weights, second_parent.weights, child.weights, random_engine, config);
    BreedFixedBuffer(first_parent.biases, second_parent.biases, child.biases, random_engine, config);
}

template <std::size_t InputSize, std::size_t OutputSize>
inline void
BreedDenseLayerParameters(const model::dense_trunk::DenseLayerParameters<InputSize, OutputSize> &first_parent,
                          const model::dense_trunk::DenseLayerParameters<InputSize, OutputSize> &second_parent,
                          model::dense_trunk::DenseLayerParameters<InputSize, OutputSize> &child,
                          BreedingRandomEngine &random_engine, const BreedingConfig &config) {
    BreedFixedBuffer(first_parent.weights, second_parent.weights, child.weights, random_engine, config);
    BreedFixedBuffer(first_parent.biases, second_parent.biases, child.biases, random_engine, config);
}

} // namespace detail

inline void BreedPolicyModelParameters(const model::policy_model::PolicyModelParameters &first_parent,
                                       const model::policy_model::PolicyModelParameters &second_parent,
                                       model::policy_model::PolicyModelParameters &child,
                                       BreedingRandomEngine &random_engine, const BreedingConfig &config = {}) {
    detail::BreedDenseLayerParameters(first_parent.input_encoder.input_to_hidden,
                                      second_parent.input_encoder.input_to_hidden, child.input_encoder.input_to_hidden,
                                      random_engine, config);
    detail::BreedDenseLayerParameters(first_parent.input_encoder.hidden_to_output,
                                      second_parent.input_encoder.hidden_to_output,
                                      child.input_encoder.hidden_to_output, random_engine, config);
    detail::BreedDenseLayerParameters(first_parent.dense_trunk.input_to_hidden0,
                                      second_parent.dense_trunk.input_to_hidden0, child.dense_trunk.input_to_hidden0,
                                      random_engine, config);
    detail::BreedDenseLayerParameters(first_parent.dense_trunk.hidden0_to_hidden1,
                                      second_parent.dense_trunk.hidden0_to_hidden1,
                                      child.dense_trunk.hidden0_to_hidden1, random_engine, config);
    detail::BreedDenseLayerParameters(first_parent.dense_trunk.hidden1_to_output,
                                      second_parent.dense_trunk.hidden1_to_output, child.dense_trunk.hidden1_to_output,
                                      random_engine, config);
}

template <std::size_t ActionCapacity>
inline void BreedOutputEmbeddingGenome(const OutputEmbeddingGenome<ActionCapacity> &first_parent,
                                       const OutputEmbeddingGenome<ActionCapacity> &second_parent,
                                       OutputEmbeddingGenome<ActionCapacity> &child,
                                       BreedingRandomEngine &random_engine, const BreedingConfig &config = {}) {
    const std::size_t shared_active_count =
        (ActiveOutputEmbeddingCount(first_parent) < ActiveOutputEmbeddingCount(second_parent))
            ? ActiveOutputEmbeddingCount(first_parent)
            : ActiveOutputEmbeddingCount(second_parent);

    child.active_count = shared_active_count;

    for (std::size_t action_index = 0; action_index < shared_active_count; ++action_index) {
        detail::BreedOutputTailRow(first_parent.trainable_tails[action_index],
                                   second_parent.trainable_tails[action_index], child.trainable_tails[action_index],
                                   random_engine, config);
    }

    for (std::size_t action_index = shared_active_count; action_index < ActionCapacity; ++action_index) {
        child.trainable_tails[action_index] = {};
    }
}

template <std::size_t ActionCapacity>
inline bool TryBreedChildGenome(const ModelGenome<ActionCapacity> &first_parent,
                                const ModelGenome<ActionCapacity> &second_parent, ModelGenome<ActionCapacity> &child,
                                BreedingRandomEngine &random_engine, const BreedingConfig &config = {}) {
    if (!IsValidBreedingConfig(config) || !IsValidModelGenome(first_parent) || !IsValidModelGenome(second_parent)) {
        return false;
    }

    BreedPolicyModelParameters(first_parent.policy_model, second_parent.policy_model, child.policy_model, random_engine,
                               config);
    BreedOutputEmbeddingGenome(first_parent.output_embedding, second_parent.output_embedding, child.output_embedding,
                               random_engine, config);
    return true;
}

template <std::size_t ActionCapacity, std::size_t PopulationSize>
inline bool TryBreedChildGenomeFromPopulation(const Population<ModelGenome<ActionCapacity>, PopulationSize> &population,
                                              const ParentPair &parent_pair, ModelGenome<ActionCapacity> &child,
                                              BreedingRandomEngine &random_engine, const BreedingConfig &config = {}) {
    if ((parent_pair.first_parent_index >= PopulationSize) || (parent_pair.second_parent_index >= PopulationSize)) {
        return false;
    }

    return TryBreedChildGenome(population.individuals[parent_pair.first_parent_index].genome,
                               population.individuals[parent_pair.second_parent_index].genome, child, random_engine,
                               config);
}

template <std::size_t ActionCapacity>
inline ModelGenome<ActionCapacity>
BreedChildGenome(const ModelGenome<ActionCapacity> &first_parent, const ModelGenome<ActionCapacity> &second_parent,
                 BreedingRandomEngine &random_engine, const BreedingConfig &config = {}) {
    ModelGenome<ActionCapacity> child{};
    const bool breed_ok = TryBreedChildGenome(first_parent, second_parent, child, random_engine, config);
    if (!breed_ok) {
        throw std::invalid_argument("Breeding requires a valid recombination config.");
    }

    return child;
}

} // namespace neuroevolution::genetic_algorithm
