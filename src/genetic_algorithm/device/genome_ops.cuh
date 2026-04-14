#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "common/float16.hpp"
#include "genetic_algorithm/breeding.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "genetic_algorithm/mutation.hpp"

namespace neuroevolution::genetic_algorithm::device_genome_ops {

struct DeviceRandomState {
    std::uint64_t state = 0;
};

__device__ inline std::uint64_t NextUInt64(DeviceRandomState &state) {
    std::uint64_t x = state.state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.state = x;
    return x * 2685821657736338717ULL;
}

__device__ inline DeviceRandomState MakeDeviceRandomState(const std::uint32_t seed, const std::uint32_t stream) {
    DeviceRandomState state{};
    state.state =
        (static_cast<std::uint64_t>(seed) << 32) ^ (static_cast<std::uint64_t>(stream) + 0x9E3779B97F4A7C15ULL);
    if (state.state == 0) {
        state.state = 0xA5A5A5A5ULL;
    }

    (void)NextUInt64(state);
    return state;
}

__device__ inline float NextUniform01(DeviceRandomState &state) {
    const std::uint32_t bits = static_cast<std::uint32_t>(NextUInt64(state) >> 32);
    return (static_cast<float>(bits) + 1.0f) / 4294967297.0f;
}

__device__ inline bool SampleBernoulli(DeviceRandomState &state, const float probability) {
    return NextUniform01(state) < probability;
}

__device__ inline std::size_t SampleIndex(DeviceRandomState &state, const std::size_t upper_bound_exclusive) {
    if (upper_bound_exclusive <= 1) {
        return 0;
    }

    return static_cast<std::size_t>(NextUInt64(state) % upper_bound_exclusive);
}

__device__ inline float SampleStandardNormal(DeviceRandomState &state) {
    constexpr float kTwoPi = 6.28318530717958647692f;
    const float u1 = NextUniform01(state);
    const float u2 = NextUniform01(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(kTwoPi * u2);
}

template <std::size_t Size>
__device__ inline void BreedAndMutateFixedBuffer(const common::FixedBuffer<common::Float16, Size> &first_parent,
                                                 const common::FixedBuffer<common::Float16, Size> &second_parent,
                                                 common::FixedBuffer<common::Float16, Size> &child,
                                                 DeviceRandomState &random_state, const BreedingConfig &breeding_config,
                                                 const MutationConfig &mutation_config) {
    for (std::size_t index = 0; index < Size; ++index) {
        float value = SampleBernoulli(random_state, breeding_config.first_parent_probability)
                          ? common::ToFloat(first_parent[index])
                          : common::ToFloat(second_parent[index]);

        if ((mutation_config.mutation_probability > 0.0f) &&
            SampleBernoulli(random_state, mutation_config.mutation_probability) &&
            (mutation_config.mutation_sigma > 0.0f)) {
            value += mutation_config.mutation_sigma * SampleStandardNormal(random_state);
        }

        child[index] = common::ToFloat16(value);
    }
}

template <std::size_t InputSize, std::size_t OutputSize>
__device__ inline void
BreedAndMutateDenseLayer(const model::input_encoder::DenseLayerParameters<InputSize, OutputSize> &first_parent,
                         const model::input_encoder::DenseLayerParameters<InputSize, OutputSize> &second_parent,
                         model::input_encoder::DenseLayerParameters<InputSize, OutputSize> &child,
                         DeviceRandomState &random_state, const BreedingConfig &breeding_config,
                         const MutationConfig &mutation_config) {
    BreedAndMutateFixedBuffer(first_parent.weights, second_parent.weights, child.weights, random_state, breeding_config,
                              mutation_config);
    BreedAndMutateFixedBuffer(first_parent.biases, second_parent.biases, child.biases, random_state, breeding_config,
                              mutation_config);
}

template <std::size_t InputSize, std::size_t OutputSize>
__device__ inline void
BreedAndMutateDenseLayer(const model::dense_trunk::DenseLayerParameters<InputSize, OutputSize> &first_parent,
                         const model::dense_trunk::DenseLayerParameters<InputSize, OutputSize> &second_parent,
                         model::dense_trunk::DenseLayerParameters<InputSize, OutputSize> &child,
                         DeviceRandomState &random_state, const BreedingConfig &breeding_config,
                         const MutationConfig &mutation_config) {
    BreedAndMutateFixedBuffer(first_parent.weights, second_parent.weights, child.weights, random_state, breeding_config,
                              mutation_config);
    BreedAndMutateFixedBuffer(first_parent.biases, second_parent.biases, child.biases, random_state, breeding_config,
                              mutation_config);
}

__device__ inline void CopyGenome(const std::uint8_t *source_genome_bytes, const std::size_t source_action_count,
                                  std::uint8_t *target_genome_bytes, const std::size_t target_action_count) {
    genome::GenomePolicyModelParameters(target_genome_bytes) = genome::GenomePolicyModelParameters(source_genome_bytes);

    const genome::TrainableActionEmbeddingTail *source_tail_rows = genome::GenomeTailRows(source_genome_bytes);
    genome::TrainableActionEmbeddingTail *target_tail_rows = genome::GenomeTailRows(target_genome_bytes);

    const std::size_t copied_action_count =
        (source_action_count < target_action_count) ? source_action_count : target_action_count;
    for (std::size_t action_index = 0; action_index < copied_action_count; ++action_index) {
        target_tail_rows[action_index] = source_tail_rows[action_index];
    }
}

__device__ inline void BreedAndMutateGenome(const std::uint8_t *first_parent_genome_bytes,
                                            const std::uint8_t *second_parent_genome_bytes,
                                            const std::size_t action_count, std::uint8_t *child_genome_bytes,
                                            DeviceRandomState &random_state, const BreedingConfig &breeding_config,
                                            const MutationConfig &mutation_config) {
    BreedAndMutateDenseLayer(
        genome::GenomePolicyModelParameters(first_parent_genome_bytes).input_encoder.input_to_hidden,
        genome::GenomePolicyModelParameters(second_parent_genome_bytes).input_encoder.input_to_hidden,
        genome::GenomePolicyModelParameters(child_genome_bytes).input_encoder.input_to_hidden, random_state,
        breeding_config, mutation_config);
    BreedAndMutateDenseLayer(
        genome::GenomePolicyModelParameters(first_parent_genome_bytes).input_encoder.hidden_to_output,
        genome::GenomePolicyModelParameters(second_parent_genome_bytes).input_encoder.hidden_to_output,
        genome::GenomePolicyModelParameters(child_genome_bytes).input_encoder.hidden_to_output, random_state,
        breeding_config, mutation_config);
    BreedAndMutateDenseLayer(
        genome::GenomePolicyModelParameters(first_parent_genome_bytes).dense_trunk.input_to_hidden0,
        genome::GenomePolicyModelParameters(second_parent_genome_bytes).dense_trunk.input_to_hidden0,
        genome::GenomePolicyModelParameters(child_genome_bytes).dense_trunk.input_to_hidden0, random_state,
        breeding_config, mutation_config);
    BreedAndMutateDenseLayer(
        genome::GenomePolicyModelParameters(first_parent_genome_bytes).dense_trunk.hidden0_to_hidden1,
        genome::GenomePolicyModelParameters(second_parent_genome_bytes).dense_trunk.hidden0_to_hidden1,
        genome::GenomePolicyModelParameters(child_genome_bytes).dense_trunk.hidden0_to_hidden1, random_state,
        breeding_config, mutation_config);
    BreedAndMutateDenseLayer(
        genome::GenomePolicyModelParameters(first_parent_genome_bytes).dense_trunk.hidden1_to_output,
        genome::GenomePolicyModelParameters(second_parent_genome_bytes).dense_trunk.hidden1_to_output,
        genome::GenomePolicyModelParameters(child_genome_bytes).dense_trunk.hidden1_to_output, random_state,
        breeding_config, mutation_config);

    const genome::TrainableActionEmbeddingTail *first_parent_tail_rows =
        genome::GenomeTailRows(first_parent_genome_bytes);
    const genome::TrainableActionEmbeddingTail *second_parent_tail_rows =
        genome::GenomeTailRows(second_parent_genome_bytes);
    genome::TrainableActionEmbeddingTail *child_tail_rows = genome::GenomeTailRows(child_genome_bytes);

    for (std::size_t action_index = 0; action_index < action_count; ++action_index) {
        BreedAndMutateFixedBuffer(first_parent_tail_rows[action_index], second_parent_tail_rows[action_index],
                                  child_tail_rows[action_index], random_state, breeding_config, mutation_config);
    }
}

} // namespace neuroevolution::genetic_algorithm::device_genome_ops
