#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <curand_kernel.h>

#include "common/float16.hpp"
#include "genetic_algorithm/breeding.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "genetic_algorithm/mutation.hpp"
#include "genetic_algorithm/output_tail_ops.hpp"

namespace neuroevolution::genetic_algorithm::device_genome_ops {

struct DeviceRandomState {
    curandStatePhilox4_32_10_t philox{};
};

__device__ inline DeviceRandomState MakeDeviceRandomState(const std::uint32_t seed, const std::uint64_t stream) {
    DeviceRandomState state{};
    curand_init(static_cast<unsigned long long>(seed), static_cast<unsigned long long>(stream), 0ULL, &state.philox);
    return state;
}

__device__ inline float NextUniform01(DeviceRandomState &state) {
    return curand_uniform(&state.philox);
}

__device__ inline bool SampleBernoulli(DeviceRandomState &state, const float probability) {
    return NextUniform01(state) < probability;
}

__device__ inline std::size_t SampleIndex(DeviceRandomState &state, const std::size_t upper_bound_exclusive) {
    if (upper_bound_exclusive <= 1) {
        return 0;
    }

    return static_cast<std::size_t>(curand(&state.philox) % upper_bound_exclusive);
}

__device__ inline float SampleStandardNormal(DeviceRandomState &state) {
    return curand_normal(&state.philox);
}

struct RandomGenomeInitializationConfig {
    float dense_weight_gain = 1.0f;
    float output_embedding_tail_stddev = 0.05f;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidRandomGenomeInitializationConfig(const RandomGenomeInitializationConfig &config) noexcept {
    return (config.dense_weight_gain > 0.0f) && (config.output_embedding_tail_stddev >= 0.0f);
}

inline NEUROEVOLUTION_HOST_DEVICE float HeNormalStddev(const std::size_t fan_in, const float gain) noexcept {
    return gain * sqrtf(2.0f / static_cast<float>(fan_in));
}

template <std::size_t Size>
__device__ inline void FillFloat16BufferWithConstant(common::FixedBuffer<common::Float16, Size> &buffer,
                                                     const float value, const std::size_t worker_index,
                                                     const std::size_t worker_count) {
    for (std::size_t index = worker_index; index < Size; index += worker_count) {
        buffer[index] = common::ToFloat16(value);
    }
}

template <std::size_t Size>
__device__ inline void FillFloat16BufferWithNormal(common::FixedBuffer<common::Float16, Size> &buffer,
                                                   DeviceRandomState &random_state, const float stddev,
                                                   const std::size_t worker_index,
                                                   const std::size_t worker_count) {
    for (std::size_t index = worker_index; index < Size; index += worker_count) {
        buffer[index] = common::ToFloat16(stddev * SampleStandardNormal(random_state));
    }
}

template <std::size_t InputSize, std::size_t OutputSize>
__device__ inline void InitializeDenseLayerRandom(
    model::input_encoder::DenseLayerParameters<InputSize, OutputSize> &layer, DeviceRandomState &random_state,
    const float weight_stddev, const std::size_t worker_index, const std::size_t worker_count) {
    FillFloat16BufferWithNormal(layer.weights, random_state, weight_stddev, worker_index, worker_count);
    FillFloat16BufferWithConstant(layer.biases, 0.0f, worker_index, worker_count);
}

template <std::size_t InputSize, std::size_t OutputSize>
__device__ inline void InitializeDenseLayerRandom(
    model::dense_trunk::DenseLayerParameters<InputSize, OutputSize> &layer, DeviceRandomState &random_state,
    const float weight_stddev, const std::size_t worker_index, const std::size_t worker_count) {
    FillFloat16BufferWithNormal(layer.weights, random_state, weight_stddev, worker_index, worker_count);
    FillFloat16BufferWithConstant(layer.biases, 0.0f, worker_index, worker_count);
}

__device__ inline void InitializeRandomPolicyModelParameters(
    genome::PolicyModelParameters &parameters, DeviceRandomState &random_state,
    const RandomGenomeInitializationConfig &config, const std::size_t worker_index,
    const std::size_t worker_count) {
    InitializeDenseLayerRandom(parameters.input_encoder.input_to_hidden, random_state,
                               HeNormalStddev(model::input_encoder::kTurnFeatureCount, config.dense_weight_gain),
                               worker_index, worker_count);
    InitializeDenseLayerRandom(parameters.input_encoder.hidden_to_output, random_state,
                               HeNormalStddev(model::input_encoder::kEncoderHiddenSize, config.dense_weight_gain),
                               worker_index, worker_count);
    InitializeDenseLayerRandom(parameters.dense_trunk.input_to_hidden0, random_state,
                               HeNormalStddev(model::dense_trunk::kDenseTrunkInputSize, config.dense_weight_gain),
                               worker_index, worker_count);
    InitializeDenseLayerRandom(parameters.dense_trunk.hidden0_to_hidden1, random_state,
                               HeNormalStddev(model::dense_trunk::kDenseTrunkHiddenSize0, config.dense_weight_gain),
                               worker_index, worker_count);
    InitializeDenseLayerRandom(parameters.dense_trunk.hidden1_to_output, random_state,
                               HeNormalStddev(model::dense_trunk::kDenseTrunkHiddenSize1, config.dense_weight_gain),
                               worker_index, worker_count);
}

__device__ inline void InitializeRandomOutputEmbeddingTailRows(
    genome::TrainableActionEmbeddingTail *tail_rows, const std::size_t action_count, DeviceRandomState &random_state,
    const RandomGenomeInitializationConfig &config, const std::size_t worker_index,
    const std::size_t worker_count) {
    constexpr std::size_t kTailFeatureCount = model::output_embedding::kTrainableFeatureDimension;
    const std::size_t flattened_tail_value_count = action_count * kTailFeatureCount;
    for (std::size_t flattened_index = worker_index; flattened_index < flattened_tail_value_count;
         flattened_index += worker_count) {
        const std::size_t action_index = flattened_index / kTailFeatureCount;
        const std::size_t feature_index = flattened_index % kTailFeatureCount;
        tail_rows[action_index][feature_index] =
            common::ToFloat16(config.output_embedding_tail_stddev * SampleStandardNormal(random_state));
    }
}

__device__ inline void InitializeRandomGenome(std::uint8_t *genome_bytes, const std::size_t action_count,
                                              DeviceRandomState &random_state,
                                              const RandomGenomeInitializationConfig &config,
                                              const std::size_t worker_index,
                                              const std::size_t worker_count) {
    InitializeRandomPolicyModelParameters(genome::GenomePolicyModelParameters(genome_bytes), random_state, config,
                                          worker_index, worker_count);
    InitializeRandomOutputEmbeddingTailRows(genome::GenomeTailRows(genome_bytes), action_count, random_state, config,
                                            worker_index, worker_count);
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

template <std::size_t Size>
__device__ inline void CopyFixedBuffer(const common::FixedBuffer<common::Float16, Size> &source,
                                       common::FixedBuffer<common::Float16, Size> &target) {
    for (std::size_t index = 0; index < Size; ++index) {
        target[index] = source[index];
    }
}

__device__ inline float SampleOutputTailRowBlendLambda(DeviceRandomState &random_state) {
    return output_tail_ops::MapArcsineUnitSampleToBlendLambda(NextUniform01(random_state));
}

template <std::size_t Size>
__device__ inline void MutateOutputTailRow(common::FixedBuffer<common::Float16, Size> &row,
                                           DeviceRandomState &random_state, const MutationConfig &mutation_config) {
    for (std::size_t index = 0; index < Size; ++index) {
        if ((mutation_config.mutation_probability > 0.0f) &&
            SampleBernoulli(random_state, mutation_config.mutation_probability) &&
            (mutation_config.mutation_sigma > 0.0f)) {
            const float value = common::ToFloat(row[index]) +
                                (mutation_config.mutation_sigma * SampleStandardNormal(random_state));
            row[index] = common::ToFloat16(value);
        }
    }

    if ((mutation_config.output_tail_row_scale_mutation_probability > 0.0f) &&
        SampleBernoulli(random_state, mutation_config.output_tail_row_scale_mutation_probability)) {
        const float scale = output_tail_ops::RowScaleFactor(SampleBernoulli(random_state, 0.5f));
        for (std::size_t index = 0; index < Size; ++index) {
            row[index] = common::ToFloat16(common::ToFloat(row[index]) * scale);
        }
    }
}

template <std::size_t Size>
__device__ inline void BreedAndMutateOutputTailRow(const common::FixedBuffer<common::Float16, Size> &first_parent,
                                                   const common::FixedBuffer<common::Float16, Size> &second_parent,
                                                   common::FixedBuffer<common::Float16, Size> &child,
                                                   DeviceRandomState &random_state,
                                                   const BreedingConfig &breeding_config,
                                                   const MutationConfig &mutation_config) {
    if ((breeding_config.output_tail_row_arithmetic_recombination_probability > 0.0f) &&
        SampleBernoulli(random_state, breeding_config.output_tail_row_arithmetic_recombination_probability)) {
        const float lambda = SampleOutputTailRowBlendLambda(random_state);
        const float other_lambda = 1.0f - lambda;
        for (std::size_t index = 0; index < Size; ++index) {
            const float blended_value = (lambda * common::ToFloat(first_parent[index])) +
                                        (other_lambda * common::ToFloat(second_parent[index]));
            child[index] = common::ToFloat16(blended_value);
        }
    } else if (SampleBernoulli(random_state, breeding_config.first_parent_probability)) {
        CopyFixedBuffer(first_parent, child);
    } else {
        CopyFixedBuffer(second_parent, child);
    }

    MutateOutputTailRow(child, random_state, mutation_config);
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
        BreedAndMutateOutputTailRow(first_parent_tail_rows[action_index], second_parent_tail_rows[action_index],
                                    child_tail_rows[action_index], random_state, breeding_config, mutation_config);
    }
}

} // namespace neuroevolution::genetic_algorithm::device_genome_ops
