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

__device__ inline std::uint32_t NextRandomUInt32(DeviceRandomState &state) {
    return curand(&state.philox);
}

__device__ inline bool SampleBernoulli(DeviceRandomState &state, const float probability) {
    if (probability <= 0.0f) {
        return false;
    }
    if (probability >= 1.0f) {
        return true;
    }

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

constexpr float kDefaultDenseWeightGain = 1.0f;
constexpr float kDefaultOutputEmbeddingTailStddev = 0.05f;

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

__device__ inline void InitializeRandomPolicyModelParameters(genome::PolicyModelParameters &parameters,
                                                            DeviceRandomState &random_state,
                                                            const std::size_t worker_index,
                                                            const std::size_t worker_count) {
    InitializeDenseLayerRandom(parameters.input_encoder.input_to_hidden, random_state,
                               HeNormalStddev(model::input_encoder::kTurnFeatureCount, kDefaultDenseWeightGain),
                               worker_index, worker_count);
    InitializeDenseLayerRandom(parameters.input_encoder.hidden_to_output, random_state,
                               HeNormalStddev(model::input_encoder::kEncoderHiddenSize, kDefaultDenseWeightGain),
                               worker_index, worker_count);
    InitializeDenseLayerRandom(parameters.dense_trunk.input_to_hidden0, random_state,
                               HeNormalStddev(model::dense_trunk::kDenseTrunkInputSize, kDefaultDenseWeightGain),
                               worker_index, worker_count);
    InitializeDenseLayerRandom(parameters.dense_trunk.hidden0_to_hidden1, random_state,
                               HeNormalStddev(model::dense_trunk::kDenseTrunkHiddenSize0, kDefaultDenseWeightGain),
                               worker_index, worker_count);
    InitializeDenseLayerRandom(parameters.dense_trunk.hidden1_to_output, random_state,
                               HeNormalStddev(model::dense_trunk::kDenseTrunkHiddenSize1, kDefaultDenseWeightGain),
                               worker_index, worker_count);
}

__device__ inline void InitializeRandomOutputEmbeddingTailRows(
    genome::TrainableActionEmbeddingTail *tail_rows, const std::size_t action_count, DeviceRandomState &random_state,
    const std::size_t worker_index, const std::size_t worker_count) {
    constexpr std::size_t kTailFeatureCount = model::output_embedding::kTrainableFeatureDimension;
    const std::size_t flattened_tail_value_count = action_count * kTailFeatureCount;
    for (std::size_t flattened_index = worker_index; flattened_index < flattened_tail_value_count;
         flattened_index += worker_count) {
        const std::size_t action_index = flattened_index / kTailFeatureCount;
        const std::size_t feature_index = flattened_index % kTailFeatureCount;
        tail_rows[action_index][feature_index] =
            common::ToFloat16(kDefaultOutputEmbeddingTailStddev * SampleStandardNormal(random_state));
    }
}

__device__ inline void InitializeRandomGenome(std::uint8_t *genome_bytes, const std::size_t action_count,
                                              DeviceRandomState &random_state,
                                              const std::size_t worker_index,
                                              const std::size_t worker_count) {
    InitializeRandomPolicyModelParameters(genome::GenomePolicyModelParameters(genome_bytes), random_state, worker_index,
                                          worker_count);
    InitializeRandomOutputEmbeddingTailRows(genome::GenomeTailRows(genome_bytes), action_count, random_state,
                                            worker_index, worker_count);
}

template <std::size_t Size>
__device__ inline void CopyFixedBuffer(const common::FixedBuffer<common::Float16, Size> &source,
                                       common::FixedBuffer<common::Float16, Size> &target) {
    for (std::size_t index = 0; index < Size; ++index) {
        target[index] = source[index];
    }
}

template <std::size_t Size>
__device__ inline void MutateFixedBuffer(common::FixedBuffer<common::Float16, Size> &buffer,
                                         DeviceRandomState &random_state, const MutationConfig &mutation_config) {
    if ((mutation_config.mutation_probability <= 0.0f) || (mutation_config.mutation_sigma <= 0.0f)) {
        return;
    }

    for (std::size_t index = 0; index < Size; ++index) {
        if (SampleBernoulli(random_state, mutation_config.mutation_probability)) {
            const float value =
                common::ToFloat(buffer[index]) + (mutation_config.mutation_sigma * SampleStandardNormal(random_state));
            buffer[index] = common::ToFloat16(value);
        }
    }
}

template <std::size_t InputSize, std::size_t OutputSize, typename DenseLayerParameters>
__device__ inline void CopyDenseLayerNeuron(const DenseLayerParameters &source, DenseLayerParameters &target,
                                            const std::size_t output_index) {
    for (std::size_t input_index = 0; input_index < InputSize; ++input_index) {
        const std::size_t weight_index = (output_index * InputSize) + input_index;
        target.weights[weight_index] = source.weights[weight_index];
    }
    target.biases[output_index] = source.biases[output_index];
}

template <typename DenseLayerParameters>
__device__ inline void CopyDenseLayer(const DenseLayerParameters &source, DenseLayerParameters &target) {
    CopyFixedBuffer(source.weights, target.weights);
    CopyFixedBuffer(source.biases, target.biases);
}

template <std::size_t InputSize, std::size_t OutputSize, typename DenseLayerParameters>
__device__ inline void CopyDenseLayerFromParent(const DenseLayerParameters &first_parent,
                                                const DenseLayerParameters &second_parent,
                                                DenseLayerParameters &child,
                                                const bool source_is_first_parent) {
    if (source_is_first_parent) {
        CopyDenseLayer(first_parent, child);
    } else {
        CopyDenseLayer(second_parent, child);
    }
}

template <std::size_t InputSize, std::size_t OutputSize, typename DenseLayerParameters>
__device__ inline void CopyDenseLayerNeuronFromParent(const DenseLayerParameters &first_parent,
                                                      const DenseLayerParameters &second_parent,
                                                      DenseLayerParameters &child,
                                                      const bool source_is_first_parent,
                                                      const std::size_t output_index) {
    if (source_is_first_parent) {
        CopyDenseLayerNeuron<InputSize, OutputSize>(first_parent, child, output_index);
    } else {
        CopyDenseLayerNeuron<InputSize, OutputSize>(second_parent, child, output_index);
    }
}

template <std::size_t InputSize, std::size_t OutputSize, typename DenseLayerParameters>
__device__ inline void RecombineDenseLayer(const DenseLayerParameters &first_parent,
                                           const DenseLayerParameters &second_parent, DenseLayerParameters &child,
                                           DeviceRandomState &random_state, const BreedingConfig &breeding_config,
                                           const bool layer_source_is_first_parent) {
    CopyDenseLayerFromParent<InputSize, OutputSize>(first_parent, second_parent, child, layer_source_is_first_parent);
    if (SampleBernoulli(random_state, breeding_config.crossover_temperature_level3)) {
        const std::size_t neuron_index = SampleIndex(random_state, OutputSize);
        CopyDenseLayerNeuronFromParent<InputSize, OutputSize>(first_parent, second_parent, child,
                                                              !layer_source_is_first_parent, neuron_index);
    }
}

template <std::size_t InputSize, std::size_t OutputSize>
__device__ inline void MutateDenseLayer(model::input_encoder::DenseLayerParameters<InputSize, OutputSize> &layer,
                                        DeviceRandomState &random_state, const MutationConfig &mutation_config) {
    MutateFixedBuffer(layer.weights, random_state, mutation_config);
    MutateFixedBuffer(layer.biases, random_state, mutation_config);
}

template <std::size_t InputSize, std::size_t OutputSize>
__device__ inline void MutateDenseLayer(model::dense_trunk::DenseLayerParameters<InputSize, OutputSize> &layer,
                                        DeviceRandomState &random_state, const MutationConfig &mutation_config) {
    MutateFixedBuffer(layer.weights, random_state, mutation_config);
    MutateFixedBuffer(layer.biases, random_state, mutation_config);
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

__device__ inline bool MaybeUseAlternateParent(DeviceRandomState &random_state, const float probability,
                                               const bool base_source_is_first_parent) {
    return SampleBernoulli(random_state, probability) ? !base_source_is_first_parent : base_source_is_first_parent;
}

__device__ inline void RecombinePolicyModelParameters(const genome::PolicyModelParameters &first_parent,
                                                      const genome::PolicyModelParameters &second_parent,
                                                      genome::PolicyModelParameters &child,
                                                      DeviceRandomState &random_state,
                                                      const BreedingConfig &breeding_config) {
    const bool input_encoder_source_is_first_parent =
        !SampleBernoulli(random_state, breeding_config.crossover_temperature_level1);
    RecombineDenseLayer<model::input_encoder::kTurnFeatureCount, model::input_encoder::kEncoderHiddenSize>(
        first_parent.input_encoder.input_to_hidden, second_parent.input_encoder.input_to_hidden,
        child.input_encoder.input_to_hidden, random_state, breeding_config,
        MaybeUseAlternateParent(random_state, breeding_config.crossover_temperature_level2,
                                input_encoder_source_is_first_parent));
    RecombineDenseLayer<model::input_encoder::kEncoderHiddenSize, model::input_encoder::kEncoderOutputSize>(
        first_parent.input_encoder.hidden_to_output, second_parent.input_encoder.hidden_to_output,
        child.input_encoder.hidden_to_output, random_state, breeding_config,
        MaybeUseAlternateParent(random_state, breeding_config.crossover_temperature_level2,
                                input_encoder_source_is_first_parent));

    const bool dense_trunk_source_is_first_parent =
        !SampleBernoulli(random_state, breeding_config.crossover_temperature_level1);
    RecombineDenseLayer<model::dense_trunk::kDenseTrunkInputSize, model::dense_trunk::kDenseTrunkHiddenSize0>(
        first_parent.dense_trunk.input_to_hidden0, second_parent.dense_trunk.input_to_hidden0,
        child.dense_trunk.input_to_hidden0, random_state, breeding_config,
        MaybeUseAlternateParent(random_state, breeding_config.crossover_temperature_level2,
                                dense_trunk_source_is_first_parent));
    RecombineDenseLayer<model::dense_trunk::kDenseTrunkHiddenSize0, model::dense_trunk::kDenseTrunkHiddenSize1>(
        first_parent.dense_trunk.hidden0_to_hidden1, second_parent.dense_trunk.hidden0_to_hidden1,
        child.dense_trunk.hidden0_to_hidden1, random_state, breeding_config,
        MaybeUseAlternateParent(random_state, breeding_config.crossover_temperature_level2,
                                dense_trunk_source_is_first_parent));
    RecombineDenseLayer<model::dense_trunk::kDenseTrunkHiddenSize1, model::dense_trunk::kDenseTrunkOutputSize>(
        first_parent.dense_trunk.hidden1_to_output, second_parent.dense_trunk.hidden1_to_output,
        child.dense_trunk.hidden1_to_output, random_state, breeding_config,
        MaybeUseAlternateParent(random_state, breeding_config.crossover_temperature_level2,
                                dense_trunk_source_is_first_parent));
}

__device__ inline void MutatePolicyModelParameters(genome::PolicyModelParameters &parameters,
                                                   DeviceRandomState &random_state,
                                                   const MutationConfig &mutation_config) {
    MutateDenseLayer(parameters.input_encoder.input_to_hidden, random_state, mutation_config);
    MutateDenseLayer(parameters.input_encoder.hidden_to_output, random_state, mutation_config);
    MutateDenseLayer(parameters.dense_trunk.input_to_hidden0, random_state, mutation_config);
    MutateDenseLayer(parameters.dense_trunk.hidden0_to_hidden1, random_state, mutation_config);
    MutateDenseLayer(parameters.dense_trunk.hidden1_to_output, random_state, mutation_config);
}

__device__ inline void CopyOutputTailRowFromParent(const genome::TrainableActionEmbeddingTail &first_parent,
                                                   const genome::TrainableActionEmbeddingTail &second_parent,
                                                   genome::TrainableActionEmbeddingTail &child,
                                                   const bool source_is_first_parent) {
    if (source_is_first_parent) {
        CopyFixedBuffer(first_parent, child);
    } else {
        CopyFixedBuffer(second_parent, child);
    }
}

__device__ inline void SpliceOutputTailRow(const genome::TrainableActionEmbeddingTail &first_parent,
                                           const genome::TrainableActionEmbeddingTail &second_parent,
                                           genome::TrainableActionEmbeddingTail &child,
                                           const bool prefix_source_is_first_parent,
                                           const std::size_t crossover_point) {
    constexpr std::size_t kTailFeatureCount = model::output_embedding::kTrainableFeatureDimension;
    for (std::size_t feature_index = 0; feature_index < kTailFeatureCount; ++feature_index) {
        const bool source_is_first_parent =
            (feature_index < crossover_point) ? prefix_source_is_first_parent : !prefix_source_is_first_parent;
        child[feature_index] = source_is_first_parent ? first_parent[feature_index] : second_parent[feature_index];
    }
}

__device__ inline std::size_t SampleOutputTailCrossoverPoint(DeviceRandomState &random_state) {
    constexpr std::size_t kTailFeatureCount = model::output_embedding::kTrainableFeatureDimension;
    if (kTailFeatureCount <= 1) {
        return 0;
    }

    return 1 + SampleIndex(random_state, kTailFeatureCount - 1);
}

__device__ inline void RecombineOutputTailRows(const genome::TrainableActionEmbeddingTail *first_parent_tail_rows,
                                               const genome::TrainableActionEmbeddingTail *second_parent_tail_rows,
                                               const std::size_t action_count,
                                               genome::TrainableActionEmbeddingTail *child_tail_rows,
                                               DeviceRandomState &random_state,
                                               const BreedingConfig &breeding_config) {
    const bool output_rows_source_is_first_parent =
        !SampleBernoulli(random_state, breeding_config.crossover_temperature_level1);
    for (std::size_t action_index = 0; action_index < action_count; ++action_index) {
        CopyOutputTailRowFromParent(first_parent_tail_rows[action_index], second_parent_tail_rows[action_index],
                                    child_tail_rows[action_index], output_rows_source_is_first_parent);
    }

    bool has_level2_row_swap = false;
    std::size_t level2_row_swap_index = 0;
    if ((action_count > 0) && SampleBernoulli(random_state, breeding_config.crossover_temperature_level2)) {
        has_level2_row_swap = true;
        level2_row_swap_index = SampleIndex(random_state, action_count);
        CopyOutputTailRowFromParent(first_parent_tail_rows[level2_row_swap_index],
                                    second_parent_tail_rows[level2_row_swap_index],
                                    child_tail_rows[level2_row_swap_index], !output_rows_source_is_first_parent);
    }

    if ((action_count > 0) && SampleBernoulli(random_state, breeding_config.crossover_temperature_level3)) {
        const std::size_t splice_index = SampleIndex(random_state, action_count);
        const bool row_source_is_first_parent =
            (has_level2_row_swap && (splice_index == level2_row_swap_index)) ? !output_rows_source_is_first_parent
                                                                             : output_rows_source_is_first_parent;
        SpliceOutputTailRow(first_parent_tail_rows[splice_index], second_parent_tail_rows[splice_index],
                            child_tail_rows[splice_index], row_source_is_first_parent,
                            SampleOutputTailCrossoverPoint(random_state));
    }
}

__device__ inline void MutateOutputTailRows(genome::TrainableActionEmbeddingTail *tail_rows,
                                            const std::size_t action_count, DeviceRandomState &random_state,
                                            const MutationConfig &mutation_config) {
    for (std::size_t action_index = 0; action_index < action_count; ++action_index) {
        MutateOutputTailRow(tail_rows[action_index], random_state, mutation_config);
    }
}

__device__ inline void BreedAndMutateGenome(const std::uint8_t *first_parent_genome_bytes,
                                            const std::uint8_t *second_parent_genome_bytes,
                                            const std::size_t action_count, std::uint8_t *child_genome_bytes,
                                            DeviceRandomState &random_state, const BreedingConfig &breeding_config,
                                            const MutationConfig &mutation_config) {
    RecombinePolicyModelParameters(genome::GenomePolicyModelParameters(first_parent_genome_bytes),
                                   genome::GenomePolicyModelParameters(second_parent_genome_bytes),
                                   genome::GenomePolicyModelParameters(child_genome_bytes), random_state,
                                   breeding_config);
    MutatePolicyModelParameters(genome::GenomePolicyModelParameters(child_genome_bytes), random_state,
                                mutation_config);

    const genome::TrainableActionEmbeddingTail *first_parent_tail_rows =
        genome::GenomeTailRows(first_parent_genome_bytes);
    const genome::TrainableActionEmbeddingTail *second_parent_tail_rows =
        genome::GenomeTailRows(second_parent_genome_bytes);
    genome::TrainableActionEmbeddingTail *child_tail_rows = genome::GenomeTailRows(child_genome_bytes);

    RecombineOutputTailRows(first_parent_tail_rows, second_parent_tail_rows, action_count, child_tail_rows, random_state,
                            breeding_config);
    MutateOutputTailRows(child_tail_rows, action_count, random_state, mutation_config);
}

} // namespace neuroevolution::genetic_algorithm::device_genome_ops
