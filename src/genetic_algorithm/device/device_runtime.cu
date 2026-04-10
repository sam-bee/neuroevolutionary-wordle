#include "genetic_algorithm/device/device_runtime.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "common/float16.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "model/policy_model/policy_model.hpp"
#include "training_folder/training_data.hpp"
#include "wordle/wordle_grid.hpp"

namespace neuroevolution::genetic_algorithm::device {

namespace {

using neuroevolution::common::FixedBuffer;
using neuroevolution::model::output_embedding::ActionEmbedding;
using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::model::output_embedding::TrySelectBestAction;
using neuroevolution::model::policy_model::PolicyVector;
using neuroevolution::model::policy_model::TryForwardPolicyModel;
using neuroevolution::training_folder::DeviceTrainingDataShard;
using neuroevolution::training_folder::IsValidTrainingDataShard;
using neuroevolution::training_folder::TrainingDataShard;
using neuroevolution::wordle::MakeWordleGrid;
using neuroevolution::wordle::TryAppendGuess;
using neuroevolution::wordle::Word;
using neuroevolution::wordle::WordleGrid;

NEUROEVOLUTION_HOST_DEVICE constexpr int DeviceStatusValue(const DeviceRuntimeStatusCode status_code) {
    return static_cast<int>(status_code);
}

struct DeviceRandomState {
    std::uint64_t state = 0;
};

__device__ std::uint64_t NextUInt64(DeviceRandomState &state) {
    std::uint64_t x = state.state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.state = x;
    return x * 2685821657736338717ULL;
}

__device__ DeviceRandomState MakeDeviceRandomState(const std::uint32_t seed, const std::uint32_t stream) {
    DeviceRandomState state{};
    state.state =
        (static_cast<std::uint64_t>(seed) << 32) ^ (static_cast<std::uint64_t>(stream) + 0x9E3779B97F4A7C15ULL);
    if (state.state == 0) {
        state.state = 0xA5A5A5A5ULL;
    }

    (void)NextUInt64(state);
    return state;
}

__device__ float NextUniform01(DeviceRandomState &state) {
    const std::uint32_t bits = static_cast<std::uint32_t>(NextUInt64(state) >> 32);
    return (static_cast<float>(bits) + 1.0f) / 4294967297.0f;
}

__device__ bool SampleBernoulli(DeviceRandomState &state, const float probability) {
    return NextUniform01(state) < probability;
}

__device__ std::size_t SampleIndex(DeviceRandomState &state, const std::size_t upper_bound_exclusive) {
    if (upper_bound_exclusive <= 1) {
        return 0;
    }

    return static_cast<std::size_t>(NextUInt64(state) % upper_bound_exclusive);
}

__device__ float SampleStandardNormal(DeviceRandomState &state) {
    constexpr float kTwoPi = 6.28318530717958647692f;
    const float u1 = NextUniform01(state);
    const float u2 = NextUniform01(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(kTwoPi * u2);
}

__device__ void SetFailureStatus(int *status, const DeviceRuntimeStatusCode status_code) {
    atomicCAS(status, DeviceStatusValue(DeviceRuntimeStatusCode::kOk), DeviceStatusValue(status_code));
}

template <std::size_t Size> __device__ void ResetFixedBuffer(FixedBuffer<bool, Size> &flags) {
    for (std::size_t index = 0; index < Size; ++index) {
        flags[index] = false;
    }
}

template <std::size_t ActionCount>
__device__ void BuildActionEmbeddingsFromTrainingShard(const DeviceGenome &genome,
                                                       const TrainingDataShard &training_shard,
                                                       FixedBuffer<ActionEmbedding, ActionCount> &action_embeddings) {
    for (std::size_t action_index = 0; action_index < training_shard.entry_count; ++action_index) {
        action_embeddings[action_index].word = training_shard.entries[action_index].word;
        for (std::size_t feature_index = 0;
             feature_index < neuroevolution::model::output_embedding::kTrainableFeatureDimension; ++feature_index) {
            action_embeddings[action_index].trainable_tail[feature_index] =
                genome.output_embedding.trainable_tails[action_index][feature_index];
        }
    }
}

__device__ DeviceRuntimeStatusCode TryPlayWordleToCompletion(const DeviceGenome &genome,
                                                             const ActionEmbedding *action_embeddings,
                                                             const std::size_t action_count, const Word &solution,
                                                             float &episode_score_out) {
    WordleGrid grid = MakeWordleGrid(solution);

    while (!grid.IsFinished()) {
        PolicyVector policy_vector{};
        if (!TryForwardPolicyModel(genome.policy_model, grid, policy_vector)) {
            return DeviceRuntimeStatusCode::kPolicyForwardFailed;
        }

        SelectedAction selected_action{};
        if (!TrySelectBestAction(policy_vector, action_embeddings, action_count, selected_action)) {
            return DeviceRuntimeStatusCode::kActionSelectionFailed;
        }

        if (!TryAppendGuess(grid, selected_action.word)) {
            return DeviceRuntimeStatusCode::kGuessAppendFailed;
        }
    }

    episode_score_out = grid.IsWon() ? 1.0f : 0.0f;
    return DeviceRuntimeStatusCode::kOk;
}

__device__ DeviceRuntimeStatusCode TryEvaluateIndividualFitness(const DeviceGenome &genome, float &fitness_out) {
    const TrainingDataShard &training_shard = DeviceTrainingDataShard();
    if (!IsValidTrainingDataShard(training_shard) || (training_shard.entry_count == 0) ||
        (training_shard.entry_count > kDeviceActionCount)) {
        return DeviceRuntimeStatusCode::kInvalidTrainingShard;
    }

    FixedBuffer<ActionEmbedding, kDeviceActionCount> action_embeddings{};
    BuildActionEmbeddingsFromTrainingShard<kDeviceActionCount>(genome, training_shard, action_embeddings);

    float score_sum = 0.0f;

    for (std::size_t entry_index = 0; entry_index < training_shard.entry_count; ++entry_index) {
        const Word solution = training_shard.entries[entry_index].word;
        float episode_score = 0.0f;
        const DeviceRuntimeStatusCode episode_status = TryPlayWordleToCompletion(
            genome, action_embeddings.values, training_shard.entry_count, solution, episode_score);
        if (episode_status != DeviceRuntimeStatusCode::kOk) {
            return episode_status;
        }

        score_sum += episode_score;
    }

    fitness_out = score_sum;
    return DeviceRuntimeStatusCode::kOk;
}

template <std::size_t PopulationSize>
__device__ bool TryFindEliteIndexByRank(const DevicePopulation &population, const std::size_t elite_rank,
                                        std::size_t &elite_index) {
    FixedBuffer<bool, PopulationSize> selected_flags{};
    ResetFixedBuffer(selected_flags);

    for (std::size_t rank = 0; rank <= elite_rank; ++rank) {
        bool found_candidate = false;
        std::size_t best_index = 0;
        float best_fitness = 0.0f;

        for (std::size_t individual_index = 0; individual_index < PopulationSize; ++individual_index) {
            const auto &individual = population.individuals[individual_index];
            if (!individual.has_fitness || selected_flags[individual_index]) {
                continue;
            }

            if (!found_candidate || (individual.fitness > best_fitness)) {
                found_candidate = true;
                best_index = individual_index;
                best_fitness = individual.fitness;
            }
        }

        if (!found_candidate) {
            return false;
        }

        selected_flags[best_index] = true;
        elite_index = best_index;
    }

    return true;
}

template <std::size_t PopulationSize>
__device__ bool TrySelectParentIndexDevice(const DevicePopulation &population, DeviceRandomState &random_state,
                                           const ParentSelectionConfig &config, std::size_t &selected_parent_index,
                                           const std::size_t excluded_index) {
    FixedBuffer<std::size_t, PopulationSize> selectable_indices{};
    std::size_t selectable_count = 0;

    for (std::size_t individual_index = 0; individual_index < PopulationSize; ++individual_index) {
        const auto &individual = population.individuals[individual_index];
        if (!individual.has_fitness || (individual_index == excluded_index)) {
            continue;
        }

        selectable_indices[selectable_count] = individual_index;
        ++selectable_count;
    }

    if (selectable_count == 0) {
        return false;
    }

    const std::size_t tournament_size =
        (config.tournament_size < selectable_count) ? config.tournament_size : selectable_count;

    for (std::size_t prefix_index = 0; prefix_index < tournament_size; ++prefix_index) {
        const std::size_t offset = SampleIndex(random_state, selectable_count - prefix_index);
        const std::size_t swap_index = prefix_index + offset;

        const std::size_t temporary = selectable_indices[prefix_index];
        selectable_indices[prefix_index] = selectable_indices[swap_index];
        selectable_indices[swap_index] = temporary;
    }

    selected_parent_index = selectable_indices[0];
    float best_fitness = population.individuals[selected_parent_index].fitness;

    for (std::size_t candidate_index = 1; candidate_index < tournament_size; ++candidate_index) {
        const std::size_t individual_index = selectable_indices[candidate_index];
        const float candidate_fitness = population.individuals[individual_index].fitness;
        if (candidate_fitness > best_fitness) {
            selected_parent_index = individual_index;
            best_fitness = candidate_fitness;
        }
    }

    return true;
}

template <std::size_t PopulationSize>
__device__ bool TrySelectParentPairDevice(const DevicePopulation &population, DeviceRandomState &random_state,
                                          const ParentSelectionConfig &config, ParentPair &parent_pair) {
    if (!TrySelectParentIndexDevice<PopulationSize>(population, random_state, config, parent_pair.first_parent_index,
                                                    kNoIndividualIndex)) {
        return false;
    }

    const std::size_t excluded_index =
        config.allow_self_parenting ? kNoIndividualIndex : parent_pair.first_parent_index;
    return TrySelectParentIndexDevice<PopulationSize>(population, random_state, config, parent_pair.second_parent_index,
                                                      excluded_index);
}

template <std::size_t Size>
__device__ void BreedAndMutateFixedBuffer(const FixedBuffer<common::Float16, Size> &first_parent,
                                          const FixedBuffer<common::Float16, Size> &second_parent,
                                          FixedBuffer<common::Float16, Size> &child, DeviceRandomState &random_state,
                                          const BreedingConfig &breeding_config,
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
__device__ void
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
__device__ void
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

__device__ void BreedAndMutateGenome(const DeviceGenome &first_parent, const DeviceGenome &second_parent,
                                     DeviceGenome &child, DeviceRandomState &random_state,
                                     const BreedingConfig &breeding_config, const MutationConfig &mutation_config) {
    BreedAndMutateDenseLayer(first_parent.policy_model.input_encoder.input_to_hidden,
                             second_parent.policy_model.input_encoder.input_to_hidden,
                             child.policy_model.input_encoder.input_to_hidden, random_state, breeding_config,
                             mutation_config);
    BreedAndMutateDenseLayer(first_parent.policy_model.input_encoder.hidden_to_output,
                             second_parent.policy_model.input_encoder.hidden_to_output,
                             child.policy_model.input_encoder.hidden_to_output, random_state, breeding_config,
                             mutation_config);
    BreedAndMutateDenseLayer(
        first_parent.policy_model.dense_trunk.input_to_hidden0, second_parent.policy_model.dense_trunk.input_to_hidden0,
        child.policy_model.dense_trunk.input_to_hidden0, random_state, breeding_config, mutation_config);
    BreedAndMutateDenseLayer(first_parent.policy_model.dense_trunk.hidden0_to_hidden1,
                             second_parent.policy_model.dense_trunk.hidden0_to_hidden1,
                             child.policy_model.dense_trunk.hidden0_to_hidden1, random_state, breeding_config,
                             mutation_config);
    BreedAndMutateDenseLayer(first_parent.policy_model.dense_trunk.hidden1_to_output,
                             second_parent.policy_model.dense_trunk.hidden1_to_output,
                             child.policy_model.dense_trunk.hidden1_to_output, random_state, breeding_config,
                             mutation_config);

    for (std::size_t action_index = 0; action_index < kDeviceActionCount; ++action_index) {
        BreedAndMutateFixedBuffer(first_parent.output_embedding.trainable_tails[action_index],
                                  second_parent.output_embedding.trainable_tails[action_index],
                                  child.output_embedding.trainable_tails[action_index], random_state, breeding_config,
                                  mutation_config);
    }
}

__device__ void MarkIndividualUnevaluated(Individual<DeviceGenome> &individual) {
    individual.fitness = 0.0f;
    individual.evaluation_count = 0;
    individual.has_fitness = false;
}

__device__ void WriteGenomeToUnevaluatedIndividual(const DeviceGenome &genome, Individual<DeviceGenome> &individual) {
    individual.genome = genome;
    MarkIndividualUnevaluated(individual);
}

__global__ void EvaluatePopulationFitnessKernel(DevicePopulation *population, int *status) {
    const std::size_t individual_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (individual_index >= kDevicePopulationSize) {
        return;
    }

    const TrainingDataShard &training_shard = DeviceTrainingDataShard();
    if (!IsValidTrainingDataShard(training_shard)) {
        SetFailureStatus(status, DeviceRuntimeStatusCode::kInvalidTrainingShard);
        return;
    }

    float fitness = 0.0f;
    const DeviceRuntimeStatusCode evaluation_status =
        TryEvaluateIndividualFitness(population->individuals[individual_index].genome, fitness);
    if (evaluation_status != DeviceRuntimeStatusCode::kOk) {
        SetFailureStatus(status, evaluation_status);
        return;
    }

    population->individuals[individual_index].fitness = fitness;
    ++population->individuals[individual_index].evaluation_count;
    population->individuals[individual_index].has_fitness = true;
}

__global__ void SummarizePopulationKernel(const DevicePopulation *population, PopulationFitnessSummary *summary,
                                          int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    bool found_best = false;
    float best_fitness = 0.0f;
    float fitness_sum = 0.0f;
    std::size_t best_index = 0;

    for (std::size_t individual_index = 0; individual_index < kDevicePopulationSize; ++individual_index) {
        const auto &individual = population->individuals[individual_index];
        if (!individual.has_fitness) {
            SetFailureStatus(status, DeviceRuntimeStatusCode::kPopulationNotEvaluated);
            return;
        }

        fitness_sum += individual.fitness;

        if (!found_best || (individual.fitness > best_fitness)) {
            found_best = true;
            best_fitness = individual.fitness;
            best_index = individual_index;
        }
    }

    summary->best_fitness = best_fitness;
    summary->average_fitness = fitness_sum / static_cast<float>(kDevicePopulationSize);
    summary->best_index = best_index;
    summary->generation_index = population->generation_index;
}

__global__ void AssembleNextGenerationKernel(const DevicePopulation *current_population,
                                             DevicePopulation *next_population, const std::uint32_t generation_seed,
                                             const GenerationAssemblyConfig config, int *status) {
    const std::size_t slot_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (slot_index >= kDevicePopulationSize) {
        return;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        next_population->generation_index = current_population->generation_index + 1;
    }

    if (slot_index < config.genetic_algorithm.elite_count) {
        std::size_t elite_index = 0;
        if (!TryFindEliteIndexByRank<kDevicePopulationSize>(*current_population, slot_index, elite_index)) {
            SetFailureStatus(status, DeviceRuntimeStatusCode::kPopulationNotEvaluated);
            return;
        }

        WriteGenomeToUnevaluatedIndividual(current_population->individuals[elite_index].genome,
                                           next_population->individuals[slot_index]);
        return;
    }

    DeviceRandomState random_state = MakeDeviceRandomState(
        generation_seed, static_cast<std::uint32_t>(slot_index + (current_population->generation_index * 4099U)));

    ParentPair parent_pair{};
    if (!TrySelectParentPairDevice<kDevicePopulationSize>(*current_population, random_state, config.parent_selection,
                                                          parent_pair)) {
        SetFailureStatus(status, DeviceRuntimeStatusCode::kParentSelectionFailed);
        return;
    }

    Individual<DeviceGenome> &child_individual = next_population->individuals[slot_index];
    BreedAndMutateGenome(current_population->individuals[parent_pair.first_parent_index].genome,
                         current_population->individuals[parent_pair.second_parent_index].genome,
                         child_individual.genome, random_state, config.breeding, config.mutation);
    MarkIndividualUnevaluated(child_individual);
}

bool CheckCuda(const cudaError_t error) { return error == cudaSuccess; }

bool ReadDeviceStatus(const DeviceRuntimeBuffers &buffers, int &status_value) {
    return CheckCuda(cudaMemcpy(&status_value, buffers.status, sizeof(int), cudaMemcpyDeviceToHost));
}

bool WriteDeviceStatus(const DeviceRuntimeBuffers &buffers, const DeviceRuntimeStatusCode status_code) {
    const int status_value = DeviceStatusValue(status_code);
    return CheckCuda(cudaMemcpy(buffers.status, &status_value, sizeof(int), cudaMemcpyHostToDevice));
}

bool ResetDeviceStatus(const DeviceRuntimeBuffers &buffers) {
    return WriteDeviceStatus(buffers, DeviceRuntimeStatusCode::kOk);
}

bool KernelCompletedSuccessfully(const DeviceRuntimeBuffers &buffers) {
    int status_value = DeviceStatusValue(DeviceRuntimeStatusCode::kCudaFailure);
    return ReadDeviceStatus(buffers, status_value) && (status_value == DeviceStatusValue(DeviceRuntimeStatusCode::kOk));
}

} // namespace

bool TryCreateDeviceRuntimeBuffers(DeviceRuntimeBuffers &buffers) {
    buffers = {};

    bool ok = true;
    ok &= CheckCuda(cudaMalloc(&buffers.current_population, sizeof(DevicePopulation)));
    ok &= CheckCuda(cudaMalloc(&buffers.next_population, sizeof(DevicePopulation)));
    ok &= CheckCuda(cudaMalloc(&buffers.summary, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMalloc(&buffers.status, sizeof(int)));

    if (!ok) {
        DestroyDeviceRuntimeBuffers(buffers);
        return false;
    }

    ok &= CheckCuda(cudaMemset(buffers.current_population, 0, sizeof(DevicePopulation)));
    ok &= CheckCuda(cudaMemset(buffers.next_population, 0, sizeof(DevicePopulation)));
    ok &= CheckCuda(cudaMemset(buffers.summary, 0, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    return ok;
}

void DestroyDeviceRuntimeBuffers(DeviceRuntimeBuffers &buffers) noexcept {
    cudaFree(buffers.current_population);
    cudaFree(buffers.next_population);
    cudaFree(buffers.summary);
    cudaFree(buffers.status);
    buffers = {};
}

bool TryUploadCurrentPopulationToDevice(const DevicePopulation &host_population, DeviceRuntimeBuffers &buffers) {
    return CheckCuda(cudaMemcpy(buffers.current_population, &host_population, sizeof(DevicePopulation),
                                cudaMemcpyHostToDevice)) &&
           CheckCuda(cudaMemset(buffers.next_population, 0, sizeof(DevicePopulation)));
}

bool TryDownloadCurrentPopulationFromDevice(const DeviceRuntimeBuffers &buffers, DevicePopulation &host_population) {
    return CheckCuda(
        cudaMemcpy(&host_population, buffers.current_population, sizeof(DevicePopulation), cudaMemcpyDeviceToHost));
}

bool TryEvaluatePopulationFitnessOnDevice(DeviceRuntimeBuffers &buffers) {
    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    EvaluatePopulationFitnessKernel<<<1, kDevicePopulationSize>>>(buffers.current_population, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    if (!KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    SummarizePopulationKernel<<<1, 1>>>(buffers.current_population, buffers.summary, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    return KernelCompletedSuccessfully(buffers);
}

bool TryReadPopulationFitnessSummaryFromDevice(const DeviceRuntimeBuffers &buffers, PopulationFitnessSummary &summary) {
    return CheckCuda(cudaMemcpy(&summary, buffers.summary, sizeof(PopulationFitnessSummary), cudaMemcpyDeviceToHost));
}

bool TryReadDeviceRuntimeStatus(const DeviceRuntimeBuffers &buffers, DeviceRuntimeStatusCode &status_code) {
    int status_value = 0;
    if (!ReadDeviceStatus(buffers, status_value)) {
        return false;
    }

    status_code = static_cast<DeviceRuntimeStatusCode>(status_value);
    return true;
}

bool TryAssembleNextGenerationOnDevice(DeviceRuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                       const GenerationAssemblyConfig &config) {
    if (!IsValidGenerationAssemblyConfig<kDevicePopulationSize>(config)) {
        (void)WriteDeviceStatus(buffers, DeviceRuntimeStatusCode::kInvalidAssemblyConfig);
        return false;
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    AssembleNextGenerationKernel<<<1, kDevicePopulationSize>>>(buffers.current_population, buffers.next_population,
                                                               generation_seed, config, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    return KernelCompletedSuccessfully(buffers);
}

void SwapDevicePopulationBuffers(DeviceRuntimeBuffers &buffers) noexcept {
    std::swap(buffers.current_population, buffers.next_population);
}

const char *DeviceRuntimeStatusCodeString(const DeviceRuntimeStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceRuntimeStatusCode::kOk:
        return "ok";
    case DeviceRuntimeStatusCode::kCudaFailure:
        return "cuda failure";
    case DeviceRuntimeStatusCode::kInvalidTrainingShard:
        return "invalid training-data shard";
    case DeviceRuntimeStatusCode::kGuessAppendFailed:
        return "could not append a model-selected guess to the Wordle grid";
    case DeviceRuntimeStatusCode::kPolicyForwardFailed:
        return "policy model forward pass failed";
    case DeviceRuntimeStatusCode::kActionSelectionFailed:
        return "output-embedding action selection failed";
    case DeviceRuntimeStatusCode::kPopulationNotEvaluated:
        return "population fitness summary requested before the population was fully evaluated";
    case DeviceRuntimeStatusCode::kInvalidAssemblyConfig:
        return "invalid next-generation assembly config";
    case DeviceRuntimeStatusCode::kParentSelectionFailed:
        return "device parent selection failed";
    }

    return "unknown device-runtime status";
}

} // namespace neuroevolution::genetic_algorithm::device
