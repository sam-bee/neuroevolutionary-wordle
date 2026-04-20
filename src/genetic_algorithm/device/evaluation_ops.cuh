#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/device/runtime_common.hpp"
#include "inference/dynamic_policy.hpp"
#include "genetic_algorithm/spatial/grid.hpp"
#include "training_folder/training_data.hpp"
#include "wordle/wordle_grid.hpp"

namespace neuroevolution::genetic_algorithm::device_evaluation_ops {

using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::spatial::CellularGridShape;
using neuroevolution::inference::dynamic_policy::DynamicPolicyBlockScratch;
using neuroevolution::inference::dynamic_policy::DynamicInferenceStatusCode;
using neuroevolution::inference::dynamic_policy::SelectNextGuessFromDynamicGenomeConcurrently;
using neuroevolution::training_folder::DeviceTrainingWordCatalog;
using neuroevolution::training_folder::DoesTrainingDataShardCoverCell;
using neuroevolution::training_folder::IsValidTrainingWordCatalog;
using neuroevolution::training_folder::IsValidWordCountSchedule;
using neuroevolution::training_folder::TrainingDataShardRuntime;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::wordle::MakeWordleGrid;
using neuroevolution::wordle::TryAppendGuess;
using neuroevolution::wordle::Word;
using neuroevolution::wordle::WordleGrid;

constexpr float kWinScoreBase = 10.0f;
constexpr float kEpisodesPerTrainingWord = 3.0f;
constexpr float kMaximumEpisodeScore =
    kWinScoreBase + static_cast<float>(neuroevolution::wordle::kMaxTurnCount - 1U);

enum class DeviceGenomeEvaluationStatusCode : int {
    kOk = 0,
    kInvalidTrainingShard = 1,
    kGuessAppendFailed = 2,
    kPolicyForwardFailed = 3,
    kActionSelectionFailed = 4,
};

template <int ThreadsPerBlock> struct GenomeEvaluationBlockScratch {
    DynamicPolicyBlockScratch<ThreadsPerBlock> dynamic_policy{};
    WordleGrid grid{};
    float score_sum = 0.0f;
    float episode_score = 0.0f;
    std::size_t local_training_word_count = 0;
    DeviceGenomeEvaluationStatusCode status = DeviceGenomeEvaluationStatusCode::kActionSelectionFailed;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidRuntimeWordCounts(const TrainingWordCatalog &training_word_catalog,
                         const device_common::RuntimeWordCounts &runtime_word_counts) noexcept {
    return (runtime_word_counts.training_word_count <= training_word_catalog.word_count) &&
           (runtime_word_counts.action_space_word_count <= training_word_catalog.word_count) &&
           (runtime_word_counts.training_word_count <= runtime_word_counts.action_space_word_count) &&
           IsValidWordCountSchedule(runtime_word_counts.training_word_schedule) &&
           (runtime_word_counts.shard_radius_growth_period_generations > 0);
}

constexpr NEUROEVOLUTION_HOST_DEVICE float MaximumPossibleFitnessForTrainingWordCount(
    const std::size_t training_word_count) noexcept {
    return kEpisodesPerTrainingWord * kMaximumEpisodeScore * static_cast<float>(training_word_count);
}

constexpr NEUROEVOLUTION_HOST_DEVICE float
MaximumPossibleFitness(const device_common::RuntimeWordCounts runtime_word_counts) noexcept {
    return MaximumPossibleFitnessForTrainingWordCount(runtime_word_counts.training_word_count);
}

constexpr NEUROEVOLUTION_HOST_DEVICE float
NormalizeFitnessForSelection(const float raw_fitness, const std::size_t training_word_count) noexcept {
    const float maximum_possible_fitness = MaximumPossibleFitnessForTrainingWordCount(training_word_count);
    if (maximum_possible_fitness <= 0.0f) {
        return spatial::kPositiveSelectionFitnessFloor;
    }

    const float normalized_fitness = raw_fitness / maximum_possible_fitness;
    if (normalized_fitness >= 1.0f) {
        return 1.0f;
    }

    return (normalized_fitness > spatial::kPositiveSelectionFitnessFloor) ? normalized_fitness
                                                                           : spatial::kPositiveSelectionFitnessFloor;
}

constexpr NEUROEVOLUTION_HOST_DEVICE float NormalizeFitnessForSelection(
    const float raw_fitness, const device_common::RuntimeWordCounts runtime_word_counts) noexcept {
    return NormalizeFitnessForSelection(raw_fitness, runtime_word_counts.training_word_count);
}

__device__ inline std::size_t WrapTrainingWordIndex(const std::size_t index, const std::size_t word_count) {
    return (word_count == 0) ? 0 : (index % word_count);
}

__device__ inline DeviceGenomeEvaluationStatusCode
TryCountLocalTrainingWords(const TrainingDataShardRuntime *active_training_shards, const std::size_t active_shard_count,
                          const CellularGridShape &grid_shape, const std::size_t cell_index,
                          std::size_t &local_training_word_count_out) {
    local_training_word_count_out = 0;
    if ((active_training_shards == nullptr) || !spatial::IsValidCellularGridShape(grid_shape) ||
        (cell_index >= grid_shape.cell_count) || (active_shard_count == 0)) {
        return DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard;
    }

    for (std::size_t shard_index = 0; shard_index < active_shard_count; ++shard_index) {
        const TrainingDataShardRuntime &shard = active_training_shards[shard_index];
        if (DoesTrainingDataShardCoverCell(shard, grid_shape, cell_index)) {
            local_training_word_count_out += shard.word_count;
        }
    }

    return (local_training_word_count_out > 0) ? DeviceGenomeEvaluationStatusCode::kOk
                                               : DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard;
}

__device__ inline DeviceGenomeEvaluationStatusCode TryResolveLocalTrainingWord(
    const TrainingWordCatalog &training_word_catalog, const TrainingDataShardRuntime *active_training_shards,
    const std::size_t active_shard_count, const CellularGridShape &grid_shape, const std::size_t cell_index,
    const std::size_t local_word_index, Word &word_out) {
    std::size_t remaining_word_index = local_word_index;

    for (std::size_t shard_index = 0; shard_index < active_shard_count; ++shard_index) {
        const TrainingDataShardRuntime &shard = active_training_shards[shard_index];
        if (!DoesTrainingDataShardCoverCell(shard, grid_shape, cell_index)) {
            continue;
        }

        if (remaining_word_index < shard.word_count) {
            const std::size_t catalog_word_index = shard.first_catalog_word_index + remaining_word_index;
            if (catalog_word_index >= training_word_catalog.word_count) {
                return DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard;
            }

            word_out = training_word_catalog.words[catalog_word_index];
            return DeviceGenomeEvaluationStatusCode::kOk;
        }

        remaining_word_index -= shard.word_count;
    }

    return DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard;
}

__device__ inline DeviceGenomeEvaluationStatusCode TryInitializePrefilledGrid(
    const TrainingWordCatalog &training_word_catalog, const TrainingDataShardRuntime *active_training_shards,
    const std::size_t active_shard_count, const CellularGridShape &grid_shape, const std::size_t cell_index,
    const Word &solution, const std::size_t first_guess_index, const std::size_t second_guess_index,
    const std::size_t local_training_word_count, WordleGrid &grid_out) {
    grid_out = MakeWordleGrid(solution);

    Word first_guess{};
    Word second_guess{};
    const DeviceGenomeEvaluationStatusCode first_guess_status =
        TryResolveLocalTrainingWord(training_word_catalog, active_training_shards, active_shard_count, grid_shape,
                                    cell_index, WrapTrainingWordIndex(first_guess_index, local_training_word_count),
                                    first_guess);
    if (first_guess_status != DeviceGenomeEvaluationStatusCode::kOk) {
        return first_guess_status;
    }

    const DeviceGenomeEvaluationStatusCode second_guess_status =
        TryResolveLocalTrainingWord(training_word_catalog, active_training_shards, active_shard_count, grid_shape,
                                    cell_index, WrapTrainingWordIndex(second_guess_index, local_training_word_count),
                                    second_guess);
    if (second_guess_status != DeviceGenomeEvaluationStatusCode::kOk) {
        return second_guess_status;
    }

    if (!TryAppendGuess(grid_out, first_guess) || !TryAppendGuess(grid_out, second_guess)) {
        return DeviceGenomeEvaluationStatusCode::kGuessAppendFailed;
    }

    return DeviceGenomeEvaluationStatusCode::kOk;
}

template <int ThreadsPerBlock>
__device__ inline DeviceGenomeEvaluationStatusCode TryPlayWordleToCompletionConcurrently(
    const std::uint8_t *genome_bytes, const TrainingWordCatalog &training_word_catalog,
    const std::size_t action_count, GenomeEvaluationBlockScratch<ThreadsPerBlock> &scratch) {
    while (!scratch.grid.IsFinished()) {
        SelectedAction selected_action{};
        const DynamicInferenceStatusCode inference_status =
            SelectNextGuessFromDynamicGenomeConcurrently<ThreadsPerBlock>(
                scratch.grid, training_word_catalog, genome_bytes, action_count, scratch.dynamic_policy,
                selected_action);
        if (inference_status == DynamicInferenceStatusCode::kPolicyForwardFailed) {
            return DeviceGenomeEvaluationStatusCode::kPolicyForwardFailed;
        }

        if (inference_status != DynamicInferenceStatusCode::kOk) {
            return DeviceGenomeEvaluationStatusCode::kActionSelectionFailed;
        }

        if (threadIdx.x == 0) {
            scratch.status = TryAppendGuess(scratch.grid, selected_action.word)
                                 ? DeviceGenomeEvaluationStatusCode::kOk
                                 : DeviceGenomeEvaluationStatusCode::kGuessAppendFailed;
        }
        __syncthreads();

        if (scratch.status != DeviceGenomeEvaluationStatusCode::kOk) {
            return scratch.status;
        }
    }

    if (threadIdx.x == 0) {
        scratch.episode_score = scratch.grid.IsWon()
                                    ? (kWinScoreBase +
                                       static_cast<float>(neuroevolution::wordle::kMaxTurnCount -
                                                          scratch.grid.turn_count))
                                    : 0.0f;
    }
    __syncthreads();
    return DeviceGenomeEvaluationStatusCode::kOk;
}

template <int ThreadsPerBlock>
__device__ inline DeviceGenomeEvaluationStatusCode TryEvaluateGenomeFitnessConcurrently(
    const std::uint8_t *genome_bytes, const std::size_t genome_action_count,
    const device_common::RuntimeWordCounts runtime_word_counts,
    const TrainingDataShardRuntime *active_training_shards, const std::size_t active_shard_count,
    const CellularGridShape &grid_shape, const std::size_t cell_index,
    GenomeEvaluationBlockScratch<ThreadsPerBlock> &scratch, float &fitness_out,
    std::size_t &local_training_word_count_out) {
    const TrainingWordCatalog &training_word_catalog = DeviceTrainingWordCatalog();
    const std::size_t selectable_action_count = (genome_action_count < runtime_word_counts.action_space_word_count)
                                                    ? genome_action_count
                                                    : runtime_word_counts.action_space_word_count;

    if (!IsValidTrainingWordCatalog(training_word_catalog) ||
        !IsValidRuntimeWordCounts(training_word_catalog, runtime_word_counts) || (genome_bytes == nullptr) ||
        (genome_action_count == 0) || (selectable_action_count == 0) ||
        (selectable_action_count > genome_action_count)) {
        return DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard;
    }

    if (threadIdx.x == 0) {
        scratch.status = TryCountLocalTrainingWords(active_training_shards, active_shard_count, grid_shape, cell_index,
                                                    scratch.local_training_word_count);
        scratch.score_sum = 0.0f;
    }
    __syncthreads();

    if (scratch.status != DeviceGenomeEvaluationStatusCode::kOk) {
        return scratch.status;
    }

    for (std::size_t entry_index = 0; entry_index < scratch.local_training_word_count; ++entry_index) {
        Word solution{};
        if (threadIdx.x == 0) {
            scratch.status =
                TryResolveLocalTrainingWord(training_word_catalog, active_training_shards, active_shard_count,
                                            grid_shape, cell_index, entry_index, solution);
            if (scratch.status == DeviceGenomeEvaluationStatusCode::kOk) {
                scratch.grid = MakeWordleGrid(solution);
            }
        }
        __syncthreads();

        if (scratch.status != DeviceGenomeEvaluationStatusCode::kOk) {
            return scratch.status;
        }

        const DeviceGenomeEvaluationStatusCode fresh_episode_status =
            TryPlayWordleToCompletionConcurrently<ThreadsPerBlock>(genome_bytes, training_word_catalog,
                                                                   selectable_action_count, scratch);
        if (threadIdx.x == 0) {
            scratch.status = fresh_episode_status;
        }
        __syncthreads();
        if (scratch.status != DeviceGenomeEvaluationStatusCode::kOk) {
            return scratch.status;
        }

        if (threadIdx.x == 0) {
            scratch.score_sum += scratch.episode_score;
            scratch.status = TryInitializePrefilledGrid(training_word_catalog, active_training_shards, active_shard_count,
                                                        grid_shape, cell_index, solution, entry_index + 1,
                                                        entry_index + 2, scratch.local_training_word_count,
                                                        scratch.grid);
        }
        __syncthreads();

        if (scratch.status != DeviceGenomeEvaluationStatusCode::kOk) {
            return scratch.status;
        }

        const DeviceGenomeEvaluationStatusCode second_episode_status =
            TryPlayWordleToCompletionConcurrently<ThreadsPerBlock>(genome_bytes, training_word_catalog,
                                                                   selectable_action_count, scratch);
        if (threadIdx.x == 0) {
            scratch.status = second_episode_status;
        }
        __syncthreads();
        if (scratch.status != DeviceGenomeEvaluationStatusCode::kOk) {
            return scratch.status;
        }

        if (threadIdx.x == 0) {
            scratch.score_sum += scratch.episode_score;
            scratch.status = TryInitializePrefilledGrid(training_word_catalog, active_training_shards, active_shard_count,
                                                        grid_shape, cell_index, solution, entry_index + 3,
                                                        entry_index + 4, scratch.local_training_word_count,
                                                        scratch.grid);
        }
        __syncthreads();

        if (scratch.status != DeviceGenomeEvaluationStatusCode::kOk) {
            return scratch.status;
        }

        const DeviceGenomeEvaluationStatusCode third_episode_status =
            TryPlayWordleToCompletionConcurrently<ThreadsPerBlock>(genome_bytes, training_word_catalog,
                                                                   selectable_action_count, scratch);
        if (threadIdx.x == 0) {
            scratch.status = third_episode_status;
        }
        __syncthreads();
        if (scratch.status != DeviceGenomeEvaluationStatusCode::kOk) {
            return scratch.status;
        }

        if (threadIdx.x == 0) {
            scratch.score_sum += scratch.episode_score;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        local_training_word_count_out = scratch.local_training_word_count;
        fitness_out = NormalizeFitnessForSelection(scratch.score_sum, scratch.local_training_word_count);
    }
    __syncthreads();

    return DeviceGenomeEvaluationStatusCode::kOk;
}

} // namespace neuroevolution::genetic_algorithm::device_evaluation_ops
