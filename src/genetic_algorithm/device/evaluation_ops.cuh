#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/device/runtime_common.hpp"
#include "genetic_algorithm/spatial/grid.hpp"
#include "inference/dynamic_policy.hpp"
#include "training_folder/training_data.hpp"
#include "wordle/wordle_grid.hpp"

namespace neuroevolution::genetic_algorithm::device_evaluation_ops {

using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::spatial::CellularGridShape;
using neuroevolution::inference::dynamic_policy::DynamicInferenceStatusCode;
using neuroevolution::inference::dynamic_policy::DynamicPolicyWarpScratch;
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
constexpr std::size_t kEpisodesPerTrainingWordCount = 3;
constexpr float kMaximumEpisodeScore =
    kWinScoreBase + static_cast<float>(neuroevolution::wordle::kMaxTurnCount - 1U);

enum class DeviceGenomeEvaluationStatusCode : int {
    kOk = 0,
    kInvalidTrainingShard = 1,
    kGuessAppendFailed = 2,
    kPolicyForwardFailed = 3,
    kActionSelectionFailed = 4,
};

template <int WarpWidth> struct GenomeEvaluationWarpScratch {
    DynamicPolicyWarpScratch<WarpWidth> dynamic_policy{};
    WordleGrid grid{};
    int status = static_cast<int>(DeviceGenomeEvaluationStatusCode::kActionSelectionFailed);
};

// Temporary compatibility shim for the abandoned block-owned evaluator.
template <int ThreadsPerBlock> struct GenomeEvaluationBlockScratch {};

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

__device__ inline DeviceGenomeEvaluationStatusCode
TryInitializeEpisodeGrid(const TrainingWordCatalog &training_word_catalog,
                         const TrainingDataShardRuntime *active_training_shards,
                         const std::size_t active_shard_count, const CellularGridShape &grid_shape,
                         const std::size_t cell_index, const std::size_t episode_index,
                         const std::size_t local_training_word_count, WordleGrid &grid_out) {
    if (local_training_word_count == 0) {
        return DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard;
    }

    const std::size_t entry_index = episode_index / kEpisodesPerTrainingWordCount;
    const std::size_t episode_variant = episode_index % kEpisodesPerTrainingWordCount;
    Word solution{};
    const DeviceGenomeEvaluationStatusCode solution_status =
        TryResolveLocalTrainingWord(training_word_catalog, active_training_shards, active_shard_count, grid_shape,
                                    cell_index, entry_index, solution);
    if (solution_status != DeviceGenomeEvaluationStatusCode::kOk) {
        return solution_status;
    }

    if (episode_variant == 0) {
        grid_out = MakeWordleGrid(solution);
        return DeviceGenomeEvaluationStatusCode::kOk;
    }

    if (episode_variant == 1) {
        return TryInitializePrefilledGrid(training_word_catalog, active_training_shards, active_shard_count, grid_shape,
                                          cell_index, solution, entry_index + 1, entry_index + 2,
                                          local_training_word_count, grid_out);
    }

    return TryInitializePrefilledGrid(training_word_catalog, active_training_shards, active_shard_count, grid_shape,
                                      cell_index, solution, entry_index + 3, entry_index + 4,
                                      local_training_word_count, grid_out);
}

template <int WarpWidth>
__device__ inline DeviceGenomeEvaluationStatusCode TryPlayWordleEpisodeConcurrently(
    const std::uint8_t *genome_bytes, const TrainingWordCatalog &training_word_catalog,
    const std::size_t action_count, GenomeEvaluationWarpScratch<WarpWidth> &scratch,
    float &episode_score_out) {
    const std::size_t lane_index = static_cast<std::size_t>(threadIdx.x % WarpWidth);

    while (!scratch.grid.IsFinished()) {
        SelectedAction selected_action{};
        const DynamicInferenceStatusCode inference_status =
            SelectNextGuessFromDynamicGenomeConcurrently<WarpWidth>(
                scratch.grid, training_word_catalog, genome_bytes, action_count, scratch.dynamic_policy,
                selected_action);
        if (inference_status == DynamicInferenceStatusCode::kPolicyForwardFailed) {
            return DeviceGenomeEvaluationStatusCode::kPolicyForwardFailed;
        }

        if (inference_status != DynamicInferenceStatusCode::kOk) {
            return DeviceGenomeEvaluationStatusCode::kActionSelectionFailed;
        }

        if (lane_index == 0) {
            scratch.status = TryAppendGuess(scratch.grid, selected_action.word)
                                 ? static_cast<int>(DeviceGenomeEvaluationStatusCode::kOk)
                                 : static_cast<int>(DeviceGenomeEvaluationStatusCode::kGuessAppendFailed);
        }
        __syncwarp();

        if (scratch.status != static_cast<int>(DeviceGenomeEvaluationStatusCode::kOk)) {
            return static_cast<DeviceGenomeEvaluationStatusCode>(scratch.status);
        }
    }

    float episode_score = 0.0f;
    if (lane_index == 0) {
        episode_score = scratch.grid.IsWon()
                            ? (kWinScoreBase +
                               static_cast<float>(neuroevolution::wordle::kMaxTurnCount - scratch.grid.turn_count))
                            : 0.0f;
    }

    episode_score_out = __shfl_sync(0xffffffffu, episode_score, 0);
    return DeviceGenomeEvaluationStatusCode::kOk;
}

template <int WarpWidth>
__device__ inline DeviceGenomeEvaluationStatusCode TryEvaluateGenomeEpisodeConcurrently(
    const std::uint8_t *genome_bytes, const std::size_t genome_action_count,
    const device_common::RuntimeWordCounts runtime_word_counts,
    const TrainingDataShardRuntime *active_training_shards, const std::size_t active_shard_count,
    const CellularGridShape &grid_shape, const std::size_t cell_index, const std::size_t episode_index,
    const std::size_t local_training_word_count, GenomeEvaluationWarpScratch<WarpWidth> &scratch,
    float &episode_score_out) {
    const TrainingWordCatalog &training_word_catalog = DeviceTrainingWordCatalog();
    const std::size_t selectable_action_count = (genome_action_count < runtime_word_counts.action_space_word_count)
                                                    ? genome_action_count
                                                    : runtime_word_counts.action_space_word_count;

    if (!IsValidTrainingWordCatalog(training_word_catalog) ||
        !IsValidRuntimeWordCounts(training_word_catalog, runtime_word_counts) || (genome_bytes == nullptr) ||
        (genome_action_count == 0) || (selectable_action_count == 0) || (local_training_word_count == 0) ||
        (selectable_action_count > genome_action_count) ||
        (episode_index >= (local_training_word_count * kEpisodesPerTrainingWordCount))) {
        return DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard;
    }

    if ((threadIdx.x % WarpWidth) == 0) {
        scratch.status = static_cast<int>(TryInitializeEpisodeGrid(
            training_word_catalog, active_training_shards, active_shard_count, grid_shape, cell_index, episode_index,
            local_training_word_count, scratch.grid));
    }
    __syncwarp();

    if (scratch.status != static_cast<int>(DeviceGenomeEvaluationStatusCode::kOk)) {
        return static_cast<DeviceGenomeEvaluationStatusCode>(scratch.status);
    }

    return TryPlayWordleEpisodeConcurrently<WarpWidth>(genome_bytes, training_word_catalog, selectable_action_count,
                                                       scratch, episode_score_out);
}

template <int ThreadsPerBlock>
__device__ inline DeviceGenomeEvaluationStatusCode TryEvaluateGenomeFitnessConcurrently(
    const std::uint8_t *, const std::size_t, const device_common::RuntimeWordCounts,
    const TrainingDataShardRuntime *, const std::size_t, const CellularGridShape &, const std::size_t,
    GenomeEvaluationBlockScratch<ThreadsPerBlock> &, float &fitness_out,
    std::size_t &local_training_word_count_out) {
    fitness_out = 0.0f;
    local_training_word_count_out = 0;
    return DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard;
}

} // namespace neuroevolution::genetic_algorithm::device_evaluation_ops
