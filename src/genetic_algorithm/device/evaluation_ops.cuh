#pragma once

#include <mma.h>

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/device/runtime_common.hpp"
#include "genetic_algorithm/spatial/grid.hpp"
#include "inference/dynamic_policy.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "training_folder/training_data.hpp"
#include "wordle/wordle_grid.hpp"

namespace neuroevolution::genetic_algorithm::device_evaluation_ops {

using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::model::output_embedding::FixedWordFeatureVector;
using neuroevolution::model::output_embedding::MaterializeFixedWordFeatures;
using neuroevolution::model::output_embedding::ScoreActionEmbedding;
using neuroevolution::spatial::CellularGridShape;
using neuroevolution::inference::dynamic_policy::DynamicInferenceStatusCode;
using neuroevolution::inference::dynamic_policy::DynamicPolicyWarpScratch;
using neuroevolution::inference::dynamic_policy::HasGridAlreadyGuessedWord;
using neuroevolution::inference::dynamic_policy::SelectNextGuessFromDynamicGenomeConcurrently;
using neuroevolution::training_folder::DeviceTrainingWordCatalog;
using neuroevolution::training_folder::DoesTrainingDataShardCoverCell;
using neuroevolution::training_folder::IsValidTrainingWordCatalog;
using neuroevolution::training_folder::IsValidWordCountSchedule;
using neuroevolution::training_folder::TrainingDataShardRuntime;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::wordle::MakeWordleGrid;
using neuroevolution::wordle::TileFeedback;
using neuroevolution::wordle::TryAppendGuess;
using neuroevolution::wordle::Word;
using neuroevolution::wordle::WordleGrid;

constexpr float kWinScoreBase = 10.0f;
constexpr float kLossProgressScoreMax = 2.0f;
constexpr float kEpisodesPerTrainingWord = 3.0f;
constexpr std::size_t kEpisodesPerTrainingWordCount = 3;
constexpr float kMaximumEpisodeScore =
    kWinScoreBase + static_cast<float>(neuroevolution::wordle::kMaxTurnCount - 1U);
constexpr std::size_t kTensorActionScoreTileRows = 16;
constexpr std::size_t kTensorActionScoreTileColumns = 16;
constexpr std::size_t kTensorActionScoreTileDepth = 16;

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

template <int WarpCount> struct GenomeEvaluationTensorActionTileScratch {
    alignas(32) common::Float16
        policy_matrix[kTensorActionScoreTileRows * model::output_embedding::kOutputEmbeddingDimension]{};
    alignas(32) common::Float16
        action_matrix[model::output_embedding::kOutputEmbeddingDimension * kTensorActionScoreTileColumns]{};
    alignas(32) float score_matrix[kTensorActionScoreTileRows * kTensorActionScoreTileColumns]{};
    SelectedAction selected_actions[WarpCount]{};
    Word action_words[kTensorActionScoreTileColumns]{};
    std::size_t action_indices[kTensorActionScoreTileColumns]{};
    std::uint8_t warp_has_episode[WarpCount]{};
    std::uint8_t warp_active[WarpCount]{};
    int tile_count = 0;
    int active_episode_count = 0;
    int failure_status = static_cast<int>(DeviceGenomeEvaluationStatusCode::kOk);
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

inline NEUROEVOLUTION_HOST_DEVICE float ScoreCompletedEpisode(const WordleGrid &grid) noexcept {
    if (grid.IsWon()) {
        return kWinScoreBase + static_cast<float>(neuroevolution::wordle::kMaxTurnCount - grid.turn_count);
    }

    float best_feedback_information = 0.0f;
    const std::size_t turn_count = (grid.turn_count < neuroevolution::wordle::kMaxTurnCount)
                                       ? grid.turn_count
                                       : neuroevolution::wordle::kMaxTurnCount;
    for (std::size_t turn_index = 0; turn_index < turn_count; ++turn_index) {
        float turn_feedback_information = 0.0f;
        for (std::size_t position = 0; position < neuroevolution::wordle::kWordLength; ++position) {
            const TileFeedback feedback = grid.turns[turn_index].feedback[position];
            if (feedback == TileFeedback::green) {
                turn_feedback_information += 2.0f;
            } else if (feedback == TileFeedback::yellow) {
                turn_feedback_information += 1.0f;
            }
        }

        if (turn_feedback_information > best_feedback_information) {
            best_feedback_information = turn_feedback_information;
        }
    }

    constexpr float kMaximumFeedbackInformation = 2.0f * static_cast<float>(neuroevolution::wordle::kWordLength);
    return (best_feedback_information / kMaximumFeedbackInformation) * kLossProgressScoreMax;
}

template <int WarpCount>
__device__ inline void ZeroTensorPolicyMatrix(GenomeEvaluationTensorActionTileScratch<WarpCount> &tile_scratch) {
    for (int index = threadIdx.x;
         index < static_cast<int>(kTensorActionScoreTileRows * model::output_embedding::kOutputEmbeddingDimension);
         index += blockDim.x) {
        tile_scratch.policy_matrix[index] = common::ToFloat16(0.0f);
    }

    __syncthreads();
}

template <int WarpWidth, int WarpCount>
__device__ inline DeviceGenomeEvaluationStatusCode RefreshActiveEpisodeTileState(
    const GenomeEvaluationWarpScratch<WarpWidth> (&warp_scratch)[WarpCount],
    GenomeEvaluationTensorActionTileScratch<WarpCount> &tile_scratch) {
    const std::size_t warp_index = static_cast<std::size_t>(threadIdx.x / WarpWidth);
    const std::size_t lane_index = static_cast<std::size_t>(threadIdx.x % WarpWidth);

    if (lane_index == 0) {
        tile_scratch.warp_active[warp_index] =
            ((tile_scratch.warp_has_episode[warp_index] != 0) && !warp_scratch[warp_index].grid.IsFinished()) ? 1U : 0U;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        tile_scratch.failure_status = static_cast<int>(DeviceGenomeEvaluationStatusCode::kOk);
        tile_scratch.active_episode_count = 0;
        for (int active_warp_index = 0; active_warp_index < WarpCount; ++active_warp_index) {
            tile_scratch.active_episode_count += tile_scratch.warp_active[active_warp_index];
            if ((tile_scratch.warp_active[active_warp_index] != 0) &&
                (warp_scratch[active_warp_index].status != static_cast<int>(DeviceGenomeEvaluationStatusCode::kOk))) {
                tile_scratch.failure_status = warp_scratch[active_warp_index].status;
                break;
            }
        }
    }
    __syncthreads();

    return static_cast<DeviceGenomeEvaluationStatusCode>(tile_scratch.failure_status);
}

template <int WarpWidth, int WarpCount>
__device__ inline void StorePolicyBatchRowsConcurrently(
    const GenomeEvaluationWarpScratch<WarpWidth> (&warp_scratch)[WarpCount],
    GenomeEvaluationTensorActionTileScratch<WarpCount> &tile_scratch) {
    const std::size_t warp_index = static_cast<std::size_t>(threadIdx.x / WarpWidth);
    const std::size_t lane_index = static_cast<std::size_t>(threadIdx.x % WarpWidth);

    if (tile_scratch.warp_active[warp_index] == 0) {
        return;
    }

    for (std::size_t feature_index = lane_index; feature_index < model::output_embedding::kOutputEmbeddingDimension;
         feature_index += WarpWidth) {
        tile_scratch.policy_matrix[(warp_index * model::output_embedding::kOutputEmbeddingDimension) + feature_index] =
            common::ToFloat16(warp_scratch[warp_index].dynamic_policy.model.policy_vector[feature_index]);
    }

    __syncwarp();
}

template <int WarpCount>
__device__ inline void LoadTensorActionTileConcurrently(
    const TrainingWordCatalog &training_word_catalog,
    const genetic_algorithm::genome::TrainableActionEmbeddingTail *tail_rows, const std::size_t tile_start,
    const std::size_t action_count, GenomeEvaluationTensorActionTileScratch<WarpCount> &tile_scratch) {
    if (threadIdx.x == 0) {
        const std::size_t remaining_action_count = action_count - tile_start;
        const std::size_t tile_count = (remaining_action_count < kTensorActionScoreTileColumns)
                                           ? remaining_action_count
                                           : kTensorActionScoreTileColumns;
        tile_scratch.tile_count = static_cast<int>(tile_count);
    }
    __syncthreads();

    for (int column_index = threadIdx.x; column_index < static_cast<int>(kTensorActionScoreTileColumns);
         column_index += blockDim.x) {
        if (column_index < tile_scratch.tile_count) {
            const std::size_t action_index = tile_start + static_cast<std::size_t>(column_index);
            const Word action_word = training_word_catalog.words[action_index];
            FixedWordFeatureVector fixed_word_features{};
            MaterializeFixedWordFeatures(action_word, fixed_word_features);
            tile_scratch.action_words[column_index] = action_word;
            tile_scratch.action_indices[column_index] = action_index;

            for (std::size_t feature_index = 0; feature_index < model::output_embedding::kWordFeatureDimension;
                 ++feature_index) {
                tile_scratch.action_matrix[(feature_index * kTensorActionScoreTileColumns) + column_index] =
                    common::ToFloat16(fixed_word_features[feature_index]);
            }

            for (std::size_t feature_index = 0; feature_index < model::output_embedding::kTrainableFeatureDimension;
                 ++feature_index) {
                tile_scratch.action_matrix[((model::output_embedding::kWordFeatureDimension + feature_index) *
                                            kTensorActionScoreTileColumns) +
                                           column_index] = tail_rows[action_index][feature_index];
            }
        } else {
            tile_scratch.action_words[column_index] = {};
            tile_scratch.action_indices[column_index] = 0;
            for (std::size_t feature_index = 0; feature_index < model::output_embedding::kOutputEmbeddingDimension;
                 ++feature_index) {
                tile_scratch.action_matrix[(feature_index * kTensorActionScoreTileColumns) + column_index] =
                    common::ToFloat16(0.0f);
            }
        }
    }

    __syncthreads();
}

template <int WarpCount>
__device__ inline void ComputeTensorActionScoreTile(
    GenomeEvaluationTensorActionTileScratch<WarpCount> &tile_scratch) {
    const std::size_t warp_index = static_cast<std::size_t>(threadIdx.x / warpSize);
    if (warp_index != 0) {
        __syncthreads();
        return;
    }

    namespace wmma = nvcuda::wmma;
    wmma::fragment<wmma::matrix_a, kTensorActionScoreTileRows, kTensorActionScoreTileColumns,
                   kTensorActionScoreTileDepth, common::Float16, wmma::row_major>
        policy_fragment;
    wmma::fragment<wmma::matrix_b, kTensorActionScoreTileRows, kTensorActionScoreTileColumns,
                   kTensorActionScoreTileDepth, common::Float16, wmma::row_major>
        action_fragment;
    wmma::fragment<wmma::accumulator, kTensorActionScoreTileRows, kTensorActionScoreTileColumns,
                   kTensorActionScoreTileDepth, float>
        accumulator_fragment;

    wmma::fill_fragment(accumulator_fragment, 0.0f);
    for (std::size_t depth_start = 0; depth_start < model::output_embedding::kOutputEmbeddingDimension;
         depth_start += kTensorActionScoreTileDepth) {
        wmma::load_matrix_sync(policy_fragment, tile_scratch.policy_matrix + depth_start,
                               model::output_embedding::kOutputEmbeddingDimension);
        wmma::load_matrix_sync(action_fragment, tile_scratch.action_matrix + (depth_start * kTensorActionScoreTileColumns),
                               kTensorActionScoreTileColumns);
        wmma::mma_sync(accumulator_fragment, policy_fragment, action_fragment, accumulator_fragment);
    }

    wmma::store_matrix_sync(tile_scratch.score_matrix, accumulator_fragment, kTensorActionScoreTileColumns,
                            wmma::mem_row_major);
    __syncthreads();
}

template <int WarpWidth, int WarpCount>
__device__ inline DeviceGenomeEvaluationStatusCode SelectNextGuessForEpisodeTileWithTensorCoresConcurrently(
    const std::uint8_t *genome_bytes, const TrainingWordCatalog &training_word_catalog,
    const std::size_t action_count, GenomeEvaluationWarpScratch<WarpWidth> (&warp_scratch)[WarpCount],
    GenomeEvaluationTensorActionTileScratch<WarpCount> &tile_scratch) {
    const std::size_t warp_index = static_cast<std::size_t>(threadIdx.x / WarpWidth);
    const std::size_t lane_index = static_cast<std::size_t>(threadIdx.x % WarpWidth);

    ZeroTensorPolicyMatrix(tile_scratch);

    if (tile_scratch.warp_active[warp_index] != 0) {
        const bool policy_forward_ok = model::policy_model::TryForwardPolicyModelConcurrently<WarpWidth>(
            genetic_algorithm::genome::GenomePolicyModelParameters(genome_bytes), warp_scratch[warp_index].grid,
            warp_scratch[warp_index].dynamic_policy.model);
        if (lane_index == 0) {
            warp_scratch[warp_index].status = policy_forward_ok ? static_cast<int>(DeviceGenomeEvaluationStatusCode::kOk)
                                                                : static_cast<int>(DeviceGenomeEvaluationStatusCode::kPolicyForwardFailed);
        }
    } else if (lane_index == 0) {
        warp_scratch[warp_index].status = static_cast<int>(DeviceGenomeEvaluationStatusCode::kOk);
    }
    __syncwarp();

    const DeviceGenomeEvaluationStatusCode forward_status =
        RefreshActiveEpisodeTileState<WarpWidth, WarpCount>(warp_scratch, tile_scratch);
    if (forward_status != DeviceGenomeEvaluationStatusCode::kOk) {
        return forward_status;
    }

    StorePolicyBatchRowsConcurrently<WarpWidth, WarpCount>(warp_scratch, tile_scratch);
    __syncthreads();

    const genetic_algorithm::genome::TrainableActionEmbeddingTail *tail_rows =
        genetic_algorithm::genome::GenomeTailRows(genome_bytes);
    SelectedAction local_best_action{};
    bool has_local_candidate = false;

    for (std::size_t tile_start = 0; tile_start < action_count; tile_start += kTensorActionScoreTileColumns) {
        LoadTensorActionTileConcurrently(training_word_catalog, tail_rows, tile_start, action_count, tile_scratch);
        ComputeTensorActionScoreTile(tile_scratch);

        if ((tile_scratch.warp_active[warp_index] != 0) && (lane_index == 0)) {
            for (int column_index = 0; column_index < tile_scratch.tile_count; ++column_index) {
                const Word &candidate_word = tile_scratch.action_words[column_index];
                if (HasGridAlreadyGuessedWord(warp_scratch[warp_index].grid, candidate_word)) {
                    continue;
                }

                const float score = tile_scratch.score_matrix[(warp_index * kTensorActionScoreTileColumns) + column_index];
                const std::size_t action_index = tile_scratch.action_indices[column_index];
                if (!has_local_candidate || (score > local_best_action.score) ||
                    ((score == local_best_action.score) && (action_index < local_best_action.action_index))) {
                    local_best_action.action_index = action_index;
                    local_best_action.word = candidate_word;
                    local_best_action.score = score;
                    has_local_candidate = true;
                }
            }
        }
        __syncthreads();
    }

    if (lane_index == 0) {
        if (tile_scratch.warp_active[warp_index] == 0) {
            tile_scratch.selected_actions[warp_index] = {};
            warp_scratch[warp_index].status = static_cast<int>(DeviceGenomeEvaluationStatusCode::kOk);
        } else if (!has_local_candidate) {
            warp_scratch[warp_index].status = static_cast<int>(DeviceGenomeEvaluationStatusCode::kActionSelectionFailed);
        } else {
            tile_scratch.selected_actions[warp_index] = local_best_action;
            warp_scratch[warp_index].status = static_cast<int>(DeviceGenomeEvaluationStatusCode::kOk);
        }
    }
    __syncwarp();

    return RefreshActiveEpisodeTileState<WarpWidth, WarpCount>(warp_scratch, tile_scratch);
}

template <int WarpWidth, int WarpCount>
__device__ inline DeviceGenomeEvaluationStatusCode TryPlayWordleEpisodeTileWithTensorCoresConcurrently(
    const std::uint8_t *genome_bytes, const TrainingWordCatalog &training_word_catalog, const std::size_t action_count,
    GenomeEvaluationWarpScratch<WarpWidth> (&warp_scratch)[WarpCount],
    GenomeEvaluationTensorActionTileScratch<WarpCount> &tile_scratch) {
    const std::size_t warp_index = static_cast<std::size_t>(threadIdx.x / WarpWidth);
    const std::size_t lane_index = static_cast<std::size_t>(threadIdx.x % WarpWidth);

    while (true) {
        const DeviceGenomeEvaluationStatusCode active_status =
            RefreshActiveEpisodeTileState<WarpWidth, WarpCount>(warp_scratch, tile_scratch);
        if (active_status != DeviceGenomeEvaluationStatusCode::kOk) {
            return active_status;
        }

        if (tile_scratch.active_episode_count == 0) {
            return DeviceGenomeEvaluationStatusCode::kOk;
        }

        const DeviceGenomeEvaluationStatusCode inference_status =
            SelectNextGuessForEpisodeTileWithTensorCoresConcurrently<WarpWidth, WarpCount>(
                genome_bytes, training_word_catalog, action_count, warp_scratch, tile_scratch);
        if (inference_status != DeviceGenomeEvaluationStatusCode::kOk) {
            return inference_status;
        }

        if ((tile_scratch.warp_active[warp_index] != 0) && (lane_index == 0)) {
            warp_scratch[warp_index].status =
                TryAppendGuess(warp_scratch[warp_index].grid, tile_scratch.selected_actions[warp_index].word)
                    ? static_cast<int>(DeviceGenomeEvaluationStatusCode::kOk)
                    : static_cast<int>(DeviceGenomeEvaluationStatusCode::kGuessAppendFailed);
        }
        __syncwarp();

        const DeviceGenomeEvaluationStatusCode append_status =
            RefreshActiveEpisodeTileState<WarpWidth, WarpCount>(warp_scratch, tile_scratch);
        if (append_status != DeviceGenomeEvaluationStatusCode::kOk) {
            return append_status;
        }
    }
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
        episode_score = ScoreCompletedEpisode(scratch.grid);
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

} // namespace neuroevolution::genetic_algorithm::device_evaluation_ops
