#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/device/runtime_common.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "genetic_algorithm/spatial/grid.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "model/policy_model/policy_model.hpp"
#include "training_folder/training_data.hpp"
#include "wordle/wordle_grid.hpp"

namespace neuroevolution::genetic_algorithm::device_evaluation_ops {

using neuroevolution::model::output_embedding::ActionEmbedding;
using neuroevolution::model::output_embedding::ScoreActionEmbedding;
using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::model::policy_model::PolicyVector;
using neuroevolution::model::policy_model::TryForwardPolicyModel;
using neuroevolution::training_folder::DeviceTrainingWordCatalog;
using neuroevolution::training_folder::IsValidTrainingWordCatalog;
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

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidRuntimeWordCounts(const TrainingWordCatalog &training_word_catalog,
                         const device_common::RuntimeWordCounts &runtime_word_counts) noexcept {
    return (runtime_word_counts.training_word_count <= training_word_catalog.word_count) &&
           (runtime_word_counts.action_space_word_count <= training_word_catalog.word_count);
}

constexpr NEUROEVOLUTION_HOST_DEVICE float
MaximumPossibleFitness(const device_common::RuntimeWordCounts runtime_word_counts) noexcept {
    return kEpisodesPerTrainingWord * kMaximumEpisodeScore * static_cast<float>(runtime_word_counts.training_word_count);
}

constexpr NEUROEVOLUTION_HOST_DEVICE float NormalizeFitnessForSelection(
    const float raw_fitness, const device_common::RuntimeWordCounts runtime_word_counts) noexcept {
    const float maximum_possible_fitness = MaximumPossibleFitness(runtime_word_counts);
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

__device__ inline float ScoreDynamicActionEmbedding(const PolicyVector &policy_vector, const Word &action_word,
                                                    const genome::TrainableActionEmbeddingTail &trainable_tail) {
    ActionEmbedding action_embedding{};
    action_embedding.word = action_word;
    action_embedding.trainable_tail = trainable_tail;
    return ScoreActionEmbedding(policy_vector, action_embedding);
}

__device__ inline bool TrySelectBestDynamicAction(const PolicyVector &policy_vector,
                                                  const TrainingWordCatalog &training_word_catalog,
                                                  const std::uint8_t *genome_bytes, const std::size_t action_count,
                                                  SelectedAction &selected_action) {
    if ((action_count == 0) || (action_count > training_word_catalog.word_count)) {
        return false;
    }

    const genome::TrainableActionEmbeddingTail *tail_rows = genome::GenomeTailRows(genome_bytes);

    selected_action.action_index = 0;
    selected_action.word = training_word_catalog.words[0];
    selected_action.score = ScoreDynamicActionEmbedding(policy_vector, selected_action.word, tail_rows[0]);

    for (std::size_t action_index = 1; action_index < action_count; ++action_index) {
        const float score = ScoreDynamicActionEmbedding(policy_vector, training_word_catalog.words[action_index],
                                                        tail_rows[action_index]);
        if (score > selected_action.score) {
            selected_action.action_index = action_index;
            selected_action.word = training_word_catalog.words[action_index];
            selected_action.score = score;
        }
    }

    return true;
}

__device__ inline std::size_t WrapTrainingWordIndex(const std::size_t index, const std::size_t word_count) {
    return (word_count == 0) ? 0 : (index % word_count);
}

__device__ inline DeviceGenomeEvaluationStatusCode
TryInitializePrefilledGrid(const TrainingWordCatalog &training_word_catalog, const Word &solution,
                           const std::size_t first_guess_index, const std::size_t second_guess_index,
                           const std::size_t active_training_word_count, WordleGrid &grid_out) {
    grid_out = MakeWordleGrid(solution);

    const Word first_guess =
        training_word_catalog.words[WrapTrainingWordIndex(first_guess_index, active_training_word_count)];
    const Word second_guess =
        training_word_catalog.words[WrapTrainingWordIndex(second_guess_index, active_training_word_count)];

    if (!TryAppendGuess(grid_out, first_guess) || !TryAppendGuess(grid_out, second_guess)) {
        return DeviceGenomeEvaluationStatusCode::kGuessAppendFailed;
    }

    return DeviceGenomeEvaluationStatusCode::kOk;
}

__device__ inline DeviceGenomeEvaluationStatusCode
TryPlayWordleToCompletion(const std::uint8_t *genome_bytes, const TrainingWordCatalog &training_word_catalog,
                          const std::size_t action_count, WordleGrid &grid, float &episode_score_out) {
    while (!grid.IsFinished()) {
        PolicyVector policy_vector{};
        if (!TryForwardPolicyModel(genome::GenomePolicyModelParameters(genome_bytes), grid, policy_vector)) {
            return DeviceGenomeEvaluationStatusCode::kPolicyForwardFailed;
        }

        SelectedAction selected_action{};
        if (!TrySelectBestDynamicAction(policy_vector, training_word_catalog, genome_bytes, action_count,
                                        selected_action)) {
            return DeviceGenomeEvaluationStatusCode::kActionSelectionFailed;
        }

        if (!TryAppendGuess(grid, selected_action.word)) {
            return DeviceGenomeEvaluationStatusCode::kGuessAppendFailed;
        }
    }

    if (!grid.IsWon()) {
        episode_score_out = 0.0f;
        return DeviceGenomeEvaluationStatusCode::kOk;
    }

    episode_score_out = kWinScoreBase + static_cast<float>(neuroevolution::wordle::kMaxTurnCount - grid.turn_count);
    return DeviceGenomeEvaluationStatusCode::kOk;
}

__device__ inline DeviceGenomeEvaluationStatusCode
TryEvaluateGenomeFitness(const std::uint8_t *genome_bytes, const std::size_t genome_action_count,
                         const device_common::RuntimeWordCounts runtime_word_counts, float &fitness_out) {
    const TrainingWordCatalog &training_word_catalog = DeviceTrainingWordCatalog();
    const std::size_t active_training_word_count = runtime_word_counts.training_word_count;
    const std::size_t selectable_action_count = (genome_action_count < runtime_word_counts.action_space_word_count)
                                                    ? genome_action_count
                                                    : runtime_word_counts.action_space_word_count;

    if (!IsValidTrainingWordCatalog(training_word_catalog) ||
        !IsValidRuntimeWordCounts(training_word_catalog, runtime_word_counts) || (genome_bytes == nullptr) ||
        (genome_action_count == 0) || (active_training_word_count == 0) || (selectable_action_count == 0) ||
        (selectable_action_count > genome_action_count)) {
        return DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard;
    }

    float score_sum = 0.0f;

    for (std::size_t entry_index = 0; entry_index < active_training_word_count; ++entry_index) {
        const Word solution = training_word_catalog.words[entry_index];

        {
            WordleGrid fresh_grid = MakeWordleGrid(solution);
            float episode_score = 0.0f;
            const DeviceGenomeEvaluationStatusCode episode_status = TryPlayWordleToCompletion(
                genome_bytes, training_word_catalog, selectable_action_count, fresh_grid, episode_score);
            if (episode_status != DeviceGenomeEvaluationStatusCode::kOk) {
                return episode_status;
            }

            score_sum += episode_score;
        }

        {
            WordleGrid prefilled_grid{};
            const DeviceGenomeEvaluationStatusCode initialize_status =
                TryInitializePrefilledGrid(training_word_catalog, solution, entry_index + 1, entry_index + 2,
                                           active_training_word_count, prefilled_grid);
            if (initialize_status != DeviceGenomeEvaluationStatusCode::kOk) {
                return initialize_status;
            }

            float episode_score = 0.0f;
            const DeviceGenomeEvaluationStatusCode episode_status = TryPlayWordleToCompletion(
                genome_bytes, training_word_catalog, selectable_action_count, prefilled_grid, episode_score);
            if (episode_status != DeviceGenomeEvaluationStatusCode::kOk) {
                return episode_status;
            }

            score_sum += episode_score;
        }

        {
            WordleGrid prefilled_grid{};
            const DeviceGenomeEvaluationStatusCode initialize_status =
                TryInitializePrefilledGrid(training_word_catalog, solution, entry_index + 3, entry_index + 4,
                                           active_training_word_count, prefilled_grid);
            if (initialize_status != DeviceGenomeEvaluationStatusCode::kOk) {
                return initialize_status;
            }

            float episode_score = 0.0f;
            const DeviceGenomeEvaluationStatusCode episode_status = TryPlayWordleToCompletion(
                genome_bytes, training_word_catalog, selectable_action_count, prefilled_grid, episode_score);
            if (episode_status != DeviceGenomeEvaluationStatusCode::kOk) {
                return episode_status;
            }

            score_sum += episode_score;
        }
    }

    fitness_out = NormalizeFitnessForSelection(score_sum, runtime_word_counts);
    return DeviceGenomeEvaluationStatusCode::kOk;
}

} // namespace neuroevolution::genetic_algorithm::device_evaluation_ops
