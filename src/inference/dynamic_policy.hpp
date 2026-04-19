#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "model/policy_model/policy_model.hpp"
#include "training_folder/training_data.hpp"
#include "wordle/wordle_grid.hpp"

namespace neuroevolution::inference::dynamic_policy {

using neuroevolution::model::output_embedding::ActionEmbedding;
using neuroevolution::model::output_embedding::ScoreActionEmbedding;
using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::model::policy_model::PolicyVector;
using neuroevolution::model::policy_model::TryForwardPolicyModel;
using neuroevolution::training_folder::IsValidTrainingWordCatalog;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::wordle::Word;
using neuroevolution::wordle::WordleGrid;

enum class DynamicInferenceStatusCode : int {
    kOk = 0,
    kPolicyForwardFailed = 1,
    kActionSelectionFailed = 2,
};

inline NEUROEVOLUTION_HOST_DEVICE float ScoreDynamicActionEmbedding(
    const PolicyVector &policy_vector, const Word &action_word,
    const genetic_algorithm::genome::TrainableActionEmbeddingTail &trainable_tail) noexcept {
    ActionEmbedding action_embedding{};
    action_embedding.word = action_word;
    action_embedding.trainable_tail = trainable_tail;
    return ScoreActionEmbedding(policy_vector, action_embedding);
}

inline NEUROEVOLUTION_HOST_DEVICE bool TrySelectBestDynamicAction(const PolicyVector &policy_vector,
                                                                  const TrainingWordCatalog &training_word_catalog,
                                                                  const std::uint8_t *genome_bytes,
                                                                  const std::size_t action_count,
                                                                  SelectedAction &selected_action) noexcept {
    if (!IsValidTrainingWordCatalog(training_word_catalog) || (genome_bytes == nullptr) || (action_count == 0) ||
        (action_count > training_word_catalog.word_count)) {
        return false;
    }

    const genetic_algorithm::genome::TrainableActionEmbeddingTail *tail_rows =
        genetic_algorithm::genome::GenomeTailRows(genome_bytes);

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

inline NEUROEVOLUTION_HOST_DEVICE DynamicInferenceStatusCode SelectNextGuessFromDynamicGenome(
    const WordleGrid &grid, const TrainingWordCatalog &training_word_catalog, const std::uint8_t *genome_bytes,
    const std::size_t action_count, SelectedAction &selected_action) noexcept {
    if ((genome_bytes == nullptr) || (action_count == 0)) {
        return DynamicInferenceStatusCode::kActionSelectionFailed;
    }

    PolicyVector policy_vector{};
    if (!TryForwardPolicyModel(genetic_algorithm::genome::GenomePolicyModelParameters(genome_bytes), grid,
                               policy_vector)) {
        return DynamicInferenceStatusCode::kPolicyForwardFailed;
    }

    return TrySelectBestDynamicAction(policy_vector, training_word_catalog, genome_bytes, action_count,
                                      selected_action)
               ? DynamicInferenceStatusCode::kOk
               : DynamicInferenceStatusCode::kActionSelectionFailed;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TrySelectNextGuessFromDynamicGenome(
    const WordleGrid &grid, const TrainingWordCatalog &training_word_catalog, const std::uint8_t *genome_bytes,
    const std::size_t action_count, SelectedAction &selected_action) noexcept {
    return SelectNextGuessFromDynamicGenome(grid, training_word_catalog, genome_bytes, action_count,
                                            selected_action) == DynamicInferenceStatusCode::kOk;
}

} // namespace neuroevolution::inference::dynamic_policy
