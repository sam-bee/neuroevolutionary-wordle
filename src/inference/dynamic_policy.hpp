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

constexpr int kDynamicPolicyThreadsPerBlock = 128;

enum class DynamicInferenceStatusCode : int {
    kOk = 0,
    kPolicyForwardFailed = 1,
    kActionSelectionFailed = 2,
};

template <int ThreadsPerBlock> struct DynamicPolicyBlockScratch {
    PolicyVector policy_vector{};
    SelectedAction best_actions[ThreadsPerBlock]{};
    int has_candidate[ThreadsPerBlock]{};
    int status = static_cast<int>(DynamicInferenceStatusCode::kActionSelectionFailed);
};

inline NEUROEVOLUTION_HOST_DEVICE float ScoreDynamicActionEmbedding(
    const PolicyVector &policy_vector, const Word &action_word,
    const genetic_algorithm::genome::TrainableActionEmbeddingTail &trainable_tail) noexcept {
    ActionEmbedding action_embedding{};
    action_embedding.word = action_word;
    action_embedding.trainable_tail = trainable_tail;
    return ScoreActionEmbedding(policy_vector, action_embedding);
}

inline NEUROEVOLUTION_HOST_DEVICE bool HasGridAlreadyGuessedWord(const WordleGrid &grid,
                                                                 const Word &candidate_word) noexcept {
    if (!wordle::IsValidWordleGrid(grid) || !wordle::IsValidWord(candidate_word)) {
        return false;
    }

    for (std::size_t turn_index = 0; turn_index < grid.turn_count; ++turn_index) {
        if (grid.turns[turn_index].guess == candidate_word) {
            return true;
        }
    }

    return false;
}

template <int ThreadsPerBlock>
inline __device__ DynamicInferenceStatusCode SelectNextGuessFromDynamicGenomeConcurrently(
    const WordleGrid &grid, const TrainingWordCatalog &training_word_catalog, const std::uint8_t *genome_bytes,
    const std::size_t action_count, DynamicPolicyBlockScratch<ThreadsPerBlock> &scratch,
    SelectedAction &selected_action) noexcept {
    if (!IsValidTrainingWordCatalog(training_word_catalog) || (genome_bytes == nullptr) || (action_count == 0) ||
        (action_count > training_word_catalog.word_count) || !wordle::IsValidWordleGrid(grid) ||
        (blockDim.x != ThreadsPerBlock)) {
        if (threadIdx.x == 0) {
            scratch.status = static_cast<int>(DynamicInferenceStatusCode::kActionSelectionFailed);
        }
        __syncthreads();
        return DynamicInferenceStatusCode::kActionSelectionFailed;
    }

    if (threadIdx.x == 0) {
        if (!TryForwardPolicyModel(genetic_algorithm::genome::GenomePolicyModelParameters(genome_bytes), grid,
                                   scratch.policy_vector)) {
            scratch.status = static_cast<int>(DynamicInferenceStatusCode::kPolicyForwardFailed);
        } else {
            scratch.status = static_cast<int>(DynamicInferenceStatusCode::kOk);
        }
    }

    __syncthreads();

    if (scratch.status != static_cast<int>(DynamicInferenceStatusCode::kOk)) {
        return static_cast<DynamicInferenceStatusCode>(scratch.status);
    }

    const genetic_algorithm::genome::TrainableActionEmbeddingTail *tail_rows =
        genetic_algorithm::genome::GenomeTailRows(genome_bytes);

    SelectedAction local_best_action{};
    bool has_local_candidate = false;
    for (std::size_t action_index = static_cast<std::size_t>(threadIdx.x); action_index < action_count;
         action_index += static_cast<std::size_t>(blockDim.x)) {
        const Word &candidate_word = training_word_catalog.words[action_index];
        if (HasGridAlreadyGuessedWord(grid, candidate_word)) {
            continue;
        }

        const float score = ScoreDynamicActionEmbedding(scratch.policy_vector, candidate_word, tail_rows[action_index]);
        if (!has_local_candidate || (score > local_best_action.score) ||
            ((score == local_best_action.score) && (action_index < local_best_action.action_index))) {
            local_best_action.action_index = action_index;
            local_best_action.word = candidate_word;
            local_best_action.score = score;
            has_local_candidate = true;
        }
    }

    scratch.has_candidate[threadIdx.x] = has_local_candidate ? 1 : 0;
    if (has_local_candidate) {
        scratch.best_actions[threadIdx.x] = local_best_action;
    }

    __syncthreads();

    for (int offset = (ThreadsPerBlock / 2); offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            const int peer_index = threadIdx.x + offset;
            if ((scratch.has_candidate[peer_index] != 0) &&
                ((scratch.has_candidate[threadIdx.x] == 0) ||
                 (scratch.best_actions[peer_index].score > scratch.best_actions[threadIdx.x].score) ||
                 ((scratch.best_actions[peer_index].score == scratch.best_actions[threadIdx.x].score) &&
                  (scratch.best_actions[peer_index].action_index < scratch.best_actions[threadIdx.x].action_index)))) {
                scratch.best_actions[threadIdx.x] = scratch.best_actions[peer_index];
                scratch.has_candidate[threadIdx.x] = 1;
            }
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (scratch.has_candidate[0] == 0) {
            scratch.status = static_cast<int>(DynamicInferenceStatusCode::kActionSelectionFailed);
        } else {
            selected_action = scratch.best_actions[0];
            scratch.status = static_cast<int>(DynamicInferenceStatusCode::kOk);
        }
    }

    __syncthreads();
    return static_cast<DynamicInferenceStatusCode>(scratch.status);
}

} // namespace neuroevolution::inference::dynamic_policy
