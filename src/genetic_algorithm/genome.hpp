#pragma once

#include <cstddef>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "model/policy_model/policy_model.hpp"

namespace neuroevolution::genetic_algorithm {

template <std::size_t ActionCount> struct OutputEmbeddingGenome {
    common::FixedBuffer<model::output_embedding::TrainableActionEmbeddingTail, ActionCount> trainable_tails{};
};

template <std::size_t ActionCount> struct ModelGenome {
    model::policy_model::PolicyModelParameters policy_model{};
    OutputEmbeddingGenome<ActionCount> output_embedding{};
};

template <std::size_t ActionCount>
inline NEUROEVOLUTION_HOST_DEVICE bool TryMaterializeActionEmbeddings(
    const OutputEmbeddingGenome<ActionCount> &genome,
    const common::FixedBuffer<wordle::Word, ActionCount> &action_words,
    common::FixedBuffer<model::output_embedding::ActionEmbedding, ActionCount> &action_embeddings) noexcept {
    for (std::size_t action_index = 0; action_index < ActionCount; ++action_index) {
        if (!wordle::IsValidWord(action_words[action_index])) {
            return false;
        }

        action_embeddings[action_index].word = action_words[action_index];

        for (std::size_t feature_index = 0; feature_index < model::output_embedding::kTrainableFeatureDimension;
             ++feature_index) {
            action_embeddings[action_index].trainable_tail[feature_index] =
                genome.trainable_tails[action_index][feature_index];
        }
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm
