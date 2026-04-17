#pragma once

#include <cstddef>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "model/policy_model/policy_model.hpp"

namespace neuroevolution::genetic_algorithm {

template <std::size_t ActionCapacity> struct OutputEmbeddingGenome {
    std::size_t active_count = ActionCapacity;
    common::FixedBuffer<model::output_embedding::TrainableActionEmbeddingTail, ActionCapacity> trainable_tails{};
};

template <std::size_t ActionCapacity>
constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidOutputEmbeddingGenome(const OutputEmbeddingGenome<ActionCapacity> &genome) noexcept {
    return genome.active_count <= ActionCapacity;
}

template <std::size_t ActionCapacity>
constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
ActiveOutputEmbeddingCount(const OutputEmbeddingGenome<ActionCapacity> &genome) noexcept {
    return IsValidOutputEmbeddingGenome(genome) ? genome.active_count : 0;
}

template <std::size_t ActionCapacity>
inline NEUROEVOLUTION_HOST_DEVICE bool TryResizeOutputEmbeddingGenome(OutputEmbeddingGenome<ActionCapacity> &genome,
                                                                      const std::size_t active_count) noexcept {
    if (active_count > ActionCapacity) {
        return false;
    }

    genome.active_count = active_count;
    return true;
}

template <std::size_t ActionCapacity> struct ModelGenome {
    model::policy_model::PolicyModelParameters policy_model{};
    OutputEmbeddingGenome<ActionCapacity> output_embedding{};
};

template <std::size_t ActionCapacity>
constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidModelGenome(const ModelGenome<ActionCapacity> &genome) noexcept {
    return IsValidOutputEmbeddingGenome(genome.output_embedding);
}

template <std::size_t ActionCapacity>
inline NEUROEVOLUTION_HOST_DEVICE bool TryMaterializeActionEmbeddings(
    const OutputEmbeddingGenome<ActionCapacity> &genome,
    const common::FixedBuffer<wordle::Word, ActionCapacity> &action_words,
    common::FixedBuffer<model::output_embedding::ActionEmbedding, ActionCapacity> &action_embeddings,
    const std::size_t action_count) noexcept {
    if (!IsValidOutputEmbeddingGenome(genome) || (action_count > ActionCapacity) ||
        (action_count > genome.active_count)) {
        return false;
    }

    for (std::size_t action_index = 0; action_index < action_count; ++action_index) {
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

template <std::size_t ActionCapacity>
inline NEUROEVOLUTION_HOST_DEVICE bool TryMaterializeActionEmbeddings(
    const OutputEmbeddingGenome<ActionCapacity> &genome,
    const common::FixedBuffer<wordle::Word, ActionCapacity> &action_words,
    common::FixedBuffer<model::output_embedding::ActionEmbedding, ActionCapacity> &action_embeddings) noexcept {
    return TryMaterializeActionEmbeddings(genome, action_words, action_embeddings, ActiveOutputEmbeddingCount(genome));
}

} // namespace neuroevolution::genetic_algorithm
