#pragma once

#include <cstddef>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "common/float16.hpp"
#include "genetic_algorithm/genome.hpp"
#include "wordle/hint_grid.hpp"

namespace neuroevolution::genetic_algorithm {

inline NEUROEVOLUTION_HOST_DEVICE bool
TrySeedOutputEmbeddingTailFromHintGrids(const model::policy_model::PolicyModelParameters &policy_model,
                                        const wordle::Word &target_word,
                                        model::output_embedding::TrainableActionEmbeddingTail &tail_out) noexcept {
    tail_out = {};

    if (!wordle::IsValidWord(target_word)) {
        return false;
    }

    wordle::HintGridGroup hint_grid_group{};
    if (!wordle::TryBuildHintGridGroup(target_word, hint_grid_group)) {
        return false;
    }

    common::FixedBuffer<float, model::output_embedding::kTrainableFeatureDimension> feature_sums{};

    for (std::size_t grid_index = 0; grid_index < wordle::kHintGridGroupSize; ++grid_index) {
        model::policy_model::PolicyVector policy_vector{};
        if (!model::policy_model::TryForwardPolicyModel(policy_model, hint_grid_group.grids[grid_index],
                                                        policy_vector)) {
            return false;
        }

        for (std::size_t feature_index = 0; feature_index < model::output_embedding::kTrainableFeatureDimension;
             ++feature_index) {
            feature_sums[feature_index] +=
                policy_vector[model::output_embedding::kWordFeatureDimension + feature_index];
        }
    }

    constexpr float kReciprocalHintGridCount = 1.0f / static_cast<float>(wordle::kHintGridGroupSize);
    for (std::size_t feature_index = 0; feature_index < model::output_embedding::kTrainableFeatureDimension;
         ++feature_index) {
        tail_out[feature_index] = common::ToFloat16(feature_sums[feature_index] * kReciprocalHintGridCount);
    }

    return true;
}

template <std::size_t ActionCapacity>
inline NEUROEVOLUTION_HOST_DEVICE bool TryInjectNewOutputEmbedding(ModelGenome<ActionCapacity> &genome,
                                                                   const wordle::Word &target_word) noexcept {
    if (!IsValidModelGenome(genome)) {
        return false;
    }

    const std::size_t append_index = ActiveOutputEmbeddingCount(genome.output_embedding);
    if (append_index >= ActionCapacity) {
        return false;
    }

    model::output_embedding::TrainableActionEmbeddingTail injected_tail{};
    if (!TrySeedOutputEmbeddingTailFromHintGrids(genome.policy_model, target_word, injected_tail)) {
        return false;
    }

    genome.output_embedding.trainable_tails[append_index] = injected_tail;
    genome.output_embedding.active_count = append_index + 1;
    return true;
}

} // namespace neuroevolution::genetic_algorithm
