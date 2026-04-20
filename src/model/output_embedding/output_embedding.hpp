#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "common/float16.hpp"
#include "model/dense_trunk/dense_trunk.hpp"
#include "wordle/word.hpp"

namespace neuroevolution::model::output_embedding {

using ParameterScalar = common::Float16;
static_assert(sizeof(ParameterScalar) == 2, "Output embedding trainable parameters must be stored in fp16.");

using PolicyVector = dense_trunk::PolicyVector;

constexpr std::size_t kOutputEmbeddingDimension = dense_trunk::kDenseTrunkOutputSize;
constexpr std::size_t kWordFeatureDimension = wordle::kAlphabetSize;
constexpr std::size_t kTrainableFeatureDimension = kOutputEmbeddingDimension - kWordFeatureDimension;

static_assert(kOutputEmbeddingDimension > kWordFeatureDimension,
              "Output embedding dimension must leave room for trainable features.");

using ActionEmbeddingVector = common::FixedBuffer<float, kOutputEmbeddingDimension>;
using FixedWordFeatureVector = common::FixedBuffer<float, kWordFeatureDimension>;
using TrainableActionEmbeddingTail = common::FixedBuffer<ParameterScalar, kTrainableFeatureDimension>;
using TrainableActionEmbeddingTailVector = common::FixedBuffer<float, kTrainableFeatureDimension>;

struct ActionEmbedding {
    wordle::Word word{};
    TrainableActionEmbeddingTail trainable_tail{};
};

struct SelectedAction {
    wordle::Word word{};
    float score = 0.0f;
    std::size_t action_index = 0;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidActionEmbedding(const ActionEmbedding &action_embedding) noexcept {
    return wordle::IsValidWord(action_embedding.word);
}

namespace detail {

inline NEUROEVOLUTION_HOST_DEVICE void
CountWordLetters(const wordle::Word &word,
                 common::FixedBuffer<std::uint8_t, wordle::kAlphabetSize> &letter_counts) noexcept {
    for (std::size_t letter_index = 0; letter_index < wordle::kAlphabetSize; ++letter_index) {
        letter_counts[letter_index] = 0;
    }

    for (std::size_t position = 0; position < wordle::kWordLength; ++position) {
        ++letter_counts[word.letter_indices[position]];
    }
}

inline NEUROEVOLUTION_HOST_DEVICE float FixedWordFeatureValue(const std::uint8_t letter_count) noexcept {
    return (letter_count == 0) ? -1.0f : static_cast<float>(letter_count);
}

} // namespace detail

inline NEUROEVOLUTION_HOST_DEVICE void MaterializeFixedWordFeatures(const wordle::Word &word,
                                                                    FixedWordFeatureVector &fixed_features) noexcept {
    common::FixedBuffer<std::uint8_t, wordle::kAlphabetSize> letter_counts{};
    detail::CountWordLetters(word, letter_counts);

    for (std::size_t letter_index = 0; letter_index < kWordFeatureDimension; ++letter_index) {
        fixed_features[letter_index] = detail::FixedWordFeatureValue(letter_counts[letter_index]);
    }
}

inline NEUROEVOLUTION_HOST_DEVICE void MaterializeActionEmbedding(const ActionEmbedding &action_embedding,
                                                                  ActionEmbeddingVector &embedding_vector) noexcept {
    FixedWordFeatureVector fixed_features{};
    MaterializeFixedWordFeatures(action_embedding.word, fixed_features);

    for (std::size_t letter_index = 0; letter_index < kWordFeatureDimension; ++letter_index) {
        embedding_vector[letter_index] = fixed_features[letter_index];
    }

    for (std::size_t trainable_index = 0; trainable_index < kTrainableFeatureDimension; ++trainable_index) {
        embedding_vector[kWordFeatureDimension + trainable_index] =
            common::ToFloat(action_embedding.trainable_tail[trainable_index]);
    }
}

inline NEUROEVOLUTION_HOST_DEVICE ActionEmbeddingVector
MaterializeActionEmbedding(const ActionEmbedding &action_embedding) noexcept {
    ActionEmbeddingVector embedding_vector{};
    MaterializeActionEmbedding(action_embedding, embedding_vector);
    return embedding_vector;
}

inline NEUROEVOLUTION_HOST_DEVICE float ScoreActionEmbedding(const PolicyVector &policy_vector,
                                                             const FixedWordFeatureVector &fixed_word_features,
                                                             const TrainableActionEmbeddingTail &trainable_tail) noexcept {
    float score = 0.0f;

    for (std::size_t letter_index = 0; letter_index < kWordFeatureDimension; ++letter_index) {
        score += policy_vector[letter_index] * fixed_word_features[letter_index];
    }

    for (std::size_t trainable_index = 0; trainable_index < kTrainableFeatureDimension; ++trainable_index) {
        score += policy_vector[kWordFeatureDimension + trainable_index] * common::ToFloat(trainable_tail[trainable_index]);
    }

    return score;
}

inline NEUROEVOLUTION_HOST_DEVICE float ScoreActionEmbedding(const PolicyVector &policy_vector,
                                                             const FixedWordFeatureVector &fixed_word_features,
                                                             const TrainableActionEmbeddingTailVector &trainable_tail) noexcept {
    float score = 0.0f;

    for (std::size_t letter_index = 0; letter_index < kWordFeatureDimension; ++letter_index) {
        score += policy_vector[letter_index] * fixed_word_features[letter_index];
    }

    for (std::size_t trainable_index = 0; trainable_index < kTrainableFeatureDimension; ++trainable_index) {
        score += policy_vector[kWordFeatureDimension + trainable_index] * trainable_tail[trainable_index];
    }

    return score;
}

inline NEUROEVOLUTION_HOST_DEVICE float ScoreActionEmbedding(const PolicyVector &policy_vector,
                                                             const ActionEmbedding &action_embedding) noexcept {
    FixedWordFeatureVector fixed_word_features{};
    MaterializeFixedWordFeatures(action_embedding.word, fixed_word_features);
    return ScoreActionEmbedding(policy_vector, fixed_word_features, action_embedding.trainable_tail);
}

inline NEUROEVOLUTION_HOST_DEVICE bool TrySelectBestAction(const PolicyVector &policy_vector,
                                                           const ActionEmbedding *action_embeddings,
                                                           const std::size_t action_count,
                                                           SelectedAction &selected_action) noexcept {
    if ((action_embeddings == nullptr) || (action_count == 0) || !IsValidActionEmbedding(action_embeddings[0])) {
        return false;
    }

    selected_action.action_index = 0;
    selected_action.word = action_embeddings[0].word;
    selected_action.score = ScoreActionEmbedding(policy_vector, action_embeddings[0]);

    for (std::size_t action_index = 1; action_index < action_count; ++action_index) {
        if (!IsValidActionEmbedding(action_embeddings[action_index])) {
            return false;
        }

        const float score = ScoreActionEmbedding(policy_vector, action_embeddings[action_index]);
        if (score > selected_action.score) {
            selected_action.action_index = action_index;
            selected_action.word = action_embeddings[action_index].word;
            selected_action.score = score;
        }
    }

    return true;
}

wordle::Word SelectBestActionWord(const PolicyVector &policy_vector, const ActionEmbedding *action_embeddings,
                                  std::size_t action_count);

} // namespace neuroevolution::model::output_embedding
