#pragma once

#include <cstdint>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "model/input_encoder/encoder_spec.hpp"
#include "wordle/turn.hpp"

namespace neuroevolution::model::input_encoder {

struct TurnInputFeatures {
    // Layout is fixed as:
    // [5 x 26 guess-letter one-hot | 5 x 3 feedback one-hot].
    common::FixedBuffer<std::uint8_t, kTurnFeatureCount> discrete{};
};

using TurnInputVector = common::FixedBuffer<float, kTurnFeatureCount>;

namespace detail {

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t FeedbackIndex(const wordle::TileFeedback feedback) noexcept {
    return static_cast<std::size_t>(feedback);
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryEncodeTurnFeatures(const wordle::Turn &turn,
                                                             TurnInputFeatures &features) noexcept {
    if (!wordle::IsValidTurn(turn)) {
        return false;
    }

    for (std::size_t feature_index = 0; feature_index < kTurnFeatureCount; ++feature_index) {
        features.discrete[feature_index] = 0u;
    }

    for (std::size_t position = 0; position < wordle::kWordLength; ++position) {
        const std::size_t letter_index = turn.guess.letter_indices[position];
        const std::size_t letter_feature_index = GuessLetterFeatureOffset(position, letter_index);
        features.discrete[letter_feature_index] = 1u;

        const std::size_t feedback_feature_index =
            FeedbackFeatureOffset(position, detail::FeedbackIndex(turn.feedback[position]));
        features.discrete[feedback_feature_index] = 1u;
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE void MaterializeTurnInputInPlace(const TurnInputFeatures &features,
                                                                   TurnInputVector &materialized) noexcept {
    for (std::size_t feature_index = 0; feature_index < kTurnFeatureCount; ++feature_index) {
        materialized[feature_index] = static_cast<float>(features.discrete[feature_index]);
    }
}

} // namespace detail

TurnInputFeatures EncodeTurnFeatures(const wordle::Turn &turn);

} // namespace neuroevolution::model::input_encoder
