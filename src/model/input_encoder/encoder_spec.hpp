#pragma once

#include <cstddef>

#include "common/cuda_compat.hpp"
#include "wordle/turn.hpp"

namespace neuroevolution::model::input_encoder {

constexpr std::size_t kGuessLetterCount = wordle::kWordLength;
constexpr std::size_t kFeedbackCount = wordle::kWordLength;
constexpr std::size_t kAlphabetSize = wordle::kAlphabetSize;
constexpr std::size_t kTileStateCount = 3;

constexpr std::size_t kGuessLetterFeatureCount =
    kGuessLetterCount * kAlphabetSize;
constexpr std::size_t kFeedbackFeatureCount = kFeedbackCount * kTileStateCount;
constexpr std::size_t kTurnFeatureCount =
    kGuessLetterFeatureCount + kFeedbackFeatureCount;

constexpr std::size_t kEncoderHiddenSize = 128;
constexpr std::size_t kEncoderOutputSize = 64;

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
GuessLetterFeatureOffset(const std::size_t position,
                         const std::size_t letter_index) noexcept {
  return (position * kAlphabetSize) + letter_index;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
FeedbackFeatureOffset(const std::size_t position,
                      const std::size_t feedback_index) noexcept {
  return kGuessLetterFeatureCount + (position * kTileStateCount) +
         feedback_index;
}

} // namespace neuroevolution::model::input_encoder
