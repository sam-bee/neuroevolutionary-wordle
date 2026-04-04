#pragma once

#include <cstddef>

namespace neuroevolution::model::input_encoder {

constexpr std::size_t kGuessLetterCount = 5;
constexpr std::size_t kFeedbackCount = 5;
constexpr std::size_t kAlphabetSize = 26;
constexpr std::size_t kTileStateCount = 3;

constexpr std::size_t kGuessLetterFeatureCount =
    kGuessLetterCount * kAlphabetSize;
constexpr std::size_t kFeedbackFeatureCount = kFeedbackCount * kTileStateCount;
constexpr std::size_t kTurnFeatureCount =
    kGuessLetterFeatureCount + kFeedbackFeatureCount;

constexpr std::size_t kEncoderHiddenSize = 128;
constexpr std::size_t kEncoderOutputSize = 64;

constexpr std::size_t kTurnSlotCount = 5;

constexpr std::size_t
GuessLetterFeatureOffset(const std::size_t position,
                         const std::size_t letter_index) noexcept {
  return (position * kAlphabetSize) + letter_index;
}

constexpr std::size_t
FeedbackFeatureOffset(const std::size_t position,
                      const std::size_t feedback_index) noexcept {
  return kGuessLetterFeatureCount + (position * kTileStateCount) +
         feedback_index;
}

} // namespace neuroevolution::model::input_encoder
