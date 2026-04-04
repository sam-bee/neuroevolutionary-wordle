#include "model/input_encoder/turn_features.hpp"

#include <stdexcept>

namespace neuroevolution::model::input_encoder {
namespace {

constexpr std::size_t FeedbackIndex(const wordle::TileFeedback feedback) {
  switch (feedback) {
  case wordle::TileFeedback::green:
    return 0;
  case wordle::TileFeedback::yellow:
    return 1;
  case wordle::TileFeedback::grey:
    return 2;
  }

  return 0;
}

} // namespace

TurnInputFeatures EncodeTurnFeatures(const wordle::Turn &turn) {
  if (!wordle::IsValidTurn(turn)) {
    throw std::invalid_argument(
        "Turn contains invalid letter indices or feedback values.");
  }

  TurnInputFeatures features{};

  for (std::size_t position = 0; position < wordle::kWordLength; ++position) {
    const std::size_t letter_index = turn.letter_indices[position];
    const std::size_t letter_feature_index =
        GuessLetterFeatureOffset(position, letter_index);
    features.discrete[letter_feature_index] = 1u;

    const std::size_t feedback_feature_index =
        FeedbackFeatureOffset(position, FeedbackIndex(turn.feedback[position]));
    features.discrete[feedback_feature_index] = 1u;
  }

  return features;
}

TurnInputVector MaterializeTurnInput(const TurnInputFeatures &features) {
  TurnInputVector materialized{};

  for (std::size_t feature_index = 0; feature_index < kTurnFeatureCount;
       ++feature_index) {
    materialized[feature_index] =
        static_cast<float>(features.discrete[feature_index]);
  }

  return materialized;
}

} // namespace neuroevolution::model::input_encoder
