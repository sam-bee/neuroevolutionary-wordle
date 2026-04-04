#pragma once

#include <array>
#include <cstdint>

#include "model/input_encoder/encoder_spec.hpp"
#include "wordle/turn.hpp"

namespace neuroevolution::model::input_encoder {

struct TurnInputFeatures {
  // Layout is fixed as:
  // [5 x 26 guess-letter one-hot | 5 x 3 feedback one-hot].
  std::array<std::uint8_t, kTurnFeatureCount> discrete{};
};

using TurnInputVector = std::array<float, kTurnFeatureCount>;

TurnInputFeatures EncodeTurnFeatures(const wordle::Turn &turn);

TurnInputVector MaterializeTurnInput(const TurnInputFeatures &features);

} // namespace neuroevolution::model::input_encoder
