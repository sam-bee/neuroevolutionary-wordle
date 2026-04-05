#include "model/input_encoder/turn_features.hpp"

#include <stdexcept>

namespace neuroevolution::model::input_encoder {

TurnInputFeatures EncodeTurnFeatures(const wordle::Turn &turn) {
  TurnInputFeatures features{};

  if (!detail::TryEncodeTurnFeatures(turn, features)) {
    throw std::invalid_argument(
        "Turn contains invalid letter indices or feedback values.");
  }

  return features;
}

} // namespace neuroevolution::model::input_encoder
