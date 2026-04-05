#include "model/input_encoder/shared_encoder.hpp"

#include <stdexcept>

namespace neuroevolution::model::input_encoder {

EncodedTurnVector ForwardOccupiedTurn(const SharedEncoderParameters &parameters,
                                      const wordle::Turn &turn) {
  EncodedTurnVector encoded_turn{};
  if (!detail::TryForwardOccupiedTurn(parameters, turn, encoded_turn)) {
    throw std::invalid_argument(
        "Turn contains invalid letter indices or feedback values.");
  }

  return encoded_turn;
}

} // namespace neuroevolution::model::input_encoder
