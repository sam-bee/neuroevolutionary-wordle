#include "model/input_encoder/shared_encoder.hpp"

#include <stdexcept>

namespace neuroevolution::model::input_encoder {

EncodedTurnVector
ForwardSharedEncoder(const SharedEncoderParameters &parameters,
                     const TurnInputVector &input_vector) {
  EncodedTurnVector encoded_turn{};
  EncoderHiddenVector hidden{};
  ApplyDenseLayer(parameters.input_to_hidden, input_vector, hidden);
  ApplyReLU(hidden);
  ApplyDenseLayer(parameters.hidden_to_output, hidden, encoded_turn);
  return encoded_turn;
}

EncodedTurnVector ForwardOccupiedTurn(const SharedEncoderParameters &parameters,
                                      const wordle::Turn &turn) {
  EncodedTurnVector encoded_turn{};
  if (!TryForwardOccupiedTurn(parameters, turn, encoded_turn)) {
    throw std::invalid_argument(
        "Turn contains invalid letter indices or feedback values.");
  }

  return encoded_turn;
}

} // namespace neuroevolution::model::input_encoder
