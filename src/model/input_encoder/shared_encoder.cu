#include "model/input_encoder/shared_encoder.hpp"

namespace neuroevolution::model::input_encoder {
namespace {

template <std::size_t InputSize, std::size_t OutputSize>
std::array<float, OutputSize>
ApplyDenseLayer(const DenseLayerParameters<InputSize, OutputSize> &layer,
                const std::array<float, InputSize> &input) {
  std::array<float, OutputSize> output{};

  for (std::size_t output_index = 0; output_index < OutputSize;
       ++output_index) {
    float sum = layer.biases[output_index];

    for (std::size_t input_index = 0; input_index < InputSize; ++input_index) {
      sum += layer.WeightAt(output_index, input_index) * input[input_index];
    }

    output[output_index] = sum;
  }

  return output;
}

void ApplyReLU(std::array<float, kEncoderHiddenSize> &activations) {
  for (float &activation : activations) {
    if (activation < 0.0f) {
      activation = 0.0f;
    }
  }
}

} // namespace

EncodedTurnVector
ForwardSharedEncoder(const SharedEncoderParameters &parameters,
                     const TurnInputVector &input_vector) {
  auto hidden = ApplyDenseLayer(parameters.input_to_hidden, input_vector);
  ApplyReLU(hidden);
  return ApplyDenseLayer(parameters.hidden_to_output, hidden);
}

EncodedTurnVector ForwardOccupiedTurn(const SharedEncoderParameters &parameters,
                                      const wordle::Turn &turn) {
  const TurnInputFeatures features = EncodeTurnFeatures(turn);
  const TurnInputVector input_vector = MaterializeTurnInput(features);
  return ForwardSharedEncoder(parameters, input_vector);
}

EncodedTurnVector EmptyTurnEncoding() noexcept { return EncodedTurnVector{}; }

} // namespace neuroevolution::model::input_encoder
