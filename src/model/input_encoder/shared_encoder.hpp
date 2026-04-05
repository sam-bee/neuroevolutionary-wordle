#pragma once

#include <cstddef>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "model/input_encoder/encoder_spec.hpp"
#include "model/input_encoder/turn_features.hpp"

namespace neuroevolution::model::input_encoder {

template <std::size_t InputSize, std::size_t OutputSize>
struct DenseLayerParameters {
  // Weights are stored row-major by output neuron, then input index.
  common::FixedBuffer<float, InputSize * OutputSize> weights{};
  common::FixedBuffer<float, OutputSize> biases{};

  constexpr NEUROEVOLUTION_HOST_DEVICE float
  WeightAt(const std::size_t output_index,
           const std::size_t input_index) const noexcept {
    return weights[(output_index * InputSize) + input_index];
  }
};

struct SharedEncoderParameters {
  DenseLayerParameters<kTurnFeatureCount, kEncoderHiddenSize> input_to_hidden{};
  DenseLayerParameters<kEncoderHiddenSize, kEncoderOutputSize>
      hidden_to_output{};
};

using EncoderHiddenVector = common::FixedBuffer<float, kEncoderHiddenSize>;
using EncodedTurnVector = common::FixedBuffer<float, kEncoderOutputSize>;

template <std::size_t InputSize, std::size_t OutputSize>
inline NEUROEVOLUTION_HOST_DEVICE void
ApplyDenseLayer(const DenseLayerParameters<InputSize, OutputSize> &layer,
                const common::FixedBuffer<float, InputSize> &input,
                common::FixedBuffer<float, OutputSize> &output) noexcept {
  for (std::size_t output_index = 0; output_index < OutputSize; ++output_index) {
    float sum = layer.biases[output_index];

    for (std::size_t input_index = 0; input_index < InputSize; ++input_index) {
      sum += layer.WeightAt(output_index, input_index) * input[input_index];
    }

    output[output_index] = sum;
  }
}

inline NEUROEVOLUTION_HOST_DEVICE void
ApplyReLU(EncoderHiddenVector &activations) noexcept {
  for (std::size_t activation_index = 0;
       activation_index < kEncoderHiddenSize; ++activation_index) {
    if (activations[activation_index] < 0.0f) {
      activations[activation_index] = 0.0f;
    }
  }
}

EncodedTurnVector
ForwardSharedEncoder(const SharedEncoderParameters &parameters,
                     const TurnInputVector &input_vector);

inline NEUROEVOLUTION_HOST_DEVICE bool
TryForwardOccupiedTurn(const SharedEncoderParameters &parameters,
                       const wordle::Turn &turn,
                       EncodedTurnVector &encoded_turn) noexcept {
  TurnInputFeatures features{};
  if (!TryEncodeTurnFeatures(turn, features)) {
    return false;
  }

  TurnInputVector input_vector{};
  MaterializeTurnInputInPlace(features, input_vector);
  EncoderHiddenVector hidden{};
  ApplyDenseLayer(parameters.input_to_hidden, input_vector, hidden);
  ApplyReLU(hidden);
  ApplyDenseLayer(parameters.hidden_to_output, hidden, encoded_turn);
  return true;
}

EncodedTurnVector ForwardOccupiedTurn(const SharedEncoderParameters &parameters,
                                      const wordle::Turn &turn);

} // namespace neuroevolution::model::input_encoder
