#pragma once

#include <array>
#include <cstddef>

#include "model/input_encoder/encoder_spec.hpp"
#include "model/input_encoder/turn_features.hpp"

namespace neuroevolution::model::input_encoder {

template <std::size_t InputSize, std::size_t OutputSize>
struct DenseLayerParameters {
  // Weights are stored row-major by output neuron, then input index.
  std::array<float, InputSize * OutputSize> weights{};
  std::array<float, OutputSize> biases{};

  constexpr float WeightAt(const std::size_t output_index,
                           const std::size_t input_index) const noexcept {
    return weights[(output_index * InputSize) + input_index];
  }
};

struct SharedEncoderParameters {
  DenseLayerParameters<kTurnFeatureCount, kEncoderHiddenSize> input_to_hidden{};
  DenseLayerParameters<kEncoderHiddenSize, kEncoderOutputSize>
      hidden_to_output{};
};

using EncodedTurnVector = std::array<float, kEncoderOutputSize>;

EncodedTurnVector
ForwardSharedEncoder(const SharedEncoderParameters &parameters,
                     const TurnInputVector &input_vector);

EncodedTurnVector ForwardOccupiedTurn(const SharedEncoderParameters &parameters,
                                      const wordle::Turn &turn);

EncodedTurnVector EmptyTurnEncoding() noexcept;

} // namespace neuroevolution::model::input_encoder
