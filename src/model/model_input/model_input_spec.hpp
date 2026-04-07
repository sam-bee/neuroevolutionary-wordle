#pragma once

#include <cstddef>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "model/input_encoder/encoder_spec.hpp"

namespace neuroevolution::model::model_input {

constexpr std::size_t kModelInputTurnCount = 5;
constexpr std::size_t kModelInputVectorSize = kModelInputTurnCount * input_encoder::kEncoderOutputSize;

using ModelInputStateVector = common::FixedBuffer<float, kModelInputVectorSize>;

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t ModelInputTurnOffset(const std::size_t turn_index) noexcept {
    return turn_index * input_encoder::kEncoderOutputSize;
}

} // namespace neuroevolution::model::model_input
