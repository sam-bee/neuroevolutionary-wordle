#pragma once

#include <cstddef>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "model/input_encoder/encoder_spec.hpp"
#include "model/input_encoder/shared_encoder.hpp"
#include "wordle/wordle_grid.hpp"

namespace neuroevolution::model::input_encoder {

using ModelInputStateVector = common::FixedBuffer<float, kModelInputVectorSize>;

namespace detail {

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidModelInputStateTurnCount(const std::size_t turn_count) noexcept {
    // WordleGrid can hold 6 turns, but the policy input only represents the pre-next-guess decision state,
    // so this encoder intentionally stops at the first 5 chronological turns.
    return turn_count <= kModelInputTurnCount;
}

inline NEUROEVOLUTION_HOST_DEVICE void ZeroModelInputState(ModelInputStateVector &model_input_state) noexcept {
    for (std::size_t value_index = 0; value_index < kModelInputVectorSize; ++value_index) {
        model_input_state[value_index] = 0.0f;
    }
}

inline NEUROEVOLUTION_HOST_DEVICE void WriteEncodedTurnToModelInput(const EncodedTurnVector &encoded_turn,
                                                                    const std::size_t turn_index,
                                                                    ModelInputStateVector &model_input_state) noexcept {
    const std::size_t slot_offset = ModelInputTurnOffset(turn_index);

    for (std::size_t value_index = 0; value_index < kEncoderOutputSize; ++value_index) {
        model_input_state[slot_offset + value_index] = encoded_turn[value_index];
    }
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryEncodeWordleGridState(const SharedEncoderParameters &parameters,
                                                                const wordle::WordleGrid &grid,
                                                                ModelInputStateVector &model_input_state) noexcept {
    if (!wordle::IsValidWordleGrid(grid) || !IsValidModelInputStateTurnCount(grid.turn_count) || grid.IsFinished()) {
        return false;
    }

    ZeroModelInputState(model_input_state);

    for (std::size_t turn_index = 0; turn_index < grid.turn_count; ++turn_index) {
        EncodedTurnVector encoded_turn{};
        if (!TryForwardOccupiedTurn(parameters, grid.turns[turn_index], encoded_turn)) {
            return false;
        }

        WriteEncodedTurnToModelInput(encoded_turn, turn_index, model_input_state);
    }

    return true;
}

} // namespace detail

ModelInputStateVector EncodeWordleGridState(const SharedEncoderParameters &parameters, const wordle::WordleGrid &grid);

} // namespace neuroevolution::model::input_encoder
