#pragma once

#include <cstddef>

#include "common/cuda_compat.hpp"
#include "model/input_encoder/shared_encoder.hpp"
#include "model/model_input/model_input_spec.hpp"
#include "wordle/wordle_grid.hpp"

namespace neuroevolution::model::model_input {

namespace detail {

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidModelInputStateTurnCount(const std::size_t turn_count) noexcept {
    // WordleGrid can hold 6 turns, but the policy input only represents the pre-next-guess decision state,
    // so this adapter intentionally stops at the first 5 chronological turns.
    return turn_count <= kModelInputTurnCount;
}

inline NEUROEVOLUTION_HOST_DEVICE void ZeroModelInputState(ModelInputStateVector &model_input_state) noexcept {
    for (std::size_t value_index = 0; value_index < kModelInputVectorSize; ++value_index) {
        model_input_state[value_index] = 0.0f;
    }
}

inline NEUROEVOLUTION_HOST_DEVICE void WriteVirginFlagToModelInput(const wordle::WordleGrid &grid,
                                                                   ModelInputStateVector &model_input_state) noexcept {
    model_input_state[kModelInputVirginFlagOffset] = grid.isVirgin() ? 1.0f : 0.0f;
}

inline NEUROEVOLUTION_HOST_DEVICE void
WriteEncodedTurnToModelInput(const input_encoder::EncodedTurnVector &encoded_turn, const std::size_t turn_index,
                             ModelInputStateVector &model_input_state) noexcept {
    const std::size_t slot_offset = ModelInputTurnOffset(turn_index);

    for (std::size_t value_index = 0; value_index < input_encoder::kEncoderOutputSize; ++value_index) {
        model_input_state[slot_offset + value_index] = encoded_turn[value_index];
    }
}

} // namespace detail

inline NEUROEVOLUTION_HOST_DEVICE bool
TryEncodeWordleGridState(const input_encoder::SharedEncoderParameters &parameters, const wordle::WordleGrid &grid,
                         ModelInputStateVector &model_input_state) noexcept {
    if (!wordle::IsValidWordleGrid(grid) || !detail::IsValidModelInputStateTurnCount(grid.turn_count) ||
        grid.IsFinished()) {
        return false;
    }

    detail::ZeroModelInputState(model_input_state);
    detail::WriteVirginFlagToModelInput(grid, model_input_state);

    for (std::size_t turn_index = 0; turn_index < grid.turn_count; ++turn_index) {
        input_encoder::EncodedTurnVector encoded_turn{};
        if (!input_encoder::TryForwardOccupiedTurn(parameters, grid.turns[turn_index], encoded_turn)) {
            return false;
        }

        detail::WriteEncodedTurnToModelInput(encoded_turn, turn_index, model_input_state);
    }

    return true;
}

ModelInputStateVector EncodeWordleGridState(const input_encoder::SharedEncoderParameters &parameters,
                                            const wordle::WordleGrid &grid);

} // namespace neuroevolution::model::model_input
