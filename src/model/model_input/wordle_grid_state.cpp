#include "model/model_input/wordle_grid_state.hpp"

#include <stdexcept>

namespace neuroevolution::model::model_input {

ModelInputStateVector EncodeWordleGridState(const input_encoder::SharedEncoderParameters &parameters,
                                            const wordle::WordleGrid &grid) {
    ModelInputStateVector model_input_state{};

    if (!TryEncodeWordleGridState(parameters, grid, model_input_state)) {
        throw std::invalid_argument(
            "Wordle grid must be a valid, non-terminal decision state with at most five turns.");
    }

    return model_input_state;
}

} // namespace neuroevolution::model::model_input
