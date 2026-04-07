#pragma once

#include "common/cuda_compat.hpp"
#include "model/dense_trunk/dense_trunk.hpp"
#include "model/model_input/wordle_grid_state.hpp"

namespace neuroevolution::model::policy_model {

struct PolicyModelParameters {
    input_encoder::SharedEncoderParameters input_encoder{};
    dense_trunk::DenseTrunkParameters dense_trunk{};
};

using PolicyVector = dense_trunk::PolicyVector;

inline NEUROEVOLUTION_HOST_DEVICE bool TryForwardPolicyModel(const PolicyModelParameters &parameters,
                                                             const wordle::WordleGrid &grid,
                                                             PolicyVector &policy_vector) noexcept {
    model_input::ModelInputStateVector model_input_state{};
    if (!model_input::TryEncodeWordleGridState(parameters.input_encoder, grid, model_input_state)) {
        return false;
    }

    dense_trunk::ForwardDenseTrunk(parameters.dense_trunk, model_input_state, policy_vector);
    return true;
}

PolicyVector ForwardPolicyModel(const PolicyModelParameters &parameters, const wordle::WordleGrid &grid);

} // namespace neuroevolution::model::policy_model
