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

#if defined(__CUDACC__)
template <int WarpWidth> struct PolicyModelWarpScratch {
    model_input::ModelInputStateVector model_input_state{};
    input_encoder::TurnInputVector turn_input{};
    input_encoder::EncoderHiddenVector encoder_hidden{};
    input_encoder::EncodedTurnVector encoded_turn{};
    dense_trunk::DenseTrunkHiddenVector0 dense_hidden0{};
    dense_trunk::DenseTrunkHiddenVector1 dense_hidden1{};
    PolicyVector policy_vector{};
};
#endif

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

#if defined(__CUDACC__)
template <int WarpWidth>
inline __device__ bool TryForwardPolicyModelConcurrently(const PolicyModelParameters &parameters,
                                                         const wordle::WordleGrid &grid,
                                                         PolicyModelWarpScratch<WarpWidth> &scratch) noexcept {
    if (!model_input::TryEncodeWordleGridStateConcurrently<WarpWidth>(
            parameters.input_encoder, grid, scratch.model_input_state, scratch.turn_input, scratch.encoder_hidden,
            scratch.encoded_turn)) {
        return false;
    }

    dense_trunk::ForwardDenseTrunkConcurrently<WarpWidth>(parameters.dense_trunk, scratch.model_input_state,
                                                          scratch.dense_hidden0, scratch.dense_hidden1,
                                                          scratch.policy_vector);
    return true;
}
#endif

PolicyVector ForwardPolicyModel(const PolicyModelParameters &parameters, const wordle::WordleGrid &grid);

} // namespace neuroevolution::model::policy_model
