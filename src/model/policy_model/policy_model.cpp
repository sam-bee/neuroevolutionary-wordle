#include "model/policy_model/policy_model.hpp"

#include <stdexcept>

namespace neuroevolution::model::policy_model {

PolicyVector ForwardPolicyModel(const PolicyModelParameters &parameters, const wordle::WordleGrid &grid) {
    PolicyVector policy_vector{};
    if (!TryForwardPolicyModel(parameters, grid, policy_vector)) {
        throw std::invalid_argument("Wordle grid must be a valid, non-terminal decision state for policy inference.");
    }

    return policy_vector;
}

} // namespace neuroevolution::model::policy_model
