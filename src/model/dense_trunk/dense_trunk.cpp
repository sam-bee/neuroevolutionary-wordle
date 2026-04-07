#include "model/dense_trunk/dense_trunk.hpp"

namespace neuroevolution::model::dense_trunk {

PolicyVector ForwardDenseTrunk(const DenseTrunkParameters &parameters, const DenseTrunkInputVector &input_vector) {
    PolicyVector policy_vector{};
    ForwardDenseTrunk(parameters, input_vector, policy_vector);
    return policy_vector;
}

} // namespace neuroevolution::model::dense_trunk
