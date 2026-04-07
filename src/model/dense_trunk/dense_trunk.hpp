#pragma once

#include <cstddef>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "common/float16.hpp"
#include "model/model_input/model_input_spec.hpp"

namespace neuroevolution::model::dense_trunk {

using ParameterScalar = common::Float16;
static_assert(sizeof(ParameterScalar) == 2, "Dense trunk parameters must be stored in fp16.");

constexpr std::size_t kDenseTrunkInputSize = model_input::kModelInputVectorSize;
constexpr std::size_t kDenseTrunkHiddenSize0 = 256;
constexpr std::size_t kDenseTrunkHiddenSize1 = 128;
constexpr std::size_t kDenseTrunkOutputSize = 64;

using DenseTrunkInputVector = model_input::ModelInputStateVector;
using DenseTrunkHiddenVector0 = common::FixedBuffer<float, kDenseTrunkHiddenSize0>;
using DenseTrunkHiddenVector1 = common::FixedBuffer<float, kDenseTrunkHiddenSize1>;
using PolicyVector = common::FixedBuffer<float, kDenseTrunkOutputSize>;

template <std::size_t InputSize, std::size_t OutputSize> struct DenseLayerParameters {
    // Trainable parameters are stored in fp16 row-major order by output neuron, then input index.
    common::FixedBuffer<ParameterScalar, InputSize * OutputSize> weights{};
    common::FixedBuffer<ParameterScalar, OutputSize> biases{};

    constexpr NEUROEVOLUTION_HOST_DEVICE float WeightAt(const std::size_t output_index,
                                                        const std::size_t input_index) const noexcept {
        return common::ToFloat(weights[(output_index * InputSize) + input_index]);
    }
};

struct DenseTrunkParameters {
    DenseLayerParameters<kDenseTrunkInputSize, kDenseTrunkHiddenSize0> input_to_hidden0{};
    DenseLayerParameters<kDenseTrunkHiddenSize0, kDenseTrunkHiddenSize1> hidden0_to_hidden1{};
    DenseLayerParameters<kDenseTrunkHiddenSize1, kDenseTrunkOutputSize> hidden1_to_output{};
};

namespace detail {

template <std::size_t InputSize, std::size_t OutputSize>
inline NEUROEVOLUTION_HOST_DEVICE void ApplyDenseLayer(const DenseLayerParameters<InputSize, OutputSize> &layer,
                                                       const common::FixedBuffer<float, InputSize> &input,
                                                       common::FixedBuffer<float, OutputSize> &output) noexcept {
    for (std::size_t output_index = 0; output_index < OutputSize; ++output_index) {
        float sum = common::ToFloat(layer.biases[output_index]);

        for (std::size_t input_index = 0; input_index < InputSize; ++input_index) {
            sum += layer.WeightAt(output_index, input_index) * input[input_index];
        }

        output[output_index] = sum;
    }
}

template <std::size_t Size>
inline NEUROEVOLUTION_HOST_DEVICE void ApplyReLU(common::FixedBuffer<float, Size> &activations) noexcept {
    for (std::size_t activation_index = 0; activation_index < Size; ++activation_index) {
        if (activations[activation_index] < 0.0f) {
            activations[activation_index] = 0.0f;
        }
    }
}

} // namespace detail

inline NEUROEVOLUTION_HOST_DEVICE void ForwardDenseTrunk(const DenseTrunkParameters &parameters,
                                                         const DenseTrunkInputVector &input_vector,
                                                         PolicyVector &policy_vector) noexcept {
    DenseTrunkHiddenVector0 hidden0{};
    detail::ApplyDenseLayer(parameters.input_to_hidden0, input_vector, hidden0);
    detail::ApplyReLU(hidden0);

    DenseTrunkHiddenVector1 hidden1{};
    detail::ApplyDenseLayer(parameters.hidden0_to_hidden1, hidden0, hidden1);
    detail::ApplyReLU(hidden1);

    detail::ApplyDenseLayer(parameters.hidden1_to_output, hidden1, policy_vector);
}

PolicyVector ForwardDenseTrunk(const DenseTrunkParameters &parameters, const DenseTrunkInputVector &input_vector);

} // namespace neuroevolution::model::dense_trunk
