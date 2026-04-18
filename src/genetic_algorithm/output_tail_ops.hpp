#pragma once

#include <cmath>

#include "common/cuda_compat.hpp"

namespace neuroevolution::genetic_algorithm::output_tail_ops {

constexpr float kDefaultRowArithmeticRecombinationProbability = 0.01f;
constexpr float kDefaultRowScaleMutationProbability = 0.005f;
constexpr float kRowBlendMinimumLambda = 0.2f;
constexpr float kRowBlendMaximumLambda = 0.8f;
constexpr float kRowScaleIncreaseFactor = 1.02f;
constexpr float kRowScaleDecreaseFactor = 0.98f;

constexpr float kPi = 3.14159265358979323846f;

constexpr NEUROEVOLUTION_HOST_DEVICE float RowBlendSpan() noexcept {
    return kRowBlendMaximumLambda - kRowBlendMinimumLambda;
}

inline NEUROEVOLUTION_HOST_DEVICE float MapArcsineUnitSampleToBlendLambda(const float unit_sample) noexcept {
    const float beta_sample = 0.5f * (1.0f - cosf(kPi * unit_sample));
    return kRowBlendMinimumLambda + (RowBlendSpan() * beta_sample);
}

constexpr NEUROEVOLUTION_HOST_DEVICE float RowScaleFactor(const bool scale_up) noexcept {
    return scale_up ? kRowScaleIncreaseFactor : kRowScaleDecreaseFactor;
}

} // namespace neuroevolution::genetic_algorithm::output_tail_ops
