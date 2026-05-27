#pragma once

#include "common/cuda_compat.hpp"

namespace neuroevolution::genetic_algorithm::output_tail_ops {

constexpr float kRowScaleIncreaseFactor = 1.02f;
constexpr float kRowScaleDecreaseFactor = 0.98f;

constexpr NEUROEVOLUTION_HOST_DEVICE float RowScaleFactor(const bool scale_up) noexcept {
    return scale_up ? kRowScaleIncreaseFactor : kRowScaleDecreaseFactor;
}

} // namespace neuroevolution::genetic_algorithm::output_tail_ops
