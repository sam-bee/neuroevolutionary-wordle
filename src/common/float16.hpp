#pragma once

#include <cuda_fp16.h>

#include "common/cuda_compat.hpp"

namespace neuroevolution::common {

using Float16 = __half;

inline NEUROEVOLUTION_HOST_DEVICE Float16 ToFloat16(const float value) noexcept { return __float2half(value); }

inline NEUROEVOLUTION_HOST_DEVICE float ToFloat(const Float16 value) noexcept { return __half2float(value); }

} // namespace neuroevolution::common
