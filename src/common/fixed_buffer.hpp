#pragma once

#include <cstddef>

#include "common/cuda_compat.hpp"

namespace neuroevolution::common {

template <typename T, std::size_t Size> struct FixedBuffer {
    T values[Size]{};

    constexpr NEUROEVOLUTION_HOST_DEVICE T &operator[](const std::size_t index) noexcept { return values[index]; }

    constexpr NEUROEVOLUTION_HOST_DEVICE const T &operator[](const std::size_t index) const noexcept {
        return values[index];
    }
};

} // namespace neuroevolution::common
