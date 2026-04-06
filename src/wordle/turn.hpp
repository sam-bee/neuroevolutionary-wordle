#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "wordle/word.hpp"

namespace neuroevolution::wordle {

enum class TileFeedback : std::uint8_t {
    green = 0,
    yellow = 1,
    grey = 2,
};

struct Turn {
    Word guess{};
    common::FixedBuffer<TileFeedback, kWordLength> feedback{};
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidFeedback(const TileFeedback value) noexcept {
    switch (value) {
    case TileFeedback::green:
    case TileFeedback::yellow:
    case TileFeedback::grey:
        return true;
    }

    return false;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidTurn(const Turn &turn) noexcept {
    if (!IsValidWord(turn.guess)) {
        return false;
    }

    for (std::size_t position = 0; position < kWordLength; ++position) {
        if (!IsValidFeedback(turn.feedback[position])) {
            return false;
        }
    }

    return true;
}

} // namespace neuroevolution::wordle
