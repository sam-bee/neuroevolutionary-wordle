#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "wordle/feedback.hpp"
#include "wordle/word.hpp"

namespace neuroevolution::wordle {

struct Turn {
    Word guess{};
    Feedback feedback{};
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidTurn(const Turn &turn) noexcept {
    if (!IsValidWord(turn.guess)) {
        return false;
    }

    return turn.feedback.IsValid();
}

} // namespace neuroevolution::wordle
