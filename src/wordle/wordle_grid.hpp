#pragma once

#include <cstddef>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "wordle/turn.hpp"

namespace neuroevolution::wordle {

constexpr std::size_t kMaxTurnCount = 6;

struct WordleGrid {
    Word solution{};
    common::FixedBuffer<Turn, kMaxTurnCount> turns{};
    std::size_t turn_count = 0;

    constexpr NEUROEVOLUTION_HOST_DEVICE bool isVirgin() const noexcept { return turn_count == 0; }

    constexpr NEUROEVOLUTION_HOST_DEVICE bool IsWon() const noexcept {
        const std::size_t turns_to_check = (turn_count < kMaxTurnCount) ? turn_count : kMaxTurnCount;

        for (std::size_t turn_index = 0; turn_index < turns_to_check; ++turn_index) {
            if (turns[turn_index].guess == solution) {
                return true;
            }
        }

        return false;
    }

    constexpr NEUROEVOLUTION_HOST_DEVICE bool IsFinished() const noexcept {
        return IsWon() || (turn_count >= kMaxTurnCount);
    }
};

inline NEUROEVOLUTION_HOST_DEVICE Turn MakeTurnWithFeedback(const Word &guess, const Word &solution) noexcept {
    Turn turn{};
    turn.guess = guess;
    (void)TryProvideFeedback(guess, solution, turn.feedback);

    return turn;
}

constexpr NEUROEVOLUTION_HOST_DEVICE WordleGrid MakeWordleGrid(const Word &solution) noexcept {
    WordleGrid grid{};
    grid.solution = solution;
    return grid;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidTurnCount(const std::size_t turn_count) noexcept {
    return turn_count <= kMaxTurnCount;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidWordleGrid(const WordleGrid &grid) noexcept {
    if (!IsValidWord(grid.solution) || !IsValidTurnCount(grid.turn_count)) {
        return false;
    }

    for (std::size_t turn_index = 0; turn_index < grid.turn_count; ++turn_index) {
        if (!IsValidTurn(grid.turns[turn_index])) {
            return false;
        }
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryAppendGuess(WordleGrid &grid, const Word &guess) noexcept {
    if (!IsValidWordleGrid(grid) || !IsValidWord(guess) || grid.IsFinished()) {
        return false;
    }

    grid.turns[grid.turn_count] = MakeTurnWithFeedback(guess, grid.solution);
    ++grid.turn_count;
    return true;
}

} // namespace neuroevolution::wordle
