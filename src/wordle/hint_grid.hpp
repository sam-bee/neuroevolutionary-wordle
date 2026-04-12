#pragma once

#include <cstddef>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "wordle/feedback.hpp"
#include "wordle/wordle_grid.hpp"

namespace neuroevolution::wordle {

constexpr std::size_t kHintGridGroupSize = 3;
constexpr std::size_t kSwapHintGridCount = 2;

struct HintGridGroup {
    common::FixedBuffer<WordleGrid, kHintGridGroupSize> grids{};
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsIsogram(const Word &word) noexcept {
    if (!IsValidWord(word)) {
        return false;
    }

    for (std::size_t position = 0; position < kWordLength; ++position) {
        for (std::size_t later_position = position + 1; later_position < kWordLength; ++later_position) {
            if (word.letter_indices[position] == word.letter_indices[later_position]) {
                return false;
            }
        }
    }

    return true;
}

constexpr NEUROEVOLUTION_HOST_DEVICE Word RotateWordLeft(const Word &word, const std::size_t rotation) noexcept {
    Word rotated{};

    for (std::size_t position = 0; position < kWordLength; ++position) {
        rotated.letter_indices[position] = word.letter_indices[(position + rotation) % kWordLength];
    }

    return rotated;
}

constexpr NEUROEVOLUTION_HOST_DEVICE Word SwapWordPositions(const Word &word, const std::size_t first_position,
                                                            const std::size_t second_position) noexcept {
    Word swapped = word;
    const std::uint8_t first_value = swapped.letter_indices[first_position];
    swapped.letter_indices[first_position] = swapped.letter_indices[second_position];
    swapped.letter_indices[second_position] = first_value;
    return swapped;
}

namespace detail {

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t CountFeedbackTiles(const Feedback &feedback,
                                                                    const TileFeedback tile) noexcept {
    std::size_t tile_count = 0;

    for (std::size_t position = 0; position < kWordLength; ++position) {
        if (feedback[position] == tile) {
            ++tile_count;
        }
    }

    return tile_count;
}

template <std::size_t Size>
constexpr NEUROEVOLUTION_HOST_DEVICE bool ContainsWord(const common::FixedBuffer<Word, Size> &words,
                                                       const std::size_t word_count,
                                                       const Word &candidate) noexcept {
    for (std::size_t word_index = 0; word_index < word_count; ++word_index) {
        if (words[word_index] == candidate) {
            return true;
        }
    }

    return false;
}

} // namespace detail

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsAllYellowFeedback(const Feedback &feedback) noexcept {
    return detail::CountFeedbackTiles(feedback, TileFeedback::yellow) == kWordLength;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool HasThreeGreenTwoYellowFeedback(const Feedback &feedback) noexcept {
    return (detail::CountFeedbackTiles(feedback, TileFeedback::green) == 3) &&
           (detail::CountFeedbackTiles(feedback, TileFeedback::yellow) == 2);
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryBuildCyclicHintGrid(const Word &solution, WordleGrid &grid_out) noexcept {
    grid_out = {};

    if (!IsValidWord(solution)) {
        return false;
    }

    grid_out = MakeWordleGrid(solution);

    common::FixedBuffer<Word, kWordLength - 1> appended_guesses{};
    std::size_t appended_guess_count = 0;

    for (std::size_t rotation = 1; rotation < kWordLength; ++rotation) {
        const Word guess = RotateWordLeft(solution, rotation);
        if ((guess == solution) || detail::ContainsWord(appended_guesses, appended_guess_count, guess)) {
            continue;
        }

        Feedback feedback{};
        if (!TryProvideFeedback(guess, solution, feedback) || !IsAllYellowFeedback(feedback)) {
            continue;
        }

        if (!TryAppendGuess(grid_out, guess)) {
            return false;
        }

        appended_guesses[appended_guess_count] = guess;
        ++appended_guess_count;
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryBuildSwapHintGrid(const Word &solution, const std::size_t first_position,
                                                            const std::size_t second_position,
                                                            WordleGrid &grid_out) noexcept {
    grid_out = {};

    if (!IsValidWord(solution) || (first_position >= kWordLength) || (second_position >= kWordLength) ||
        (first_position == second_position) ||
        (solution.letter_indices[first_position] == solution.letter_indices[second_position])) {
        return false;
    }

    const Word guess = SwapWordPositions(solution, first_position, second_position);
    Feedback feedback{};
    if (!TryProvideFeedback(guess, solution, feedback) || !HasThreeGreenTwoYellowFeedback(feedback)) {
        return false;
    }

    grid_out = MakeWordleGrid(solution);
    return TryAppendGuess(grid_out, guess);
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryBuildHintGridGroup(const Word &solution, HintGridGroup &group_out) noexcept {
    group_out = {};

    if (!TryBuildCyclicHintGrid(solution, group_out.grids[0])) {
        return false;
    }

    common::FixedBuffer<Word, kSwapHintGridCount> selected_swap_guesses{};
    std::size_t selected_swap_count = 0;

    for (std::size_t first_position = 0; first_position < kWordLength; ++first_position) {
        for (std::size_t second_position = first_position + 1; second_position < kWordLength; ++second_position) {
            WordleGrid swap_grid{};
            if (!TryBuildSwapHintGrid(solution, first_position, second_position, swap_grid)) {
                continue;
            }

            const Word guess = swap_grid.turns[0].guess;
            if (detail::ContainsWord(selected_swap_guesses, selected_swap_count, guess)) {
                continue;
            }

            group_out.grids[1 + selected_swap_count] = swap_grid;
            selected_swap_guesses[selected_swap_count] = guess;
            ++selected_swap_count;

            if (selected_swap_count == kSwapHintGridCount) {
                return true;
            }
        }
    }

    return false;
}

} // namespace neuroevolution::wordle
