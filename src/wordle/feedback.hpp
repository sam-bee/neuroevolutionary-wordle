#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "wordle/word.hpp"

namespace neuroevolution::wordle {

enum class TileFeedback : std::uint8_t {
    green = 0,
    yellow = 1,
    grey = 2,
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

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidFeedbackSymbol(const char value) noexcept {
    switch (value) {
    case 'G':
    case 'Y':
    case '-':
        return true;
    }

    return false;
}

constexpr NEUROEVOLUTION_HOST_DEVICE TileFeedback FeedbackFromSymbol(const char value) noexcept {
    switch (value) {
    case 'G':
        return TileFeedback::green;
    case 'Y':
        return TileFeedback::yellow;
    case '-':
        return TileFeedback::grey;
    }

    return TileFeedback::grey;
}

class Feedback;

constexpr NEUROEVOLUTION_HOST_DEVICE bool TryMakeFeedbackFromSymbols(const common::FixedBuffer<char, kWordLength> &symbols,
                                                                     Feedback &feedback) noexcept;

class Feedback {
  public:
    common::FixedBuffer<TileFeedback, kWordLength> values{};

    constexpr Feedback() = default;

    explicit Feedback(const common::FixedBuffer<char, kWordLength> &symbols) {
        if (!TryMakeFeedbackFromSymbols(symbols, *this)) {
            throw std::invalid_argument(
                "Feedback literal must contain exactly five symbols from {'G', 'Y', '-'}.");
        }
    }

    constexpr NEUROEVOLUTION_HOST_DEVICE TileFeedback &operator[](const std::size_t index) noexcept {
        return values[index];
    }

    constexpr NEUROEVOLUTION_HOST_DEVICE const TileFeedback &operator[](const std::size_t index) const noexcept {
        return values[index];
    }

    constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValid() const noexcept {
        for (std::size_t position = 0; position < kWordLength; ++position) {
            if (!IsValidFeedback(values[position])) {
                return false;
            }
        }

        return true;
    }
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool TryMakeFeedbackFromSymbols(const common::FixedBuffer<char, kWordLength> &symbols,
                                                                     Feedback &feedback) noexcept {
    for (std::size_t position = 0; position < kWordLength; ++position) {
        if (!IsValidFeedbackSymbol(symbols[position])) {
            return false;
        }

        feedback[position] = FeedbackFromSymbol(symbols[position]);
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryProvideFeedback(const Word &guess, const Word &solution,
                                                          Feedback &feedback) noexcept {
    if (!IsValidWord(guess) || !IsValidWord(solution)) {
        return false;
    }

    common::FixedBuffer<std::uint8_t, kAlphabetSize> unmatched_solution_letter_counts{};

    for (std::size_t position = 0; position < kWordLength; ++position) {
        if (guess.letter_indices[position] == solution.letter_indices[position]) {
            feedback[position] = TileFeedback::green;
            continue;
        }

        feedback[position] = TileFeedback::grey;
        ++unmatched_solution_letter_counts[solution.letter_indices[position]];
    }

    for (std::size_t position = 0; position < kWordLength; ++position) {
        if (feedback[position] == TileFeedback::green) {
            continue;
        }

        const std::uint8_t guess_letter_index = guess.letter_indices[position];
        if (unmatched_solution_letter_counts[guess_letter_index] == 0) {
            continue;
        }

        feedback[position] = TileFeedback::yellow;
        --unmatched_solution_letter_counts[guess_letter_index];
    }

    return true;
}

inline Feedback ProvideFeedback(const Word &guess, const Word &solution) {
    Feedback feedback{};
    if (!TryProvideFeedback(guess, solution, feedback)) {
        throw std::invalid_argument("Guess and solution must both contain exactly five valid letters.");
    }

    return feedback;
}

} // namespace neuroevolution::wordle
