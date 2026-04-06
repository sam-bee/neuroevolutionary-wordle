#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"

namespace neuroevolution::wordle {

constexpr std::size_t kWordLength = 5;
constexpr std::size_t kAlphabetSize = 26;

struct Word {
    common::FixedBuffer<std::uint8_t, kWordLength> letter_indices{};

    constexpr NEUROEVOLUTION_HOST_DEVICE bool operator==(const Word &other) const noexcept {
        for (std::size_t position = 0; position < kWordLength; ++position) {
            if (letter_indices[position] != other.letter_indices[position]) {
                return false;
            }
        }

        return true;
    }
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsAsciiUppercaseLetter(const char value) noexcept {
    return (value >= 'A') && (value <= 'Z');
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidLetterIndex(const std::uint8_t letter_index) noexcept {
    return letter_index < kAlphabetSize;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::uint8_t LetterIndexFromAscii(const char value) noexcept {
    return static_cast<std::uint8_t>(value - 'A');
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool TryMakeWordFromAscii(const char (&letters)[kWordLength + 1],
                                                               Word &word) noexcept {
    if (letters[kWordLength] != '\0') {
        return false;
    }

    for (std::size_t position = 0; position < kWordLength; ++position) {
        if (!IsAsciiUppercaseLetter(letters[position])) {
            return false;
        }

        word.letter_indices[position] = LetterIndexFromAscii(letters[position]);
    }

    return true;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidWord(const Word &word) noexcept {
    for (std::size_t position = 0; position < kWordLength; ++position) {
        if (!IsValidLetterIndex(word.letter_indices[position])) {
            return false;
        }
    }

    return true;
}

} // namespace neuroevolution::wordle
