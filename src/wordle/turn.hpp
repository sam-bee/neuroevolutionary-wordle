#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace neuroevolution::wordle {

constexpr std::size_t kWordLength = 5;
constexpr std::size_t kAlphabetSize = 26;

enum class TileFeedback : std::uint8_t {
  green = 0,
  yellow = 1,
  grey = 2,
};

struct Turn {
  std::array<std::uint8_t, kWordLength> letter_indices{};
  std::array<TileFeedback, kWordLength> feedback{};
};

constexpr bool IsValidLetterIndex(const std::uint8_t letter_index) noexcept {
  return letter_index < kAlphabetSize;
}

constexpr bool IsValidFeedback(const TileFeedback value) noexcept {
  switch (value) {
  case TileFeedback::green:
  case TileFeedback::yellow:
  case TileFeedback::grey:
    return true;
  }

  return false;
}

constexpr bool IsValidTurn(const Turn &turn) noexcept {
  for (std::size_t position = 0; position < kWordLength; ++position) {
    if (!IsValidLetterIndex(turn.letter_indices[position])) {
      return false;
    }

    if (!IsValidFeedback(turn.feedback[position])) {
      return false;
    }
  }

  return true;
}

} // namespace neuroevolution::wordle
