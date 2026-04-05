#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"

namespace neuroevolution::wordle {

constexpr std::size_t kWordLength = 5;
constexpr std::size_t kAlphabetSize = 26;

enum class TileFeedback : std::uint8_t {
  green = 0,
  yellow = 1,
  grey = 2,
};

struct Turn {
  common::FixedBuffer<std::uint8_t, kWordLength> letter_indices{};
  common::FixedBuffer<TileFeedback, kWordLength> feedback{};
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidLetterIndex(const std::uint8_t letter_index) noexcept {
  return letter_index < kAlphabetSize;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidFeedback(const TileFeedback value) noexcept {
  switch (value) {
  case TileFeedback::green:
  case TileFeedback::yellow:
  case TileFeedback::grey:
    return true;
  }

  return false;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidTurn(const Turn &turn) noexcept {
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
