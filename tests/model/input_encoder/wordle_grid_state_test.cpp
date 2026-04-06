#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "model/input_encoder/encoder_spec.hpp"
#include "model/input_encoder/shared_encoder.hpp"
#include "model/input_encoder/wordle_grid_state.hpp"
#include "shared_encoder_fixture.hpp"
#include "wordle/word.hpp"
#include "wordle/wordle_grid.hpp"

namespace {

using neuroevolution::model::input_encoder::EncodedTurnVector;
using neuroevolution::model::input_encoder::EncodeWordleGridState;
using neuroevolution::model::input_encoder::ForwardOccupiedTurn;
using neuroevolution::model::input_encoder::kEncoderOutputSize;
using neuroevolution::model::input_encoder::kModelInputTurnCount;
using neuroevolution::model::input_encoder::ModelInputStateVector;
using neuroevolution::model::input_encoder::ModelInputTurnOffset;
using neuroevolution::tests::input_encoder::SharedEncoderGoldenFixture;
using neuroevolution::wordle::MakeWordleGrid;
using neuroevolution::wordle::TryAppendGuess;
using neuroevolution::wordle::Word;
using neuroevolution::wordle::WordleGrid;

constexpr float kTolerance = 1.0e-6f;

Word MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    Word word{};

    if (!neuroevolution::wordle::TryMakeWordFromAscii(letters, word)) {
        throw std::invalid_argument("Word test literal must contain exactly five uppercase ASCII letters.");
    }

    return word;
}

void AppendGuessOrThrow(WordleGrid &grid, const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    if (!TryAppendGuess(grid, MakeWord(letters))) {
        throw std::invalid_argument("Test guess append unexpectedly failed.");
    }
}

bool ExpectSlotEquals(const ModelInputStateVector &actual, const EncodedTurnVector &expected,
                      const std::size_t slot_index, const std::string_view label) {
    bool ok = true;
    const std::size_t slot_offset = ModelInputTurnOffset(slot_index);

    for (std::size_t value_index = 0; value_index < kEncoderOutputSize; ++value_index) {
        const float delta = std::fabs(actual[slot_offset + value_index] - expected[value_index]);
        if (delta > kTolerance) {
            std::cerr << "FAIL: " << label << " slot " << slot_index << " value " << value_index << " expected "
                      << expected[value_index] << ", got " << actual[slot_offset + value_index] << '\n';
            ok = false;
        }
    }

    return ok;
}

bool ExpectSlotZero(const ModelInputStateVector &actual, const std::size_t slot_index, const std::string_view label) {
    bool ok = true;
    const std::size_t slot_offset = ModelInputTurnOffset(slot_index);

    for (std::size_t value_index = 0; value_index < kEncoderOutputSize; ++value_index) {
        const float actual_value = actual[slot_offset + value_index];
        if (actual_value != 0.0f) {
            std::cerr << "FAIL: " << label << " slot " << slot_index << " value " << value_index << " expected 0, got "
                      << actual_value << '\n';
            ok = false;
        }
    }

    return ok;
}

bool TestEncodeWordleGridStateEmptyGridUsesZeroVectors() {
    const SharedEncoderGoldenFixture fixture{};
    const WordleGrid grid = MakeWordleGrid(MakeWord("SOLAR"));
    const ModelInputStateVector encoded_state = EncodeWordleGridState(fixture.parameters, grid);

    bool ok = true;
    for (std::size_t slot_index = 0; slot_index < kModelInputTurnCount; ++slot_index) {
        ok &= ExpectSlotZero(encoded_state, slot_index, "Empty grid");
    }

    return ok;
}

bool TestEncodeWordleGridStateOneTurnZeroFillsRemainingSlots() {
    const SharedEncoderGoldenFixture fixture{};
    WordleGrid grid = MakeWordleGrid(MakeWord("SOLAR"));
    AppendGuessOrThrow(grid, "CRANE");

    const ModelInputStateVector encoded_state = EncodeWordleGridState(fixture.parameters, grid);
    const EncodedTurnVector expected_first_turn = ForwardOccupiedTurn(fixture.parameters, grid.turns[0]);

    bool ok = true;
    ok &= ExpectSlotEquals(encoded_state, expected_first_turn, 0, "One-turn grid");
    for (std::size_t slot_index = 1; slot_index < kModelInputTurnCount; ++slot_index) {
        ok &= ExpectSlotZero(encoded_state, slot_index, "One-turn grid");
    }

    return ok;
}

bool TestEncodeWordleGridStatePreservesChronologicalOrderingForPartialGrid() {
    const SharedEncoderGoldenFixture fixture{};
    WordleGrid grid = MakeWordleGrid(MakeWord("SOLAR"));
    AppendGuessOrThrow(grid, "CRANE");
    AppendGuessOrThrow(grid, "ALERT");
    AppendGuessOrThrow(grid, "ROAST");

    const ModelInputStateVector encoded_state = EncodeWordleGridState(fixture.parameters, grid);

    bool ok = true;
    ok &= ExpectSlotEquals(encoded_state, ForwardOccupiedTurn(fixture.parameters, grid.turns[0]), 0, "Partial grid");
    ok &= ExpectSlotEquals(encoded_state, ForwardOccupiedTurn(fixture.parameters, grid.turns[1]), 1, "Partial grid");
    ok &= ExpectSlotEquals(encoded_state, ForwardOccupiedTurn(fixture.parameters, grid.turns[2]), 2, "Partial grid");
    ok &= ExpectSlotZero(encoded_state, 3, "Partial grid");
    ok &= ExpectSlotZero(encoded_state, 4, "Partial grid");

    return ok;
}

bool TestEncodeWordleGridStateFiveTurnGridFillsAllSlots() {
    const SharedEncoderGoldenFixture fixture{};
    WordleGrid grid = MakeWordleGrid(MakeWord("SOLAR"));
    AppendGuessOrThrow(grid, "CRANE");
    AppendGuessOrThrow(grid, "ALERT");
    AppendGuessOrThrow(grid, "ROAST");
    AppendGuessOrThrow(grid, "POLAR");
    AppendGuessOrThrow(grid, "SONAR");

    const ModelInputStateVector encoded_state = EncodeWordleGridState(fixture.parameters, grid);

    bool ok = true;
    for (std::size_t slot_index = 0; slot_index < kModelInputTurnCount; ++slot_index) {
        ok &= ExpectSlotEquals(encoded_state, ForwardOccupiedTurn(fixture.parameters, grid.turns[slot_index]),
                               slot_index, "Five-turn grid");
    }

    return ok;
}

bool TestEncodeWordleGridStateRejectsSixTurnTerminalBoard() {
    const SharedEncoderGoldenFixture fixture{};
    WordleGrid grid = MakeWordleGrid(MakeWord("SOLAR"));
    AppendGuessOrThrow(grid, "CRANE");
    AppendGuessOrThrow(grid, "ALERT");
    AppendGuessOrThrow(grid, "ROAST");
    AppendGuessOrThrow(grid, "POLAR");
    AppendGuessOrThrow(grid, "SONAR");
    AppendGuessOrThrow(grid, "SOLAR");

    try {
        (void)EncodeWordleGridState(fixture.parameters, grid);
    } catch (const std::invalid_argument &) {
        return true;
    }

    std::cerr << "FAIL: six-turn terminal grid should not be encodable as a five-turn decision state\n";
    return false;
}

} // namespace

int main() {
    if (!TestEncodeWordleGridStateEmptyGridUsesZeroVectors()) {
        return 1;
    }

    if (!TestEncodeWordleGridStateOneTurnZeroFillsRemainingSlots()) {
        return 1;
    }

    if (!TestEncodeWordleGridStatePreservesChronologicalOrderingForPartialGrid()) {
        return 1;
    }

    if (!TestEncodeWordleGridStateFiveTurnGridFillsAllSlots()) {
        return 1;
    }

    if (!TestEncodeWordleGridStateRejectsSixTurnTerminalBoard()) {
        return 1;
    }

    std::cout << "PASS: wordle_grid_state_test\n";
    return 0;
}
