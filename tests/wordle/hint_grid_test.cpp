#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "../model/input_encoder/shared_encoder_fixture.hpp"
#include "model/input_encoder/shared_encoder.hpp"
#include "model/model_input/wordle_grid_state.hpp"
#include "wordle/feedback.hpp"
#include "wordle/hint_grid.hpp"
#include "wordle/word.hpp"
#include "wordle/wordle_grid.hpp"

namespace {

using neuroevolution::model::input_encoder::EncodedTurnVector;
using neuroevolution::model::input_encoder::ForwardOccupiedTurn;
using neuroevolution::model::input_encoder::kEncoderOutputSize;
using neuroevolution::model::model_input::EncodeWordleGridState;
using neuroevolution::model::model_input::kModelInputTurnCount;
using neuroevolution::model::model_input::kModelInputVirginFlagOffset;
using neuroevolution::model::model_input::ModelInputStateVector;
using neuroevolution::model::model_input::ModelInputTurnOffset;
using neuroevolution::tests::input_encoder::SharedEncoderGoldenFixture;
using neuroevolution::wordle::Feedback;
using neuroevolution::wordle::HasThreeGreenTwoYellowFeedback;
using neuroevolution::wordle::HintGridGroup;
using neuroevolution::wordle::IsAllYellowFeedback;
using neuroevolution::wordle::IsValidWordleGrid;
using neuroevolution::wordle::kHintGridGroupSize;
using neuroevolution::wordle::RotateWordLeft;
using neuroevolution::wordle::TryBuildCyclicHintGrid;
using neuroevolution::wordle::TryBuildHintGridGroup;
using neuroevolution::wordle::TryBuildSwapHintGrid;
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

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

bool ExpectWordEquals(const Word &actual, const Word &expected, const std::string_view label) {
    bool ok = true;

    for (std::size_t position = 0; position < neuroevolution::wordle::kWordLength; ++position) {
        if (actual.letter_indices[position] != expected.letter_indices[position]) {
            std::cerr << "FAIL: " << label << " mismatch at position " << position << '\n';
            ok = false;
        }
    }

    return ok;
}

bool ExpectFeedbackMatches(const Feedback &actual, const Feedback &expected, const std::string_view label) {
    bool ok = true;

    for (std::size_t position = 0; position < neuroevolution::wordle::kWordLength; ++position) {
        if (actual[position] != expected[position]) {
            std::cerr << "FAIL: " << label << " mismatch at position " << position << '\n';
            ok = false;
        }
    }

    return ok;
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
        if (actual[slot_offset + value_index] != 0.0f) {
            std::cerr << "FAIL: " << label << " slot " << slot_index << " value " << value_index
                      << " expected 0, got " << actual[slot_offset + value_index] << '\n';
            ok = false;
        }
    }

    return ok;
}

bool ExpectGridEncodesAsDecisionState(const SharedEncoderGoldenFixture &fixture, const WordleGrid &grid,
                                      const std::string_view label) {
    const ModelInputStateVector encoded_state = EncodeWordleGridState(fixture.parameters, grid);

    bool ok = true;
    ok &= ExpectTrue(encoded_state[kModelInputVirginFlagOffset] == (grid.isVirgin() ? 1.0f : 0.0f),
                     "Encoded virgin flag should match grid occupancy");

    for (std::size_t slot_index = 0; slot_index < grid.turn_count; ++slot_index) {
        ok &= ExpectSlotEquals(encoded_state, ForwardOccupiedTurn(fixture.parameters, grid.turns[slot_index]), slot_index,
                               label);
    }

    for (std::size_t slot_index = grid.turn_count; slot_index < kModelInputTurnCount; ++slot_index) {
        ok &= ExpectSlotZero(encoded_state, slot_index, label);
    }

    return ok;
}

bool TestBuildCyclicHintGridUsesFourAllYellowRotationsForIsogram() {
    const Word solution = MakeWord("SPARE");
    WordleGrid grid{};

    const bool build_ok = TryBuildCyclicHintGrid(solution, grid);

    bool ok = true;
    ok &= ExpectTrue(build_ok, "Expected cyclic hint-grid construction to succeed for an isogram");
    ok &= ExpectTrue(IsValidWordleGrid(grid), "Constructed cyclic hint grid should be valid");
    ok &= ExpectTrue(grid.turn_count == 4, "Isogram cyclic hint grid should use four turns");
    ok &= ExpectTrue(!grid.IsFinished(), "Four-turn cyclic hint grid should remain a decision state");

    for (std::size_t rotation = 1; rotation < neuroevolution::wordle::kWordLength; ++rotation) {
        const Word expected_guess = RotateWordLeft(solution, rotation);
        ok &= ExpectWordEquals(grid.turns[rotation - 1].guess, expected_guess, "Cyclic hint-grid rotation guess");
        ok &= ExpectTrue(IsAllYellowFeedback(grid.turns[rotation - 1].feedback),
                         "Isogram cyclic hint-grid feedback should be all yellow");
    }

    return ok;
}

bool TestBuildCyclicHintGridSupportsRepeatedLettersWithFewerAllYellowTurns() {
    const Word solution = MakeWord("AABCD");
    WordleGrid grid{};

    const bool build_ok = TryBuildCyclicHintGrid(solution, grid);

    bool ok = true;
    ok &= ExpectTrue(build_ok, "Expected cyclic hint-grid construction to support repeated letters");
    ok &= ExpectTrue(IsValidWordleGrid(grid), "Repeated-letter cyclic hint grid should be valid");
    ok &= ExpectTrue(grid.turn_count == 2, "Repeated-letter cyclic hint grid should skip non-all-yellow rotations");
    ok &= ExpectWordEquals(grid.turns[0].guess, RotateWordLeft(solution, 2), "First repeated-letter rotation guess");
    ok &= ExpectWordEquals(grid.turns[1].guess, RotateWordLeft(solution, 3), "Second repeated-letter rotation guess");
    ok &= ExpectTrue(IsAllYellowFeedback(grid.turns[0].feedback), "First repeated-letter feedback should be all yellow");
    ok &= ExpectTrue(IsAllYellowFeedback(grid.turns[1].feedback), "Second repeated-letter feedback should be all yellow");
    return ok;
}

bool TestBuildSwapHintGridCreatesThreeGreenTwoYellowSingleGuess() {
    const Word solution = MakeWord("LEVEL");
    WordleGrid grid{};

    const bool build_ok = TryBuildSwapHintGrid(solution, 0, 1, grid);

    bool ok = true;
    ok &= ExpectTrue(build_ok, "Expected swap hint-grid construction to succeed for different letters");
    ok &= ExpectTrue(IsValidWordleGrid(grid), "Swap hint grid should be valid");
    ok &= ExpectTrue(grid.turn_count == 1, "Swap hint grid should contain a single guess");
    ok &= ExpectWordEquals(grid.turns[0].guess, MakeWord("ELVEL"), "Swap hint guess");
    ok &= ExpectTrue(HasThreeGreenTwoYellowFeedback(grid.turns[0].feedback),
                     "Swap hint feedback should be three green and two yellow");
    ok &= ExpectFeedbackMatches(grid.turns[0].feedback, Feedback{{'Y', 'Y', 'G', 'G', 'G'}},
                                "Swap hint feedback pattern");
    return ok;
}

bool TestBuildHintGridGroupProducesCyclicAndTwoSwapGrids() {
    const Word solution = MakeWord("SPARE");
    HintGridGroup group{};

    const bool build_ok = TryBuildHintGridGroup(solution, group);

    bool ok = true;
    ok &= ExpectTrue(build_ok, "Expected hint-grid group construction to succeed");
    ok &= ExpectTrue(group.grids[0].turn_count == 4, "First group grid should be the four-turn cyclic grid");
    ok &= ExpectTrue(group.grids[1].turn_count == 1, "Second group grid should be a single-swap hint grid");
    ok &= ExpectTrue(group.grids[2].turn_count == 1, "Third group grid should be a second single-swap hint grid");
    ok &= ExpectWordEquals(group.grids[1].turns[0].guess, MakeWord("PSARE"), "First swap-grid guess");
    ok &= ExpectWordEquals(group.grids[2].turns[0].guess, MakeWord("APSRE"), "Second swap-grid guess");
    ok &= ExpectTrue(HasThreeGreenTwoYellowFeedback(group.grids[1].turns[0].feedback),
                     "First swap-grid feedback should be three green and two yellow");
    ok &= ExpectTrue(HasThreeGreenTwoYellowFeedback(group.grids[2].turns[0].feedback),
                     "Second swap-grid feedback should be three green and two yellow");
    return ok;
}

bool TestBuildHintGridGroupRejectsWordsWithoutTwoDistinctSwapHints() {
    HintGridGroup group{};
    const bool build_ok = TryBuildHintGridGroup(MakeWord("AAAAA"), group);

    bool ok = true;
    ok &= ExpectTrue(!build_ok, "Expected hint-grid group construction to reject monochrome words");
    for (std::size_t grid_index = 0; grid_index < kHintGridGroupSize; ++grid_index) {
        ok &= ExpectTrue(group.grids[grid_index].turn_count == 0,
                         "Failed hint-grid group construction should leave all grids empty");
    }

    return ok;
}

bool TestEveryHintGridInGroupCanBeEncodedAsModelInput() {
    const SharedEncoderGoldenFixture fixture{};
    HintGridGroup group{};
    if (!TryBuildHintGridGroup(MakeWord("AABCD"), group)) {
        std::cerr << "FAIL: repeated-letter hint-grid group should build successfully\n";
        return false;
    }

    bool ok = true;
    for (std::size_t grid_index = 0; grid_index < kHintGridGroupSize; ++grid_index) {
        ok &= ExpectGridEncodesAsDecisionState(fixture, group.grids[grid_index], "Hint-grid group encoding");
    }

    return ok;
}

} // namespace

int main() {
    if (!TestBuildCyclicHintGridUsesFourAllYellowRotationsForIsogram()) {
        return 1;
    }

    if (!TestBuildCyclicHintGridSupportsRepeatedLettersWithFewerAllYellowTurns()) {
        return 1;
    }

    if (!TestBuildSwapHintGridCreatesThreeGreenTwoYellowSingleGuess()) {
        return 1;
    }

    if (!TestBuildHintGridGroupProducesCyclicAndTwoSwapGrids()) {
        return 1;
    }

    if (!TestBuildHintGridGroupRejectsWordsWithoutTwoDistinctSwapHints()) {
        return 1;
    }

    if (!TestEveryHintGridInGroupCanBeEncodedAsModelInput()) {
        return 1;
    }

    std::cout << "PASS: hint_grid_test\n";
    return 0;
}
