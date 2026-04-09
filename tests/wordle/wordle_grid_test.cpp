#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "wordle/word.hpp"
#include "wordle/wordle_grid.hpp"

namespace {

using neuroevolution::wordle::Feedback;
using neuroevolution::wordle::HasGuess;
using neuroevolution::wordle::IsValidWordleGrid;
using neuroevolution::wordle::kMaxTurnCount;
using neuroevolution::wordle::MakeWordleGrid;
using neuroevolution::wordle::ProvideFeedback;
using neuroevolution::wordle::TileFeedback;
using neuroevolution::wordle::TryAppendGuess;
using neuroevolution::wordle::Word;
using neuroevolution::wordle::WordleGrid;

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

Word MakeWordFromIndices(const std::array<std::uint8_t, neuroevolution::wordle::kWordLength> &letter_indices) {
    Word word{};

    for (std::size_t position = 0; position < neuroevolution::wordle::kWordLength; ++position) {
        word.letter_indices[position] = letter_indices[position];
    }

    return word;
}

Word MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    Word word{};

    if (!neuroevolution::wordle::TryMakeWordFromAscii(letters, word)) {
        throw std::invalid_argument("Word test literal must contain exactly five uppercase ASCII letters.");
    }

    return word;
}

bool ExpectWordEquals(const Word &actual, const Word &expected, const std::string_view label) {
    bool ok = true;

    for (std::size_t position = 0; position < neuroevolution::wordle::kWordLength; ++position) {
        if (actual.letter_indices[position] != expected.letter_indices[position]) {
            std::cerr << "FAIL: " << label << " mismatch at position " << position << ", expected "
                      << static_cast<int>(expected.letter_indices[position]) << ", got "
                      << static_cast<int>(actual.letter_indices[position]) << '\n';
            ok = false;
        }
    }

    return ok;
}

bool ExpectFeedbackEquals(const Feedback &actual, const Feedback &expected, const std::string_view label) {
    bool ok = true;

    auto symbol_for = [](const TileFeedback value) -> char {
        switch (value) {
        case TileFeedback::green:
            return 'G';
        case TileFeedback::yellow:
            return 'Y';
        case TileFeedback::grey:
            return '-';
        }

        return '?';
    };

    for (std::size_t position = 0; position < neuroevolution::wordle::kWordLength; ++position) {
        if (actual[position] != expected[position]) {
            std::cerr << "FAIL: " << label << " mismatch at position " << position << ", expected "
                      << symbol_for(expected[position]) << ", got " << symbol_for(actual[position]) << '\n';
            ok = false;
        }
    }

    return ok;
}

bool TestFeedbackParsesReadableSymbols() {
    const Feedback feedback{{'G', 'Y', '-', '-', 'G'}};

    bool ok = true;
    ok &= ExpectTrue(feedback.IsValid(), "Feedback literal should produce a valid feedback object");
    ok &= ExpectTrue(feedback[0] == TileFeedback::green, "Expected 'G' to map to green");
    ok &= ExpectTrue(feedback[1] == TileFeedback::yellow, "Expected 'Y' to map to yellow");
    ok &= ExpectTrue(feedback[2] == TileFeedback::grey, "Expected '-' to map to grey");
    ok &= ExpectTrue(feedback[3] == TileFeedback::grey, "Expected second '-' to map to grey");
    ok &= ExpectTrue(feedback[4] == TileFeedback::green, "Expected trailing 'G' to map to green");
    return ok;
}

bool TestProvideFeedbackMatchesWordleRules() {
    const Feedback feedback = ProvideFeedback(MakeWord("CRANE"), MakeWord("SOLAR"));
    return ExpectFeedbackEquals(feedback, Feedback{{'-', 'Y', 'Y', '-', '-'}}, "CRANE vs SOLAR");
}

bool TestProvideFeedbackHandlesDuplicateLettersInGuess() {
    const Feedback feedback = ProvideFeedback(MakeWord("EERIE"), MakeWord("STEEL"));
    return ExpectFeedbackEquals(feedback, Feedback{{'Y', 'Y', '-', '-', '-'}}, "EERIE vs STEEL");
}

bool TestProvideFeedbackHandlesDuplicateLettersInSolution() {
    const Feedback feedback = ProvideFeedback(MakeWord("HELLO"), MakeWord("LEVEL"));
    return ExpectFeedbackEquals(feedback, Feedback{{'-', 'G', 'Y', 'Y', '-'}}, "HELLO vs LEVEL");
}

bool TestProvideFeedbackHandlesExcessDuplicateLettersInGuess() {
    const Feedback feedback = ProvideFeedback(MakeWord("LEVEL"), MakeWord("HELLO"));
    return ExpectFeedbackEquals(feedback, Feedback{{'Y', 'G', '-', '-', 'Y'}}, "LEVEL vs HELLO");
}

bool TestWordleGridStartsEmptyAndStoresSolution() {
    const Word solution = MakeWord("SOLAR");
    const WordleGrid grid = MakeWordleGrid(solution);

    bool ok = true;
    ok &= ExpectTrue(grid.turn_count == 0, "New grid should start with zero turns");
    ok &= ExpectWordEquals(grid.solution, solution, "Stored solution");
    ok &= ExpectTrue(grid.isVirgin(), "New grid should report itself as virgin");
    ok &= ExpectTrue(!grid.IsFinished(), "New grid should not be finished");
    ok &= ExpectTrue(!grid.IsWon(), "New grid should not be won");
    ok &= ExpectTrue(IsValidWordleGrid(grid), "New grid should be valid");
    return ok;
}

bool TestWordleGridAppendsTurnsWithCorrectFeedback() {
    const Word solution = MakeWord("SOLAR");
    const Word guess = MakeWord("CRANE");

    WordleGrid grid = MakeWordleGrid(solution);

    bool ok = true;
    ok &= ExpectTrue(TryAppendGuess(grid, guess), "Appending a valid guess should succeed");
    ok &= ExpectTrue(grid.turn_count == 1, "Grid should contain one turn after the first guess");
    ok &= ExpectWordEquals(grid.turns[0].guess, guess, "Stored guess");
    ok &= ExpectFeedbackEquals(grid.turns[0].feedback, Feedback{{'-', 'Y', 'Y', '-', '-'}},
                               "Stored feedback for CRANE vs SOLAR");
    ok &= ExpectTrue(!grid.isVirgin(), "Grid should no longer be virgin after a guess");
    ok &= ExpectTrue(!grid.IsFinished(), "Grid should not be finished after one non-winning guess");
    ok &= ExpectTrue(!grid.IsWon(), "Grid should not be won after a non-matching guess");
    ok &= ExpectTrue(IsValidWordleGrid(grid), "Grid should remain valid after appending a guess");
    return ok;
}

bool TestWordleGridTracksPreviousGuesses() {
    const Word solution = MakeWord("SOLAR");
    const Word first_guess = MakeWord("CRANE");
    const Word second_guess = MakeWord("ALERT");
    const Word unseen_guess = MakeWord("MIGHT");

    WordleGrid grid = MakeWordleGrid(solution);

    bool ok = true;
    ok &= ExpectTrue(!HasGuess(grid, first_guess), "Virgin grid should report no previous guesses");
    ok &= ExpectTrue(TryAppendGuess(grid, first_guess), "First guess should append successfully");
    ok &= ExpectTrue(TryAppendGuess(grid, second_guess), "Second guess should append successfully");
    ok &= ExpectTrue(HasGuess(grid, first_guess), "Grid should report the first guess as already played");
    ok &= ExpectTrue(HasGuess(grid, second_guess), "Grid should report the second guess as already played");
    ok &= ExpectTrue(!HasGuess(grid, unseen_guess), "Grid should not report unseen guesses as already played");
    return ok;
}

bool TestWordleGridFinishesAfterSixTurns() {
    const Word solution = MakeWord("SOLAR");
    const Word guess = MakeWord("CRANE");

    WordleGrid grid = MakeWordleGrid(solution);

    bool ok = true;

    for (std::size_t turn_index = 0; turn_index < kMaxTurnCount; ++turn_index) {
        ok &= ExpectTrue(TryAppendGuess(grid, guess), "Expected guess append within six-turn limit to succeed");
    }

    ok &= ExpectTrue(grid.turn_count == kMaxTurnCount, "Grid should cap out at six stored turns");
    ok &= ExpectTrue(grid.IsFinished(), "Grid should report finished after six turns");
    ok &= ExpectTrue(!grid.IsWon(), "Grid should not be won when none of the six guesses match");
    ok &= ExpectTrue(!TryAppendGuess(grid, guess), "Appending a seventh guess should fail");
    return ok;
}

bool TestWordleGridReportsWinWhenAnyGuessMatchesSolution() {
    const Word solution = MakeWord("SOLAR");
    const Word opening_guess = MakeWord("CRANE");
    const Word extra_guess = MakeWord("ALERT");

    WordleGrid grid = MakeWordleGrid(solution);

    bool ok = true;
    ok &= ExpectTrue(TryAppendGuess(grid, opening_guess), "Opening non-winning guess should append successfully");
    ok &= ExpectTrue(TryAppendGuess(grid, solution), "Solution guess should append successfully");
    ok &= ExpectTrue(grid.IsWon(), "Grid should report won when a stored guess matches the solution");
    ok &= ExpectTrue(grid.IsFinished(), "Grid should report finished immediately after a winning guess");
    ok &= ExpectTrue(!TryAppendGuess(grid, extra_guess), "Solved grid should reject any later guesses");
    return ok;
}

bool TestWordleGridRejectsInvalidWords() {
    const Word invalid_solution = MakeWordFromIndices({0, 1, 2, 3, 26});
    const Word valid_guess = MakeWord("CRANE");
    WordleGrid invalid_grid = MakeWordleGrid(invalid_solution);

    bool ok = true;
    ok &= ExpectTrue(!IsValidWordleGrid(invalid_grid), "Grid with an invalid solution should be invalid");
    ok &= ExpectTrue(!TryAppendGuess(invalid_grid, valid_guess), "Grid with an invalid solution should reject guesses");
    return ok;
}

} // namespace

int main() {
    if (!TestFeedbackParsesReadableSymbols()) {
        return 1;
    }

    if (!TestProvideFeedbackMatchesWordleRules()) {
        return 1;
    }

    if (!TestProvideFeedbackHandlesExcessDuplicateLettersInGuess()) {
        return 1;
    }

    if (!TestProvideFeedbackHandlesDuplicateLettersInSolution()) {
        return 1;
    }

    if (!TestProvideFeedbackHandlesDuplicateLettersInGuess()) {
        return 1;
    }

    if (!TestWordleGridStartsEmptyAndStoresSolution()) {
        return 1;
    }

    if (!TestWordleGridAppendsTurnsWithCorrectFeedback()) {
        return 1;
    }

    if (!TestWordleGridTracksPreviousGuesses()) {
        return 1;
    }

    if (!TestWordleGridFinishesAfterSixTurns()) {
        return 1;
    }

    if (!TestWordleGridReportsWinWhenAnyGuessMatchesSolution()) {
        return 1;
    }

    if (!TestWordleGridRejectsInvalidWords()) {
        return 1;
    }

    std::cout << "PASS: wordle_grid_test\n";
    return 0;
}
