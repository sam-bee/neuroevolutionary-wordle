#include <cstddef>
#include <iostream>
#include <string>
#include <string_view>

#include "play_wordle/interactive_ui.hpp"
#include "wordle/word.hpp"
#include "wordle/wordle_grid.hpp"

namespace {

using neuroevolution::play_wordle::DoesCatalogContainWord;
using neuroevolution::play_wordle::ParseSolutionPrompt;
using neuroevolution::play_wordle::RenderWordleGridAnsi;
using neuroevolution::play_wordle::SolutionPromptParseStatus;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::wordle::MakeWordleGrid;
using neuroevolution::wordle::TryAppendGuess;
using neuroevolution::wordle::TryMakeWordFromAscii;
using neuroevolution::wordle::Word;
using neuroevolution::wordle::WordleGrid;

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

bool MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1], Word &word_out) {
    return TryMakeWordFromAscii(letters, word_out);
}

TrainingWordCatalog MakeCatalog() {
    TrainingWordCatalog catalog{};
    (void)MakeWord("CRANE", catalog.words[0]);
    (void)MakeWord("SLATE", catalog.words[1]);
    (void)MakeWord("TRACE", catalog.words[2]);
    catalog.word_count = 3;
    return catalog;
}

bool TestParseSolutionPromptAcceptsCatalogWords() {
    const TrainingWordCatalog catalog = MakeCatalog();

    const auto result = ParseSolutionPrompt(" slate ", catalog);
    bool ok = true;
    ok &= ExpectTrue(result.status == SolutionPromptParseStatus::kSolutionAccepted,
                     "Expected lowercase catalog word input to be accepted");
    ok &= ExpectTrue(result.message.empty(), "Expected successful parse result to have no message");
    ok &= ExpectTrue(result.solution == catalog.words[1], "Expected parser to normalize solution word to uppercase");
    ok &= ExpectTrue(DoesCatalogContainWord(catalog, result.solution),
                     "Expected accepted solution to still exist in the action catalog");
    return ok;
}

bool TestParseSolutionPromptRecognizesExitAndRejectsUnknownWords() {
    const TrainingWordCatalog catalog = MakeCatalog();

    const auto exit_result = ParseSolutionPrompt("/exit", catalog);
    const auto unknown_word_result = ParseSolutionPrompt("spoon", catalog);

    bool ok = true;
    ok &= ExpectTrue(exit_result.status == SolutionPromptParseStatus::kExitRequested,
                     "Expected /exit to request process shutdown");
    ok &= ExpectTrue(unknown_word_result.status == SolutionPromptParseStatus::kInvalid,
                     "Expected unknown solutions to be rejected");
    ok &= ExpectTrue(unknown_word_result.message.find("model action space") != std::string::npos,
                     "Expected rejection message to mention the model action space");
    return ok;
}

bool TestRenderWordleGridAnsiShowsColoredTilesAndEmptyRows() {
    Word solution{};
    Word guess{};
    bool ok = MakeWord("LEVEL", solution);
    ok &= MakeWord("HELLO", guess);
    if (!ok) {
        return false;
    }

    WordleGrid grid = MakeWordleGrid(solution);
    ok &= TryAppendGuess(grid, guess);
    if (!ok) {
        return false;
    }

    const std::string rendered = RenderWordleGridAnsi(grid);
    ok &= ExpectTrue(rendered.find("\033[30;43m") != std::string::npos,
                     "Expected rendered grid to include yellow tile styling");
    ok &= ExpectTrue(rendered.find("\033[30;42m") != std::string::npos,
                     "Expected rendered grid to include green tile styling");
    ok &= ExpectTrue(rendered.find("\033[37;100m") != std::string::npos,
                     "Expected rendered grid to include grey tile styling");
    ok &= ExpectTrue(rendered.find("\033[2;37m . \033[0m") != std::string::npos,
                     "Expected rendered grid to show dim empty rows");
    ok &= ExpectTrue(rendered.find('H') != std::string::npos, "Expected rendered grid to include guessed letters");

    std::size_t newline_count = 0;
    for (const char character : rendered) {
        if (character == '\n') {
            ++newline_count;
        }
    }

    ok &= ExpectTrue(newline_count == (neuroevolution::wordle::kMaxTurnCount - 1),
                     "Expected rendered grid to contain one line per Wordle row");
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= TestParseSolutionPromptAcceptsCatalogWords();
    ok &= TestParseSolutionPromptRecognizesExitAndRejectsUnknownWords();
    ok &= TestRenderWordleGridAnsiShowsColoredTilesAndEmptyRows();

    if (!ok) {
        return 1;
    }

    std::cout << "PASS: interactive_ui_test\n";
    return 0;
}
