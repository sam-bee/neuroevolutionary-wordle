#include "play_wordle/interactive_ui.hpp"

#include <cctype>
#include <sstream>

#include "wordle/feedback.hpp"

namespace neuroevolution::play_wordle {

namespace {

constexpr const char *kAnsiReset = "\033[0m";
constexpr const char *kAnsiGreenTile = "\033[30;42m";
constexpr const char *kAnsiYellowTile = "\033[30;43m";
constexpr const char *kAnsiGreyTile = "\033[37;100m";
constexpr const char *kAnsiEmptyTile = "\033[2;37m";

std::string TrimAsciiWhitespace(const std::string_view input) {
    std::size_t begin = 0;
    while ((begin < input.size()) && std::isspace(static_cast<unsigned char>(input[begin])) != 0) {
        ++begin;
    }

    std::size_t end = input.size();
    while ((end > begin) && std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
        --end;
    }

    return std::string(input.substr(begin, end - begin));
}

const char *TileAnsiPrefix(const wordle::TileFeedback feedback) noexcept {
    switch (feedback) {
    case wordle::TileFeedback::green:
        return kAnsiGreenTile;
    case wordle::TileFeedback::yellow:
        return kAnsiYellowTile;
    case wordle::TileFeedback::grey:
        return kAnsiGreyTile;
    }

    return kAnsiGreyTile;
}

bool TryMakeUppercaseWord(const std::string_view input, wordle::Word &word_out) {
    if (input.size() != wordle::kWordLength) {
        return false;
    }

    char letters[wordle::kWordLength + 1]{};
    for (std::size_t position = 0; position < wordle::kWordLength; ++position) {
        const unsigned char raw = static_cast<unsigned char>(input[position]);
        if (!std::isalpha(raw)) {
            return false;
        }

        letters[position] = static_cast<char>(std::toupper(raw));
    }

    letters[wordle::kWordLength] = '\0';
    return wordle::TryMakeWordFromAscii(letters, word_out);
}

void AppendRenderedTile(std::ostringstream &stream, const char *ansi_prefix, const char letter) {
    stream << ansi_prefix << ' ' << letter << ' ' << kAnsiReset;
}

} // namespace

std::string WordToAsciiString(const wordle::Word &word) {
    std::string text(wordle::kWordLength, 'A');
    for (std::size_t position = 0; position < wordle::kWordLength; ++position) {
        text[position] = static_cast<char>('A' + word.letter_indices[position]);
    }

    return text;
}

bool DoesCatalogContainWord(const training_folder::TrainingWordCatalog &catalog, const wordle::Word &word) noexcept {
    if (!training_folder::IsValidTrainingWordCatalog(catalog)) {
        return false;
    }

    for (std::size_t word_index = 0; word_index < catalog.word_count; ++word_index) {
        if (catalog.words[word_index] == word) {
            return true;
        }
    }

    return false;
}

SolutionPromptParseResult ParseSolutionPrompt(const std::string_view input,
                                              const training_folder::TrainingWordCatalog &action_space_words) {
    SolutionPromptParseResult result{};
    const std::string trimmed_input = TrimAsciiWhitespace(input);
    if (trimmed_input.empty()) {
        result.message = "Enter a five-letter solution word or /exit.";
        return result;
    }

    if (trimmed_input == "/exit") {
        result.status = SolutionPromptParseStatus::kExitRequested;
        result.message.clear();
        return result;
    }

    if (!training_folder::IsValidTrainingWordCatalog(action_space_words)) {
        result.message = "Loaded action space is invalid.";
        return result;
    }

    if (!TryMakeUppercaseWord(trimmed_input, result.solution)) {
        result.message = "Solution must be exactly five ASCII letters.";
        return result;
    }

    if (!DoesCatalogContainWord(action_space_words, result.solution)) {
        result.message = "Solution must exist in the model action space.";
        return result;
    }

    result.status = SolutionPromptParseStatus::kSolutionAccepted;
    result.message.clear();
    return result;
}

std::string RenderWordleGridAnsi(const wordle::WordleGrid &grid) {
    std::ostringstream stream{};
    for (std::size_t turn_index = 0; turn_index < wordle::kMaxTurnCount; ++turn_index) {
        if (turn_index < grid.turn_count) {
            const wordle::Turn &turn = grid.turns[turn_index];
            const std::string guess_text = WordToAsciiString(turn.guess);
            for (std::size_t position = 0; position < wordle::kWordLength; ++position) {
                AppendRenderedTile(stream, TileAnsiPrefix(turn.feedback[position]), guess_text[position]);
                if (position + 1 < wordle::kWordLength) {
                    stream << ' ';
                }
            }
        } else {
            for (std::size_t position = 0; position < wordle::kWordLength; ++position) {
                AppendRenderedTile(stream, kAnsiEmptyTile, '.');
                if (position + 1 < wordle::kWordLength) {
                    stream << ' ';
                }
            }
        }

        if (turn_index + 1 < wordle::kMaxTurnCount) {
            stream << '\n';
        }
    }

    return stream.str();
}

} // namespace neuroevolution::play_wordle
