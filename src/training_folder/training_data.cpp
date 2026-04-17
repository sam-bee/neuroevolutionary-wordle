#include "training_folder/training_data.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace neuroevolution::training_folder {

namespace {

bool TryParseWordFromLine(const std::string &line, wordle::Word &word) {
    if (line.size() != wordle::kWordLength) {
        return false;
    }

    for (std::size_t position = 0; position < wordle::kWordLength; ++position) {
        const char value = line[position];
        if (!wordle::IsAsciiUppercaseLetter(value)) {
            return false;
        }

        word.letter_indices[position] = wordle::LetterIndexFromAscii(value);
    }

    return true;
}

std::string NormalizeLine(std::string line) {
    if (!line.empty() && (line.back() == '\r')) {
        line.pop_back();
    }

    return line;
}

} // namespace

std::filesystem::path DefaultActionSpacePath() {
    return std::filesystem::path(NEUROEVOLUTION_PROJECT_SOURCE_DIR) / "data" / "action-space-randomised.txt";
}

bool TryLoadTrainingWordCatalogFromActionSpace(const std::filesystem::path &action_space_path,
                                               TrainingWordCatalog &catalog) {
    catalog = {};

    std::ifstream input_file(action_space_path);
    if (!input_file.is_open()) {
        return false;
    }

    std::string line{};
    while (std::getline(input_file, line)) {
        const std::string normalized_line = NormalizeLine(line);
        if (normalized_line.empty()) {
            continue;
        }

        if (catalog.word_count >= kTrainingWordCatalogCapacity) {
            return false;
        }

        wordle::Word word{};
        if (!TryParseWordFromLine(normalized_line, word)) {
            return false;
        }

        catalog.words[catalog.word_count] = word;
        ++catalog.word_count;
    }

    return catalog.word_count > 0;
}

TrainingWordCatalog LoadTrainingWordCatalogFromActionSpace(const std::filesystem::path &action_space_path) {
    TrainingWordCatalog catalog{};
    if (!TryLoadTrainingWordCatalogFromActionSpace(action_space_path, catalog)) {
        throw std::runtime_error(
            "Could not load the training-word catalog from the configured action-space word list.");
    }

    return catalog;
}

TrainingWordCatalog LoadTrainingWordCatalogFromActionSpace() {
    return LoadTrainingWordCatalogFromActionSpace(DefaultActionSpacePath());
}

} // namespace neuroevolution::training_folder
