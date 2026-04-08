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
    return std::filesystem::path(NEUROEVOLUTION_PROJECT_SOURCE_DIR) / "data" / "action-space.txt";
}

bool TryLoadInitialTrainingDataShardFromActionSpace(const std::filesystem::path &action_space_path,
                                                    TrainingDataShard &shard) {
    shard = {};

    std::ifstream input_file(action_space_path);
    if (!input_file.is_open()) {
        return false;
    }

    std::string line{};
    while ((shard.entry_count < kInitialTrainingDataShardEntryCount) && std::getline(input_file, line)) {
        const std::string normalized_line = NormalizeLine(line);
        if (normalized_line.empty()) {
            continue;
        }

        wordle::Word word{};
        if (!TryParseWordFromLine(normalized_line, word)) {
            return false;
        }

        shard.entries[shard.entry_count].word = word;
        ++shard.entry_count;
    }

    return shard.entry_count == kInitialTrainingDataShardEntryCount;
}

TrainingDataShard LoadInitialTrainingDataShardFromActionSpace(const std::filesystem::path &action_space_path) {
    TrainingDataShard shard{};
    if (!TryLoadInitialTrainingDataShardFromActionSpace(action_space_path, shard)) {
        throw std::runtime_error("Could not load the initial training-data shard from the action-space word list.");
    }

    return shard;
}

TrainingDataShard LoadInitialTrainingDataShardFromActionSpace() {
    return LoadInitialTrainingDataShardFromActionSpace(DefaultActionSpacePath());
}

} // namespace neuroevolution::training_folder
