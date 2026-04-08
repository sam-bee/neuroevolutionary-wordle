#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::training_folder::DefaultActionSpacePath;
using neuroevolution::training_folder::IsValidTrainingDataShard;
using neuroevolution::training_folder::LoadInitialTrainingDataShardFromActionSpace;
using neuroevolution::training_folder::TrainingDataShard;
using neuroevolution::wordle::TryMakeWordFromAscii;
using neuroevolution::wordle::Word;

Word MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    Word word{};

    if (!TryMakeWordFromAscii(letters, word)) {
        throw std::invalid_argument(
            "Training-data test word literal must contain exactly five uppercase ASCII letters.");
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

bool TestLoadInitialTrainingDataShardReadsFirstFiveActionSpaceWords() {
    const std::filesystem::path action_space_path = DefaultActionSpacePath();
    const TrainingDataShard shard = LoadInitialTrainingDataShardFromActionSpace(action_space_path);

    bool ok = true;
    ok &= ExpectTrue(std::filesystem::exists(action_space_path), "Expected action-space file to exist");
    ok &= ExpectTrue(shard.entry_count == 5, "Expected initial training-data shard to contain five entries");
    ok &= ExpectTrue(IsValidTrainingDataShard(shard), "Expected loaded training-data shard to be valid");
    ok &= ExpectWordEquals(shard.entries[0].word, MakeWord("AARGH"), "first training-data word");
    ok &= ExpectWordEquals(shard.entries[1].word, MakeWord("ABACK"), "second training-data word");
    ok &= ExpectWordEquals(shard.entries[2].word, MakeWord("ABASE"), "third training-data word");
    ok &= ExpectWordEquals(shard.entries[3].word, MakeWord("ABATE"), "fourth training-data word");
    ok &= ExpectWordEquals(shard.entries[4].word, MakeWord("ABBAS"), "fifth training-data word");
    return ok;
}

} // namespace

int main() {
    if (!TestLoadInitialTrainingDataShardReadsFirstFiveActionSpaceWords()) {
        return 1;
    }

    std::cout << "PASS: training_data_test\n";
    return 0;
}
