#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::training_folder::ActiveTrainingDataEntryCountForGeneration;
using neuroevolution::training_folder::DefaultActionSpacePath;
using neuroevolution::training_folder::IsValidTrainingDataShard;
using neuroevolution::training_folder::kPhasedCurriculumSecondShardGeneration;
using neuroevolution::training_folder::kTrainingDataCurriculumEntryCount;
using neuroevolution::training_folder::kTrainingDataEntriesPerShard;
using neuroevolution::training_folder::LoadInitialTrainingDataShardFromActionSpace;
using neuroevolution::training_folder::TrainingDataShard;
using neuroevolution::wordle::Word;

constexpr std::array<const char *, kTrainingDataCurriculumEntryCount> kExpectedTrainingWords = {
    "MINOS", "VODKA", "RAZOR", "GRADS", "CURLS", "BILGE", "GREET", "PYLON", "ENTER", "READY",
    "VERDE", "AUGER", "FOOTS", "BRACE", "PURTY", "SPORT", "TIRES", "FRISK", "AFFIX", "CHUMS",
};

Word MakeWord(const std::string_view letters) {
    if (letters.size() != neuroevolution::wordle::kWordLength) {
        throw std::invalid_argument("Training-data test word view must contain exactly five characters.");
    }

    Word word{};
    for (std::size_t position = 0; position < neuroevolution::wordle::kWordLength; ++position) {
        const char value = letters[position];
        if (!neuroevolution::wordle::IsAsciiUppercaseLetter(value)) {
            throw std::invalid_argument("Training-data test word view must contain only uppercase ASCII letters.");
        }

        word.letter_indices[position] = neuroevolution::wordle::LetterIndexFromAscii(value);
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

bool ExpectTrainingShardMatchesExpectedWords(const TrainingDataShard &shard, const std::string_view label_prefix) {
    bool ok = true;

    ok &= ExpectTrue(shard.entry_count == kExpectedTrainingWords.size(),
                     std::string(label_prefix) + " should contain the expected number of entries");

    const std::size_t comparison_count =
        (shard.entry_count < kExpectedTrainingWords.size()) ? shard.entry_count : kExpectedTrainingWords.size();

    for (std::size_t entry_index = 0; entry_index < comparison_count; ++entry_index) {
        ok &= ExpectWordEquals(shard.entries[entry_index].word, MakeWord(kExpectedTrainingWords[entry_index]),
                               std::string(label_prefix) + " word " + std::to_string(entry_index));
    }

    return ok;
}

bool TestLoadInitialTrainingDataShardReadsTopTwentyRandomisedActionSpaceWords() {
    const std::filesystem::path action_space_path = DefaultActionSpacePath();
    const TrainingDataShard shard = LoadInitialTrainingDataShardFromActionSpace(action_space_path);

    bool ok = true;
    ok &= ExpectTrue(std::filesystem::exists(action_space_path), "Expected randomized action-space file to exist");
    ok &= ExpectTrue(IsValidTrainingDataShard(shard), "Expected loaded training-data shard to be valid");
    ok &= ExpectTrainingShardMatchesExpectedWords(shard, "initial training-data shard");
    return ok;
}

bool TestPhasedCurriculumUsesOneShardBeforeGenerationOneHundredAndBothAfterwards() {
    const TrainingDataShard shard = LoadInitialTrainingDataShardFromActionSpace();

    bool ok = true;
    ok &= ExpectTrue(ActiveTrainingDataEntryCountForGeneration(shard, 0) == kTrainingDataEntriesPerShard,
                     "Expected phased curriculum generation zero to use only the first training-data shard");
    ok &=
        ExpectTrue(ActiveTrainingDataEntryCountForGeneration(shard, kPhasedCurriculumSecondShardGeneration - 1) ==
                       kTrainingDataEntriesPerShard,
                   "Expected phased curriculum to keep using only the first training-data shard before generation 100");
    ok &= ExpectTrue(ActiveTrainingDataEntryCountForGeneration(shard, kPhasedCurriculumSecondShardGeneration) ==
                         kTrainingDataCurriculumEntryCount,
                     "Expected phased curriculum generation 100 to include both training-data shards");
    ok &= ExpectTrue(ActiveTrainingDataEntryCountForGeneration(shard, kPhasedCurriculumSecondShardGeneration + 1) ==
                         kTrainingDataCurriculumEntryCount,
                     "Expected phased curriculum to keep both training-data shards active after generation 100");
    return ok;
}

} // namespace

int main() {
    if (!TestLoadInitialTrainingDataShardReadsTopTwentyRandomisedActionSpaceWords() ||
        !TestPhasedCurriculumUsesOneShardBeforeGenerationOneHundredAndBothAfterwards()) {
        return 1;
    }

    std::cout << "PASS: training_data_test\n";
    return 0;
}
