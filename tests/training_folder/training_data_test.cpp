#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::training_folder::ActiveTrainingWordCountForGeneration;
using neuroevolution::training_folder::DefaultActionSpacePath;
using neuroevolution::training_folder::IsValidTrainingWordCatalog;
using neuroevolution::training_folder::kPhasedCurriculumSecondShardGeneration;
using neuroevolution::training_folder::kTrainingDataCurriculumEntryCount;
using neuroevolution::training_folder::kTrainingDataEntriesPerShard;
using neuroevolution::training_folder::kTrainingWordCatalogCapacity;
using neuroevolution::training_folder::LoadTrainingWordCatalogFromActionSpace;
using neuroevolution::training_folder::TrainingWordCatalog;
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

bool ExpectTrainingWordCatalogMatchesExpectedPrefix(const TrainingWordCatalog &catalog,
                                                    const std::string_view label_prefix) {
    bool ok = true;

    ok &= ExpectTrue(catalog.word_count == kTrainingWordCatalogCapacity,
                     std::string(label_prefix) + " should contain the full training-word catalog");

    for (std::size_t entry_index = 0; entry_index < kExpectedTrainingWords.size(); ++entry_index) {
        ok &= ExpectWordEquals(catalog.words[entry_index], MakeWord(kExpectedTrainingWords[entry_index]),
                               std::string(label_prefix) + " word " + std::to_string(entry_index));
    }

    return ok;
}

bool TestLoadTrainingWordCatalogReadsRandomisedActionSpaceWords() {
    const std::filesystem::path action_space_path = DefaultActionSpacePath();
    const TrainingWordCatalog catalog = LoadTrainingWordCatalogFromActionSpace(action_space_path);

    bool ok = true;
    ok &= ExpectTrue(std::filesystem::exists(action_space_path), "Expected randomized action-space file to exist");
    ok &= ExpectTrue(IsValidTrainingWordCatalog(catalog), "Expected loaded training-word catalog to be valid");
    ok &= ExpectTrainingWordCatalogMatchesExpectedPrefix(catalog, "training-word catalog");
    return ok;
}

bool TestPhasedCurriculumUsesOneShardBeforeGenerationOneHundredAndBothAfterwards() {
    bool ok = true;
    ok &= ExpectTrue(ActiveTrainingWordCountForGeneration(kTrainingDataCurriculumEntryCount, 0) ==
                         kTrainingDataEntriesPerShard,
                     "Expected phased curriculum generation zero to use only the first training-data shard");
    ok &= ExpectTrue(
        ActiveTrainingWordCountForGeneration(kTrainingDataCurriculumEntryCount,
                                             kPhasedCurriculumSecondShardGeneration - 1) == kTrainingDataEntriesPerShard,
        "Expected phased curriculum to keep using only the first training-data shard before generation 100");
    ok &= ExpectTrue(ActiveTrainingWordCountForGeneration(kTrainingDataCurriculumEntryCount,
                                                          kPhasedCurriculumSecondShardGeneration) ==
                         kTrainingDataCurriculumEntryCount,
                     "Expected phased curriculum generation 100 to include both training-data shards");
    ok &= ExpectTrue(ActiveTrainingWordCountForGeneration(kTrainingDataCurriculumEntryCount,
                                                          kPhasedCurriculumSecondShardGeneration + 1) ==
                         kTrainingDataCurriculumEntryCount,
                     "Expected phased curriculum to keep both training-data shards active after generation 100");
    return ok;
}

} // namespace

int main() {
    if (!TestLoadTrainingWordCatalogReadsRandomisedActionSpaceWords() ||
        !TestPhasedCurriculumUsesOneShardBeforeGenerationOneHundredAndBothAfterwards()) {
        return 1;
    }

    std::cout << "PASS: training_data_test\n";
    return 0;
}
