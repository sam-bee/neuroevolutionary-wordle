#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::training_folder::DefaultActionSpacePath;
using neuroevolution::training_folder::IsValidTrainingWordCatalog;
using neuroevolution::training_folder::kDefaultInitialActiveWordCount;
using neuroevolution::training_folder::kTrainingWordCatalogCapacity;
using neuroevolution::training_folder::LoadTrainingWordCatalogFromActionSpace;
using neuroevolution::training_folder::ScheduledWordCountForGeneration;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::training_folder::WordCountSchedule;
using neuroevolution::wordle::Word;

constexpr std::array<const char *, kDefaultInitialActiveWordCount> kExpectedTrainingWords = {
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

bool TestWordCountScheduleAdvancesByConfiguredStepAndClampsToCatalog() {
    constexpr WordCountSchedule kSchedule{
        .initial_word_count = 50,
        .word_count_step = 50,
        .word_count_step_period_generations = 20,
    };

    bool ok = true;
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 0) == 50,
                     "Expected schedule generation zero to use the initial word count");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 19) == 50,
                     "Expected schedule to hold steady before the first configured period boundary");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 20) == 100,
                     "Expected schedule to add one configured step at generation 20");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 39) == 100,
                     "Expected schedule to hold the stepped count until the next boundary");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 40) == 150,
                     "Expected schedule to add the second configured step at generation 40");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, 120, 1000) == 120,
                     "Expected schedule to clamp to the catalog size when repeated steps would exceed it");
    return ok;
}

bool TestWordCountScheduleCanStayFixedWidthWithZeroStep() {
    constexpr WordCountSchedule kSchedule{
        .initial_word_count = 37,
        .word_count_step = 0,
        .word_count_step_period_generations = 5,
    };

    bool ok = true;
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 0) == 37,
                     "Expected zero-step schedule generation zero to use the initial word count");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 4) == 37,
                     "Expected zero-step schedule to stay fixed before a period boundary");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 500) == 37,
                     "Expected zero-step schedule to remain fixed for later generations");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, 20, 500) == 20,
                     "Expected zero-step schedule to still clamp to the catalog size");
    return ok;
}

} // namespace

int main() {
    if (!TestLoadTrainingWordCatalogReadsRandomisedActionSpaceWords() ||
        !TestWordCountScheduleAdvancesByConfiguredStepAndClampsToCatalog() ||
        !TestWordCountScheduleCanStayFixedWidthWithZeroStep()) {
        return 1;
    }

    std::cout << "PASS: training_data_test\n";
    return 0;
}
