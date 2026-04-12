#pragma once

#include <cstddef>
#include <filesystem>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "wordle/word.hpp"

namespace neuroevolution::training_folder {

constexpr std::size_t kTrainingDataShardCount = 2;
constexpr std::size_t kTrainingDataEntriesPerShard = 10;
constexpr std::size_t kTrainingDataCurriculumEntryCount = kTrainingDataShardCount * kTrainingDataEntriesPerShard;
constexpr std::size_t kPhasedCurriculumSecondShardGeneration = 100;
constexpr std::size_t kTrainingWordCatalogCapacity = 4739;

struct WordCountSchedule {
    std::size_t initial_word_count = kTrainingDataCurriculumEntryCount;
    std::size_t word_count_step = 1;
    std::size_t word_count_step_period_generations = 1;
};

struct TrainingWordCatalog {
    common::FixedBuffer<wordle::Word, kTrainingWordCatalogCapacity> words{};
    std::size_t word_count = 0;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidTrainingWordCatalog(const TrainingWordCatalog &catalog) noexcept {
    if (catalog.word_count > kTrainingWordCatalogCapacity) {
        return false;
    }

    for (std::size_t word_index = 0; word_index < catalog.word_count; ++word_index) {
        if (!wordle::IsValidWord(catalog.words[word_index])) {
            return false;
        }
    }

    return true;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidWordCountSchedule(const WordCountSchedule &schedule) noexcept {
    return (schedule.initial_word_count > 0) && (schedule.word_count_step > 0) &&
           (schedule.word_count_step_period_generations > 0);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t ScheduledWordCountForGeneration(
    const WordCountSchedule &schedule, const std::size_t catalog_word_count, const std::size_t generation_index) noexcept {
    if (!IsValidWordCountSchedule(schedule) || (catalog_word_count == 0)) {
        return 0;
    }

    const std::size_t initial_word_count =
        (schedule.initial_word_count < catalog_word_count) ? schedule.initial_word_count : catalog_word_count;
    const std::size_t completed_schedule_steps = generation_index / schedule.word_count_step_period_generations;
    if (completed_schedule_steps == 0) {
        return initial_word_count;
    }

    if (schedule.word_count_step > ((catalog_word_count - initial_word_count) / completed_schedule_steps)) {
        return catalog_word_count;
    }

    const std::size_t scheduled_word_count = initial_word_count + (completed_schedule_steps * schedule.word_count_step);
    return (scheduled_word_count < catalog_word_count) ? scheduled_word_count : catalog_word_count;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
ActiveTrainingWordCountForGeneration(const std::size_t training_word_count, const std::size_t generation_index) noexcept {
    if (training_word_count == 0) {
        return 0;
    }

    const std::size_t first_shard_word_count =
        (training_word_count < kTrainingDataEntriesPerShard) ? training_word_count : kTrainingDataEntriesPerShard;
    if (generation_index < kPhasedCurriculumSecondShardGeneration) {
        return first_shard_word_count;
    }

    return training_word_count;
}

std::filesystem::path DefaultActionSpacePath();

bool TryLoadTrainingWordCatalogFromActionSpace(const std::filesystem::path &action_space_path,
                                               TrainingWordCatalog &catalog);

TrainingWordCatalog LoadTrainingWordCatalogFromActionSpace(const std::filesystem::path &action_space_path);

TrainingWordCatalog LoadTrainingWordCatalogFromActionSpace();

bool UploadTrainingWordCatalogToDeviceConstantMemory(const TrainingWordCatalog &catalog);

void UploadTrainingWordCatalogToDeviceConstantMemoryOrThrow(const TrainingWordCatalog &catalog);

#if defined(__CUDACC__)
__device__ const TrainingWordCatalog &DeviceTrainingWordCatalog() noexcept;
#endif

} // namespace neuroevolution::training_folder
