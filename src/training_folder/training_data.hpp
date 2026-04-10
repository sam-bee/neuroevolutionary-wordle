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
constexpr std::size_t kInitialTrainingDataShardEntryCount = kTrainingDataCurriculumEntryCount;

struct TrainingDataEntry {
    wordle::Word word{};
};

struct TrainingDataShard {
    common::FixedBuffer<TrainingDataEntry, kInitialTrainingDataShardEntryCount> entries{};
    std::size_t entry_count = 0;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidTrainingDataEntry(const TrainingDataEntry &entry) noexcept {
    return wordle::IsValidWord(entry.word);
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidTrainingDataShard(const TrainingDataShard &shard) noexcept {
    if (shard.entry_count > kInitialTrainingDataShardEntryCount) {
        return false;
    }

    for (std::size_t entry_index = 0; entry_index < shard.entry_count; ++entry_index) {
        if (!IsValidTrainingDataEntry(shard.entries[entry_index])) {
            return false;
        }
    }

    return true;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
ActiveTrainingDataEntryCountForGeneration(const TrainingDataShard &shard, const std::size_t generation_index) noexcept {
    if (!IsValidTrainingDataShard(shard) || (shard.entry_count == 0)) {
        return 0;
    }

    const std::size_t first_shard_entry_count =
        (shard.entry_count < kTrainingDataEntriesPerShard) ? shard.entry_count : kTrainingDataEntriesPerShard;
    if (generation_index < kPhasedCurriculumSecondShardGeneration) {
        return first_shard_entry_count;
    }

    return shard.entry_count;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
SelectableTrainingActionCount(const TrainingDataShard &shard) noexcept {
    if (!IsValidTrainingDataShard(shard)) {
        return 0;
    }

    return shard.entry_count;
}

std::filesystem::path DefaultActionSpacePath();

bool TryLoadInitialTrainingDataShardFromActionSpace(const std::filesystem::path &action_space_path,
                                                    TrainingDataShard &shard);

TrainingDataShard LoadInitialTrainingDataShardFromActionSpace(const std::filesystem::path &action_space_path);

TrainingDataShard LoadInitialTrainingDataShardFromActionSpace();

bool UploadTrainingDataShardToDeviceConstantMemory(const TrainingDataShard &shard);

void UploadTrainingDataShardToDeviceConstantMemoryOrThrow(const TrainingDataShard &shard);

#if defined(__CUDACC__)
__device__ const TrainingDataShard &DeviceTrainingDataShard() noexcept;
#endif

} // namespace neuroevolution::training_folder
