#pragma once

#include <cstddef>
#include <filesystem>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "wordle/word.hpp"

namespace neuroevolution::training_folder {

constexpr std::size_t kInitialTrainingDataShardEntryCount = 5;

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
