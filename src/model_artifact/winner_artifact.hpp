#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "training_folder/training_data.hpp"

namespace neuroevolution::model_artifact {

struct WinnerArtifactMetadata {
    float best_fitness = 0.0f;
    std::size_t generation_index = 0;
    std::size_t best_index = 0;
    std::uint32_t best_slot_index = static_cast<std::uint32_t>(-1);
    std::size_t action_count = 0;
    std::size_t genome_byte_count = 0;
    std::uint32_t seed = 0;
    std::filesystem::path action_space_path{};
};

struct WinnerArtifactPaths {
    std::filesystem::path binary_path{};
    std::filesystem::path metadata_path{};
    std::string timestamp_local{};
};

bool TryWriteWinnerArtifact(const std::filesystem::path &directory, const std::uint8_t *genome_bytes,
                            const training_folder::TrainingWordCatalog &action_space_words,
                            const WinnerArtifactMetadata &metadata, WinnerArtifactPaths &paths_out);

} // namespace neuroevolution::model_artifact
