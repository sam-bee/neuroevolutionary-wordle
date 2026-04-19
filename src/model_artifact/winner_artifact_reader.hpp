#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "model_artifact/winner_artifact.hpp"

namespace neuroevolution::model_artifact {

struct LoadedWinnerArtifact {
    WinnerArtifactMetadata metadata{};
    training_folder::TrainingWordCatalog action_space_words{};
    std::unique_ptr<std::uint8_t[]> genome_bytes{};
    std::string timestamp_local{};
};

bool TryReadWinnerArtifact(const std::filesystem::path &binary_path, const std::filesystem::path &metadata_path,
                           LoadedWinnerArtifact &artifact_out);

} // namespace neuroevolution::model_artifact
