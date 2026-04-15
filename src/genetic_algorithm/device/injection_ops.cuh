#pragma once

#include <cstddef>
#include <cstdint>

#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "genetic_algorithm/output_embedding_injection.hpp"
#include "training_folder/training_data.hpp"

namespace neuroevolution::genetic_algorithm::device_injection_ops {

enum class DeviceOutputEmbeddingInjectionStatusCode : int {
    kOk = 0,
    kInvalidTrainingShard = 1,
    kInjectionFailed = 2,
};

__device__ inline DeviceOutputEmbeddingInjectionStatusCode
TryInjectExpandedOutputEmbeddingTails(std::uint8_t *genome_bytes, const std::size_t parent_action_count,
                                      const std::size_t first_catalog_word_index, const std::size_t injection_count) {
    if ((genome_bytes == nullptr) || (parent_action_count == 0) || (injection_count == 0) ||
        (first_catalog_word_index != parent_action_count)) {
        return DeviceOutputEmbeddingInjectionStatusCode::kInjectionFailed;
    }

    const training_folder::TrainingWordCatalog &training_word_catalog = training_folder::DeviceTrainingWordCatalog();
    if (!training_folder::IsValidTrainingWordCatalog(training_word_catalog) ||
        (first_catalog_word_index >= training_word_catalog.word_count) ||
        (injection_count > (training_word_catalog.word_count - first_catalog_word_index))) {
        return DeviceOutputEmbeddingInjectionStatusCode::kInvalidTrainingShard;
    }

    genome::TrainableActionEmbeddingTail *tail_rows = genome::GenomeTailRows(genome_bytes);
    for (std::size_t injection_offset = 0; injection_offset < injection_count; ++injection_offset) {
        if (!TrySeedOutputEmbeddingTailFromHintGrids(
                genome::GenomePolicyModelParameters(genome_bytes),
                training_word_catalog.words[first_catalog_word_index + injection_offset],
                tail_rows[parent_action_count + injection_offset])) {
            return DeviceOutputEmbeddingInjectionStatusCode::kInjectionFailed;
        }
    }

    return DeviceOutputEmbeddingInjectionStatusCode::kOk;
}

} // namespace neuroevolution::genetic_algorithm::device_injection_ops
