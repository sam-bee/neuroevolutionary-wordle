#pragma once

#include <cstddef>
#include <cstdint>

#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "genetic_algorithm/output_embedding_injection.hpp"
#include "training_folder/training_data.hpp"

namespace neuroevolution::genetic_algorithm::device_injection_ops {

template <int WarpWidth>
using OutputEmbeddingInjectionWarpScratch = genetic_algorithm::OutputEmbeddingInjectionWarpScratch<WarpWidth>;

enum class DeviceOutputEmbeddingInjectionStatusCode : int {
    kOk = 0,
    kInvalidTrainingShard = 1,
    kInjectionFailed = 2,
};

template <int WarpWidth>
__device__ inline DeviceOutputEmbeddingInjectionStatusCode TryInjectExpandedOutputEmbeddingTailsConcurrently(
    std::uint8_t *genome_bytes, const std::size_t parent_action_count, const std::size_t first_catalog_word_index,
    const std::size_t injection_count, OutputEmbeddingInjectionWarpScratch<WarpWidth> &scratch) {
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
    const std::size_t lane_index = static_cast<std::size_t>(threadIdx.x % WarpWidth);
    if (lane_index == 0) {
        common::FixedBuffer<float, training_folder::kTrainingWordCatalogCapacity> existing_tail_norms{};
        scratch.status = output_embedding_injection::TryComputeMedianTailNorm(
            tail_rows, parent_action_count, existing_tail_norms, scratch.target_norm);
    }
    __syncwarp();

    if (scratch.status == 0) {
        return DeviceOutputEmbeddingInjectionStatusCode::kInjectionFailed;
    }

    for (std::size_t injection_offset = 0; injection_offset < injection_count; ++injection_offset) {
        if (!TrySeedOutputEmbeddingTailFromHintGridsConcurrently<WarpWidth>(
                genome::GenomePolicyModelParameters(genome_bytes),
                training_word_catalog.words[first_catalog_word_index + injection_offset],
                tail_rows[parent_action_count + injection_offset], scratch)) {
            return DeviceOutputEmbeddingInjectionStatusCode::kInjectionFailed;
        }
        if (!TryScaleTrainableTailToNormConcurrently<WarpWidth>(tail_rows[parent_action_count + injection_offset],
                                                                scratch.target_norm, scratch)) {
            return DeviceOutputEmbeddingInjectionStatusCode::kInjectionFailed;
        }
    }

    return DeviceOutputEmbeddingInjectionStatusCode::kOk;
}

} // namespace neuroevolution::genetic_algorithm::device_injection_ops
