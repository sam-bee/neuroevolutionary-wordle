#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "model/policy_model/policy_model.hpp"

namespace neuroevolution::genetic_algorithm::genome {

using TrainableActionEmbeddingTail = model::output_embedding::TrainableActionEmbeddingTail;
using PolicyModelParameters = model::policy_model::PolicyModelParameters;

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t DynamicGenomeAlignment() noexcept {
    return (alignof(PolicyModelParameters) > alignof(TrainableActionEmbeddingTail))
               ? alignof(PolicyModelParameters)
               : alignof(TrainableActionEmbeddingTail);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t RoundUpBytes(const std::size_t value,
                                                              const std::size_t alignment) noexcept {
    return (alignment == 0) ? value : ((value + alignment - 1) / alignment) * alignment;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t DynamicGenomeTailOffsetBytes() noexcept {
    return RoundUpBytes(sizeof(PolicyModelParameters), alignof(TrainableActionEmbeddingTail));
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
ComputeDynamicGenomeStrideBytes(const std::size_t action_count) noexcept {
    return (action_count == 0)
               ? 0
               : RoundUpBytes(DynamicGenomeTailOffsetBytes() + (action_count * sizeof(TrainableActionEmbeddingTail)),
                              DynamicGenomeAlignment());
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
PopulationSizeForGenotypeBudgetBytes(const std::size_t genotype_memory_budget_bytes, const std::size_t action_count,
                                     const std::size_t population_size_ceiling = 0) noexcept {
    const std::size_t genome_stride_bytes = ComputeDynamicGenomeStrideBytes(action_count);
    if ((genome_stride_bytes == 0) || (genotype_memory_budget_bytes < genome_stride_bytes)) {
        return 0;
    }

    const std::size_t uncapped_population_size = genotype_memory_budget_bytes / genome_stride_bytes;
    if ((population_size_ceiling == 0) || (uncapped_population_size < population_size_ceiling)) {
        return uncapped_population_size;
    }

    return population_size_ceiling;
}

inline NEUROEVOLUTION_HOST_DEVICE PolicyModelParameters &
GenomePolicyModelParameters(std::uint8_t *genome_bytes) noexcept {
    return *reinterpret_cast<PolicyModelParameters *>(genome_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE const PolicyModelParameters &
GenomePolicyModelParameters(const std::uint8_t *genome_bytes) noexcept {
    return *reinterpret_cast<const PolicyModelParameters *>(genome_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE TrainableActionEmbeddingTail *GenomeTailRows(std::uint8_t *genome_bytes) noexcept {
    return reinterpret_cast<TrainableActionEmbeddingTail *>(genome_bytes + DynamicGenomeTailOffsetBytes());
}

inline NEUROEVOLUTION_HOST_DEVICE const TrainableActionEmbeddingTail *
GenomeTailRows(const std::uint8_t *genome_bytes) noexcept {
    return reinterpret_cast<const TrainableActionEmbeddingTail *>(genome_bytes + DynamicGenomeTailOffsetBytes());
}

} // namespace neuroevolution::genetic_algorithm::genome
