#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/population_initialization.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "model/policy_model/policy_model.hpp"

namespace neuroevolution::genetic_algorithm::genome {

using TrainableActionEmbeddingTail = model::output_embedding::TrainableActionEmbeddingTail;
using PolicyModelParameters = model::policy_model::PolicyModelParameters;

struct DynamicPopulationLayout {
    std::size_t active_individual_count = 0;
    std::size_t generation_index = 0;
    std::size_t action_count = 0;
    std::size_t genome_stride_bytes = 0;
    std::size_t genotype_bytes = 0;
};

struct AlignedGenomeStorageDeleter {
    void operator()(std::uint8_t *pointer) const noexcept;
};

using AlignedGenomeStorage = std::unique_ptr<std::uint8_t[], AlignedGenomeStorageDeleter>;

struct HostPopulation {
    DynamicPopulationLayout layout{};
    AlignedGenomeStorage genomes{};
};

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

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t ComputeDynamicGenomeStrideBytes(
    const std::size_t action_count) noexcept {
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

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidDynamicPopulationLayout(
    const DynamicPopulationLayout &layout) noexcept {
    return (layout.active_individual_count > 0) && (layout.action_count > 0) &&
           (layout.genome_stride_bytes == ComputeDynamicGenomeStrideBytes(layout.action_count)) &&
           (layout.genotype_bytes == (layout.active_individual_count * layout.genome_stride_bytes));
}

inline NEUROEVOLUTION_HOST_DEVICE std::uint8_t *GenomeBytesAt(std::uint8_t *population_genomes,
                                                              const DynamicPopulationLayout &layout,
                                                              const std::size_t individual_index) noexcept {
    return population_genomes + (individual_index * layout.genome_stride_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE const std::uint8_t *GenomeBytesAt(const std::uint8_t *population_genomes,
                                                                    const DynamicPopulationLayout &layout,
                                                                    const std::size_t individual_index) noexcept {
    return population_genomes + (individual_index * layout.genome_stride_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE PolicyModelParameters &GenomePolicyModelParameters(
    std::uint8_t *genome_bytes) noexcept {
    return *reinterpret_cast<PolicyModelParameters *>(genome_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE const PolicyModelParameters &GenomePolicyModelParameters(
    const std::uint8_t *genome_bytes) noexcept {
    return *reinterpret_cast<const PolicyModelParameters *>(genome_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE TrainableActionEmbeddingTail *GenomeTailRows(std::uint8_t *genome_bytes) noexcept {
    return reinterpret_cast<TrainableActionEmbeddingTail *>(genome_bytes + DynamicGenomeTailOffsetBytes());
}

inline NEUROEVOLUTION_HOST_DEVICE const TrainableActionEmbeddingTail *GenomeTailRows(
    const std::uint8_t *genome_bytes) noexcept {
    return reinterpret_cast<const TrainableActionEmbeddingTail *>(genome_bytes + DynamicGenomeTailOffsetBytes());
}

inline std::uint8_t *HostGenomeBytesAt(HostPopulation &population, const std::size_t individual_index) noexcept {
    return GenomeBytesAt(population.genomes.get(), population.layout, individual_index);
}

inline const std::uint8_t *HostGenomeBytesAt(const HostPopulation &population,
                                             const std::size_t individual_index) noexcept {
    return GenomeBytesAt(population.genomes.get(), population.layout, individual_index);
}

bool TryAllocateHostGenomeStorage(HostPopulation &population);

bool TryInitializeRandomHostPopulation(
    HostPopulation &population, std::size_t population_size, std::size_t action_count, std::uint32_t seed,
    const PopulationInitializationConfig &config = {});

} // namespace neuroevolution::genetic_algorithm::genome
