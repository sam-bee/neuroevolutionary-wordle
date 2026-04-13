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
using DynamicArenaSlotId = std::uint32_t;

constexpr DynamicArenaSlotId kInvalidDynamicArenaSlotId = static_cast<DynamicArenaSlotId>(-1);

struct DynamicTailSchema {
    std::size_t action_count = 0;
    std::size_t chunk_action_capacity = 0;
    std::size_t chunk_count = 0;
};

struct DynamicTailRowLocation {
    std::size_t chunk_index = 0;
    std::size_t action_offset_in_chunk = 0;
};

struct DynamicBodyView {
    PolicyModelParameters *parameters = nullptr;
};

struct ConstDynamicBodyView {
    const PolicyModelParameters *parameters = nullptr;
};

struct DynamicTailChunkView {
    TrainableActionEmbeddingTail *rows = nullptr;
    std::size_t chunk_index = 0;
    std::size_t first_action_index = 0;
    std::size_t active_action_count = 0;
    std::size_t action_capacity = 0;
};

struct ConstDynamicTailChunkView {
    const TrainableActionEmbeddingTail *rows = nullptr;
    std::size_t chunk_index = 0;
    std::size_t first_action_index = 0;
    std::size_t active_action_count = 0;
    std::size_t action_capacity = 0;
};

struct DynamicGenomeView {
    DynamicBodyView body{};
    TrainableActionEmbeddingTail *tail_row_storage = nullptr;
    TrainableActionEmbeddingTail *tail_row_arena = nullptr;
    const DynamicArenaSlotId *tail_row_slot_ids = nullptr;
    DynamicTailSchema tail_schema{};
};

struct ConstDynamicGenomeView {
    ConstDynamicBodyView body{};
    const TrainableActionEmbeddingTail *tail_row_storage = nullptr;
    const TrainableActionEmbeddingTail *tail_row_arena = nullptr;
    const DynamicArenaSlotId *tail_row_slot_ids = nullptr;
    DynamicTailSchema tail_schema{};
};

struct DynamicPopulationLayout {
    std::size_t active_individual_count = 0;
    std::size_t generation_index = 0;
    std::size_t schema_epoch = 0;
    std::size_t action_count = 0;
    std::size_t tail_chunk_action_capacity = 0;
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

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t RoundUpDivide(const std::size_t numerator,
                                                               const std::size_t denominator) noexcept {
    return (denominator == 0) ? 0 : ((numerator + denominator - 1) / denominator);
}

constexpr NEUROEVOLUTION_HOST_DEVICE DynamicTailSchema MakeDynamicTailSchema(
    const std::size_t action_count, const std::size_t chunk_action_capacity = 0) noexcept {
    if (action_count == 0) {
        return {};
    }

    const std::size_t resolved_chunk_action_capacity =
        (chunk_action_capacity == 0) ? action_count : chunk_action_capacity;
    if (resolved_chunk_action_capacity == 0) {
        return {};
    }

    return {
        .action_count = action_count,
        .chunk_action_capacity = resolved_chunk_action_capacity,
        .chunk_count = RoundUpDivide(action_count, resolved_chunk_action_capacity),
    };
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidDynamicTailSchema(const DynamicTailSchema &schema) noexcept {
    return (schema.action_count > 0) && (schema.chunk_action_capacity > 0) && (schema.chunk_count > 0) &&
           (schema.chunk_count == RoundUpDivide(schema.action_count, schema.chunk_action_capacity));
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

constexpr NEUROEVOLUTION_HOST_DEVICE DynamicTailSchema
DynamicTailSchemaForLayout(const DynamicPopulationLayout &layout) noexcept {
    return MakeDynamicTailSchema(layout.action_count, layout.tail_chunk_action_capacity);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t ActiveActionCountInTailChunk(
    const DynamicTailSchema &schema, const std::size_t chunk_index) noexcept {
    if (!IsValidDynamicTailSchema(schema) || (chunk_index >= schema.chunk_count)) {
        return 0;
    }

    const std::size_t first_action_index = chunk_index * schema.chunk_action_capacity;
    const std::size_t remaining_action_count = schema.action_count - first_action_index;
    return (remaining_action_count < schema.chunk_action_capacity) ? remaining_action_count
                                                                   : schema.chunk_action_capacity;
}

constexpr NEUROEVOLUTION_HOST_DEVICE DynamicTailRowLocation TailRowLocationForActionIndex(
    const DynamicTailSchema &schema, const std::size_t action_index) noexcept {
    if (!IsValidDynamicTailSchema(schema) || (action_index >= schema.action_count)) {
        return {};
    }

    return {
        .chunk_index = action_index / schema.chunk_action_capacity,
        .action_offset_in_chunk = action_index % schema.chunk_action_capacity,
    };
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t TailRowStorageIndex(
    const DynamicTailSchema &schema, const DynamicTailRowLocation &location) noexcept {
    return (location.chunk_index * schema.chunk_action_capacity) + location.action_offset_in_chunk;
}

constexpr NEUROEVOLUTION_HOST_DEVICE DynamicPopulationLayout MakeDynamicPopulationLayout(
    const std::size_t active_individual_count, const std::size_t generation_index, const std::size_t action_count,
    const std::size_t tail_chunk_action_capacity = 0, const std::size_t schema_epoch = 0) noexcept {
    const DynamicTailSchema tail_schema = MakeDynamicTailSchema(action_count, tail_chunk_action_capacity);
    const std::size_t genome_stride_bytes = ComputeDynamicGenomeStrideBytes(action_count);
    return {
        .active_individual_count = active_individual_count,
        .generation_index = generation_index,
        .schema_epoch = schema_epoch,
        .action_count = action_count,
        .tail_chunk_action_capacity = tail_schema.chunk_action_capacity,
        .genome_stride_bytes = genome_stride_bytes,
        .genotype_bytes = active_individual_count * genome_stride_bytes,
    };
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidDynamicPopulationLayout(
    const DynamicPopulationLayout &layout) noexcept {
    return (layout.active_individual_count > 0) && IsValidDynamicTailSchema(DynamicTailSchemaForLayout(layout)) &&
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

inline NEUROEVOLUTION_HOST_DEVICE DynamicBodyView GenomeBodyView(std::uint8_t *genome_bytes) noexcept {
    return {reinterpret_cast<PolicyModelParameters *>(genome_bytes)};
}

inline NEUROEVOLUTION_HOST_DEVICE ConstDynamicBodyView GenomeBodyView(const std::uint8_t *genome_bytes) noexcept {
    return {reinterpret_cast<const PolicyModelParameters *>(genome_bytes)};
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

inline NEUROEVOLUTION_HOST_DEVICE DynamicGenomeView GenomeView(std::uint8_t *genome_bytes,
                                                               const DynamicPopulationLayout &layout) noexcept {
    return {
        .body = GenomeBodyView(genome_bytes),
        .tail_row_storage = GenomeTailRows(genome_bytes),
        .tail_row_arena = nullptr,
        .tail_row_slot_ids = nullptr,
        .tail_schema = DynamicTailSchemaForLayout(layout),
    };
}

inline NEUROEVOLUTION_HOST_DEVICE ConstDynamicGenomeView GenomeView(const std::uint8_t *genome_bytes,
                                                                    const DynamicPopulationLayout &layout) noexcept {
    return {
        .body = GenomeBodyView(genome_bytes),
        .tail_row_storage = GenomeTailRows(genome_bytes),
        .tail_row_arena = nullptr,
        .tail_row_slot_ids = nullptr,
        .tail_schema = DynamicTailSchemaForLayout(layout),
    };
}

inline NEUROEVOLUTION_HOST_DEVICE DynamicGenomeView ArenaGenomeView(
    PolicyModelParameters *body_slots, TrainableActionEmbeddingTail *tail_row_arena, const DynamicArenaSlotId *body_slot_ids,
    const DynamicArenaSlotId *tail_row_slot_ids, const std::size_t tail_row_slot_id_stride,
    const DynamicPopulationLayout &layout, const std::size_t individual_index) noexcept {
    return {
        .body = {body_slots + body_slot_ids[individual_index]},
        .tail_row_storage = nullptr,
        .tail_row_arena = tail_row_arena,
        .tail_row_slot_ids = tail_row_slot_ids + (individual_index * tail_row_slot_id_stride),
        .tail_schema = DynamicTailSchemaForLayout(layout),
    };
}

inline NEUROEVOLUTION_HOST_DEVICE ConstDynamicGenomeView ArenaGenomeView(
    const PolicyModelParameters *body_slots, const TrainableActionEmbeddingTail *tail_row_arena,
    const DynamicArenaSlotId *body_slot_ids, const DynamicArenaSlotId *tail_row_slot_ids,
    const std::size_t tail_row_slot_id_stride, const DynamicPopulationLayout &layout,
    const std::size_t individual_index) noexcept {
    return {
        .body = {body_slots + body_slot_ids[individual_index]},
        .tail_row_storage = nullptr,
        .tail_row_arena = tail_row_arena,
        .tail_row_slot_ids = tail_row_slot_ids + (individual_index * tail_row_slot_id_stride),
        .tail_schema = DynamicTailSchemaForLayout(layout),
    };
}

inline NEUROEVOLUTION_HOST_DEVICE DynamicGenomeView GenomeViewAt(std::uint8_t *population_genomes,
                                                                 const DynamicPopulationLayout &layout,
                                                                 const std::size_t individual_index) noexcept {
    return GenomeView(GenomeBytesAt(population_genomes, layout, individual_index), layout);
}

inline NEUROEVOLUTION_HOST_DEVICE ConstDynamicGenomeView GenomeViewAt(const std::uint8_t *population_genomes,
                                                                      const DynamicPopulationLayout &layout,
                                                                      const std::size_t individual_index) noexcept {
    return GenomeView(GenomeBytesAt(population_genomes, layout, individual_index), layout);
}

inline NEUROEVOLUTION_HOST_DEVICE PolicyModelParameters &GenomeBodyParameters(
    const DynamicGenomeView &genome_view) noexcept {
    return *genome_view.body.parameters;
}

inline NEUROEVOLUTION_HOST_DEVICE const PolicyModelParameters &GenomeBodyParameters(
    const ConstDynamicGenomeView &genome_view) noexcept {
    return *genome_view.body.parameters;
}

inline NEUROEVOLUTION_HOST_DEVICE DynamicTailChunkView GenomeTailChunk(const DynamicGenomeView &genome_view,
                                                                       const std::size_t chunk_index) noexcept {
    const std::size_t active_action_count = ActiveActionCountInTailChunk(genome_view.tail_schema, chunk_index);
    if ((active_action_count == 0) || (genome_view.tail_row_storage == nullptr)) {
        return {};
    }

    return {
        .rows = genome_view.tail_row_storage + (chunk_index * genome_view.tail_schema.chunk_action_capacity),
        .chunk_index = chunk_index,
        .first_action_index = chunk_index * genome_view.tail_schema.chunk_action_capacity,
        .active_action_count = active_action_count,
        .action_capacity = genome_view.tail_schema.chunk_action_capacity,
    };
}

inline NEUROEVOLUTION_HOST_DEVICE ConstDynamicTailChunkView GenomeTailChunk(
    const ConstDynamicGenomeView &genome_view, const std::size_t chunk_index) noexcept {
    const std::size_t active_action_count = ActiveActionCountInTailChunk(genome_view.tail_schema, chunk_index);
    if ((active_action_count == 0) || (genome_view.tail_row_storage == nullptr)) {
        return {};
    }

    return {
        .rows = genome_view.tail_row_storage + (chunk_index * genome_view.tail_schema.chunk_action_capacity),
        .chunk_index = chunk_index,
        .first_action_index = chunk_index * genome_view.tail_schema.chunk_action_capacity,
        .active_action_count = active_action_count,
        .action_capacity = genome_view.tail_schema.chunk_action_capacity,
    };
}

inline NEUROEVOLUTION_HOST_DEVICE TrainableActionEmbeddingTail &GenomeTailRow(
    const DynamicGenomeView &genome_view, const std::size_t action_index) noexcept {
    const DynamicTailRowLocation location = TailRowLocationForActionIndex(genome_view.tail_schema, action_index);
    const std::size_t storage_index = TailRowStorageIndex(genome_view.tail_schema, location);
    if (genome_view.tail_row_storage != nullptr) {
        return genome_view.tail_row_storage[storage_index];
    }

    return genome_view.tail_row_arena[genome_view.tail_row_slot_ids[storage_index]];
}

inline NEUROEVOLUTION_HOST_DEVICE const TrainableActionEmbeddingTail &GenomeTailRow(
    const ConstDynamicGenomeView &genome_view, const std::size_t action_index) noexcept {
    const DynamicTailRowLocation location = TailRowLocationForActionIndex(genome_view.tail_schema, action_index);
    const std::size_t storage_index = TailRowStorageIndex(genome_view.tail_schema, location);
    if (genome_view.tail_row_storage != nullptr) {
        return genome_view.tail_row_storage[storage_index];
    }

    return genome_view.tail_row_arena[genome_view.tail_row_slot_ids[storage_index]];
}

inline std::uint8_t *HostGenomeBytesAt(HostPopulation &population, const std::size_t individual_index) noexcept {
    return GenomeBytesAt(population.genomes.get(), population.layout, individual_index);
}

inline const std::uint8_t *HostGenomeBytesAt(const HostPopulation &population,
                                             const std::size_t individual_index) noexcept {
    return GenomeBytesAt(population.genomes.get(), population.layout, individual_index);
}

inline DynamicGenomeView HostGenomeViewAt(HostPopulation &population, const std::size_t individual_index) noexcept {
    return GenomeView(HostGenomeBytesAt(population, individual_index), population.layout);
}

inline ConstDynamicGenomeView HostGenomeViewAt(const HostPopulation &population,
                                               const std::size_t individual_index) noexcept {
    return GenomeView(HostGenomeBytesAt(population, individual_index), population.layout);
}

bool TryAllocateHostGenomeStorage(HostPopulation &population);

bool TryInitializeRandomHostPopulation(
    HostPopulation &population, std::size_t population_size, std::size_t action_count, std::uint32_t seed,
    const PopulationInitializationConfig &config = {});

} // namespace neuroevolution::genetic_algorithm::genome
