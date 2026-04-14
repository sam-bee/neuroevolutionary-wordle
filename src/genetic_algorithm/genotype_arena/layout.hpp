#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"

namespace neuroevolution::genetic_algorithm::genotype_arena {

using genome::DynamicGenomeAlignment;
using genome::DynamicGenomeTailOffsetBytes;
using genome::GenomePolicyModelParameters;
using genome::GenomeTailRows;
using genome::PolicyModelParameters;
using genome::TrainableActionEmbeddingTail;

constexpr std::uint32_t kInvalidArenaSlotIndex = static_cast<std::uint32_t>(-1);

struct GenotypeArenaLayout {
    std::size_t action_count = 0;
    std::size_t slot_stride_bytes = 0;
    std::size_t slot_count = 0;
    std::size_t arena_bytes = 0;
};

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t ComputeArenaSlotStrideBytes(const std::size_t action_count) noexcept {
    return genome::ComputeDynamicGenomeStrideBytes(action_count);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
ArenaSlotCountForByteBudget(const std::size_t arena_byte_budget_bytes, const std::size_t action_count,
                            const std::size_t slot_count_ceiling = 0) noexcept {
    return genome::PopulationSizeForGenotypeBudgetBytes(arena_byte_budget_bytes, action_count, slot_count_ceiling);
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidGenotypeArenaLayout(const GenotypeArenaLayout &layout) noexcept {
    return (layout.action_count > 0) && (layout.slot_count > 0) &&
           (layout.slot_stride_bytes == ComputeArenaSlotStrideBytes(layout.action_count)) &&
           (layout.arena_bytes == (layout.slot_count * layout.slot_stride_bytes));
}

inline NEUROEVOLUTION_HOST_DEVICE std::uint8_t *
ArenaSlotBytesAt(std::uint8_t *arena_bytes, const GenotypeArenaLayout &layout, const std::size_t slot_index) noexcept {
    return arena_bytes + (slot_index * layout.slot_stride_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE const std::uint8_t *ArenaSlotBytesAt(const std::uint8_t *arena_bytes,
                                                                       const GenotypeArenaLayout &layout,
                                                                       const std::size_t slot_index) noexcept {
    return arena_bytes + (slot_index * layout.slot_stride_bytes);
}

} // namespace neuroevolution::genetic_algorithm::genotype_arena
