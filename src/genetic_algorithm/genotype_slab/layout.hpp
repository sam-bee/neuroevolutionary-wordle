#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"

namespace neuroevolution::genetic_algorithm::genotype_slab {

using genome::DynamicGenomeAlignment;
using genome::DynamicGenomeTailOffsetBytes;
using genome::GenomePolicyModelParameters;
using genome::GenomeTailRows;
using genome::PolicyModelParameters;
using genome::TrainableActionEmbeddingTail;

constexpr std::uint32_t kInvalidSlabSlotIndex = static_cast<std::uint32_t>(-1);

struct GenotypeSlabLayout {
    std::size_t action_count = 0;
    std::size_t slot_stride_bytes = 0;
    std::size_t slot_count = 0;
    std::size_t slab_bytes = 0;
};

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t ComputeSlabSlotStrideBytes(const std::size_t action_count) noexcept {
    return genome::ComputeDynamicGenomeStrideBytes(action_count);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
ComputeOutputEmbeddingGrowthBytes(const std::size_t action_count_increment) noexcept {
    return action_count_increment * sizeof(TrainableActionEmbeddingTail);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t SlabUsedBytes(const GenotypeSlabLayout &layout) noexcept {
    return layout.slot_count * layout.slot_stride_bytes;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
SlabSlotCountForByteBudget(const std::size_t slab_byte_budget_bytes, const std::size_t action_count,
                           const std::size_t slot_count_ceiling = 0) noexcept {
    return genome::PopulationSizeForGenotypeBudgetBytes(slab_byte_budget_bytes, action_count, slot_count_ceiling);
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool TryCreateSlabLayoutForByteBudget(GenotypeSlabLayout &layout,
                                                                           const std::size_t slab_byte_budget_bytes,
                                                                           const std::size_t action_count) noexcept {
    layout = {};
    const std::size_t slot_stride_bytes = ComputeSlabSlotStrideBytes(action_count);
    const std::size_t slot_count = SlabSlotCountForByteBudget(slab_byte_budget_bytes, action_count);
    if ((slab_byte_budget_bytes == 0) || (slot_stride_bytes == 0) || (slot_count == 0) ||
        (slot_count >= static_cast<std::size_t>(kInvalidSlabSlotIndex))) {
        return false;
    }

    layout.action_count = action_count;
    layout.slot_stride_bytes = slot_stride_bytes;
    layout.slot_count = slot_count;
    layout.slab_bytes = slab_byte_budget_bytes;
    return true;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool TryCreateExpandedSlabLayout(const GenotypeSlabLayout &current_layout,
                                                                      const std::size_t next_action_count,
                                                                      GenotypeSlabLayout &next_layout) noexcept {
    if (!TryCreateSlabLayoutForByteBudget(next_layout, current_layout.slab_bytes, next_action_count)) {
        return false;
    }

    return next_layout.action_count >= current_layout.action_count;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidGenotypeSlabLayout(const GenotypeSlabLayout &layout) noexcept {
    return (layout.action_count > 0) && (layout.slot_count > 0) &&
           (layout.slot_count < static_cast<std::size_t>(kInvalidSlabSlotIndex)) &&
           (layout.slot_stride_bytes == ComputeSlabSlotStrideBytes(layout.action_count)) &&
           (layout.slot_count == SlabSlotCountForByteBudget(layout.slab_bytes, layout.action_count)) &&
           (layout.slab_bytes >= (layout.slot_count * layout.slot_stride_bytes));
}

inline NEUROEVOLUTION_HOST_DEVICE std::uint8_t *
SlabSlotBytesAt(std::uint8_t *slab_bytes, const GenotypeSlabLayout &layout, const std::size_t slot_index) noexcept {
    return slab_bytes + (slot_index * layout.slot_stride_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE const std::uint8_t *SlabSlotBytesAt(const std::uint8_t *slab_bytes,
                                                                      const GenotypeSlabLayout &layout,
                                                                      const std::size_t slot_index) noexcept {
    return slab_bytes + (slot_index * layout.slot_stride_bytes);
}

} // namespace neuroevolution::genetic_algorithm::genotype_slab
