#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"

namespace neuroevolution::genetic_algorithm::genotype_pool {

using genome::DynamicGenomeAlignment;
using genome::DynamicGenomeTailOffsetBytes;
using genome::GenomePolicyModelParameters;
using genome::GenomeTailRows;
using genome::PolicyModelParameters;
using genome::TrainableActionEmbeddingTail;

constexpr std::uint32_t kInvalidPoolSlotIndex = static_cast<std::uint32_t>(-1);

struct GenotypePoolLayout {
    std::size_t action_count = 0;
    std::size_t slot_stride_bytes = 0;
    std::size_t slot_count = 0;
    std::size_t pool_bytes = 0;
};

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t ComputePoolSlotStrideBytes(const std::size_t action_count) noexcept {
    return genome::ComputeDynamicGenomeStrideBytes(action_count);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
ComputeOutputEmbeddingGrowthBytes(const std::size_t action_count_increment) noexcept {
    return action_count_increment * sizeof(TrainableActionEmbeddingTail);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t PoolUsedBytes(const GenotypePoolLayout &layout) noexcept {
    return layout.slot_count * layout.slot_stride_bytes;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
PoolSlotCountForByteBudget(const std::size_t pool_byte_budget_bytes, const std::size_t action_count,
                           const std::size_t slot_count_ceiling = 0) noexcept {
    return genome::PopulationSizeForGenotypeBudgetBytes(pool_byte_budget_bytes, action_count, slot_count_ceiling);
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool TryCreatePoolLayoutForByteBudget(GenotypePoolLayout &layout,
                                                                           const std::size_t pool_byte_budget_bytes,
                                                                           const std::size_t action_count) noexcept {
    layout = {};
    const std::size_t slot_stride_bytes = ComputePoolSlotStrideBytes(action_count);
    const std::size_t slot_count = PoolSlotCountForByteBudget(pool_byte_budget_bytes, action_count);
    if ((pool_byte_budget_bytes == 0) || (slot_stride_bytes == 0) || (slot_count == 0)) {
        return false;
    }

    layout.action_count = action_count;
    layout.slot_stride_bytes = slot_stride_bytes;
    layout.slot_count = slot_count;
    layout.pool_bytes = pool_byte_budget_bytes;
    return true;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool TryCreateExpandedPoolLayout(const GenotypePoolLayout &current_layout,
                                                                      const std::size_t next_action_count,
                                                                      GenotypePoolLayout &next_layout) noexcept {
    if (!TryCreatePoolLayoutForByteBudget(next_layout, current_layout.pool_bytes, next_action_count)) {
        return false;
    }

    return next_layout.action_count >= current_layout.action_count;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidGenotypePoolLayout(const GenotypePoolLayout &layout) noexcept {
    return (layout.action_count > 0) && (layout.slot_count > 0) &&
           (layout.slot_stride_bytes == ComputePoolSlotStrideBytes(layout.action_count)) &&
           (layout.slot_count == PoolSlotCountForByteBudget(layout.pool_bytes, layout.action_count)) &&
           (layout.pool_bytes >= (layout.slot_count * layout.slot_stride_bytes));
}

inline NEUROEVOLUTION_HOST_DEVICE std::uint8_t *
PoolSlotBytesAt(std::uint8_t *pool_bytes, const GenotypePoolLayout &layout, const std::size_t slot_index) noexcept {
    return pool_bytes + (slot_index * layout.slot_stride_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE const std::uint8_t *PoolSlotBytesAt(const std::uint8_t *pool_bytes,
                                                                      const GenotypePoolLayout &layout,
                                                                      const std::size_t slot_index) noexcept {
    return pool_bytes + (slot_index * layout.slot_stride_bytes);
}

} // namespace neuroevolution::genetic_algorithm::genotype_pool
