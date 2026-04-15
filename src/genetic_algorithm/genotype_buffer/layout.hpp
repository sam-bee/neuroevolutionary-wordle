#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"

namespace neuroevolution::genetic_algorithm::genotype_buffer {

using genome::DynamicGenomeAlignment;
using genome::DynamicGenomeTailOffsetBytes;
using genome::GenomePolicyModelParameters;
using genome::GenomeTailRows;
using genome::PolicyModelParameters;
using genome::TrainableActionEmbeddingTail;

constexpr std::uint32_t kInvalidBufferSlotIndex = static_cast<std::uint32_t>(-1);

struct GenotypeBufferLayout {
    std::size_t action_count = 0;
    std::size_t slot_stride_bytes = 0;
    std::size_t slot_count = 0;
    std::size_t buffer_bytes = 0;
};

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t ComputeBufferSlotStrideBytes(const std::size_t action_count) noexcept {
    return genome::ComputeDynamicGenomeStrideBytes(action_count);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
ComputeOutputEmbeddingGrowthBytes(const std::size_t action_count_increment) noexcept {
    return action_count_increment * sizeof(TrainableActionEmbeddingTail);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t BufferUsedBytes(const GenotypeBufferLayout &layout) noexcept {
    return layout.slot_count * layout.slot_stride_bytes;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
BufferSlotCountForByteBudget(const std::size_t buffer_byte_budget_bytes, const std::size_t action_count,
                             const std::size_t slot_count_ceiling = 0) noexcept {
    return genome::PopulationSizeForGenotypeBudgetBytes(buffer_byte_budget_bytes, action_count, slot_count_ceiling);
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool TryCreateBufferLayoutForByteBudget(GenotypeBufferLayout &layout,
                                                                             const std::size_t buffer_byte_budget_bytes,
                                                                             const std::size_t action_count) noexcept {
    layout = {};
    const std::size_t slot_stride_bytes = ComputeBufferSlotStrideBytes(action_count);
    const std::size_t slot_count = BufferSlotCountForByteBudget(buffer_byte_budget_bytes, action_count);
    if ((buffer_byte_budget_bytes == 0) || (slot_stride_bytes == 0) || (slot_count == 0)) {
        return false;
    }

    layout.action_count = action_count;
    layout.slot_stride_bytes = slot_stride_bytes;
    layout.slot_count = slot_count;
    layout.buffer_bytes = buffer_byte_budget_bytes;
    return true;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool TryCreateExpandedBufferLayout(const GenotypeBufferLayout &current_layout,
                                                                        const std::size_t next_action_count,
                                                                        GenotypeBufferLayout &next_layout) noexcept {
    if (!TryCreateBufferLayoutForByteBudget(next_layout, current_layout.buffer_bytes, next_action_count)) {
        return false;
    }

    return next_layout.action_count >= current_layout.action_count;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidGenotypeBufferLayout(const GenotypeBufferLayout &layout) noexcept {
    return (layout.action_count > 0) && (layout.slot_count > 0) &&
           (layout.slot_stride_bytes == ComputeBufferSlotStrideBytes(layout.action_count)) &&
           (layout.slot_count == BufferSlotCountForByteBudget(layout.buffer_bytes, layout.action_count)) &&
           (layout.buffer_bytes >= (layout.slot_count * layout.slot_stride_bytes));
}

inline NEUROEVOLUTION_HOST_DEVICE std::uint8_t *BufferSlotBytesAt(std::uint8_t *buffer_bytes,
                                                                  const GenotypeBufferLayout &layout,
                                                                  const std::size_t slot_index) noexcept {
    return buffer_bytes + (slot_index * layout.slot_stride_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE const std::uint8_t *BufferSlotBytesAt(const std::uint8_t *buffer_bytes,
                                                                        const GenotypeBufferLayout &layout,
                                                                        const std::size_t slot_index) noexcept {
    return buffer_bytes + (slot_index * layout.slot_stride_bytes);
}

} // namespace neuroevolution::genetic_algorithm::genotype_buffer
