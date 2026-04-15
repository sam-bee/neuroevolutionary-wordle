#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genotype_pool/generation.hpp"

namespace neuroevolution::genetic_algorithm::genotype_pool {

namespace detail {

inline NEUROEVOLUTION_HOST_DEVICE void ZeroPoolBytes(std::uint8_t *bytes, const std::size_t byte_count) noexcept {
    if (bytes == nullptr) {
        return;
    }

    for (std::size_t byte_index = 0; byte_index < byte_count; ++byte_index) {
        bytes[byte_index] = 0;
    }
}

inline NEUROEVOLUTION_HOST_DEVICE void MovePoolBytesOverlapping(const std::uint8_t *source_bytes,
                                                                std::uint8_t *target_bytes,
                                                                const std::size_t byte_count) noexcept {
    if ((source_bytes == nullptr) || (target_bytes == nullptr) || (source_bytes == target_bytes)) {
        return;
    }

    if (target_bytes < source_bytes) {
        for (std::size_t byte_index = 0; byte_index < byte_count; ++byte_index) {
            target_bytes[byte_index] = source_bytes[byte_index];
        }
        return;
    }

    for (std::size_t byte_index = byte_count; byte_index > 0; --byte_index) {
        target_bytes[byte_index - 1] = source_bytes[byte_index - 1];
    }
}

inline NEUROEVOLUTION_HOST_DEVICE std::uint32_t FindReferencedParentIndexOwningSlot(
    const std::uint32_t *generation_slot_indices, const std::size_t active_individual_count,
    const std::uint32_t *parent_reference_counts, const std::uint32_t slot_index) noexcept {
    if ((generation_slot_indices == nullptr) || (active_individual_count == 0) ||
        (parent_reference_counts == nullptr)) {
        return kInvalidPoolSlotIndex;
    }

    for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
        if ((parent_reference_counts[parent_index] > 0) && (generation_slot_indices[parent_index] == slot_index)) {
            return static_cast<std::uint32_t>(parent_index);
        }
    }

    return kInvalidPoolSlotIndex;
}

} // namespace detail

inline NEUROEVOLUTION_HOST_DEVICE bool TryCompactAndRepackPoolForExpandedActionCount(
    GenotypePoolLayout &pool_layout, std::uint8_t *pool_storage, PoolSlotState *slot_states,
    std::uint32_t *free_slot_stack, std::size_t &free_slot_count, std::uint32_t *generation_slot_indices,
    const std::size_t active_individual_count, const std::uint32_t *parent_reference_counts,
    const std::size_t next_action_count) noexcept {
    // Stop-the-world reference-counting GC for growth: collect unreferenced parents, compact survivors,
    // then repack them to the right using the expanded slot stride.
    if (!IsValidGenotypePoolLayout(pool_layout) || (pool_storage == nullptr) || (slot_states == nullptr) ||
        (free_slot_stack == nullptr) || (generation_slot_indices == nullptr) || (active_individual_count == 0) ||
        (parent_reference_counts == nullptr) || (next_action_count <= pool_layout.action_count)) {
        return false;
    }

    GenotypePoolLayout next_layout{};
    if (!TryCreateExpandedPoolLayout(pool_layout, next_action_count, next_layout)) {
        return false;
    }

    const std::size_t current_slot_stride_bytes = pool_layout.slot_stride_bytes;
    std::size_t survivor_count = 0;

    for (std::uint32_t source_slot_index = 0; source_slot_index < pool_layout.slot_count; ++source_slot_index) {
        const std::uint32_t parent_index = detail::FindReferencedParentIndexOwningSlot(
            generation_slot_indices, active_individual_count, parent_reference_counts, source_slot_index);
        if (parent_index == kInvalidPoolSlotIndex) {
            continue;
        }

        if (survivor_count >= next_layout.slot_count) {
            return false;
        }

        if (survivor_count != source_slot_index) {
            detail::MovePoolBytesOverlapping(PoolSlotBytesAt(pool_storage, pool_layout, source_slot_index),
                                             PoolSlotBytesAt(pool_storage, pool_layout, survivor_count),
                                             current_slot_stride_bytes);
            detail::ZeroPoolBytes(PoolSlotBytesAt(pool_storage, pool_layout, source_slot_index),
                                  current_slot_stride_bytes);
        }

        generation_slot_indices[parent_index] = static_cast<std::uint32_t>(survivor_count);
        ++survivor_count;
    }

    for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
        if (parent_reference_counts[parent_index] == 0) {
            generation_slot_indices[parent_index] = kInvalidPoolSlotIndex;
        }
    }

    if ((survivor_count == 0) || (survivor_count >= next_layout.slot_count)) {
        return false;
    }

    const std::size_t destination_base_slot = next_layout.slot_count - survivor_count;
    for (std::size_t compacted_slot = survivor_count; compacted_slot > 0; --compacted_slot) {
        const std::size_t source_slot_index = compacted_slot - 1;
        const std::size_t destination_slot_index = destination_base_slot + source_slot_index;
        std::uint8_t *destination_bytes = PoolSlotBytesAt(pool_storage, next_layout, destination_slot_index);
        detail::MovePoolBytesOverlapping(PoolSlotBytesAt(pool_storage, pool_layout, source_slot_index),
                                         destination_bytes, current_slot_stride_bytes);
        detail::ZeroPoolBytes(destination_bytes + current_slot_stride_bytes,
                              next_layout.slot_stride_bytes - current_slot_stride_bytes);
    }

    for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
        if (generation_slot_indices[parent_index] == kInvalidPoolSlotIndex) {
            continue;
        }

        generation_slot_indices[parent_index] =
            static_cast<std::uint32_t>(destination_base_slot + generation_slot_indices[parent_index]);
    }

    for (std::size_t slot_index = 0; slot_index < next_layout.slot_count; ++slot_index) {
        slot_states[slot_index] = {};
    }

    for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
        const std::uint32_t slot_index = generation_slot_indices[parent_index];
        if (slot_index == kInvalidPoolSlotIndex) {
            continue;
        }

        slot_states[slot_index].occupied = true;
        slot_states[slot_index].reference_count = 1;
    }

    free_slot_count = destination_base_slot;
    for (std::size_t free_slot_index = 0; free_slot_index < free_slot_count; ++free_slot_index) {
        free_slot_stack[free_slot_index] = static_cast<std::uint32_t>((free_slot_count - 1) - free_slot_index);
        detail::ZeroPoolBytes(PoolSlotBytesAt(pool_storage, next_layout, free_slot_index),
                              next_layout.slot_stride_bytes);
    }

    detail::ZeroPoolBytes(pool_storage + PoolUsedBytes(next_layout),
                          next_layout.pool_bytes - PoolUsedBytes(next_layout));
    pool_layout = next_layout;
    return true;
}

inline bool TryCompactAndRepackPoolForExpandedActionCount(HostGenotypePool &pool, PoolGeneration &generation,
                                                          const std::uint32_t *parent_reference_counts,
                                                          const std::size_t next_action_count) {
    return TryCompactAndRepackPoolForExpandedActionCount(
        pool.layout, pool.storage.get(), pool.slot_states.get(), pool.free_slot_stack.get(), pool.free_slot_count,
        generation.slot_indices.get(), generation.active_individual_count, parent_reference_counts, next_action_count);
}

} // namespace neuroevolution::genetic_algorithm::genotype_pool
