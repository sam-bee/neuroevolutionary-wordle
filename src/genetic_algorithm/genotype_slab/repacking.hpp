#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genotype_slab/generation.hpp"

namespace neuroevolution::genetic_algorithm::genotype_slab {

namespace detail {

inline NEUROEVOLUTION_HOST_DEVICE void ZeroSlabBytes(std::uint8_t *bytes, const std::size_t byte_count) noexcept {
    if (bytes == nullptr) {
        return;
    }

    for (std::size_t byte_index = 0; byte_index < byte_count; ++byte_index) {
        bytes[byte_index] = 0;
    }
}

inline NEUROEVOLUTION_HOST_DEVICE void MoveSlabBytesOverlapping(const std::uint8_t *source_bytes,
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
        return kInvalidSlabSlotIndex;
    }

    for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
        if ((parent_reference_counts[parent_index] > 0) && (generation_slot_indices[parent_index] == slot_index)) {
            return static_cast<std::uint32_t>(parent_index);
        }
    }

    return kInvalidSlabSlotIndex;
}

struct SlabRepackPreflight {
    GenotypeSlabLayout next_layout{};
    std::size_t survivor_count = 0;
    std::size_t destination_base_slot = 0;
};

inline NEUROEVOLUTION_HOST_DEVICE bool ReferencedParentSlotIsUnique(const std::uint32_t *generation_slot_indices,
                                                                    const std::uint32_t *parent_reference_counts,
                                                                    const std::size_t parent_index) noexcept {
    for (std::size_t previous_parent_index = 0; previous_parent_index < parent_index; ++previous_parent_index) {
        if ((parent_reference_counts[previous_parent_index] > 0) &&
            (generation_slot_indices[previous_parent_index] == generation_slot_indices[parent_index])) {
            return false;
        }
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryPreflightCompactionAndRepackForExpandedActionCount(
    const GenotypeSlabLayout &slab_layout, const SlabSlotState *slot_states,
    const std::uint32_t *generation_slot_indices, const std::size_t active_individual_count,
    const std::uint32_t *parent_reference_counts, const std::size_t next_action_count,
    const std::size_t required_free_slot_count, SlabRepackPreflight &preflight) noexcept {
    preflight = {};
    if (!IsValidGenotypeSlabLayout(slab_layout) || (slot_states == nullptr) || (generation_slot_indices == nullptr) ||
        (active_individual_count == 0) || (parent_reference_counts == nullptr) ||
        (next_action_count <= slab_layout.action_count)) {
        return false;
    }

    GenotypeSlabLayout next_layout{};
    if (!TryCreateExpandedSlabLayout(slab_layout, next_action_count, next_layout)) {
        return false;
    }

    std::size_t survivor_count = 0;
    for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
        if (parent_reference_counts[parent_index] == 0) {
            continue;
        }

        const std::uint32_t slot_index = generation_slot_indices[parent_index];
        if ((slot_index == kInvalidSlabSlotIndex) || (slot_index >= slab_layout.slot_count) ||
            !slot_states[slot_index].occupied || (slot_states[slot_index].reference_count == 0) ||
            !ReferencedParentSlotIsUnique(generation_slot_indices, parent_reference_counts, parent_index)) {
            return false;
        }

        ++survivor_count;
    }

    if ((survivor_count == 0) || ((survivor_count + required_free_slot_count) > next_layout.slot_count)) {
        return false;
    }

    preflight.next_layout = next_layout;
    preflight.survivor_count = survivor_count;
    preflight.destination_base_slot = next_layout.slot_count - survivor_count;
    return true;
}

} // namespace detail

inline NEUROEVOLUTION_HOST_DEVICE bool TryCompactAndRepackSlabForExpandedActionCount(
    GenotypeSlabLayout &slab_layout, std::uint8_t *slab_storage, SlabSlotState *slot_states,
    std::uint32_t *free_slot_stack, std::uint32_t &free_slot_count, std::uint32_t *generation_slot_indices,
    const std::size_t active_individual_count, const std::uint32_t *parent_reference_counts,
    const std::size_t next_action_count, const std::size_t required_free_slot_count = 1) noexcept {
    // Stop-the-world reference-counting GC for growth: collect unreferenced parents, compact survivors,
    // then repack them to the right using the expanded slot stride.
    if (!IsValidGenotypeSlabLayout(slab_layout) || (slab_storage == nullptr) || (slot_states == nullptr) ||
        (free_slot_stack == nullptr) || (generation_slot_indices == nullptr) || (active_individual_count == 0) ||
        (parent_reference_counts == nullptr) || (next_action_count <= slab_layout.action_count)) {
        return false;
    }

    detail::SlabRepackPreflight preflight{};
    if (!detail::TryPreflightCompactionAndRepackForExpandedActionCount(
            slab_layout, slot_states, generation_slot_indices, active_individual_count, parent_reference_counts,
            next_action_count, required_free_slot_count, preflight)) {
        return false;
    }

    const GenotypeSlabLayout next_layout = preflight.next_layout;
    const std::size_t current_slot_stride_bytes = slab_layout.slot_stride_bytes;
    std::size_t survivor_count = 0;

    for (std::uint32_t source_slot_index = 0; source_slot_index < slab_layout.slot_count; ++source_slot_index) {
        const std::uint32_t parent_index = detail::FindReferencedParentIndexOwningSlot(
            generation_slot_indices, active_individual_count, parent_reference_counts, source_slot_index);
        if (parent_index == kInvalidSlabSlotIndex) {
            continue;
        }

        if (survivor_count != source_slot_index) {
            detail::MoveSlabBytesOverlapping(SlabSlotBytesAt(slab_storage, slab_layout, source_slot_index),
                                             SlabSlotBytesAt(slab_storage, slab_layout, survivor_count),
                                             current_slot_stride_bytes);
            detail::ZeroSlabBytes(SlabSlotBytesAt(slab_storage, slab_layout, source_slot_index),
                                  current_slot_stride_bytes);
        }

        generation_slot_indices[parent_index] = static_cast<std::uint32_t>(survivor_count);
        ++survivor_count;
    }

    for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
        if (parent_reference_counts[parent_index] == 0) {
            generation_slot_indices[parent_index] = kInvalidSlabSlotIndex;
        }
    }

    if (survivor_count != preflight.survivor_count) {
        return false;
    }

    const std::size_t destination_base_slot = preflight.destination_base_slot;
    for (std::size_t compacted_slot = survivor_count; compacted_slot > 0; --compacted_slot) {
        const std::size_t source_slot_index = compacted_slot - 1;
        const std::size_t destination_slot_index = destination_base_slot + source_slot_index;
        std::uint8_t *destination_bytes = SlabSlotBytesAt(slab_storage, next_layout, destination_slot_index);
        detail::MoveSlabBytesOverlapping(SlabSlotBytesAt(slab_storage, slab_layout, source_slot_index),
                                         destination_bytes, current_slot_stride_bytes);
        detail::ZeroSlabBytes(destination_bytes + current_slot_stride_bytes,
                              next_layout.slot_stride_bytes - current_slot_stride_bytes);
    }

    for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
        if (generation_slot_indices[parent_index] == kInvalidSlabSlotIndex) {
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
        if (slot_index == kInvalidSlabSlotIndex) {
            continue;
        }

        slot_states[slot_index].occupied = true;
        slot_states[slot_index].reference_count = 1;
    }

    free_slot_count = static_cast<std::uint32_t>(destination_base_slot);
    for (std::uint32_t free_slot_index = 0; free_slot_index < free_slot_count; ++free_slot_index) {
        free_slot_stack[free_slot_index] = (free_slot_count - 1) - free_slot_index;
        detail::ZeroSlabBytes(SlabSlotBytesAt(slab_storage, next_layout, free_slot_index),
                              next_layout.slot_stride_bytes);
    }

    detail::ZeroSlabBytes(slab_storage + SlabUsedBytes(next_layout),
                          next_layout.slab_bytes - SlabUsedBytes(next_layout));
    slab_layout = next_layout;
    return true;
}

inline bool TryCompactAndRepackSlabForExpandedActionCount(HostGenotypeSlab &buffer, SlabGeneration &generation,
                                                          const std::uint32_t *parent_reference_counts,
                                                          const std::size_t next_action_count,
                                                          const std::size_t required_free_slot_count = 1) {
    return TryCompactAndRepackSlabForExpandedActionCount(
        buffer.layout, buffer.storage.get(), buffer.slot_states.get(), buffer.free_slot_stack.get(),
        buffer.free_slot_count, generation.slot_indices.get(), generation.active_individual_count,
        parent_reference_counts, next_action_count, required_free_slot_count);
}

} // namespace neuroevolution::genetic_algorithm::genotype_slab
