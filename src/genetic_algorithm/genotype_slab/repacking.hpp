#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genotype_slab/generation.hpp"

namespace neuroevolution::genetic_algorithm::genotype_slab {

namespace detail {

inline NEUROEVOLUTION_HOST_DEVICE void SynchronizeRepackWorkers() noexcept {
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
}

inline NEUROEVOLUTION_HOST_DEVICE void ZeroSlabBytes(std::uint8_t *bytes, const std::size_t byte_count,
                                                     const std::size_t worker_index = 0,
                                                     const std::size_t worker_count = 1) noexcept {
    if (bytes == nullptr) {
        return;
    }

    for (std::size_t byte_index = worker_index; byte_index < byte_count; byte_index += worker_count) {
        bytes[byte_index] = 0;
    }

    SynchronizeRepackWorkers();
}

inline NEUROEVOLUTION_HOST_DEVICE void MoveSlabBytesOverlapping(const std::uint8_t *source_bytes,
                                                                std::uint8_t *target_bytes,
                                                                const std::size_t byte_count,
                                                                const std::size_t worker_index = 0,
                                                                const std::size_t worker_count = 1) noexcept {
    if ((source_bytes == nullptr) || (target_bytes == nullptr) || (source_bytes == target_bytes) ||
        (worker_count == 0) || (byte_count == 0)) {
        return;
    }

    if (target_bytes < source_bytes) {
        const std::size_t gap_bytes = static_cast<std::size_t>(source_bytes - target_bytes);
        const std::size_t chunk_bytes = (gap_bytes >= byte_count) ? byte_count : gap_bytes;
        for (std::size_t chunk_offset = 0; chunk_offset < byte_count; chunk_offset += chunk_bytes) {
            const std::size_t active_chunk_bytes =
                ((byte_count - chunk_offset) < chunk_bytes) ? (byte_count - chunk_offset) : chunk_bytes;
            for (std::size_t byte_index = worker_index; byte_index < active_chunk_bytes; byte_index += worker_count) {
                target_bytes[chunk_offset + byte_index] = source_bytes[chunk_offset + byte_index];
            }
            SynchronizeRepackWorkers();
        }
        return;
    }

    const std::size_t gap_bytes = static_cast<std::size_t>(target_bytes - source_bytes);
    const std::size_t chunk_bytes = (gap_bytes >= byte_count) ? byte_count : gap_bytes;
    for (std::size_t remaining_bytes = byte_count; remaining_bytes > 0;) {
        const std::size_t active_chunk_bytes = (remaining_bytes < chunk_bytes) ? remaining_bytes : chunk_bytes;
        const std::size_t chunk_offset = remaining_bytes - active_chunk_bytes;
        for (std::size_t byte_index = worker_index; byte_index < active_chunk_bytes; byte_index += worker_count) {
            target_bytes[chunk_offset + byte_index] = source_bytes[chunk_offset + byte_index];
        }
        SynchronizeRepackWorkers();
        remaining_bytes = chunk_offset;
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
    SlabRepackPreflight &preflight) noexcept {
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
            !slot_states[slot_index].occupied || (slot_states[slot_index].liveness_count == 0) ||
            !ReferencedParentSlotIsUnique(generation_slot_indices, parent_reference_counts, parent_index)) {
            return false;
        }

        ++survivor_count;
    }

    if (survivor_count == 0) {
        return false;
    }

    preflight.next_layout = next_layout;
    preflight.survivor_count = survivor_count;
    preflight.destination_base_slot = next_layout.slot_count - survivor_count;
    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryCompactReferencedParentsIntoPrefix(
    const GenotypeSlabLayout &slab_layout, std::uint8_t *slab_storage, std::uint32_t *generation_slot_indices,
    const std::size_t active_individual_count, const std::uint32_t *parent_reference_counts,
    const std::size_t survivor_count, const std::size_t worker_index = 0,
    const std::size_t worker_count = 1) noexcept {
    if (!IsValidGenotypeSlabLayout(slab_layout) || (slab_storage == nullptr) || (generation_slot_indices == nullptr) ||
        (active_individual_count == 0) || (parent_reference_counts == nullptr) || (survivor_count == 0) ||
        (survivor_count > slab_layout.slot_count)) {
        return false;
    }

    std::uint32_t next_source_slot = static_cast<std::uint32_t>(survivor_count);
    for (std::uint32_t target_slot_index = 0; target_slot_index < survivor_count; ++target_slot_index) {
        std::uint32_t parent_index = detail::FindReferencedParentIndexOwningSlot(
            generation_slot_indices, active_individual_count, parent_reference_counts, target_slot_index);
        if (parent_index == kInvalidSlabSlotIndex) {
            while (next_source_slot < slab_layout.slot_count) {
                parent_index = detail::FindReferencedParentIndexOwningSlot(
                    generation_slot_indices, active_individual_count, parent_reference_counts, next_source_slot);
                if (parent_index != kInvalidSlabSlotIndex) {
                    break;
                }
                ++next_source_slot;
            }

            if ((next_source_slot >= slab_layout.slot_count) || (parent_index == kInvalidSlabSlotIndex)) {
                return false;
            }

            detail::MoveSlabBytesOverlapping(SlabSlotBytesAt(slab_storage, slab_layout, next_source_slot),
                                             SlabSlotBytesAt(slab_storage, slab_layout, target_slot_index),
                                             slab_layout.slot_stride_bytes, worker_index, worker_count);
            ++next_source_slot;
        }

        if (worker_index == 0) {
            generation_slot_indices[parent_index] = target_slot_index;
        }
        detail::SynchronizeRepackWorkers();
    }

    if (worker_index == 0) {
        for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
            if (parent_reference_counts[parent_index] == 0) {
                generation_slot_indices[parent_index] = kInvalidSlabSlotIndex;
            }
        }
    }
    detail::SynchronizeRepackWorkers();

    for (std::uint32_t source_slot_index = static_cast<std::uint32_t>(survivor_count);
         source_slot_index < slab_layout.slot_count; ++source_slot_index) {
        if (detail::FindReferencedParentIndexOwningSlot(generation_slot_indices, active_individual_count,
                                                        parent_reference_counts, source_slot_index) !=
            kInvalidSlabSlotIndex) {
            return false;
        }
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryRepackCompactedParentsForExpandedActionCount(
    const GenotypeSlabLayout &current_layout, const GenotypeSlabLayout &next_layout, std::uint8_t *slab_storage,
    SlabSlotState *slot_states, std::uint32_t *free_slot_stack, std::uint32_t &free_slot_count,
    std::uint32_t *generation_slot_indices, const std::size_t active_individual_count,
    const std::size_t survivor_count, const std::size_t destination_base_slot, const std::size_t worker_index = 0,
    const std::size_t worker_count = 1) noexcept {
    if (!IsValidGenotypeSlabLayout(current_layout) || !IsValidGenotypeSlabLayout(next_layout) ||
        (slab_storage == nullptr) || (slot_states == nullptr) || (free_slot_stack == nullptr) ||
        (generation_slot_indices == nullptr) || (active_individual_count == 0) || (survivor_count == 0) ||
        (survivor_count > current_layout.slot_count) || (destination_base_slot >= next_layout.slot_count) ||
        ((destination_base_slot + survivor_count) > next_layout.slot_count)) {
        return false;
    }

    const std::size_t current_slot_stride_bytes = current_layout.slot_stride_bytes;
    for (std::size_t compacted_slot = survivor_count; compacted_slot > 0; --compacted_slot) {
        const std::size_t source_slot_index = compacted_slot - 1;
        const std::size_t destination_slot_index = destination_base_slot + source_slot_index;
        std::uint8_t *destination_bytes = SlabSlotBytesAt(slab_storage, next_layout, destination_slot_index);
        detail::MoveSlabBytesOverlapping(SlabSlotBytesAt(slab_storage, current_layout, source_slot_index),
                                         destination_bytes, current_slot_stride_bytes, worker_index, worker_count);
        detail::ZeroSlabBytes(destination_bytes + current_slot_stride_bytes,
                              next_layout.slot_stride_bytes - current_slot_stride_bytes, worker_index, worker_count);
    }

    if (worker_index == 0) {
        for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
            if (generation_slot_indices[parent_index] == kInvalidSlabSlotIndex) {
                continue;
            }

            generation_slot_indices[parent_index] =
                static_cast<std::uint32_t>(destination_base_slot + generation_slot_indices[parent_index]);
        }
    }
    detail::SynchronizeRepackWorkers();

    if (worker_index == 0) {
        for (std::size_t slot_index = 0; slot_index < next_layout.slot_count; ++slot_index) {
            slot_states[slot_index] = {};
        }
    }
    detail::SynchronizeRepackWorkers();

    if (worker_index == 0) {
        for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
            const std::uint32_t slot_index = generation_slot_indices[parent_index];
            if (slot_index == kInvalidSlabSlotIndex) {
                continue;
            }

            slot_states[slot_index].occupied = true;
            slot_states[slot_index].liveness_count = 1;
        }
    }
    detail::SynchronizeRepackWorkers();

    if (worker_index == 0) {
        free_slot_count = static_cast<std::uint32_t>(destination_base_slot);
        for (std::uint32_t free_slot_index = 0; free_slot_index < free_slot_count; ++free_slot_index) {
            free_slot_stack[free_slot_index] = (free_slot_count - 1) - free_slot_index;
        }
    }
    detail::SynchronizeRepackWorkers();

    return true;
}

} // namespace detail

inline NEUROEVOLUTION_HOST_DEVICE bool TryCompactAndRepackSlabForExpandedActionCount(
    GenotypeSlabLayout &slab_layout, std::uint8_t *slab_storage, SlabSlotState *slot_states,
    std::uint32_t *free_slot_stack, std::uint32_t &free_slot_count, std::uint32_t *generation_slot_indices,
    const std::size_t active_individual_count, const std::uint32_t *parent_reference_counts,
    const std::size_t next_action_count, const std::size_t worker_index = 0,
    const std::size_t worker_count = 1) noexcept {
    // Growth widens the slab by compacting referenced survivors and repacking them to the right
    // using the expanded slot stride.
    if (!IsValidGenotypeSlabLayout(slab_layout) || (slab_storage == nullptr) || (slot_states == nullptr) ||
        (free_slot_stack == nullptr) || (generation_slot_indices == nullptr) || (active_individual_count == 0) ||
        (parent_reference_counts == nullptr) || (next_action_count <= slab_layout.action_count)) {
        return false;
    }

    detail::SlabRepackPreflight preflight{};
    if (!detail::TryPreflightCompactionAndRepackForExpandedActionCount(
            slab_layout, slot_states, generation_slot_indices, active_individual_count, parent_reference_counts,
            next_action_count, preflight)) {
        return false;
    }

    const GenotypeSlabLayout next_layout = preflight.next_layout;
    if (!detail::TryCompactReferencedParentsIntoPrefix(
            slab_layout, slab_storage, generation_slot_indices, active_individual_count, parent_reference_counts,
            preflight.survivor_count, worker_index, worker_count) ||
        !detail::TryRepackCompactedParentsForExpandedActionCount(
            slab_layout, next_layout, slab_storage, slot_states, free_slot_stack, free_slot_count,
            generation_slot_indices, active_individual_count, preflight.survivor_count,
            preflight.destination_base_slot, worker_index, worker_count)) {
        return false;
    }

    if (worker_index == 0) {
        slab_layout = next_layout;
    }
    detail::SynchronizeRepackWorkers();
    return true;
}

inline bool TryCompactAndRepackSlabForExpandedActionCount(HostGenotypeSlab &buffer, SlabGeneration &generation,
                                                          const std::uint32_t *parent_reference_counts,
                                                          const std::size_t next_action_count) {
    return TryCompactAndRepackSlabForExpandedActionCount(
        buffer.layout, buffer.storage.get(), buffer.slot_states.get(), buffer.free_slot_stack.get(),
        buffer.free_slot_count, generation.slot_indices.get(), generation.active_individual_count,
        parent_reference_counts, next_action_count);
}

} // namespace neuroevolution::genetic_algorithm::genotype_slab
