#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"

namespace neuroevolution::genetic_algorithm::genome {

inline NEUROEVOLUTION_HOST_DEVICE DynamicArenaSlotId *
TailRowSlotIdsForIndividual(DynamicArenaSlotId *tail_row_slot_ids, const std::size_t tail_row_slot_id_stride,
                            const std::size_t individual_index) noexcept {
    return tail_row_slot_ids + (individual_index * tail_row_slot_id_stride);
}

inline NEUROEVOLUTION_HOST_DEVICE const DynamicArenaSlotId *
TailRowSlotIdsForIndividual(const DynamicArenaSlotId *tail_row_slot_ids, const std::size_t tail_row_slot_id_stride,
                            const std::size_t individual_index) noexcept {
    return tail_row_slot_ids + (individual_index * tail_row_slot_id_stride);
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidArenaSlotId(const DynamicArenaSlotId slot_id,
                                                             const std::size_t slot_capacity) noexcept {
    return (slot_id != kInvalidDynamicArenaSlotId) && (static_cast<std::size_t>(slot_id) < slot_capacity);
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryPushArenaSlotId(DynamicArenaSlotId *free_slot_ids,
                                                          std::uint32_t &free_slot_count,
                                                          const std::size_t free_slot_capacity,
                                                          const DynamicArenaSlotId slot_id) noexcept {
    if ((free_slot_ids == nullptr) || !IsValidArenaSlotId(slot_id, free_slot_capacity) ||
        (free_slot_count >= free_slot_capacity)) {
        return false;
    }

    free_slot_ids[free_slot_count] = slot_id;
    ++free_slot_count;
    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryPopArenaSlotId(DynamicArenaSlotId *free_slot_ids,
                                                         std::uint32_t &free_slot_count,
                                                         DynamicArenaSlotId &slot_id_out) noexcept {
    if ((free_slot_ids == nullptr) || (free_slot_count == 0)) {
        slot_id_out = kInvalidDynamicArenaSlotId;
        return false;
    }

    --free_slot_count;
    slot_id_out = free_slot_ids[free_slot_count];
    return slot_id_out != kInvalidDynamicArenaSlotId;
}

inline NEUROEVOLUTION_HOST_DEVICE void InvalidateArenaGenomeSlotIds(
    DynamicArenaSlotId *body_slot_ids, DynamicArenaSlotId *tail_row_slot_ids, const std::size_t tail_row_slot_id_stride,
    const DynamicPopulationLayout &layout, const std::size_t individual_index) noexcept {
    body_slot_ids[individual_index] = kInvalidDynamicArenaSlotId;

    DynamicArenaSlotId *individual_tail_row_slot_ids =
        TailRowSlotIdsForIndividual(tail_row_slot_ids, tail_row_slot_id_stride, individual_index);
    for (std::size_t action_index = 0; action_index < layout.action_count; ++action_index) {
        individual_tail_row_slot_ids[action_index] = kInvalidDynamicArenaSlotId;
    }
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryAssignArenaGenomeSlotIds(
    DynamicArenaSlotId *body_slot_ids, DynamicArenaSlotId *tail_row_slot_ids, const std::size_t tail_row_slot_id_stride,
    const DynamicPopulationLayout &layout, const std::size_t individual_index, DynamicArenaSlotId *body_free_slot_ids,
    std::uint32_t &body_free_slot_count, const std::size_t body_free_slot_capacity,
    DynamicArenaSlotId *tail_row_free_slot_ids, std::uint32_t &tail_row_free_slot_count,
    const std::size_t tail_row_free_slot_capacity) noexcept {
    DynamicArenaSlotId allocated_body_slot_id = kInvalidDynamicArenaSlotId;
    if (!TryPopArenaSlotId(body_free_slot_ids, body_free_slot_count, allocated_body_slot_id)) {
        return false;
    }

    body_slot_ids[individual_index] = allocated_body_slot_id;

    DynamicArenaSlotId *individual_tail_row_slot_ids =
        TailRowSlotIdsForIndividual(tail_row_slot_ids, tail_row_slot_id_stride, individual_index);
    for (std::size_t action_index = 0; action_index < layout.action_count; ++action_index) {
        individual_tail_row_slot_ids[action_index] = kInvalidDynamicArenaSlotId;
        if (TryPopArenaSlotId(tail_row_free_slot_ids, tail_row_free_slot_count, individual_tail_row_slot_ids[action_index])) {
            continue;
        }

        for (std::size_t rollback_action_index = 0; rollback_action_index < action_index; ++rollback_action_index) {
            (void)TryPushArenaSlotId(tail_row_free_slot_ids, tail_row_free_slot_count, tail_row_free_slot_capacity,
                                     individual_tail_row_slot_ids[rollback_action_index]);
            individual_tail_row_slot_ids[rollback_action_index] = kInvalidDynamicArenaSlotId;
        }

        (void)TryPushArenaSlotId(body_free_slot_ids, body_free_slot_count, body_free_slot_capacity,
                                 allocated_body_slot_id);
        body_slot_ids[individual_index] = kInvalidDynamicArenaSlotId;
        return false;
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryRecycleArenaGenomeSlotIds(
    DynamicArenaSlotId *body_slot_ids, DynamicArenaSlotId *tail_row_slot_ids, const std::size_t tail_row_slot_id_stride,
    const DynamicPopulationLayout &layout, const std::size_t individual_index, DynamicArenaSlotId *body_free_slot_ids,
    std::uint32_t &body_free_slot_count, const std::size_t body_free_slot_capacity,
    DynamicArenaSlotId *tail_row_free_slot_ids, std::uint32_t &tail_row_free_slot_count,
    const std::size_t tail_row_free_slot_capacity) noexcept {
    const DynamicArenaSlotId body_slot_id = body_slot_ids[individual_index];
    if (!IsValidArenaSlotId(body_slot_id, body_free_slot_capacity)) {
        return false;
    }

    DynamicArenaSlotId *individual_tail_row_slot_ids =
        TailRowSlotIdsForIndividual(tail_row_slot_ids, tail_row_slot_id_stride, individual_index);
    for (std::size_t action_index = 0; action_index < layout.action_count; ++action_index) {
        const DynamicArenaSlotId tail_row_slot_id =
            individual_tail_row_slot_ids[layout.action_count - action_index - 1];
        if (!TryPushArenaSlotId(tail_row_free_slot_ids, tail_row_free_slot_count, tail_row_free_slot_capacity,
                                tail_row_slot_id)) {
            return false;
        }
    }

    if (!TryPushArenaSlotId(body_free_slot_ids, body_free_slot_count, body_free_slot_capacity, body_slot_id)) {
        return false;
    }

    InvalidateArenaGenomeSlotIds(body_slot_ids, tail_row_slot_ids, tail_row_slot_id_stride, layout, individual_index);
    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryConsumeParentUseAndMaybeRecycleArenaGenomeSlotIds(
    const std::size_t parent_index, std::uint32_t *remaining_use_counts, DynamicArenaSlotId *body_slot_ids,
    DynamicArenaSlotId *tail_row_slot_ids, const std::size_t tail_row_slot_id_stride,
    const DynamicPopulationLayout &layout, DynamicArenaSlotId *body_free_slot_ids, std::uint32_t &body_free_slot_count,
    const std::size_t body_free_slot_capacity, DynamicArenaSlotId *tail_row_free_slot_ids,
    std::uint32_t &tail_row_free_slot_count, const std::size_t tail_row_free_slot_capacity) noexcept {
    if ((remaining_use_counts == nullptr) || (remaining_use_counts[parent_index] == 0)) {
        return false;
    }

    --remaining_use_counts[parent_index];
    if (remaining_use_counts[parent_index] > 0) {
        return true;
    }

    return TryRecycleArenaGenomeSlotIds(body_slot_ids, tail_row_slot_ids, tail_row_slot_id_stride, layout, parent_index,
                                        body_free_slot_ids, body_free_slot_count, body_free_slot_capacity,
                                        tail_row_free_slot_ids, tail_row_free_slot_count, tail_row_free_slot_capacity);
}

} // namespace neuroevolution::genetic_algorithm::genome
