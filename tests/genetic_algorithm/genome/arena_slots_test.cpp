#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "genetic_algorithm/genome/arena_slots.hpp"

namespace {

using neuroevolution::genetic_algorithm::genome::DynamicArenaSlotId;
using neuroevolution::genetic_algorithm::genome::DynamicPopulationLayout;
using neuroevolution::genetic_algorithm::genome::IsValidArenaSlotId;
using neuroevolution::genetic_algorithm::genome::MakeDynamicPopulationLayout;
using neuroevolution::genetic_algorithm::genome::TailRowSlotIdsForIndividual;
using neuroevolution::genetic_algorithm::genome::TryAssignArenaGenomeSlotIds;
using neuroevolution::genetic_algorithm::genome::TryConsumeParentUseAndMaybeRecycleArenaGenomeSlotIds;
using neuroevolution::genetic_algorithm::genome::kInvalidDynamicArenaSlotId;

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

bool TestArenaGenomeSlotIdsCanBeReusedAfterFinalParentUse() {
    constexpr std::size_t kTailStride = 4;

    DynamicArenaSlotId current_body_slot_ids[2]{0, 1};
    DynamicArenaSlotId current_tail_row_slot_ids[2 * kTailStride]{
        0, 1, kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId,
        2, 3, kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId,
    };
    DynamicArenaSlotId next_body_slot_ids[2]{kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId};
    DynamicArenaSlotId next_tail_row_slot_ids[2 * kTailStride]{
        kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId,
        kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId,
    };

    DynamicArenaSlotId body_free_slot_ids[4]{3, 2, kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId};
    DynamicArenaSlotId tail_row_free_slot_ids[8]{7, 6, 5, 4, kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId,
                                                 kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId};
    std::uint32_t body_free_slot_count = 2;
    std::uint32_t tail_row_free_slot_count = 4;
    std::uint32_t remaining_use_counts[2]{0, 2};

    const DynamicPopulationLayout current_layout = MakeDynamicPopulationLayout(2, 0, 2, 1);
    const DynamicPopulationLayout next_layout = MakeDynamicPopulationLayout(2, 1, 2, 1);

    bool ok = true;
    ok &= ExpectTrue(TryAssignArenaGenomeSlotIds(next_body_slot_ids, next_tail_row_slot_ids, kTailStride, next_layout, 0,
                                                 body_free_slot_ids, body_free_slot_count, 4, tail_row_free_slot_ids,
                                                 tail_row_free_slot_count, 8),
                     "Expected first child slot assignment to succeed from the initial free pool");
    ok &= ExpectTrue(next_body_slot_ids[0] == 2, "Expected first child to consume the most recently free body slot");
    ok &= ExpectTrue(TailRowSlotIdsForIndividual(next_tail_row_slot_ids, kTailStride, 0)[0] == 4,
                     "Expected first child action zero to consume the most recently free tail slot");
    ok &= ExpectTrue(TailRowSlotIdsForIndividual(next_tail_row_slot_ids, kTailStride, 0)[1] == 5,
                     "Expected first child action one to consume the next free tail slot");
    ok &= ExpectTrue(body_free_slot_count == 1, "Expected first child allocation to reduce body free-count");
    ok &= ExpectTrue(tail_row_free_slot_count == 2, "Expected first child allocation to reduce tail free-count");
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(TryConsumeParentUseAndMaybeRecycleArenaGenomeSlotIds(
                         1, remaining_use_counts, current_body_slot_ids, current_tail_row_slot_ids, kTailStride,
                         current_layout, body_free_slot_ids, body_free_slot_count, 4, tail_row_free_slot_ids,
                         tail_row_free_slot_count, 8),
                     "Expected first parent use consumption to succeed without retiring the parent");
    ok &= ExpectTrue(remaining_use_counts[1] == 1, "Expected first parent use to leave one remaining use");
    ok &= ExpectTrue(IsValidArenaSlotId(current_body_slot_ids[1], 4),
                     "Expected parent body slot to remain live until the final use is consumed");
    ok &= ExpectTrue(body_free_slot_count == 1, "Expected no body slot to be recycled before the final use");
    ok &= ExpectTrue(tail_row_free_slot_count == 2, "Expected no tail slots to be recycled before the final use");
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(TryConsumeParentUseAndMaybeRecycleArenaGenomeSlotIds(
                         1, remaining_use_counts, current_body_slot_ids, current_tail_row_slot_ids, kTailStride,
                         current_layout, body_free_slot_ids, body_free_slot_count, 4, tail_row_free_slot_ids,
                         tail_row_free_slot_count, 8),
                     "Expected final parent use to recycle the parent's slots");
    ok &= ExpectTrue(remaining_use_counts[1] == 0, "Expected final parent use to exhaust the parent");
    ok &= ExpectTrue(current_body_slot_ids[1] == kInvalidDynamicArenaSlotId,
                     "Expected recycled parent body slot id to be invalidated");
    ok &= ExpectTrue(TailRowSlotIdsForIndividual(current_tail_row_slot_ids, kTailStride, 1)[0] ==
                         kInvalidDynamicArenaSlotId,
                     "Expected recycled parent tail slot ids to be invalidated");
    ok &= ExpectTrue(body_free_slot_count == 2, "Expected recycled parent body slot to return to the free pool");
    ok &= ExpectTrue(tail_row_free_slot_count == 4, "Expected recycled parent tail slots to return to the free pool");
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(TryAssignArenaGenomeSlotIds(next_body_slot_ids, next_tail_row_slot_ids, kTailStride, next_layout, 1,
                                                 body_free_slot_ids, body_free_slot_count, 4, tail_row_free_slot_ids,
                                                 tail_row_free_slot_count, 8),
                     "Expected second child slot assignment to reuse the recycled parent slots");
    ok &= ExpectTrue(next_body_slot_ids[1] == 1, "Expected second child to reuse the retired parent body slot");
    ok &= ExpectTrue(TailRowSlotIdsForIndividual(next_tail_row_slot_ids, kTailStride, 1)[0] == 2,
                     "Expected second child action zero to reuse the retired parent tail slot order");
    ok &= ExpectTrue(TailRowSlotIdsForIndividual(next_tail_row_slot_ids, kTailStride, 1)[1] == 3,
                     "Expected second child action one to reuse the retired parent tail slot order");
    return ok;
}

bool TestAssignArenaGenomeSlotIdsRollsBackOnInsufficientTailCapacity() {
    constexpr std::size_t kTailStride = 3;

    DynamicArenaSlotId body_slot_ids[1]{kInvalidDynamicArenaSlotId};
    DynamicArenaSlotId tail_row_slot_ids[kTailStride]{
        kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId, kInvalidDynamicArenaSlotId};
    DynamicArenaSlotId body_free_slot_ids[2]{1, kInvalidDynamicArenaSlotId};
    DynamicArenaSlotId tail_row_free_slot_ids[2]{0, kInvalidDynamicArenaSlotId};
    std::uint32_t body_free_slot_count = 1;
    std::uint32_t tail_row_free_slot_count = 1;
    const DynamicPopulationLayout layout = MakeDynamicPopulationLayout(1, 0, 2, 1);

    bool ok = true;
    ok &= ExpectTrue(!TryAssignArenaGenomeSlotIds(body_slot_ids, tail_row_slot_ids, kTailStride, layout, 0,
                                                  body_free_slot_ids, body_free_slot_count, 2, tail_row_free_slot_ids,
                                                  tail_row_free_slot_count, 2),
                     "Expected genome-slot assignment to fail when there are not enough tail slots");
    ok &= ExpectTrue(body_slot_ids[0] == kInvalidDynamicArenaSlotId,
                     "Expected failed assignment to roll back the body slot id");
    ok &= ExpectTrue(body_free_slot_count == 1, "Expected failed assignment to restore the body free-count");
    ok &= ExpectTrue(tail_row_free_slot_count == 1, "Expected failed assignment to restore the tail free-count");
    return ok;
}

} // namespace

int main() {
    if (!TestArenaGenomeSlotIdsCanBeReusedAfterFinalParentUse()) {
        return 1;
    }

    if (!TestAssignArenaGenomeSlotIdsRollsBackOnInsufficientTailCapacity()) {
        return 1;
    }

    std::cout << "PASS: arena_slots_test\n";
    return 0;
}
