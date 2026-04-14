#include "genetic_algorithm/genotype_arena/assembly.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

namespace neuroevolution::genetic_algorithm::genotype_arena {

namespace {

bool TryCleanupFailedAssembly(HostGenotypeArena &arena, ArenaGeneration &next_generation) {
    if (IsValidArenaGeneration(next_generation)) {
        (void)TryReleaseArenaGenerationSlots(arena, next_generation);
        next_generation = {};
    }

    return false;
}

bool IsValidParentIndex(const ArenaGeneration &generation, const std::uint32_t parent_index) {
    return IsValidArenaGeneration(generation) && (parent_index < generation.active_individual_count) &&
           (generation.slot_indices[parent_index] != kInvalidArenaSlotIndex);
}

bool TryConsumeParentDuty(HostGenotypeArena &arena, ArenaGeneration &current_generation,
                          std::uint32_t *remaining_parent_duties, const std::uint32_t parent_index) {
    if (!IsValidParentIndex(current_generation, parent_index) || (remaining_parent_duties == nullptr) ||
        (remaining_parent_duties[parent_index] == 0)) {
        return false;
    }

    --remaining_parent_duties[parent_index];
    if (remaining_parent_duties[parent_index] == 0) {
        if (!TryReleaseArenaSlot(arena, current_generation.slot_indices[parent_index])) {
            return false;
        }

        current_generation.slot_indices[parent_index] = kInvalidArenaSlotIndex;
    }

    return true;
}

} // namespace

bool TryAssembleNextGenerationWithoutElitism(HostGenotypeArena &arena, ArenaGeneration &current_generation,
                                             const ArenaAssemblyPlan &plan, ArenaGeneration &next_generation,
                                             const ArenaAssemblyCallbacks &callbacks) {
    next_generation = {};
    if (!IsValidHostGenotypeArena(arena) || !IsValidArenaGeneration(current_generation) ||
        !IsValidArenaAssemblyPlan(plan) || (callbacks.assemble_child_genome == nullptr)) {
        return false;
    }

    std::unique_ptr<std::uint32_t[]> remaining_parent_duties(
        new (std::nothrow) std::uint32_t[current_generation.active_individual_count]());
    if (remaining_parent_duties == nullptr) {
        return false;
    }

    for (std::size_t child_index = 0; child_index < plan.child_count; ++child_index) {
        const ArenaParentPair &parent_pair = plan.parent_pairs[child_index];
        if (!IsValidParentIndex(current_generation, parent_pair.first_parent_index) ||
            !IsValidParentIndex(current_generation, parent_pair.second_parent_index)) {
            return false;
        }

        ++remaining_parent_duties[parent_pair.first_parent_index];
        ++remaining_parent_duties[parent_pair.second_parent_index];
    }

    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        if ((remaining_parent_duties[parent_index] == 0) &&
            (current_generation.slot_indices[parent_index] != kInvalidArenaSlotIndex)) {
            if (!TryReleaseArenaSlot(arena, current_generation.slot_indices[parent_index])) {
                return false;
            }

            current_generation.slot_indices[parent_index] = kInvalidArenaSlotIndex;
        }
    }

    if (!TryCreateArenaGeneration(next_generation, plan.child_count, current_generation.generation_index + 1)) {
        return false;
    }

    for (std::size_t child_index = 0; child_index < plan.child_count; ++child_index) {
        const ArenaParentPair &parent_pair = plan.parent_pairs[child_index];
        const std::uint32_t first_parent_slot = current_generation.slot_indices[parent_pair.first_parent_index];
        const std::uint32_t second_parent_slot = current_generation.slot_indices[parent_pair.second_parent_index];
        if ((first_parent_slot == kInvalidArenaSlotIndex) || (second_parent_slot == kInvalidArenaSlotIndex)) {
            return TryCleanupFailedAssembly(arena, next_generation);
        }

        std::uint32_t child_slot = kInvalidArenaSlotIndex;
        if (!TryAllocateArenaSlot(arena, child_slot)) {
            return TryCleanupFailedAssembly(arena, next_generation);
        }

        if (!callbacks.assemble_child_genome(HostArenaSlotBytesAt(arena, first_parent_slot),
                                             HostArenaSlotBytesAt(arena, second_parent_slot), arena.layout.action_count,
                                             HostArenaSlotBytesAt(arena, child_slot), callbacks.user_data)) {
            (void)TryReleaseArenaSlot(arena, child_slot);
            return TryCleanupFailedAssembly(arena, next_generation);
        }

        if (!TrySetArenaGenerationSlot(next_generation, child_index, child_slot)) {
            (void)TryReleaseArenaSlot(arena, child_slot);
            return TryCleanupFailedAssembly(arena, next_generation);
        }

        if (!TryConsumeParentDuty(arena, current_generation, remaining_parent_duties.get(),
                                  parent_pair.first_parent_index) ||
            !TryConsumeParentDuty(arena, current_generation, remaining_parent_duties.get(),
                                  parent_pair.second_parent_index)) {
            return TryCleanupFailedAssembly(arena, next_generation);
        }
    }

    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        if (current_generation.slot_indices[parent_index] != kInvalidArenaSlotIndex) {
            return TryCleanupFailedAssembly(arena, next_generation);
        }
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm::genotype_arena
