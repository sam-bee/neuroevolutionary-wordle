#include "genetic_algorithm/genotype_pool/assembly.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

namespace neuroevolution::genetic_algorithm::genotype_pool {

namespace {

bool TryCleanupFailedAssembly(HostGenotypePool &pool, PoolGeneration &next_generation) {
    if (IsValidPoolGeneration(next_generation)) {
        (void)TryReleasePoolGenerationSlots(pool, next_generation);
        next_generation = {};
    }

    return false;
}

bool IsValidParentIndex(const PoolGeneration &generation, const std::uint32_t parent_index) {
    return IsValidPoolGeneration(generation) && (parent_index < generation.active_individual_count) &&
           (generation.slot_indices[parent_index] != kInvalidPoolSlotIndex);
}

bool TryReleaseParentReference(HostGenotypePool &pool, PoolGeneration &current_generation,
                               std::uint32_t *parent_reference_counts, const std::uint32_t parent_index) {
    if (!IsValidParentIndex(current_generation, parent_index) || (parent_reference_counts == nullptr) ||
        (parent_reference_counts[parent_index] == 0)) {
        return false;
    }

    --parent_reference_counts[parent_index];
    if (parent_reference_counts[parent_index] == 0) {
        if (!TryReleasePoolSlot(pool, current_generation.slot_indices[parent_index])) {
            return false;
        }

        current_generation.slot_indices[parent_index] = kInvalidPoolSlotIndex;
    }

    return true;
}

} // namespace

bool TryAssembleNextGeneration(HostGenotypePool &pool, PoolGeneration &current_generation, const PoolAssemblyPlan &plan,
                               PoolGeneration &next_generation, const PoolAssemblyCallbacks &callbacks) {
    next_generation = {};
    if (!IsValidHostGenotypePool(pool) || !IsValidPoolGeneration(current_generation) ||
        !IsValidPoolAssemblyPlan(plan) || (callbacks.assemble_child_genome == nullptr)) {
        return false;
    }

    std::unique_ptr<std::uint32_t[]> parent_reference_counts(
        new (std::nothrow) std::uint32_t[current_generation.active_individual_count]());
    if (parent_reference_counts == nullptr) {
        return false;
    }

    for (std::size_t child_index = 0; child_index < plan.child_count; ++child_index) {
        const PoolParentPair &parent_pair = plan.parent_pairs[child_index];
        if (!IsValidParentIndex(current_generation, parent_pair.first_parent_index) ||
            !IsValidParentIndex(current_generation, parent_pair.second_parent_index)) {
            return false;
        }

        ++parent_reference_counts[parent_pair.first_parent_index];
        ++parent_reference_counts[parent_pair.second_parent_index];
    }

    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        if ((parent_reference_counts[parent_index] == 0) &&
            (current_generation.slot_indices[parent_index] != kInvalidPoolSlotIndex)) {
            if (!TryReleasePoolSlot(pool, current_generation.slot_indices[parent_index])) {
                return false;
            }

            current_generation.slot_indices[parent_index] = kInvalidPoolSlotIndex;
        }
    }

    if (!TryCreatePoolGeneration(next_generation, plan.child_count, current_generation.generation_index + 1)) {
        return false;
    }

    for (std::size_t child_index = 0; child_index < plan.child_count; ++child_index) {
        const PoolParentPair &parent_pair = plan.parent_pairs[child_index];
        const std::uint32_t first_parent_slot = current_generation.slot_indices[parent_pair.first_parent_index];
        const std::uint32_t second_parent_slot = current_generation.slot_indices[parent_pair.second_parent_index];
        if ((first_parent_slot == kInvalidPoolSlotIndex) || (second_parent_slot == kInvalidPoolSlotIndex)) {
            return TryCleanupFailedAssembly(pool, next_generation);
        }

        std::uint32_t child_slot = kInvalidPoolSlotIndex;
        if (!TryAllocatePoolSlot(pool, child_slot)) {
            return TryCleanupFailedAssembly(pool, next_generation);
        }

        if (!callbacks.assemble_child_genome(HostPoolSlotBytesAt(pool, first_parent_slot),
                                             HostPoolSlotBytesAt(pool, second_parent_slot), pool.layout.action_count,
                                             HostPoolSlotBytesAt(pool, child_slot), callbacks.user_data)) {
            (void)TryReleasePoolSlot(pool, child_slot);
            return TryCleanupFailedAssembly(pool, next_generation);
        }

        if (!TrySetPoolGenerationSlot(next_generation, child_index, child_slot)) {
            (void)TryReleasePoolSlot(pool, child_slot);
            return TryCleanupFailedAssembly(pool, next_generation);
        }

        if (!TryReleaseParentReference(pool, current_generation, parent_reference_counts.get(),
                                       parent_pair.first_parent_index) ||
            !TryReleaseParentReference(pool, current_generation, parent_reference_counts.get(),
                                       parent_pair.second_parent_index)) {
            return TryCleanupFailedAssembly(pool, next_generation);
        }
    }

    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        if (current_generation.slot_indices[parent_index] != kInvalidPoolSlotIndex) {
            return TryCleanupFailedAssembly(pool, next_generation);
        }
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm::genotype_pool
