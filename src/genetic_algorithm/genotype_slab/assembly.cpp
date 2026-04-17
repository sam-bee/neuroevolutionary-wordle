#include "genetic_algorithm/genotype_slab/assembly.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include "genetic_algorithm/genotype_slab/reference_counter.hpp"

namespace neuroevolution::genetic_algorithm::genotype_slab {

namespace {

bool TryCleanupFailedAssembly(HostGenotypeSlab &buffer, SlabGeneration &next_generation) {
    if (IsValidSlabGeneration(next_generation)) {
        (void)TryReleaseSlabGenerationSlots(buffer, next_generation);
        next_generation = {};
    }

    return false;
}

} // namespace

bool TryAssembleNextGeneration(HostGenotypeSlab &buffer, SlabGeneration &current_generation,
                               const SlabAssemblyPlan &plan, SlabGeneration &next_generation,
                               const SlabAssemblyCallbacks &callbacks) {
    next_generation = {};
    if (!IsValidHostGenotypeSlab(buffer) || !IsValidSlabGeneration(current_generation) ||
        !IsValidSlabAssemblyPlan(plan) || (callbacks.assemble_child_genome == nullptr)) {
        return false;
    }

    std::unique_ptr<std::uint32_t[]> parent_reference_counts(
        new (std::nothrow) std::uint32_t[current_generation.active_individual_count]());
    if (parent_reference_counts == nullptr) {
        return false;
    }

    if (!TryBuildParentReferenceCounts(MakeSlabGenerationView(current_generation), plan.parent_pairs.get(),
                                       plan.child_count, parent_reference_counts.get())) {
        return false;
    }

    if (!TryCollectZeroReferenceParents(MakeGenotypeSlabView(buffer), MakeSlabGenerationView(current_generation),
                                        parent_reference_counts.get())) {
        return false;
    }

    if (!TryCreateSlabGeneration(next_generation, plan.child_count, current_generation.generation_index + 1)) {
        return false;
    }

    for (std::size_t child_index = 0; child_index < plan.child_count; ++child_index) {
        const SlabParentPair &parent_pair = plan.parent_pairs[child_index];
        const std::uint32_t first_parent_slot = current_generation.slot_indices[parent_pair.first_parent_index];
        const std::uint32_t second_parent_slot = current_generation.slot_indices[parent_pair.second_parent_index];
        if ((first_parent_slot == kInvalidSlabSlotIndex) || (second_parent_slot == kInvalidSlabSlotIndex)) {
            return TryCleanupFailedAssembly(buffer, next_generation);
        }

        std::uint32_t child_slot = kInvalidSlabSlotIndex;
        if (!TryAllocateSlabSlot(buffer, child_slot)) {
            return TryCleanupFailedAssembly(buffer, next_generation);
        }

        if (!callbacks.assemble_child_genome(
                HostSlabSlotBytesAt(buffer, first_parent_slot), HostSlabSlotBytesAt(buffer, second_parent_slot),
                buffer.layout.action_count, HostSlabSlotBytesAt(buffer, child_slot), callbacks.user_data)) {
            (void)TryReleaseSlabSlot(buffer, child_slot);
            return TryCleanupFailedAssembly(buffer, next_generation);
        }

        if (!TrySetSlabGenerationSlot(next_generation, child_index, child_slot)) {
            (void)TryReleaseSlabSlot(buffer, child_slot);
            return TryCleanupFailedAssembly(buffer, next_generation);
        }

        if (!TryReleaseParentReference(MakeGenotypeSlabView(buffer), MakeSlabGenerationView(current_generation),
                                       parent_reference_counts.get(), parent_pair.first_parent_index) ||
            !TryReleaseParentReference(MakeGenotypeSlabView(buffer), MakeSlabGenerationView(current_generation),
                                       parent_reference_counts.get(), parent_pair.second_parent_index)) {
            return TryCleanupFailedAssembly(buffer, next_generation);
        }
    }

    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        if (current_generation.slot_indices[parent_index] != kInvalidSlabSlotIndex) {
            return TryCleanupFailedAssembly(buffer, next_generation);
        }
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm::genotype_slab
