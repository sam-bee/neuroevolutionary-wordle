#include "genetic_algorithm/genotype_buffer/assembly.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include "genetic_algorithm/genotype_buffer/reference_counter.hpp"

namespace neuroevolution::genetic_algorithm::genotype_buffer {

namespace {

bool TryCleanupFailedAssembly(HostGenotypeBuffer &buffer, BufferGeneration &next_generation) {
    if (IsValidBufferGeneration(next_generation)) {
        (void)TryReleaseBufferGenerationSlots(buffer, next_generation);
        next_generation = {};
    }

    return false;
}

} // namespace

bool TryAssembleNextGeneration(HostGenotypeBuffer &buffer, BufferGeneration &current_generation,
                               const BufferAssemblyPlan &plan, BufferGeneration &next_generation,
                               const BufferAssemblyCallbacks &callbacks) {
    next_generation = {};
    if (!IsValidHostGenotypeBuffer(buffer) || !IsValidBufferGeneration(current_generation) ||
        !IsValidBufferAssemblyPlan(plan) || (callbacks.assemble_child_genome == nullptr)) {
        return false;
    }

    std::unique_ptr<std::uint32_t[]> parent_reference_counts(
        new (std::nothrow) std::uint32_t[current_generation.active_individual_count]());
    if (parent_reference_counts == nullptr) {
        return false;
    }

    if (!TryBuildParentReferenceCounts(MakeBufferGenerationView(current_generation), plan.parent_pairs.get(),
                                       plan.child_count, parent_reference_counts.get())) {
        return false;
    }

    if (!TryCollectZeroReferenceParents(MakeGenotypeBufferView(buffer), MakeBufferGenerationView(current_generation),
                                        parent_reference_counts.get())) {
        return false;
    }

    if (!TryCreateBufferGeneration(next_generation, plan.child_count, current_generation.generation_index + 1)) {
        return false;
    }

    for (std::size_t child_index = 0; child_index < plan.child_count; ++child_index) {
        const BufferParentPair &parent_pair = plan.parent_pairs[child_index];
        const std::uint32_t first_parent_slot = current_generation.slot_indices[parent_pair.first_parent_index];
        const std::uint32_t second_parent_slot = current_generation.slot_indices[parent_pair.second_parent_index];
        if ((first_parent_slot == kInvalidBufferSlotIndex) || (second_parent_slot == kInvalidBufferSlotIndex)) {
            return TryCleanupFailedAssembly(buffer, next_generation);
        }

        std::uint32_t child_slot = kInvalidBufferSlotIndex;
        if (!TryAllocateBufferSlot(buffer, child_slot)) {
            return TryCleanupFailedAssembly(buffer, next_generation);
        }

        if (!callbacks.assemble_child_genome(
                HostBufferSlotBytesAt(buffer, first_parent_slot), HostBufferSlotBytesAt(buffer, second_parent_slot),
                buffer.layout.action_count, HostBufferSlotBytesAt(buffer, child_slot), callbacks.user_data)) {
            (void)TryReleaseBufferSlot(buffer, child_slot);
            return TryCleanupFailedAssembly(buffer, next_generation);
        }

        if (!TrySetBufferGenerationSlot(next_generation, child_index, child_slot)) {
            (void)TryReleaseBufferSlot(buffer, child_slot);
            return TryCleanupFailedAssembly(buffer, next_generation);
        }

        if (!TryReleaseParentReference(MakeGenotypeBufferView(buffer), MakeBufferGenerationView(current_generation),
                                       parent_reference_counts.get(), parent_pair.first_parent_index) ||
            !TryReleaseParentReference(MakeGenotypeBufferView(buffer), MakeBufferGenerationView(current_generation),
                                       parent_reference_counts.get(), parent_pair.second_parent_index)) {
            return TryCleanupFailedAssembly(buffer, next_generation);
        }
    }

    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        if (current_generation.slot_indices[parent_index] != kInvalidBufferSlotIndex) {
            return TryCleanupFailedAssembly(buffer, next_generation);
        }
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm::genotype_buffer
