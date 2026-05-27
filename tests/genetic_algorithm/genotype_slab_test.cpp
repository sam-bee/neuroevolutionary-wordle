#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "genetic_algorithm/genotype_slab/assembly.hpp"
#include "genetic_algorithm/genotype_slab/generation.hpp"
#include "genetic_algorithm/genotype_slab/reference_counter.hpp"
#include "genetic_algorithm/genotype_slab/slab_allocator.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::genetic_algorithm::genome::ComputeDynamicGenomeStrideBytes;
using neuroevolution::genetic_algorithm::genotype_slab::ClearSlabGenerationFitness;
using neuroevolution::genetic_algorithm::genotype_slab::ComputeSlabSlotStrideBytes;
using neuroevolution::genetic_algorithm::genotype_slab::GenomePolicyModelParameters;
using neuroevolution::genetic_algorithm::genotype_slab::GenomeTailRows;
using neuroevolution::genetic_algorithm::genotype_slab::GenotypeSlabLayout;
using neuroevolution::genetic_algorithm::genotype_slab::GenotypeSlabView;
using neuroevolution::genetic_algorithm::genotype_slab::HostGenotypeSlab;
using neuroevolution::genetic_algorithm::genotype_slab::HostSlabSlotBytesAt;
using neuroevolution::genetic_algorithm::genotype_slab::IsValidGenotypeSlabLayout;
using neuroevolution::genetic_algorithm::genotype_slab::IsValidSlabGeneration;
using neuroevolution::genetic_algorithm::genotype_slab::kInvalidSlabSlotIndex;
using neuroevolution::genetic_algorithm::genotype_slab::MakeGenotypeSlabView;
using neuroevolution::genetic_algorithm::genotype_slab::MakeSlabGenerationView;
using neuroevolution::genetic_algorithm::genotype_slab::SlabAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_slab::SlabGeneration;
using neuroevolution::genetic_algorithm::genotype_slab::SlabGenerationView;
using neuroevolution::genetic_algorithm::genotype_slab::SlabSlotCountForByteBudget;
using neuroevolution::genetic_algorithm::genotype_slab::SlabSlotState;
using neuroevolution::genetic_algorithm::genotype_slab::TryAllocateSlabSlot;
using neuroevolution::genetic_algorithm::genotype_slab::TryBuildParentReferenceCounts;
using neuroevolution::genetic_algorithm::genotype_slab::TryCloneSlabSlotIntoGeneration;
using neuroevolution::genetic_algorithm::genotype_slab::TryCollectZeroReferenceParents;
using neuroevolution::genetic_algorithm::genotype_slab::TryCopyGenomeBytesIntoSlabSlot;
using neuroevolution::genetic_algorithm::genotype_slab::TryCreateHostGenotypeSlab;
using neuroevolution::genetic_algorithm::genotype_slab::TryCreateSlabAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_slab::TryCreateSlabGeneration;
using neuroevolution::genetic_algorithm::genotype_slab::TryReleaseParentReference;
using neuroevolution::genetic_algorithm::genotype_slab::TryReleaseSlabGenerationSlots;
using neuroevolution::genetic_algorithm::genotype_slab::TryReleaseSlabSlot;
using neuroevolution::genetic_algorithm::genotype_slab::TryRetainSlabSlot;
using neuroevolution::genetic_algorithm::genotype_slab::TrySetSlabGenerationSlot;

constexpr float kTolerance = 1.0e-6f;

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

bool ExpectNear(const float actual, const float expected, const std::string_view label) {
    if (std::fabs(actual - expected) > kTolerance) {
        std::cerr << "FAIL: " << label << " expected " << expected << ", got " << actual << '\n';
        return false;
    }

    return true;
}

bool TestSlabLayoutReusesDynamicGenomeStrideMath() {
    constexpr std::size_t kActionCount = 20;
    constexpr std::size_t kSlotCount = 6;

    GenotypeSlabLayout layout{};
    layout.action_count = kActionCount;
    layout.slot_stride_bytes = ComputeSlabSlotStrideBytes(kActionCount);
    layout.slot_count = kSlotCount;
    layout.slab_bytes = kSlotCount * layout.slot_stride_bytes;

    bool ok = true;
    ok &= ExpectTrue(layout.slot_stride_bytes == ComputeDynamicGenomeStrideBytes(kActionCount),
                     "Expected slab slot stride to match dynamic genome stride math");
    ok &= ExpectTrue(SlabSlotCountForByteBudget(layout.slab_bytes, kActionCount) == kSlotCount,
                     "Expected slab byte budget helper to recover the slot count");
    ok &= ExpectTrue(IsValidGenotypeSlabLayout(layout), "Expected constructed slab layout to be valid");
    return ok;
}

bool TestRawViewsMutateUnderlyingBufferAndGenerationState() {
    HostGenotypeSlab buffer{};
    SlabGeneration generation{};
    bool ok = TryCreateHostGenotypeSlab(buffer, 2, 4);
    ok &= TryCreateSlabGeneration(generation, 2, 3);
    ok &= ExpectTrue(ok, "Expected raw-view test fixtures to allocate");
    if (!ok) {
        return false;
    }

    GenotypeSlabView buffer_view = MakeGenotypeSlabView(buffer);
    SlabGenerationView generation_view = MakeSlabGenerationView(generation);
    std::uint32_t slot_index = kInvalidSlabSlotIndex;

    ok &= TryAllocateSlabSlot(buffer_view, slot_index);
    ok &= TrySetSlabGenerationSlot(generation_view, 0, slot_index);
    ok &= ExpectTrue(ok, "Expected view-based slot allocation and assignment to succeed");
    ok &= ExpectTrue(buffer.free_slot_count == 1, "Expected the host owner to observe the view-based allocation");
    ok &= ExpectTrue(generation.slot_indices[0] == slot_index,
                     "Expected the host generation to observe the view-based slot assignment");
    return ok;
}

bool TestHostSlabAllocatesReleasesAndReusesFixedWidthSlots() {
    HostGenotypeSlab buffer{};
    bool ok = TryCreateHostGenotypeSlab(buffer, 3, 4);
    ok &= ExpectTrue(ok, "Expected host genotype slab creation to succeed");
    if (!ok) {
        return false;
    }

    std::uint32_t slot0 = kInvalidSlabSlotIndex;
    std::uint32_t slot1 = kInvalidSlabSlotIndex;
    std::uint32_t slot2 = kInvalidSlabSlotIndex;
    std::uint32_t overflow_slot = kInvalidSlabSlotIndex;

    ok &= TryAllocateSlabSlot(buffer, slot0);
    ok &= TryAllocateSlabSlot(buffer, slot1);
    ok &= TryAllocateSlabSlot(buffer, slot2);
    ok &= ExpectTrue(ok, "Expected every slab slot to allocate exactly once");
    ok &= ExpectTrue((slot0 == 0U) && (slot1 == 1U) && (slot2 == 2U),
                     "Expected the free-slot stack to hand out deterministic slot indices");
    ok &= ExpectTrue(!TryAllocateSlabSlot(buffer, overflow_slot),
                     "Expected allocation to fail once the fixed-width slab is full");
    ok &= ExpectTrue(buffer.free_slot_count == 0, "Expected a full slab to report no free slots");

    ok &= TryRetainSlabSlot(buffer, slot1);
    ok &= ExpectTrue(buffer.slot_states[slot1].liveness_count == 2,
                     "Expected retaining a slot to increment its reference count");
    ok &= TryReleaseSlabSlot(buffer, slot1);
    ok &= ExpectTrue(buffer.slot_states[slot1].occupied, "Expected the slot to stay occupied while references remain");
    ok &= ExpectTrue(buffer.slot_states[slot1].liveness_count == 1,
                     "Expected releasing one reference to leave the slot live");
    ok &= TryReleaseSlabSlot(buffer, slot1);
    ok &= ExpectTrue(!buffer.slot_states[slot1].occupied,
                     "Expected the slot to become free after the last reference is released");
    ok &= ExpectTrue(buffer.free_slot_count == 1,
                     "Expected releasing the last reference to return the slot to the buffer");

    std::uint32_t reused_slot = kInvalidSlabSlotIndex;
    ok &= TryAllocateSlabSlot(buffer, reused_slot);
    ok &= ExpectTrue(reused_slot == slot1, "Expected the most recently released slot to be reused first");
    return ok;
}

bool TestReferenceCounterBuildsCollectsAndReleasesParentSlots() {
    HostGenotypeSlab buffer{};
    SlabGeneration current_generation{};
    SlabAssemblyPlan plan{};
    bool ok = TryCreateHostGenotypeSlab(buffer, 4, 4);
    ok &= TryCreateSlabGeneration(current_generation, 3, 11);
    ok &= TryCreateSlabAssemblyPlan(plan, 2);
    ok &= ExpectTrue(ok, "Expected reference-counter fixtures to allocate");
    if (!ok) {
        return false;
    }

    std::uint32_t parent_slots[3]{};
    for (std::size_t parent_index = 0; parent_index < 3; ++parent_index) {
        ok &= TryAllocateSlabSlot(buffer, parent_slots[parent_index]);
        ok &= TrySetSlabGenerationSlot(current_generation, parent_index, parent_slots[parent_index]);
    }
    ok &= ExpectTrue(ok, "Expected reference-counter parent slots to allocate");
    if (!ok) {
        return false;
    }

    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};
    plan.parent_pairs[1] = {.first_parent_index = 0, .second_parent_index = 0};

    std::uint32_t parent_reference_counts[3]{};
    ok &= TryBuildParentReferenceCounts(MakeSlabGenerationView(current_generation), plan.parent_pairs.get(),
                                        plan.child_count, parent_reference_counts);
    ok &= ExpectTrue(parent_reference_counts[0] == 3U,
                     "Expected repeated parent selection to add every parent reference");
    ok &= ExpectTrue(parent_reference_counts[1] == 1U, "Expected singly selected parent to get one parent reference");
    ok &= ExpectTrue(parent_reference_counts[2] == 0U, "Expected unselected parent to remain zero-reference");

    ok &= TryCollectZeroReferenceParents(MakeGenotypeSlabView(buffer), MakeSlabGenerationView(current_generation),
                                         parent_reference_counts);
    ok &= ExpectTrue(current_generation.slot_indices[2] == kInvalidSlabSlotIndex,
                     "Expected zero-reference parent to be garbage-collected");
    ok &= ExpectTrue(buffer.free_slot_count == 2U,
                     "Expected zero-reference collection to return the parent slot to the buffer");

    ok &= TryReleaseParentReference(MakeGenotypeSlabView(buffer), MakeSlabGenerationView(current_generation),
                                    parent_reference_counts, 1);
    ok &= ExpectTrue(current_generation.slot_indices[1] == kInvalidSlabSlotIndex,
                     "Expected final reference release to clear the singly selected parent");
    ok &= ExpectTrue(buffer.free_slot_count == 3U,
                     "Expected final reference release to return the singly selected parent slot");

    ok &= TryReleaseParentReference(MakeGenotypeSlabView(buffer), MakeSlabGenerationView(current_generation),
                                    parent_reference_counts, 0);
    ok &= TryReleaseParentReference(MakeGenotypeSlabView(buffer), MakeSlabGenerationView(current_generation),
                                    parent_reference_counts, 0);
    ok &= ExpectTrue(current_generation.slot_indices[0] == parent_slots[0],
                     "Expected parent with remaining references to stay live");
    ok &= ExpectTrue(buffer.free_slot_count == 3U, "Expected non-final releases to leave the parent slot occupied");

    ok &= TryReleaseParentReference(MakeGenotypeSlabView(buffer), MakeSlabGenerationView(current_generation),
                                    parent_reference_counts, 0);
    ok &= ExpectTrue(current_generation.slot_indices[0] == kInvalidSlabSlotIndex,
                     "Expected the last parent reference to garbage-collect the parent slot");
    ok &= ExpectTrue(buffer.free_slot_count == 4U, "Expected every parent slot to be free after final releases");
    ok &= ExpectTrue(!TryReleaseParentReference(MakeGenotypeSlabView(buffer),
                                                MakeSlabGenerationView(current_generation), parent_reference_counts, 0),
                     "Expected releasing a collected parent reference to fail");
    return ok;
}

bool TestSlabSlotAccessorsAndCopyHelperRoundTripGenomeBytes() {
    HostGenotypeSlab buffer{};
    bool ok = TryCreateHostGenotypeSlab(buffer, 2, 4);
    ok &= ExpectTrue(ok, "Expected host genotype slab creation for copy test to succeed");
    if (!ok) {
        return false;
    }

    std::uint32_t source_slot = kInvalidSlabSlotIndex;
    std::uint32_t target_slot = kInvalidSlabSlotIndex;
    ok &= TryAllocateSlabSlot(buffer, source_slot);
    ok &= TryAllocateSlabSlot(buffer, target_slot);
    ok &= ExpectTrue(ok, "Expected copy test slots to allocate");
    if (!ok) {
        return false;
    }

    std::uint8_t *source_genome_bytes = HostSlabSlotBytesAt(buffer, source_slot);
    GenomePolicyModelParameters(source_genome_bytes).dense_trunk.hidden1_to_output.biases[0] = 1.25f;
    GenomePolicyModelParameters(source_genome_bytes).input_encoder.input_to_hidden.weights[0] = -0.75f;
    GenomeTailRows(source_genome_bytes)[2][1] = 2.5f;

    ok &= TryCopyGenomeBytesIntoSlabSlot(buffer, target_slot, source_genome_bytes, buffer.layout.slot_stride_bytes);
    ok &= ExpectTrue(ok, "Expected fixed-width slot copy helper to succeed");
    if (!ok) {
        return false;
    }

    const std::uint8_t *target_genome_bytes = HostSlabSlotBytesAt(buffer, target_slot);
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(target_genome_bytes).dense_trunk.hidden1_to_output.biases[0]),
                     1.25f, "copied dense trunk bias");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(target_genome_bytes).input_encoder.input_to_hidden.weights[0]),
                     -0.75f, "copied encoder weight");
    ok &= ExpectNear(ToFloat(GenomeTailRows(target_genome_bytes)[2][1]), 2.5f, "copied trainable tail value");
    return ok;
}

bool TestGenerationLifecycleClonesAndReleasesSlotOwnership() {
    HostGenotypeSlab buffer{};
    SlabGeneration current_generation{};
    SlabGeneration next_generation{};
    bool ok = TryCreateHostGenotypeSlab(buffer, 3, 4);
    ok &= TryCreateSlabGeneration(current_generation, 1, 2);
    ok &= TryCreateSlabGeneration(next_generation, 1, 3);
    ok &= ExpectTrue(ok, "Expected generation lifecycle fixtures to allocate");
    if (!ok) {
        return false;
    }

    std::uint32_t source_slot = kInvalidSlabSlotIndex;
    ok &= TryAllocateSlabSlot(buffer, source_slot);
    ok &= TrySetSlabGenerationSlot(current_generation, 0, source_slot);
    GenomePolicyModelParameters(HostSlabSlotBytesAt(buffer, source_slot)).dense_trunk.hidden1_to_output.biases[0] =
        4.5f;
    GenomeTailRows(HostSlabSlotBytesAt(buffer, source_slot))[1][0] = -1.25f;

    std::uint32_t cloned_slot = kInvalidSlabSlotIndex;
    ok &= TryCloneSlabSlotIntoGeneration(buffer, next_generation, 0, source_slot, cloned_slot);
    ok &= ExpectTrue(ok, "Expected generation lifecycle clone helper to allocate a child slot");
    ok &= ExpectTrue(cloned_slot != source_slot, "Expected cloned generation slots to use fresh buffer storage");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(buffer, cloned_slot))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     4.5f, "cloned slot dense trunk bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(buffer, cloned_slot))[1][0]), -1.25f,
                     "cloned slot trainable tail value");
    if (!ok) {
        return false;
    }

    ok &= TryReleaseSlabGenerationSlots(buffer, current_generation);
    ok &= ExpectTrue(ok, "Expected releasing a generation to release its slot ownership");
    ok &= ExpectTrue(current_generation.slot_indices[0] == kInvalidSlabSlotIndex,
                     "Expected releasing a generation to clear its slot handles");
    ok &= ExpectTrue(!buffer.slot_states[source_slot].occupied,
                     "Expected released generation slots to return to the buffer");
    ok &= ExpectTrue(buffer.slot_states[cloned_slot].occupied, "Expected cloned child slots to remain live");
    return ok;
}

bool TestSlabGenerationSeparatesSlotHandlesFromFitnessBookkeeping() {
    SlabGeneration generation{};
    bool ok = TryCreateSlabGeneration(generation, 3, 7);
    ok &= ExpectTrue(ok, "Expected slab generation allocation to succeed");
    ok &= ExpectTrue(IsValidSlabGeneration(generation), "Expected allocated slab generation to be valid");
    ok &= ExpectTrue(generation.generation_index == 7, "Expected slab generation to keep its generation index");
    if (!ok) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        ok &= ExpectTrue(generation.slot_indices[individual_index] == kInvalidSlabSlotIndex,
                         "Expected fresh generation slots to start unset");
        ok &= ExpectNear(generation.fitness[individual_index], 0.0f,
                         "Expected fresh generation fitness values to start cleared");
        ok &= ExpectTrue(generation.evaluation_counts[individual_index] == 0,
                         "Expected fresh generation evaluation counts to start cleared");
        ok &= ExpectTrue(generation.has_fitness[individual_index] == 0,
                         "Expected fresh generation fitness flags to start cleared");
    }

    generation.slot_indices[1] = 5;
    generation.fitness[1] = 9.0f;
    generation.evaluation_counts[1] = 3;
    generation.has_fitness[1] = 1;

    ClearSlabGenerationFitness(generation);

    ok &= ExpectTrue(generation.slot_indices[1] == 5,
                     "Expected slot handles to survive a fitness reset because they are separate generation metadata");
    ok &= ExpectNear(generation.fitness[1], 0.0f, "Expected fitness reset to clear stored fitness");
    ok &= ExpectTrue(generation.evaluation_counts[1] == 0, "Expected fitness reset to clear evaluation counts");
    ok &= ExpectTrue(generation.has_fitness[1] == 0, "Expected fitness reset to clear fitness flags");
    return ok;
}

} // namespace

int main() {
    if (!TestSlabLayoutReusesDynamicGenomeStrideMath()) {
        return 1;
    }

    if (!TestRawViewsMutateUnderlyingBufferAndGenerationState()) {
        return 1;
    }

    if (!TestHostSlabAllocatesReleasesAndReusesFixedWidthSlots()) {
        return 1;
    }

    if (!TestReferenceCounterBuildsCollectsAndReleasesParentSlots()) {
        return 1;
    }

    if (!TestSlabSlotAccessorsAndCopyHelperRoundTripGenomeBytes()) {
        return 1;
    }

    if (!TestGenerationLifecycleClonesAndReleasesSlotOwnership()) {
        return 1;
    }

    if (!TestSlabGenerationSeparatesSlotHandlesFromFitnessBookkeeping()) {
        return 1;
    }

    std::cout << "PASS: genotype_slab_test\n";
    return 0;
}
