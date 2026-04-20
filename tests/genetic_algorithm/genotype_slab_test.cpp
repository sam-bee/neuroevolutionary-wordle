#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "genetic_algorithm/genotype_slab/assembly.hpp"
#include "genetic_algorithm/genotype_slab/generation.hpp"
#include "genetic_algorithm/genotype_slab/reference_counter.hpp"
#include "genetic_algorithm/genotype_slab/repacking.hpp"
#include "genetic_algorithm/genotype_slab/slab_allocator.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::common::ToFloat16;
using neuroevolution::genetic_algorithm::genome::ComputeDynamicGenomeStrideBytes;
using neuroevolution::genetic_algorithm::genotype_slab::ClearSlabGenerationFitness;
using neuroevolution::genetic_algorithm::genotype_slab::ComputeOutputEmbeddingGrowthBytes;
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
using neuroevolution::genetic_algorithm::genotype_slab::SlabAssemblyCallbacks;
using neuroevolution::genetic_algorithm::genotype_slab::SlabAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_slab::SlabGeneration;
using neuroevolution::genetic_algorithm::genotype_slab::SlabGenerationView;
using neuroevolution::genetic_algorithm::genotype_slab::SlabSlotCountForByteBudget;
using neuroevolution::genetic_algorithm::genotype_slab::SlabSlotState;
using neuroevolution::genetic_algorithm::genotype_slab::TryAllocateSlabSlot;
using neuroevolution::genetic_algorithm::genotype_slab::TryAssembleNextGeneration;
using neuroevolution::genetic_algorithm::genotype_slab::TryBuildParentReferenceCounts;
using neuroevolution::genetic_algorithm::genotype_slab::TryCloneSlabSlotIntoGeneration;
using neuroevolution::genetic_algorithm::genotype_slab::TryCollectZeroReferenceParents;
using neuroevolution::genetic_algorithm::genotype_slab::TryCompactAndRepackSlabForExpandedActionCount;
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

struct TestAssemblyState {
    std::uint32_t assembled_child_count = 0;
};

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

bool AssembleSummedChildGenome(const std::uint8_t *first_parent_genome_bytes,
                               const std::uint8_t *second_parent_genome_bytes, const std::size_t action_count,
                               std::uint8_t *child_genome_bytes, void *user_data) {
    (void)action_count;
    auto *assembly_state = static_cast<TestAssemblyState *>(user_data);
    if ((first_parent_genome_bytes == nullptr) || (second_parent_genome_bytes == nullptr) ||
        (child_genome_bytes == nullptr) || (assembly_state == nullptr)) {
        return false;
    }

    const float first_bias =
        ToFloat(GenomePolicyModelParameters(first_parent_genome_bytes).dense_trunk.hidden1_to_output.biases[0]);
    const float second_bias =
        ToFloat(GenomePolicyModelParameters(second_parent_genome_bytes).dense_trunk.hidden1_to_output.biases[0]);

    GenomePolicyModelParameters(child_genome_bytes).dense_trunk.hidden1_to_output.biases[0] =
        ToFloat16(first_bias + second_bias);
    GenomeTailRows(child_genome_bytes)[0][0] = ToFloat16(static_cast<float>(assembly_state->assembled_child_count + 1));
    ++assembly_state->assembled_child_count;
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
    ok &= ExpectTrue(ComputeOutputEmbeddingGrowthBytes(50) == 3800,
                     "Expected a fifty-word shard increment to add 3800 trainable output-embedding bytes");
    return ok;
}

bool TestHostSlabCompactsAndRepacksForExpandedActionCount() {
    constexpr std::size_t kInitialActionCount = 4;
    constexpr std::size_t kExpandedActionCount = 8;

    HostGenotypeSlab buffer{};
    SlabGeneration current_generation{};
    bool ok = TryCreateHostGenotypeSlab(buffer, 6, kInitialActionCount);
    ok &= TryCreateSlabGeneration(current_generation, 3, 9);
    ok &= ExpectTrue(ok, "Expected repack fixtures to allocate");
    if (!ok) {
        return false;
    }

    std::uint32_t slots[6]{};
    for (std::size_t slot_offset = 0; slot_offset < 6; ++slot_offset) {
        ok &= TryAllocateSlabSlot(buffer, slots[slot_offset]);
    }
    ok &= ExpectTrue(ok, "Expected repack fixtures to allocate six deterministic slab slots");
    ok &= TrySetSlabGenerationSlot(current_generation, 0, slots[5]);
    ok &= TrySetSlabGenerationSlot(current_generation, 1, slots[1]);
    ok &= TrySetSlabGenerationSlot(current_generation, 2, slots[4]);
    ok &= TryReleaseSlabSlot(buffer, slots[0]);
    ok &= TryReleaseSlabSlot(buffer, slots[2]);
    ok &= TryReleaseSlabSlot(buffer, slots[3]);
    if (!ok) {
        return false;
    }

    GenomePolicyModelParameters(HostSlabSlotBytesAt(buffer, slots[5])).dense_trunk.hidden1_to_output.biases[0] = 10.0f;
    GenomeTailRows(HostSlabSlotBytesAt(buffer, slots[5]))[1][0] = 1.5f;
    GenomePolicyModelParameters(HostSlabSlotBytesAt(buffer, slots[4])).dense_trunk.hidden1_to_output.biases[0] = 20.0f;
    GenomeTailRows(HostSlabSlotBytesAt(buffer, slots[4]))[3][0] = 2.5f;

    std::uint32_t parent_reference_counts[3]{1U, 0U, 2U};
    const std::size_t original_slab_bytes = buffer.layout.slab_bytes;

    ok &= TryCompactAndRepackSlabForExpandedActionCount(buffer, current_generation, parent_reference_counts,
                                                        kExpandedActionCount);
    ok &= ExpectTrue(ok, "Expected repacking to preserve live parents while expanding slot size");
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(buffer.layout.action_count == kExpandedActionCount,
                     "Expected repacking to switch the slab to the expanded action count");
    ok &= ExpectTrue(buffer.layout.slab_bytes == original_slab_bytes,
                     "Expected repacking to keep the total slab byte budget fixed");
    ok &= ExpectTrue(buffer.layout.slot_count == SlabSlotCountForByteBudget(original_slab_bytes, kExpandedActionCount),
                     "Expected repacking to recompute the slot count for the expanded slot size");
    ok &= ExpectTrue(current_generation.slot_indices[0] == 4U,
                     "Expected the highest-address survivor to land in the right-most expanded slot");
    ok &= ExpectTrue(current_generation.slot_indices[1] == kInvalidSlabSlotIndex,
                     "Expected zero-reference parents to be collected during repacking");
    ok &= ExpectTrue(current_generation.slot_indices[2] == 3U,
                     "Expected survivors to preserve source-slot order during the left compaction step");
    ok &= ExpectTrue(buffer.free_slot_count == 3U,
                     "Expected repacking to leave the lower expanded slots free for child assembly");
    ok &= ExpectTrue(buffer.slot_states[3].occupied && buffer.slot_states[4].occupied,
                     "Expected repacking to mark only the survivor slots as occupied");

    ok &= ExpectNear(
        ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(buffer, 4)).dense_trunk.hidden1_to_output.biases[0]),
        10.0f, "repacked parent 0 bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(buffer, 4))[1][0]), 1.5f,
                     "repacked parent 0 preserved tail");
    ok &= ExpectNear(
        ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(buffer, 3)).dense_trunk.hidden1_to_output.biases[0]),
        20.0f, "repacked parent 2 bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(buffer, 3))[3][0]), 2.5f,
                     "repacked parent 2 preserved tail");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(buffer, 4))[kInitialActionCount][0]), 0.0f,
                     "expected newly appended trainable tails to start zeroed");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(buffer, 3))[kExpandedActionCount - 1][0]), 0.0f,
                     "expected all appended trainable tails to be cleared before injection");
    return ok;
}

bool TestHostSlabCompactingLeavesInRangeSurvivorsInPlace() {
    constexpr std::size_t kInitialActionCount = 4;
    constexpr std::size_t kExpandedActionCount = 8;

    HostGenotypeSlab buffer{};
    SlabGeneration current_generation{};
    bool ok = TryCreateHostGenotypeSlab(buffer, 6, kInitialActionCount);
    ok &= TryCreateSlabGeneration(current_generation, 3, 10);
    ok &= ExpectTrue(ok, "Expected compacting-range fixtures to allocate");
    if (!ok) {
        return false;
    }

    std::uint32_t slots[6]{};
    for (std::size_t slot_offset = 0; slot_offset < 6; ++slot_offset) {
        ok &= TryAllocateSlabSlot(buffer, slots[slot_offset]);
    }
    ok &= ExpectTrue(ok, "Expected compacting-range fixtures to allocate six deterministic slab slots");
    ok &= TrySetSlabGenerationSlot(current_generation, 0, slots[1]);
    ok &= TrySetSlabGenerationSlot(current_generation, 1, slots[0]);
    ok &= TrySetSlabGenerationSlot(current_generation, 2, slots[5]);
    ok &= TryReleaseSlabSlot(buffer, slots[2]);
    ok &= TryReleaseSlabSlot(buffer, slots[3]);
    ok &= TryReleaseSlabSlot(buffer, slots[4]);
    if (!ok) {
        return false;
    }

    GenomePolicyModelParameters(HostSlabSlotBytesAt(buffer, slots[1])).dense_trunk.hidden1_to_output.biases[0] = 10.0f;
    GenomeTailRows(HostSlabSlotBytesAt(buffer, slots[1]))[1][0] = 1.5f;
    GenomePolicyModelParameters(HostSlabSlotBytesAt(buffer, slots[5])).dense_trunk.hidden1_to_output.biases[0] = 20.0f;
    GenomeTailRows(HostSlabSlotBytesAt(buffer, slots[5]))[3][0] = 2.5f;

    std::uint32_t parent_reference_counts[3]{1U, 0U, 1U};
    ok &= TryCompactAndRepackSlabForExpandedActionCount(buffer, current_generation, parent_reference_counts,
                                                        kExpandedActionCount);
    ok &= ExpectTrue(ok, "Expected repacking to preserve live parents while widening from compacted range");
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(current_generation.slot_indices[0] == 4U,
                     "Expected in-range survivor to keep the higher compacted slot when another survivor fills the hole");
    ok &= ExpectTrue(current_generation.slot_indices[1] == kInvalidSlabSlotIndex,
                     "Expected zero-reference parent to be collected during widening");
    ok &= ExpectTrue(current_generation.slot_indices[2] == 3U,
                     "Expected out-of-range survivor to fill the compacted prefix hole");
    ok &= ExpectNear(
        ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(buffer, 4)).dense_trunk.hidden1_to_output.biases[0]),
        10.0f, "kept-in-place parent bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(buffer, 4))[1][0]), 1.5f,
                     "kept-in-place parent preserved tail");
    ok &= ExpectNear(
        ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(buffer, 3)).dense_trunk.hidden1_to_output.biases[0]),
        20.0f, "moved-into-hole parent bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(buffer, 3))[3][0]), 2.5f,
                     "moved-into-hole parent preserved tail");
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

bool TestAssemblyWithReferenceCountingGcReusesParentSlots() {
    HostGenotypeSlab buffer{};
    SlabGeneration current_generation{};
    SlabGeneration next_generation{};
    bool ok = TryCreateHostGenotypeSlab(buffer, 3, 4);
    ok &= TryCreateSlabGeneration(current_generation, 3, 5);
    ok &= ExpectTrue(ok, "Expected assembly test fixtures to allocate");
    if (!ok) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = kInvalidSlabSlotIndex;
        ok &= TryAllocateSlabSlot(buffer, slot_index);
        ok &= TrySetSlabGenerationSlot(current_generation, individual_index, slot_index);
        GenomePolicyModelParameters(HostSlabSlotBytesAt(buffer, slot_index)).dense_trunk.hidden1_to_output.biases[0] =
            ToFloat16(10.0f * static_cast<float>(individual_index + 1));
    }
    ok &= ExpectTrue(ok, "Expected assembly test parents to occupy all slab slots");
    if (!ok) {
        return false;
    }

    SlabAssemblyPlan plan{};
    ok &= TryCreateSlabAssemblyPlan(plan, 2);
    ok &= ExpectTrue(ok, "Expected assembly plan allocation to succeed");
    if (!ok) {
        return false;
    }

    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};
    plan.parent_pairs[1] = {.first_parent_index = 0, .second_parent_index = 0};

    TestAssemblyState assembly_state{};
    SlabAssemblyCallbacks callbacks{};
    callbacks.assemble_child_genome = AssembleSummedChildGenome;
    callbacks.user_data = &assembly_state;

    ok &= TryAssembleNextGeneration(buffer, current_generation, plan, next_generation, callbacks);
    ok &= ExpectTrue(ok, "Expected slab assembly to succeed by collecting and reusing parent slots");
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(next_generation.generation_index == 6, "Expected slab assembly to increment the generation index");
    ok &= ExpectTrue(next_generation.slot_indices[0] == 2U,
                     "Expected the first child to reuse the garbage-collected zero-reference parent slot");
    ok &= ExpectTrue(next_generation.slot_indices[1] == 1U,
                     "Expected the second child to reuse the parent slot freed after its final parent reference");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(buffer, next_generation.slot_indices[0]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     30.0f, "first assembled child bias");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(buffer, next_generation.slot_indices[1]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     20.0f, "second assembled child bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(buffer, next_generation.slot_indices[0]))[0][0]), 1.0f,
                     "first assembled child marker");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(buffer, next_generation.slot_indices[1]))[0][0]), 2.0f,
                     "second assembled child marker");
    ok &= ExpectTrue(current_generation.slot_indices[0] == kInvalidSlabSlotIndex,
                     "Expected the first parent to be released after its final parent reference");
    ok &= ExpectTrue(current_generation.slot_indices[1] == kInvalidSlabSlotIndex,
                     "Expected the second parent to be released after its only parent reference");
    ok &= ExpectTrue(current_generation.slot_indices[2] == kInvalidSlabSlotIndex,
                     "Expected zero-reference parents to be garbage-collected before child assembly begins");
    ok &= ExpectTrue(buffer.free_slot_count == 1,
                     "Expected one parent slot to be free after two children occupy the reused slots");
    return ok;
}

bool TestAssemblyFailsWhenBufferIsGenuinelyFull() {
    HostGenotypeSlab buffer{};
    SlabGeneration current_generation{};
    SlabGeneration next_generation{};
    bool ok = TryCreateHostGenotypeSlab(buffer, 2, 4);
    ok &= TryCreateSlabGeneration(current_generation, 2, 4);
    ok &= ExpectTrue(ok, "Expected full-slab failure fixtures to allocate");
    if (!ok) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = kInvalidSlabSlotIndex;
        ok &= TryAllocateSlabSlot(buffer, slot_index);
        ok &= TrySetSlabGenerationSlot(current_generation, individual_index, slot_index);
    }
    ok &= ExpectTrue(ok, "Expected full-slab failure fixtures to consume every slot");
    if (!ok) {
        return false;
    }

    SlabAssemblyPlan plan{};
    ok &= TryCreateSlabAssemblyPlan(plan, 1);
    ok &= ExpectTrue(ok, "Expected single-child assembly plan allocation to succeed");
    if (!ok) {
        return false;
    }

    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};

    TestAssemblyState assembly_state{};
    SlabAssemblyCallbacks callbacks{};
    callbacks.assemble_child_genome = AssembleSummedChildGenome;
    callbacks.user_data = &assembly_state;

    ok &= ExpectTrue(!TryAssembleNextGeneration(buffer, current_generation, plan, next_generation, callbacks),
                     "Expected assembly to fail when the buffer is full and every parent still has parent references");
    ok &= ExpectTrue(!IsValidSlabGeneration(next_generation),
                     "Expected failed assembly to leave the next generation output unset");
    ok &= ExpectTrue(current_generation.slot_indices[0] == 0U,
                     "Expected failed full-slab assembly to leave the first parent slot intact");
    ok &= ExpectTrue(current_generation.slot_indices[1] == 1U,
                     "Expected failed full-slab assembly to leave the second parent slot intact");
    ok &= ExpectTrue(buffer.free_slot_count == 0, "Expected failed full-slab assembly to leave the slab full");
    return ok;
}

} // namespace

int main() {
    if (!TestSlabLayoutReusesDynamicGenomeStrideMath()) {
        return 1;
    }

    if (!TestHostSlabCompactsAndRepacksForExpandedActionCount()) {
        return 1;
    }

    if (!TestHostSlabCompactingLeavesInRangeSurvivorsInPlace()) {
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

    if (!TestAssemblyWithReferenceCountingGcReusesParentSlots()) {
        return 1;
    }

    if (!TestAssemblyFailsWhenBufferIsGenuinelyFull()) {
        return 1;
    }

    std::cout << "PASS: genotype_slab_test\n";
    return 0;
}
