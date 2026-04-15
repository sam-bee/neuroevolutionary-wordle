#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "genetic_algorithm/genotype_pool/assembly.hpp"
#include "genetic_algorithm/genotype_pool/generation.hpp"
#include "genetic_algorithm/genotype_pool/pool.hpp"
#include "genetic_algorithm/genotype_pool/repacking.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::common::ToFloat16;
using neuroevolution::genetic_algorithm::genome::ComputeDynamicGenomeStrideBytes;
using neuroevolution::genetic_algorithm::genotype_pool::ClearPoolGenerationFitness;
using neuroevolution::genetic_algorithm::genotype_pool::ComputeOutputEmbeddingGrowthBytes;
using neuroevolution::genetic_algorithm::genotype_pool::ComputePoolSlotStrideBytes;
using neuroevolution::genetic_algorithm::genotype_pool::GenomePolicyModelParameters;
using neuroevolution::genetic_algorithm::genotype_pool::GenomeTailRows;
using neuroevolution::genetic_algorithm::genotype_pool::GenotypePoolLayout;
using neuroevolution::genetic_algorithm::genotype_pool::GenotypePoolView;
using neuroevolution::genetic_algorithm::genotype_pool::HostGenotypePool;
using neuroevolution::genetic_algorithm::genotype_pool::HostPoolSlotBytesAt;
using neuroevolution::genetic_algorithm::genotype_pool::IsValidGenotypePoolLayout;
using neuroevolution::genetic_algorithm::genotype_pool::IsValidPoolGeneration;
using neuroevolution::genetic_algorithm::genotype_pool::kInvalidPoolSlotIndex;
using neuroevolution::genetic_algorithm::genotype_pool::MakeGenotypePoolView;
using neuroevolution::genetic_algorithm::genotype_pool::MakePoolGenerationView;
using neuroevolution::genetic_algorithm::genotype_pool::PoolAssemblyCallbacks;
using neuroevolution::genetic_algorithm::genotype_pool::PoolAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_pool::PoolGeneration;
using neuroevolution::genetic_algorithm::genotype_pool::PoolGenerationView;
using neuroevolution::genetic_algorithm::genotype_pool::PoolSlotCountForByteBudget;
using neuroevolution::genetic_algorithm::genotype_pool::TryAllocatePoolSlot;
using neuroevolution::genetic_algorithm::genotype_pool::TryAssembleNextGeneration;
using neuroevolution::genetic_algorithm::genotype_pool::TryClonePoolSlotIntoGeneration;
using neuroevolution::genetic_algorithm::genotype_pool::TryCompactAndRepackPoolForExpandedActionCount;
using neuroevolution::genetic_algorithm::genotype_pool::TryCopyGenomeBytesIntoPoolSlot;
using neuroevolution::genetic_algorithm::genotype_pool::TryCreateHostGenotypePool;
using neuroevolution::genetic_algorithm::genotype_pool::TryCreatePoolAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_pool::TryCreatePoolGeneration;
using neuroevolution::genetic_algorithm::genotype_pool::TryReleasePoolGenerationSlots;
using neuroevolution::genetic_algorithm::genotype_pool::TryReleasePoolSlot;
using neuroevolution::genetic_algorithm::genotype_pool::TryRetainPoolSlot;
using neuroevolution::genetic_algorithm::genotype_pool::TrySetPoolGenerationSlot;

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

bool TestPoolLayoutReusesDynamicGenomeStrideMath() {
    constexpr std::size_t kActionCount = 20;
    constexpr std::size_t kSlotCount = 6;

    GenotypePoolLayout layout{};
    layout.action_count = kActionCount;
    layout.slot_stride_bytes = ComputePoolSlotStrideBytes(kActionCount);
    layout.slot_count = kSlotCount;
    layout.pool_bytes = kSlotCount * layout.slot_stride_bytes;

    bool ok = true;
    ok &= ExpectTrue(layout.slot_stride_bytes == ComputeDynamicGenomeStrideBytes(kActionCount),
                     "Expected pool slot stride to match dynamic genome stride math");
    ok &= ExpectTrue(PoolSlotCountForByteBudget(layout.pool_bytes, kActionCount) == kSlotCount,
                     "Expected pool byte budget helper to recover the slot count");
    ok &= ExpectTrue(IsValidGenotypePoolLayout(layout), "Expected constructed pool layout to be valid");
    ok &= ExpectTrue(ComputeOutputEmbeddingGrowthBytes(50) == 3800,
                     "Expected a fifty-word shard increment to add 3800 trainable output-embedding bytes");
    return ok;
}

bool TestHostPoolCompactsAndRepacksForExpandedActionCount() {
    constexpr std::size_t kInitialActionCount = 4;
    constexpr std::size_t kExpandedActionCount = 8;

    HostGenotypePool pool{};
    PoolGeneration current_generation{};
    bool ok = TryCreateHostGenotypePool(pool, 6, kInitialActionCount);
    ok &= TryCreatePoolGeneration(current_generation, 3, 9);
    ok &= ExpectTrue(ok, "Expected repack fixtures to allocate");
    if (!ok) {
        return false;
    }

    std::uint32_t slots[6]{};
    for (std::size_t slot_offset = 0; slot_offset < 6; ++slot_offset) {
        ok &= TryAllocatePoolSlot(pool, slots[slot_offset]);
    }
    ok &= ExpectTrue(ok, "Expected repack fixtures to allocate six deterministic pool slots");
    ok &= TrySetPoolGenerationSlot(current_generation, 0, slots[5]);
    ok &= TrySetPoolGenerationSlot(current_generation, 1, slots[1]);
    ok &= TrySetPoolGenerationSlot(current_generation, 2, slots[4]);
    ok &= TryReleasePoolSlot(pool, slots[0]);
    ok &= TryReleasePoolSlot(pool, slots[2]);
    ok &= TryReleasePoolSlot(pool, slots[3]);
    if (!ok) {
        return false;
    }

    GenomePolicyModelParameters(HostPoolSlotBytesAt(pool, slots[5])).dense_trunk.hidden1_to_output.biases[0] = 10.0f;
    GenomeTailRows(HostPoolSlotBytesAt(pool, slots[5]))[1][0] = 1.5f;
    GenomePolicyModelParameters(HostPoolSlotBytesAt(pool, slots[4])).dense_trunk.hidden1_to_output.biases[0] = 20.0f;
    GenomeTailRows(HostPoolSlotBytesAt(pool, slots[4]))[3][0] = 2.5f;

    std::uint32_t parent_reference_counts[3]{1U, 0U, 2U};
    const std::size_t original_pool_bytes = pool.layout.pool_bytes;

    ok &= TryCompactAndRepackPoolForExpandedActionCount(pool, current_generation, parent_reference_counts,
                                                        kExpandedActionCount);
    ok &= ExpectTrue(ok, "Expected repacking to preserve live parents while expanding slot size");
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(pool.layout.action_count == kExpandedActionCount,
                     "Expected repacking to switch the pool to the expanded action count");
    ok &= ExpectTrue(pool.layout.pool_bytes == original_pool_bytes,
                     "Expected repacking to keep the total pool byte budget fixed");
    ok &= ExpectTrue(pool.layout.slot_count == PoolSlotCountForByteBudget(original_pool_bytes, kExpandedActionCount),
                     "Expected repacking to recompute the slot count for the expanded slot size");
    ok &= ExpectTrue(current_generation.slot_indices[0] == 4U,
                     "Expected the highest-address survivor to land in the right-most expanded slot");
    ok &= ExpectTrue(current_generation.slot_indices[1] == kInvalidPoolSlotIndex,
                     "Expected zero-reference parents to be collected during repacking");
    ok &= ExpectTrue(current_generation.slot_indices[2] == 3U,
                     "Expected survivors to preserve source-slot order during the left compaction step");
    ok &= ExpectTrue(pool.free_slot_count == 3U,
                     "Expected repacking to leave the lower expanded slots free for child assembly");
    ok &= ExpectTrue(pool.slot_states[3].occupied && pool.slot_states[4].occupied,
                     "Expected repacking to mark only the survivor slots as occupied");

    ok &= ExpectNear(
        ToFloat(GenomePolicyModelParameters(HostPoolSlotBytesAt(pool, 4)).dense_trunk.hidden1_to_output.biases[0]),
        10.0f, "repacked parent 0 bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostPoolSlotBytesAt(pool, 4))[1][0]), 1.5f,
                     "repacked parent 0 preserved tail");
    ok &= ExpectNear(
        ToFloat(GenomePolicyModelParameters(HostPoolSlotBytesAt(pool, 3)).dense_trunk.hidden1_to_output.biases[0]),
        20.0f, "repacked parent 2 bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostPoolSlotBytesAt(pool, 3))[3][0]), 2.5f,
                     "repacked parent 2 preserved tail");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostPoolSlotBytesAt(pool, 4))[kInitialActionCount][0]), 0.0f,
                     "expected newly appended trainable tails to start zeroed");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostPoolSlotBytesAt(pool, 3))[kExpandedActionCount - 1][0]), 0.0f,
                     "expected all appended trainable tails to be cleared before injection");
    return ok;
}

bool TestRawViewsMutateUnderlyingPoolAndGenerationState() {
    HostGenotypePool pool{};
    PoolGeneration generation{};
    bool ok = TryCreateHostGenotypePool(pool, 2, 4);
    ok &= TryCreatePoolGeneration(generation, 2, 3);
    ok &= ExpectTrue(ok, "Expected raw-view test fixtures to allocate");
    if (!ok) {
        return false;
    }

    GenotypePoolView pool_view = MakeGenotypePoolView(pool);
    PoolGenerationView generation_view = MakePoolGenerationView(generation);
    std::uint32_t slot_index = kInvalidPoolSlotIndex;

    ok &= TryAllocatePoolSlot(pool_view, slot_index);
    ok &= TrySetPoolGenerationSlot(generation_view, 0, slot_index);
    ok &= ExpectTrue(ok, "Expected view-based slot allocation and assignment to succeed");
    ok &= ExpectTrue(pool.free_slot_count == 1, "Expected the host owner to observe the view-based allocation");
    ok &= ExpectTrue(generation.slot_indices[0] == slot_index,
                     "Expected the host generation to observe the view-based slot assignment");
    return ok;
}

bool TestHostPoolAllocatesReleasesAndReusesFixedWidthSlots() {
    HostGenotypePool pool{};
    bool ok = TryCreateHostGenotypePool(pool, 3, 4);
    ok &= ExpectTrue(ok, "Expected host genotype pool creation to succeed");
    if (!ok) {
        return false;
    }

    std::uint32_t slot0 = kInvalidPoolSlotIndex;
    std::uint32_t slot1 = kInvalidPoolSlotIndex;
    std::uint32_t slot2 = kInvalidPoolSlotIndex;
    std::uint32_t overflow_slot = kInvalidPoolSlotIndex;

    ok &= TryAllocatePoolSlot(pool, slot0);
    ok &= TryAllocatePoolSlot(pool, slot1);
    ok &= TryAllocatePoolSlot(pool, slot2);
    ok &= ExpectTrue(ok, "Expected every pool slot to allocate exactly once");
    ok &= ExpectTrue((slot0 == 0U) && (slot1 == 1U) && (slot2 == 2U),
                     "Expected the free-slot stack to hand out deterministic slot indices");
    ok &= ExpectTrue(!TryAllocatePoolSlot(pool, overflow_slot),
                     "Expected allocation to fail once the fixed-width pool is full");
    ok &= ExpectTrue(pool.free_slot_count == 0, "Expected a full pool to report no free slots");

    ok &= TryRetainPoolSlot(pool, slot1);
    ok &= ExpectTrue(pool.slot_states[slot1].reference_count == 2,
                     "Expected retaining a slot to increment its reference count");
    ok &= TryReleasePoolSlot(pool, slot1);
    ok &= ExpectTrue(pool.slot_states[slot1].occupied, "Expected the slot to stay occupied while references remain");
    ok &= ExpectTrue(pool.slot_states[slot1].reference_count == 1,
                     "Expected releasing one reference to leave the slot live");
    ok &= TryReleasePoolSlot(pool, slot1);
    ok &= ExpectTrue(!pool.slot_states[slot1].occupied,
                     "Expected the slot to become free after the last reference is released");
    ok &= ExpectTrue(pool.free_slot_count == 1, "Expected releasing the last reference to return the slot to the pool");

    std::uint32_t reused_slot = kInvalidPoolSlotIndex;
    ok &= TryAllocatePoolSlot(pool, reused_slot);
    ok &= ExpectTrue(reused_slot == slot1, "Expected the most recently released slot to be reused first");
    return ok;
}

bool TestPoolSlotAccessorsAndCopyHelperRoundTripGenomeBytes() {
    HostGenotypePool pool{};
    bool ok = TryCreateHostGenotypePool(pool, 2, 4);
    ok &= ExpectTrue(ok, "Expected host genotype pool creation for copy test to succeed");
    if (!ok) {
        return false;
    }

    std::uint32_t source_slot = kInvalidPoolSlotIndex;
    std::uint32_t target_slot = kInvalidPoolSlotIndex;
    ok &= TryAllocatePoolSlot(pool, source_slot);
    ok &= TryAllocatePoolSlot(pool, target_slot);
    ok &= ExpectTrue(ok, "Expected copy test slots to allocate");
    if (!ok) {
        return false;
    }

    std::uint8_t *source_genome_bytes = HostPoolSlotBytesAt(pool, source_slot);
    GenomePolicyModelParameters(source_genome_bytes).dense_trunk.hidden1_to_output.biases[0] = 1.25f;
    GenomePolicyModelParameters(source_genome_bytes).input_encoder.input_to_hidden.weights[0] = -0.75f;
    GenomeTailRows(source_genome_bytes)[2][1] = 2.5f;

    ok &= TryCopyGenomeBytesIntoPoolSlot(pool, target_slot, source_genome_bytes, pool.layout.slot_stride_bytes);
    ok &= ExpectTrue(ok, "Expected fixed-width slot copy helper to succeed");
    if (!ok) {
        return false;
    }

    const std::uint8_t *target_genome_bytes = HostPoolSlotBytesAt(pool, target_slot);
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(target_genome_bytes).dense_trunk.hidden1_to_output.biases[0]),
                     1.25f, "copied dense trunk bias");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(target_genome_bytes).input_encoder.input_to_hidden.weights[0]),
                     -0.75f, "copied encoder weight");
    ok &= ExpectNear(ToFloat(GenomeTailRows(target_genome_bytes)[2][1]), 2.5f, "copied trainable tail value");
    return ok;
}

bool TestGenerationLifecycleClonesAndReleasesSlotOwnership() {
    HostGenotypePool pool{};
    PoolGeneration current_generation{};
    PoolGeneration next_generation{};
    bool ok = TryCreateHostGenotypePool(pool, 3, 4);
    ok &= TryCreatePoolGeneration(current_generation, 1, 2);
    ok &= TryCreatePoolGeneration(next_generation, 1, 3);
    ok &= ExpectTrue(ok, "Expected generation lifecycle fixtures to allocate");
    if (!ok) {
        return false;
    }

    std::uint32_t source_slot = kInvalidPoolSlotIndex;
    ok &= TryAllocatePoolSlot(pool, source_slot);
    ok &= TrySetPoolGenerationSlot(current_generation, 0, source_slot);
    GenomePolicyModelParameters(HostPoolSlotBytesAt(pool, source_slot)).dense_trunk.hidden1_to_output.biases[0] = 4.5f;
    GenomeTailRows(HostPoolSlotBytesAt(pool, source_slot))[1][0] = -1.25f;

    std::uint32_t cloned_slot = kInvalidPoolSlotIndex;
    ok &= TryClonePoolSlotIntoGeneration(pool, next_generation, 0, source_slot, cloned_slot);
    ok &= ExpectTrue(ok, "Expected generation lifecycle clone helper to allocate a child slot");
    ok &= ExpectTrue(cloned_slot != source_slot, "Expected cloned generation slots to use fresh pool storage");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostPoolSlotBytesAt(pool, cloned_slot))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     4.5f, "cloned slot dense trunk bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostPoolSlotBytesAt(pool, cloned_slot))[1][0]), -1.25f,
                     "cloned slot trainable tail value");
    if (!ok) {
        return false;
    }

    ok &= TryReleasePoolGenerationSlots(pool, current_generation);
    ok &= ExpectTrue(ok, "Expected releasing a generation to release its slot ownership");
    ok &= ExpectTrue(current_generation.slot_indices[0] == kInvalidPoolSlotIndex,
                     "Expected releasing a generation to clear its slot handles");
    ok &=
        ExpectTrue(!pool.slot_states[source_slot].occupied, "Expected released generation slots to return to the pool");
    ok &= ExpectTrue(pool.slot_states[cloned_slot].occupied, "Expected cloned child slots to remain live");
    return ok;
}

bool TestPoolGenerationSeparatesSlotHandlesFromFitnessBookkeeping() {
    PoolGeneration generation{};
    bool ok = TryCreatePoolGeneration(generation, 3, 7);
    ok &= ExpectTrue(ok, "Expected pool generation allocation to succeed");
    ok &= ExpectTrue(IsValidPoolGeneration(generation), "Expected allocated pool generation to be valid");
    ok &= ExpectTrue(generation.generation_index == 7, "Expected pool generation to keep its generation index");
    if (!ok) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        ok &= ExpectTrue(generation.slot_indices[individual_index] == kInvalidPoolSlotIndex,
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

    ClearPoolGenerationFitness(generation);

    ok &= ExpectTrue(generation.slot_indices[1] == 5,
                     "Expected slot handles to survive a fitness reset because they are separate generation metadata");
    ok &= ExpectNear(generation.fitness[1], 0.0f, "Expected fitness reset to clear stored fitness");
    ok &= ExpectTrue(generation.evaluation_counts[1] == 0, "Expected fitness reset to clear evaluation counts");
    ok &= ExpectTrue(generation.has_fitness[1] == 0, "Expected fitness reset to clear fitness flags");
    return ok;
}

bool TestAssemblyWithReferenceCountingGcReusesParentSlots() {
    HostGenotypePool pool{};
    PoolGeneration current_generation{};
    PoolGeneration next_generation{};
    bool ok = TryCreateHostGenotypePool(pool, 3, 4);
    ok &= TryCreatePoolGeneration(current_generation, 3, 5);
    ok &= ExpectTrue(ok, "Expected assembly test fixtures to allocate");
    if (!ok) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = kInvalidPoolSlotIndex;
        ok &= TryAllocatePoolSlot(pool, slot_index);
        ok &= TrySetPoolGenerationSlot(current_generation, individual_index, slot_index);
        GenomePolicyModelParameters(HostPoolSlotBytesAt(pool, slot_index)).dense_trunk.hidden1_to_output.biases[0] =
            ToFloat16(10.0f * static_cast<float>(individual_index + 1));
    }
    ok &= ExpectTrue(ok, "Expected assembly test parents to occupy all pool slots");
    if (!ok) {
        return false;
    }

    PoolAssemblyPlan plan{};
    ok &= TryCreatePoolAssemblyPlan(plan, 2);
    ok &= ExpectTrue(ok, "Expected assembly plan allocation to succeed");
    if (!ok) {
        return false;
    }

    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};
    plan.parent_pairs[1] = {.first_parent_index = 0, .second_parent_index = 0};

    TestAssemblyState assembly_state{};
    PoolAssemblyCallbacks callbacks{};
    callbacks.assemble_child_genome = AssembleSummedChildGenome;
    callbacks.user_data = &assembly_state;

    ok &= TryAssembleNextGeneration(pool, current_generation, plan, next_generation, callbacks);
    ok &= ExpectTrue(ok, "Expected pool assembly to succeed by collecting and reusing parent slots");
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(next_generation.generation_index == 6, "Expected pool assembly to increment the generation index");
    ok &= ExpectTrue(next_generation.slot_indices[0] == 2U,
                     "Expected the first child to reuse the garbage-collected zero-reference parent slot");
    ok &= ExpectTrue(next_generation.slot_indices[1] == 1U,
                     "Expected the second child to reuse the parent slot freed after its final parent reference");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostPoolSlotBytesAt(pool, next_generation.slot_indices[0]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     30.0f, "first assembled child bias");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostPoolSlotBytesAt(pool, next_generation.slot_indices[1]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     20.0f, "second assembled child bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostPoolSlotBytesAt(pool, next_generation.slot_indices[0]))[0][0]), 1.0f,
                     "first assembled child marker");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostPoolSlotBytesAt(pool, next_generation.slot_indices[1]))[0][0]), 2.0f,
                     "second assembled child marker");
    ok &= ExpectTrue(current_generation.slot_indices[0] == kInvalidPoolSlotIndex,
                     "Expected the first parent to be released after its final parent reference");
    ok &= ExpectTrue(current_generation.slot_indices[1] == kInvalidPoolSlotIndex,
                     "Expected the second parent to be released after its only parent reference");
    ok &= ExpectTrue(current_generation.slot_indices[2] == kInvalidPoolSlotIndex,
                     "Expected zero-reference parents to be garbage-collected before child assembly begins");
    ok &= ExpectTrue(pool.free_slot_count == 1,
                     "Expected one parent slot to be free after two children occupy the reused slots");
    return ok;
}

bool TestAssemblyFailsWhenPoolIsGenuinelyFull() {
    HostGenotypePool pool{};
    PoolGeneration current_generation{};
    PoolGeneration next_generation{};
    bool ok = TryCreateHostGenotypePool(pool, 2, 4);
    ok &= TryCreatePoolGeneration(current_generation, 2, 4);
    ok &= ExpectTrue(ok, "Expected full-pool failure fixtures to allocate");
    if (!ok) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = kInvalidPoolSlotIndex;
        ok &= TryAllocatePoolSlot(pool, slot_index);
        ok &= TrySetPoolGenerationSlot(current_generation, individual_index, slot_index);
    }
    ok &= ExpectTrue(ok, "Expected full-pool failure fixtures to consume every slot");
    if (!ok) {
        return false;
    }

    PoolAssemblyPlan plan{};
    ok &= TryCreatePoolAssemblyPlan(plan, 1);
    ok &= ExpectTrue(ok, "Expected single-child assembly plan allocation to succeed");
    if (!ok) {
        return false;
    }

    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};

    TestAssemblyState assembly_state{};
    PoolAssemblyCallbacks callbacks{};
    callbacks.assemble_child_genome = AssembleSummedChildGenome;
    callbacks.user_data = &assembly_state;

    ok &= ExpectTrue(!TryAssembleNextGeneration(pool, current_generation, plan, next_generation, callbacks),
                     "Expected assembly to fail when the pool is full and every parent still has parent references");
    ok &= ExpectTrue(!IsValidPoolGeneration(next_generation),
                     "Expected failed assembly to leave the next generation output unset");
    ok &= ExpectTrue(current_generation.slot_indices[0] == 0U,
                     "Expected failed full-pool assembly to leave the first parent slot intact");
    ok &= ExpectTrue(current_generation.slot_indices[1] == 1U,
                     "Expected failed full-pool assembly to leave the second parent slot intact");
    ok &= ExpectTrue(pool.free_slot_count == 0, "Expected failed full-pool assembly to leave the pool full");
    return ok;
}

} // namespace

int main() {
    if (!TestPoolLayoutReusesDynamicGenomeStrideMath()) {
        return 1;
    }

    if (!TestHostPoolCompactsAndRepacksForExpandedActionCount()) {
        return 1;
    }

    if (!TestRawViewsMutateUnderlyingPoolAndGenerationState()) {
        return 1;
    }

    if (!TestHostPoolAllocatesReleasesAndReusesFixedWidthSlots()) {
        return 1;
    }

    if (!TestPoolSlotAccessorsAndCopyHelperRoundTripGenomeBytes()) {
        return 1;
    }

    if (!TestGenerationLifecycleClonesAndReleasesSlotOwnership()) {
        return 1;
    }

    if (!TestPoolGenerationSeparatesSlotHandlesFromFitnessBookkeeping()) {
        return 1;
    }

    if (!TestAssemblyWithReferenceCountingGcReusesParentSlots()) {
        return 1;
    }

    if (!TestAssemblyFailsWhenPoolIsGenuinelyFull()) {
        return 1;
    }

    std::cout << "PASS: genotype_pool_test\n";
    return 0;
}
