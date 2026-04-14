#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "genetic_algorithm/genotype_arena/arena.hpp"
#include "genetic_algorithm/genotype_arena/assembly.hpp"
#include "genetic_algorithm/genotype_arena/generation.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::common::ToFloat16;
using neuroevolution::genetic_algorithm::genome::ComputeDynamicGenomeStrideBytes;
using neuroevolution::genetic_algorithm::genotype_arena::ArenaAssemblyCallbacks;
using neuroevolution::genetic_algorithm::genotype_arena::ArenaAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_arena::ArenaGeneration;
using neuroevolution::genetic_algorithm::genotype_arena::ArenaGenerationView;
using neuroevolution::genetic_algorithm::genotype_arena::ArenaSlotCountForByteBudget;
using neuroevolution::genetic_algorithm::genotype_arena::ClearArenaGenerationFitness;
using neuroevolution::genetic_algorithm::genotype_arena::ComputeArenaSlotStrideBytes;
using neuroevolution::genetic_algorithm::genotype_arena::GenomePolicyModelParameters;
using neuroevolution::genetic_algorithm::genotype_arena::GenomeTailRows;
using neuroevolution::genetic_algorithm::genotype_arena::GenotypeArenaLayout;
using neuroevolution::genetic_algorithm::genotype_arena::GenotypeArenaView;
using neuroevolution::genetic_algorithm::genotype_arena::HostArenaSlotBytesAt;
using neuroevolution::genetic_algorithm::genotype_arena::HostGenotypeArena;
using neuroevolution::genetic_algorithm::genotype_arena::IsValidArenaGeneration;
using neuroevolution::genetic_algorithm::genotype_arena::IsValidGenotypeArenaLayout;
using neuroevolution::genetic_algorithm::genotype_arena::kInvalidArenaSlotIndex;
using neuroevolution::genetic_algorithm::genotype_arena::MakeArenaGenerationView;
using neuroevolution::genetic_algorithm::genotype_arena::MakeGenotypeArenaView;
using neuroevolution::genetic_algorithm::genotype_arena::TryAllocateArenaSlot;
using neuroevolution::genetic_algorithm::genotype_arena::TryAssembleNextGenerationWithoutElitism;
using neuroevolution::genetic_algorithm::genotype_arena::TryCloneArenaSlotIntoGeneration;
using neuroevolution::genetic_algorithm::genotype_arena::TryCopyGenomeBytesIntoArenaSlot;
using neuroevolution::genetic_algorithm::genotype_arena::TryCreateArenaAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_arena::TryCreateArenaGeneration;
using neuroevolution::genetic_algorithm::genotype_arena::TryCreateHostGenotypeArena;
using neuroevolution::genetic_algorithm::genotype_arena::TryReleaseArenaGenerationSlots;
using neuroevolution::genetic_algorithm::genotype_arena::TryReleaseArenaSlot;
using neuroevolution::genetic_algorithm::genotype_arena::TryRetainArenaSlot;
using neuroevolution::genetic_algorithm::genotype_arena::TrySetArenaGenerationSlot;

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

bool TestArenaLayoutReusesDynamicGenomeStrideMath() {
    constexpr std::size_t kActionCount = 20;
    constexpr std::size_t kSlotCount = 6;

    GenotypeArenaLayout layout{};
    layout.action_count = kActionCount;
    layout.slot_stride_bytes = ComputeArenaSlotStrideBytes(kActionCount);
    layout.slot_count = kSlotCount;
    layout.arena_bytes = kSlotCount * layout.slot_stride_bytes;

    bool ok = true;
    ok &= ExpectTrue(layout.slot_stride_bytes == ComputeDynamicGenomeStrideBytes(kActionCount),
                     "Expected arena slot stride to match dynamic genome stride math");
    ok &= ExpectTrue(ArenaSlotCountForByteBudget(layout.arena_bytes, kActionCount) == kSlotCount,
                     "Expected arena byte budget helper to recover the slot count");
    ok &= ExpectTrue(IsValidGenotypeArenaLayout(layout), "Expected constructed arena layout to be valid");
    return ok;
}

bool TestRawViewsMutateUnderlyingArenaAndGenerationState() {
    HostGenotypeArena arena{};
    ArenaGeneration generation{};
    bool ok = TryCreateHostGenotypeArena(arena, 2, 4);
    ok &= TryCreateArenaGeneration(generation, 2, 3);
    ok &= ExpectTrue(ok, "Expected raw-view test fixtures to allocate");
    if (!ok) {
        return false;
    }

    GenotypeArenaView arena_view = MakeGenotypeArenaView(arena);
    ArenaGenerationView generation_view = MakeArenaGenerationView(generation);
    std::uint32_t slot_index = kInvalidArenaSlotIndex;

    ok &= TryAllocateArenaSlot(arena_view, slot_index);
    ok &= TrySetArenaGenerationSlot(generation_view, 0, slot_index);
    ok &= ExpectTrue(ok, "Expected view-based slot allocation and assignment to succeed");
    ok &= ExpectTrue(arena.free_slot_count == 1, "Expected the host owner to observe the view-based allocation");
    ok &= ExpectTrue(generation.slot_indices[0] == slot_index,
                     "Expected the host generation to observe the view-based slot assignment");
    return ok;
}

bool TestHostArenaAllocatesReleasesAndReusesFixedWidthSlots() {
    HostGenotypeArena arena{};
    bool ok = TryCreateHostGenotypeArena(arena, 3, 4);
    ok &= ExpectTrue(ok, "Expected host genotype arena creation to succeed");
    if (!ok) {
        return false;
    }

    std::uint32_t slot0 = kInvalidArenaSlotIndex;
    std::uint32_t slot1 = kInvalidArenaSlotIndex;
    std::uint32_t slot2 = kInvalidArenaSlotIndex;
    std::uint32_t overflow_slot = kInvalidArenaSlotIndex;

    ok &= TryAllocateArenaSlot(arena, slot0);
    ok &= TryAllocateArenaSlot(arena, slot1);
    ok &= TryAllocateArenaSlot(arena, slot2);
    ok &= ExpectTrue(ok, "Expected every arena slot to allocate exactly once");
    ok &= ExpectTrue((slot0 == 0U) && (slot1 == 1U) && (slot2 == 2U),
                     "Expected the free-slot stack to hand out deterministic slot indices");
    ok &= ExpectTrue(!TryAllocateArenaSlot(arena, overflow_slot),
                     "Expected allocation to fail once the fixed-width arena is full");
    ok &= ExpectTrue(arena.free_slot_count == 0, "Expected a full arena to report no free slots");

    ok &= TryRetainArenaSlot(arena, slot1);
    ok &= ExpectTrue(arena.slot_states[slot1].reference_count == 2,
                     "Expected retaining a slot to increment its reference count");
    ok &= TryReleaseArenaSlot(arena, slot1);
    ok &= ExpectTrue(arena.slot_states[slot1].occupied, "Expected the slot to stay occupied while references remain");
    ok &= ExpectTrue(arena.slot_states[slot1].reference_count == 1,
                     "Expected releasing one reference to leave the slot live");
    ok &= TryReleaseArenaSlot(arena, slot1);
    ok &= ExpectTrue(!arena.slot_states[slot1].occupied,
                     "Expected the slot to become free after the last reference is released");
    ok &=
        ExpectTrue(arena.free_slot_count == 1, "Expected releasing the last reference to return the slot to the pool");

    std::uint32_t reused_slot = kInvalidArenaSlotIndex;
    ok &= TryAllocateArenaSlot(arena, reused_slot);
    ok &= ExpectTrue(reused_slot == slot1, "Expected the most recently released slot to be reused first");
    return ok;
}

bool TestArenaSlotAccessorsAndCopyHelperRoundTripGenomeBytes() {
    HostGenotypeArena arena{};
    bool ok = TryCreateHostGenotypeArena(arena, 2, 4);
    ok &= ExpectTrue(ok, "Expected host genotype arena creation for copy test to succeed");
    if (!ok) {
        return false;
    }

    std::uint32_t source_slot = kInvalidArenaSlotIndex;
    std::uint32_t target_slot = kInvalidArenaSlotIndex;
    ok &= TryAllocateArenaSlot(arena, source_slot);
    ok &= TryAllocateArenaSlot(arena, target_slot);
    ok &= ExpectTrue(ok, "Expected copy test slots to allocate");
    if (!ok) {
        return false;
    }

    std::uint8_t *source_genome_bytes = HostArenaSlotBytesAt(arena, source_slot);
    GenomePolicyModelParameters(source_genome_bytes).dense_trunk.hidden1_to_output.biases[0] = 1.25f;
    GenomePolicyModelParameters(source_genome_bytes).input_encoder.input_to_hidden.weights[0] = -0.75f;
    GenomeTailRows(source_genome_bytes)[2][1] = 2.5f;

    ok &= TryCopyGenomeBytesIntoArenaSlot(arena, target_slot, source_genome_bytes, arena.layout.slot_stride_bytes);
    ok &= ExpectTrue(ok, "Expected fixed-width slot copy helper to succeed");
    if (!ok) {
        return false;
    }

    const std::uint8_t *target_genome_bytes = HostArenaSlotBytesAt(arena, target_slot);
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(target_genome_bytes).dense_trunk.hidden1_to_output.biases[0]),
                     1.25f, "copied dense trunk bias");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(target_genome_bytes).input_encoder.input_to_hidden.weights[0]),
                     -0.75f, "copied encoder weight");
    ok &= ExpectNear(ToFloat(GenomeTailRows(target_genome_bytes)[2][1]), 2.5f, "copied trainable tail value");
    return ok;
}

bool TestGenerationLifecycleClonesAndReleasesSlotOwnership() {
    HostGenotypeArena arena{};
    ArenaGeneration current_generation{};
    ArenaGeneration next_generation{};
    bool ok = TryCreateHostGenotypeArena(arena, 3, 4);
    ok &= TryCreateArenaGeneration(current_generation, 1, 2);
    ok &= TryCreateArenaGeneration(next_generation, 1, 3);
    ok &= ExpectTrue(ok, "Expected generation lifecycle fixtures to allocate");
    if (!ok) {
        return false;
    }

    std::uint32_t source_slot = kInvalidArenaSlotIndex;
    ok &= TryAllocateArenaSlot(arena, source_slot);
    ok &= TrySetArenaGenerationSlot(current_generation, 0, source_slot);
    GenomePolicyModelParameters(HostArenaSlotBytesAt(arena, source_slot)).dense_trunk.hidden1_to_output.biases[0] =
        4.5f;
    GenomeTailRows(HostArenaSlotBytesAt(arena, source_slot))[1][0] = -1.25f;

    std::uint32_t cloned_slot = kInvalidArenaSlotIndex;
    ok &= TryCloneArenaSlotIntoGeneration(arena, next_generation, 0, source_slot, cloned_slot);
    ok &= ExpectTrue(ok, "Expected generation lifecycle clone helper to allocate a child slot");
    ok &= ExpectTrue(cloned_slot != source_slot, "Expected cloned generation slots to use fresh arena storage");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostArenaSlotBytesAt(arena, cloned_slot))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     4.5f, "cloned slot dense trunk bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostArenaSlotBytesAt(arena, cloned_slot))[1][0]), -1.25f,
                     "cloned slot trainable tail value");
    if (!ok) {
        return false;
    }

    ok &= TryReleaseArenaGenerationSlots(arena, current_generation);
    ok &= ExpectTrue(ok, "Expected releasing a generation to release its slot ownership");
    ok &= ExpectTrue(current_generation.slot_indices[0] == kInvalidArenaSlotIndex,
                     "Expected releasing a generation to clear its slot handles");
    ok &= ExpectTrue(!arena.slot_states[source_slot].occupied,
                     "Expected released generation slots to return to the arena");
    ok &= ExpectTrue(arena.slot_states[cloned_slot].occupied, "Expected cloned child slots to remain live");
    return ok;
}

bool TestArenaGenerationSeparatesSlotHandlesFromFitnessBookkeeping() {
    ArenaGeneration generation{};
    bool ok = TryCreateArenaGeneration(generation, 3, 7);
    ok &= ExpectTrue(ok, "Expected arena generation allocation to succeed");
    ok &= ExpectTrue(IsValidArenaGeneration(generation), "Expected allocated arena generation to be valid");
    ok &= ExpectTrue(generation.generation_index == 7, "Expected arena generation to keep its generation index");
    if (!ok) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        ok &= ExpectTrue(generation.slot_indices[individual_index] == kInvalidArenaSlotIndex,
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

    ClearArenaGenerationFitness(generation);

    ok &= ExpectTrue(generation.slot_indices[1] == 5,
                     "Expected slot handles to survive a fitness reset because they are separate generation metadata");
    ok &= ExpectNear(generation.fitness[1], 0.0f, "Expected fitness reset to clear stored fitness");
    ok &= ExpectTrue(generation.evaluation_counts[1] == 0, "Expected fitness reset to clear evaluation counts");
    ok &= ExpectTrue(generation.has_fitness[1] == 0, "Expected fitness reset to clear fitness flags");
    return ok;
}

bool TestAssemblyWithoutElitismSweepsAndReusesParentSlots() {
    HostGenotypeArena arena{};
    ArenaGeneration current_generation{};
    ArenaGeneration next_generation{};
    bool ok = TryCreateHostGenotypeArena(arena, 3, 4);
    ok &= TryCreateArenaGeneration(current_generation, 3, 5);
    ok &= ExpectTrue(ok, "Expected assembly test fixtures to allocate");
    if (!ok) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = kInvalidArenaSlotIndex;
        ok &= TryAllocateArenaSlot(arena, slot_index);
        ok &= TrySetArenaGenerationSlot(current_generation, individual_index, slot_index);
        GenomePolicyModelParameters(HostArenaSlotBytesAt(arena, slot_index)).dense_trunk.hidden1_to_output.biases[0] =
            ToFloat16(10.0f * static_cast<float>(individual_index + 1));
    }
    ok &= ExpectTrue(ok, "Expected assembly test parents to occupy all arena slots");
    if (!ok) {
        return false;
    }

    ArenaAssemblyPlan plan{};
    ok &= TryCreateArenaAssemblyPlan(plan, 2);
    ok &= ExpectTrue(ok, "Expected assembly plan allocation to succeed");
    if (!ok) {
        return false;
    }

    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};
    plan.parent_pairs[1] = {.first_parent_index = 0, .second_parent_index = 0};

    TestAssemblyState assembly_state{};
    ArenaAssemblyCallbacks callbacks{};
    callbacks.assemble_child_genome = AssembleSummedChildGenome;
    callbacks.user_data = &assembly_state;

    ok &= TryAssembleNextGenerationWithoutElitism(arena, current_generation, plan, next_generation, callbacks);
    ok &= ExpectTrue(ok, "Expected non-elitist arena assembly to succeed by sweeping and reusing parent slots");
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(next_generation.generation_index == 6,
                     "Expected non-elitist arena assembly to increment the generation index");
    ok &= ExpectTrue(next_generation.slot_indices[0] == 2U,
                     "Expected the first child to reuse the pre-swept zero-duty parent slot");
    ok &= ExpectTrue(next_generation.slot_indices[1] == 1U,
                     "Expected the second child to reuse the parent slot freed after its final breeding duty");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostArenaSlotBytesAt(arena, next_generation.slot_indices[0]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     30.0f, "first assembled child bias");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostArenaSlotBytesAt(arena, next_generation.slot_indices[1]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     20.0f, "second assembled child bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostArenaSlotBytesAt(arena, next_generation.slot_indices[0]))[0][0]), 1.0f,
                     "first assembled child marker");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostArenaSlotBytesAt(arena, next_generation.slot_indices[1]))[0][0]), 2.0f,
                     "second assembled child marker");
    ok &= ExpectTrue(current_generation.slot_indices[0] == kInvalidArenaSlotIndex,
                     "Expected the first parent to be released after its final breeding duty");
    ok &= ExpectTrue(current_generation.slot_indices[1] == kInvalidArenaSlotIndex,
                     "Expected the second parent to be released after its only breeding duty");
    ok &= ExpectTrue(current_generation.slot_indices[2] == kInvalidArenaSlotIndex,
                     "Expected zero-duty parents to be swept before child assembly begins");
    ok &= ExpectTrue(arena.free_slot_count == 1,
                     "Expected one parent slot to be free after two children occupy the reused slots");
    return ok;
}

bool TestAssemblyWithoutElitismFailsWhenArenaIsGenuinelyFull() {
    HostGenotypeArena arena{};
    ArenaGeneration current_generation{};
    ArenaGeneration next_generation{};
    bool ok = TryCreateHostGenotypeArena(arena, 2, 4);
    ok &= TryCreateArenaGeneration(current_generation, 2, 4);
    ok &= ExpectTrue(ok, "Expected full-arena failure fixtures to allocate");
    if (!ok) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = kInvalidArenaSlotIndex;
        ok &= TryAllocateArenaSlot(arena, slot_index);
        ok &= TrySetArenaGenerationSlot(current_generation, individual_index, slot_index);
    }
    ok &= ExpectTrue(ok, "Expected full-arena failure fixtures to consume every slot");
    if (!ok) {
        return false;
    }

    ArenaAssemblyPlan plan{};
    ok &= TryCreateArenaAssemblyPlan(plan, 1);
    ok &= ExpectTrue(ok, "Expected single-child assembly plan allocation to succeed");
    if (!ok) {
        return false;
    }

    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};

    TestAssemblyState assembly_state{};
    ArenaAssemblyCallbacks callbacks{};
    callbacks.assemble_child_genome = AssembleSummedChildGenome;
    callbacks.user_data = &assembly_state;

    ok &= ExpectTrue(
        !TryAssembleNextGenerationWithoutElitism(arena, current_generation, plan, next_generation, callbacks),
        "Expected assembly to fail when the arena is full and every parent still has breeding duties");
    ok &= ExpectTrue(!IsValidArenaGeneration(next_generation),
                     "Expected failed assembly to leave the next generation output unset");
    ok &= ExpectTrue(current_generation.slot_indices[0] == 0U,
                     "Expected failed full-arena assembly to leave the first parent slot intact");
    ok &= ExpectTrue(current_generation.slot_indices[1] == 1U,
                     "Expected failed full-arena assembly to leave the second parent slot intact");
    ok &= ExpectTrue(arena.free_slot_count == 0, "Expected failed full-arena assembly to leave the arena full");
    return ok;
}

} // namespace

int main() {
    if (!TestArenaLayoutReusesDynamicGenomeStrideMath()) {
        return 1;
    }

    if (!TestRawViewsMutateUnderlyingArenaAndGenerationState()) {
        return 1;
    }

    if (!TestHostArenaAllocatesReleasesAndReusesFixedWidthSlots()) {
        return 1;
    }

    if (!TestArenaSlotAccessorsAndCopyHelperRoundTripGenomeBytes()) {
        return 1;
    }

    if (!TestGenerationLifecycleClonesAndReleasesSlotOwnership()) {
        return 1;
    }

    if (!TestArenaGenerationSeparatesSlotHandlesFromFitnessBookkeeping()) {
        return 1;
    }

    if (!TestAssemblyWithoutElitismSweepsAndReusesParentSlots()) {
        return 1;
    }

    if (!TestAssemblyWithoutElitismFailsWhenArenaIsGenuinelyFull()) {
        return 1;
    }

    std::cout << "PASS: genotype_arena_test\n";
    return 0;
}
