#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "genetic_algorithm/genotype_arena/arena.hpp"
#include "genetic_algorithm/genotype_arena/generation.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::genetic_algorithm::genome::ComputeDynamicGenomeStrideBytes;
using neuroevolution::genetic_algorithm::genotype_arena::ArenaGeneration;
using neuroevolution::genetic_algorithm::genotype_arena::ArenaSlotCountForByteBudget;
using neuroevolution::genetic_algorithm::genotype_arena::ClearArenaGenerationFitness;
using neuroevolution::genetic_algorithm::genotype_arena::ComputeArenaSlotStrideBytes;
using neuroevolution::genetic_algorithm::genotype_arena::GenomePolicyModelParameters;
using neuroevolution::genetic_algorithm::genotype_arena::GenomeTailRows;
using neuroevolution::genetic_algorithm::genotype_arena::GenotypeArenaLayout;
using neuroevolution::genetic_algorithm::genotype_arena::HostArenaSlotBytesAt;
using neuroevolution::genetic_algorithm::genotype_arena::HostGenotypeArena;
using neuroevolution::genetic_algorithm::genotype_arena::IsValidArenaGeneration;
using neuroevolution::genetic_algorithm::genotype_arena::IsValidGenotypeArenaLayout;
using neuroevolution::genetic_algorithm::genotype_arena::kInvalidArenaSlotIndex;
using neuroevolution::genetic_algorithm::genotype_arena::TryAllocateArenaSlot;
using neuroevolution::genetic_algorithm::genotype_arena::TryCopyGenomeBytesIntoArenaSlot;
using neuroevolution::genetic_algorithm::genotype_arena::TryCreateArenaGeneration;
using neuroevolution::genetic_algorithm::genotype_arena::TryCreateHostGenotypeArena;
using neuroevolution::genetic_algorithm::genotype_arena::TryReleaseArenaSlot;
using neuroevolution::genetic_algorithm::genotype_arena::TryRetainArenaSlot;

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

} // namespace

int main() {
    if (!TestArenaLayoutReusesDynamicGenomeStrideMath()) {
        return 1;
    }

    if (!TestHostArenaAllocatesReleasesAndReusesFixedWidthSlots()) {
        return 1;
    }

    if (!TestArenaSlotAccessorsAndCopyHelperRoundTripGenomeBytes()) {
        return 1;
    }

    if (!TestArenaGenerationSeparatesSlotHandlesFromFitnessBookkeeping()) {
        return 1;
    }

    std::cout << "PASS: genotype_arena_test\n";
    return 0;
}
