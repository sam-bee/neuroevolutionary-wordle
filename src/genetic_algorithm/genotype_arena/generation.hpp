#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genotype_arena/arena.hpp"

namespace neuroevolution::genetic_algorithm::genotype_arena {

struct ArenaGeneration {
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    std::unique_ptr<std::uint32_t[]> slot_indices{};
    std::unique_ptr<float[]> fitness{};
    std::unique_ptr<std::uint32_t[]> evaluation_counts{};
    std::unique_ptr<std::uint8_t[]> has_fitness{};
};

struct ArenaGenerationView {
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    std::uint32_t *slot_indices = nullptr;
    float *fitness = nullptr;
    std::uint32_t *evaluation_counts = nullptr;
    std::uint8_t *has_fitness = nullptr;
};

struct ConstArenaGenerationView {
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    const std::uint32_t *slot_indices = nullptr;
    const float *fitness = nullptr;
    const std::uint32_t *evaluation_counts = nullptr;
    const std::uint8_t *has_fitness = nullptr;
};

inline ArenaGenerationView MakeArenaGenerationView(ArenaGeneration &generation) noexcept {
    return {
        .generation_index = generation.generation_index,
        .active_individual_count = generation.active_individual_count,
        .slot_indices = generation.slot_indices.get(),
        .fitness = generation.fitness.get(),
        .evaluation_counts = generation.evaluation_counts.get(),
        .has_fitness = generation.has_fitness.get(),
    };
}

inline ConstArenaGenerationView MakeConstArenaGenerationView(const ArenaGeneration &generation) noexcept {
    return {
        .generation_index = generation.generation_index,
        .active_individual_count = generation.active_individual_count,
        .slot_indices = generation.slot_indices.get(),
        .fitness = generation.fitness.get(),
        .evaluation_counts = generation.evaluation_counts.get(),
        .has_fitness = generation.has_fitness.get(),
    };
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidArenaGenerationView(const ArenaGenerationView &generation) noexcept {
    return (generation.active_individual_count > 0) && (generation.slot_indices != nullptr) &&
           (generation.fitness != nullptr) && (generation.evaluation_counts != nullptr) &&
           (generation.has_fitness != nullptr);
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidArenaGenerationView(const ConstArenaGenerationView &generation) noexcept {
    return (generation.active_individual_count > 0) && (generation.slot_indices != nullptr) &&
           (generation.fitness != nullptr) && (generation.evaluation_counts != nullptr) &&
           (generation.has_fitness != nullptr);
}

inline bool IsValidArenaGeneration(const ArenaGeneration &generation) noexcept {
    return IsValidArenaGenerationView(MakeConstArenaGenerationView(generation));
}

inline NEUROEVOLUTION_HOST_DEVICE void ClearArenaGenerationFitness(const ArenaGenerationView generation) noexcept {
    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        generation.fitness[individual_index] = 0.0f;
        generation.evaluation_counts[individual_index] = 0;
        generation.has_fitness[individual_index] = 0;
    }
}

inline void ClearArenaGenerationFitness(ArenaGeneration &generation) noexcept {
    ClearArenaGenerationFitness(MakeArenaGenerationView(generation));
}

inline NEUROEVOLUTION_HOST_DEVICE bool TrySetArenaGenerationSlot(const ArenaGenerationView generation,
                                                                 const std::size_t individual_index,
                                                                 const std::uint32_t slot_index) noexcept {
    if (!IsValidArenaGenerationView(generation) || (individual_index >= generation.active_individual_count) ||
        (slot_index == kInvalidArenaSlotIndex)) {
        return false;
    }

    generation.slot_indices[individual_index] = slot_index;
    return true;
}

inline bool TrySetArenaGenerationSlot(ArenaGeneration &generation, const std::size_t individual_index,
                                      const std::uint32_t slot_index) noexcept {
    return TrySetArenaGenerationSlot(MakeArenaGenerationView(generation), individual_index, slot_index);
}

inline NEUROEVOLUTION_HOST_DEVICE void ClearArenaGenerationSlot(const ArenaGenerationView generation,
                                                                const std::size_t individual_index) noexcept {
    if (!IsValidArenaGenerationView(generation) || (individual_index >= generation.active_individual_count)) {
        return;
    }

    generation.slot_indices[individual_index] = kInvalidArenaSlotIndex;
}

inline void ClearArenaGenerationSlot(ArenaGeneration &generation, const std::size_t individual_index) noexcept {
    ClearArenaGenerationSlot(MakeArenaGenerationView(generation), individual_index);
}

inline bool TryCreateArenaGeneration(ArenaGeneration &generation, const std::size_t active_individual_count,
                                     const std::size_t generation_index = 0) {
    generation = {};
    if (active_individual_count == 0) {
        return false;
    }

    generation.generation_index = generation_index;
    generation.active_individual_count = active_individual_count;
    generation.slot_indices.reset(new (std::nothrow) std::uint32_t[active_individual_count]);
    generation.fitness.reset(new (std::nothrow) float[active_individual_count]);
    generation.evaluation_counts.reset(new (std::nothrow) std::uint32_t[active_individual_count]);
    generation.has_fitness.reset(new (std::nothrow) std::uint8_t[active_individual_count]);
    if (!IsValidArenaGeneration(generation)) {
        generation = {};
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        generation.slot_indices[individual_index] = kInvalidArenaSlotIndex;
    }
    ClearArenaGenerationFitness(generation);
    return true;
}

inline bool TryCloneArenaSlotIntoGeneration(HostGenotypeArena &arena, ArenaGeneration &generation,
                                            const std::size_t individual_index, const std::uint32_t source_slot_index,
                                            std::uint32_t &cloned_slot_index) {
    if (!IsValidHostGenotypeArena(arena) || !IsValidArenaGeneration(generation) ||
        (individual_index >= generation.active_individual_count) ||
        (generation.slot_indices[individual_index] != kInvalidArenaSlotIndex)) {
        return false;
    }

    if (!TryCloneArenaSlot(arena, source_slot_index, cloned_slot_index)) {
        return false;
    }

    if (!TrySetArenaGenerationSlot(generation, individual_index, cloned_slot_index)) {
        (void)TryReleaseArenaSlot(arena, cloned_slot_index);
        return false;
    }

    return true;
}

inline bool TryReleaseArenaGenerationSlots(HostGenotypeArena &arena, ArenaGeneration &generation) {
    if (!IsValidHostGenotypeArena(arena) || !IsValidArenaGeneration(generation)) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        const std::uint32_t slot_index = generation.slot_indices[individual_index];
        if (slot_index == kInvalidArenaSlotIndex) {
            continue;
        }

        if (!TryReleaseArenaSlot(arena, slot_index)) {
            return false;
        }

        generation.slot_indices[individual_index] = kInvalidArenaSlotIndex;
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm::genotype_arena
