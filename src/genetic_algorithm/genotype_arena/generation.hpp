#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include "genetic_algorithm/genotype_arena/layout.hpp"

namespace neuroevolution::genetic_algorithm::genotype_arena {

struct ArenaGeneration {
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    std::unique_ptr<std::uint32_t[]> slot_indices{};
    std::unique_ptr<float[]> fitness{};
    std::unique_ptr<std::uint32_t[]> evaluation_counts{};
    std::unique_ptr<std::uint8_t[]> has_fitness{};
};

inline bool IsValidArenaGeneration(const ArenaGeneration &generation) noexcept {
    return (generation.active_individual_count > 0) && (generation.slot_indices != nullptr) &&
           (generation.fitness != nullptr) && (generation.evaluation_counts != nullptr) &&
           (generation.has_fitness != nullptr);
}

inline void ClearArenaGenerationFitness(ArenaGeneration &generation) noexcept {
    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        generation.fitness[individual_index] = 0.0f;
        generation.evaluation_counts[individual_index] = 0;
        generation.has_fitness[individual_index] = 0;
    }
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

} // namespace neuroevolution::genetic_algorithm::genotype_arena
