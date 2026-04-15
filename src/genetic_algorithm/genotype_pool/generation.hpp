#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genotype_pool/pool.hpp"

namespace neuroevolution::genetic_algorithm::genotype_pool {

struct PoolGeneration {
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    std::unique_ptr<std::uint32_t[]> slot_indices{};
    std::unique_ptr<float[]> fitness{};
    std::unique_ptr<std::uint32_t[]> evaluation_counts{};
    std::unique_ptr<std::uint8_t[]> has_fitness{};
};

struct PoolGenerationView {
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    std::uint32_t *slot_indices = nullptr;
    float *fitness = nullptr;
    std::uint32_t *evaluation_counts = nullptr;
    std::uint8_t *has_fitness = nullptr;
};

struct ConstPoolGenerationView {
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    const std::uint32_t *slot_indices = nullptr;
    const float *fitness = nullptr;
    const std::uint32_t *evaluation_counts = nullptr;
    const std::uint8_t *has_fitness = nullptr;
};

inline PoolGenerationView MakePoolGenerationView(PoolGeneration &generation) noexcept {
    return {
        .generation_index = generation.generation_index,
        .active_individual_count = generation.active_individual_count,
        .slot_indices = generation.slot_indices.get(),
        .fitness = generation.fitness.get(),
        .evaluation_counts = generation.evaluation_counts.get(),
        .has_fitness = generation.has_fitness.get(),
    };
}

inline ConstPoolGenerationView MakeConstPoolGenerationView(const PoolGeneration &generation) noexcept {
    return {
        .generation_index = generation.generation_index,
        .active_individual_count = generation.active_individual_count,
        .slot_indices = generation.slot_indices.get(),
        .fitness = generation.fitness.get(),
        .evaluation_counts = generation.evaluation_counts.get(),
        .has_fitness = generation.has_fitness.get(),
    };
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidPoolGenerationView(const PoolGenerationView &generation) noexcept {
    return (generation.active_individual_count > 0) && (generation.slot_indices != nullptr) &&
           (generation.fitness != nullptr) && (generation.evaluation_counts != nullptr) &&
           (generation.has_fitness != nullptr);
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidPoolGenerationView(const ConstPoolGenerationView &generation) noexcept {
    return (generation.active_individual_count > 0) && (generation.slot_indices != nullptr) &&
           (generation.fitness != nullptr) && (generation.evaluation_counts != nullptr) &&
           (generation.has_fitness != nullptr);
}

inline bool IsValidPoolGeneration(const PoolGeneration &generation) noexcept {
    return IsValidPoolGenerationView(MakeConstPoolGenerationView(generation));
}

inline NEUROEVOLUTION_HOST_DEVICE void ClearPoolGenerationFitness(const PoolGenerationView generation) noexcept {
    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        generation.fitness[individual_index] = 0.0f;
        generation.evaluation_counts[individual_index] = 0;
        generation.has_fitness[individual_index] = 0;
    }
}

inline void ClearPoolGenerationFitness(PoolGeneration &generation) noexcept {
    ClearPoolGenerationFitness(MakePoolGenerationView(generation));
}

inline NEUROEVOLUTION_HOST_DEVICE bool TrySetPoolGenerationSlot(const PoolGenerationView generation,
                                                                const std::size_t individual_index,
                                                                const std::uint32_t slot_index) noexcept {
    if (!IsValidPoolGenerationView(generation) || (individual_index >= generation.active_individual_count) ||
        (slot_index == kInvalidPoolSlotIndex)) {
        return false;
    }

    generation.slot_indices[individual_index] = slot_index;
    return true;
}

inline bool TrySetPoolGenerationSlot(PoolGeneration &generation, const std::size_t individual_index,
                                     const std::uint32_t slot_index) noexcept {
    return TrySetPoolGenerationSlot(MakePoolGenerationView(generation), individual_index, slot_index);
}

inline NEUROEVOLUTION_HOST_DEVICE void ClearPoolGenerationSlot(const PoolGenerationView generation,
                                                               const std::size_t individual_index) noexcept {
    if (!IsValidPoolGenerationView(generation) || (individual_index >= generation.active_individual_count)) {
        return;
    }

    generation.slot_indices[individual_index] = kInvalidPoolSlotIndex;
}

inline void ClearPoolGenerationSlot(PoolGeneration &generation, const std::size_t individual_index) noexcept {
    ClearPoolGenerationSlot(MakePoolGenerationView(generation), individual_index);
}

inline bool TryCreatePoolGeneration(PoolGeneration &generation, const std::size_t active_individual_count,
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
    if (!IsValidPoolGeneration(generation)) {
        generation = {};
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        generation.slot_indices[individual_index] = kInvalidPoolSlotIndex;
    }
    ClearPoolGenerationFitness(generation);
    return true;
}

inline bool TryClonePoolSlotIntoGeneration(HostGenotypePool &pool, PoolGeneration &generation,
                                           const std::size_t individual_index, const std::uint32_t source_slot_index,
                                           std::uint32_t &cloned_slot_index) {
    if (!IsValidHostGenotypePool(pool) || !IsValidPoolGeneration(generation) ||
        (individual_index >= generation.active_individual_count) ||
        (generation.slot_indices[individual_index] != kInvalidPoolSlotIndex)) {
        return false;
    }

    if (!TryClonePoolSlot(pool, source_slot_index, cloned_slot_index)) {
        return false;
    }

    if (!TrySetPoolGenerationSlot(generation, individual_index, cloned_slot_index)) {
        (void)TryReleasePoolSlot(pool, cloned_slot_index);
        return false;
    }

    return true;
}

inline bool TryReleasePoolGenerationSlots(HostGenotypePool &pool, PoolGeneration &generation) {
    if (!IsValidHostGenotypePool(pool) || !IsValidPoolGeneration(generation)) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        const std::uint32_t slot_index = generation.slot_indices[individual_index];
        if (slot_index == kInvalidPoolSlotIndex) {
            continue;
        }

        if (!TryReleasePoolSlot(pool, slot_index)) {
            return false;
        }

        generation.slot_indices[individual_index] = kInvalidPoolSlotIndex;
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm::genotype_pool
