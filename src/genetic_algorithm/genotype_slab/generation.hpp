#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genotype_slab/slab_allocator.hpp"

namespace neuroevolution::genetic_algorithm::genotype_slab {

struct SlabGeneration {
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    std::unique_ptr<std::uint32_t[]> slot_indices{};
    std::unique_ptr<float[]> fitness{};
    std::unique_ptr<std::uint32_t[]> evaluation_counts{};
    std::unique_ptr<std::uint8_t[]> has_fitness{};
};

struct SlabGenerationView {
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    std::uint32_t *slot_indices = nullptr;
    float *fitness = nullptr;
    std::uint32_t *evaluation_counts = nullptr;
    std::uint8_t *has_fitness = nullptr;
};

struct ConstSlabGenerationView {
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    const std::uint32_t *slot_indices = nullptr;
    const float *fitness = nullptr;
    const std::uint32_t *evaluation_counts = nullptr;
    const std::uint8_t *has_fitness = nullptr;
};

inline SlabGenerationView MakeSlabGenerationView(SlabGeneration &generation) noexcept {
    return {
        .generation_index = generation.generation_index,
        .active_individual_count = generation.active_individual_count,
        .slot_indices = generation.slot_indices.get(),
        .fitness = generation.fitness.get(),
        .evaluation_counts = generation.evaluation_counts.get(),
        .has_fitness = generation.has_fitness.get(),
    };
}

inline ConstSlabGenerationView MakeConstSlabGenerationView(const SlabGeneration &generation) noexcept {
    return {
        .generation_index = generation.generation_index,
        .active_individual_count = generation.active_individual_count,
        .slot_indices = generation.slot_indices.get(),
        .fitness = generation.fitness.get(),
        .evaluation_counts = generation.evaluation_counts.get(),
        .has_fitness = generation.has_fitness.get(),
    };
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidSlabGenerationView(const SlabGenerationView &generation) noexcept {
    return (generation.active_individual_count > 0) && (generation.slot_indices != nullptr) &&
           (generation.fitness != nullptr) && (generation.evaluation_counts != nullptr) &&
           (generation.has_fitness != nullptr);
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidSlabGenerationView(const ConstSlabGenerationView &generation) noexcept {
    return (generation.active_individual_count > 0) && (generation.slot_indices != nullptr) &&
           (generation.fitness != nullptr) && (generation.evaluation_counts != nullptr) &&
           (generation.has_fitness != nullptr);
}

inline bool IsValidSlabGeneration(const SlabGeneration &generation) noexcept {
    return IsValidSlabGenerationView(MakeConstSlabGenerationView(generation));
}

inline NEUROEVOLUTION_HOST_DEVICE void ClearSlabGenerationFitness(const SlabGenerationView generation) noexcept {
    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        generation.fitness[individual_index] = 0.0f;
        generation.evaluation_counts[individual_index] = 0;
        generation.has_fitness[individual_index] = 0;
    }
}

inline void ClearSlabGenerationFitness(SlabGeneration &generation) noexcept {
    ClearSlabGenerationFitness(MakeSlabGenerationView(generation));
}

inline NEUROEVOLUTION_HOST_DEVICE bool TrySetSlabGenerationSlot(const SlabGenerationView generation,
                                                                const std::size_t individual_index,
                                                                const std::uint32_t slot_index) noexcept {
    if (!IsValidSlabGenerationView(generation) || (individual_index >= generation.active_individual_count) ||
        (slot_index == kInvalidSlabSlotIndex)) {
        return false;
    }

    generation.slot_indices[individual_index] = slot_index;
    return true;
}

inline bool TrySetSlabGenerationSlot(SlabGeneration &generation, const std::size_t individual_index,
                                     const std::uint32_t slot_index) noexcept {
    return TrySetSlabGenerationSlot(MakeSlabGenerationView(generation), individual_index, slot_index);
}

inline NEUROEVOLUTION_HOST_DEVICE void ClearSlabGenerationSlot(const SlabGenerationView generation,
                                                               const std::size_t individual_index) noexcept {
    if (!IsValidSlabGenerationView(generation) || (individual_index >= generation.active_individual_count)) {
        return;
    }

    generation.slot_indices[individual_index] = kInvalidSlabSlotIndex;
}

inline void ClearSlabGenerationSlot(SlabGeneration &generation, const std::size_t individual_index) noexcept {
    ClearSlabGenerationSlot(MakeSlabGenerationView(generation), individual_index);
}

inline bool TryCreateSlabGeneration(SlabGeneration &generation, const std::size_t active_individual_count,
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
    if (!IsValidSlabGeneration(generation)) {
        generation = {};
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        generation.slot_indices[individual_index] = kInvalidSlabSlotIndex;
    }
    ClearSlabGenerationFitness(generation);
    return true;
}

inline bool TryCloneSlabSlotIntoGeneration(HostGenotypeSlab &buffer, SlabGeneration &generation,
                                           const std::size_t individual_index, const std::uint32_t source_slot_index,
                                           std::uint32_t &cloned_slot_index) {
    if (!IsValidHostGenotypeSlab(buffer) || !IsValidSlabGeneration(generation) ||
        (individual_index >= generation.active_individual_count) ||
        (generation.slot_indices[individual_index] != kInvalidSlabSlotIndex)) {
        return false;
    }

    if (!TryCloneSlabSlot(buffer, source_slot_index, cloned_slot_index)) {
        return false;
    }

    if (!TrySetSlabGenerationSlot(generation, individual_index, cloned_slot_index)) {
        (void)TryReleaseSlabSlot(buffer, cloned_slot_index);
        return false;
    }

    return true;
}

inline bool TryReleaseSlabGenerationSlots(HostGenotypeSlab &buffer, SlabGeneration &generation) {
    if (!IsValidHostGenotypeSlab(buffer) || !IsValidSlabGeneration(generation)) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        const std::uint32_t slot_index = generation.slot_indices[individual_index];
        if (slot_index == kInvalidSlabSlotIndex) {
            continue;
        }

        if (!TryReleaseSlabSlot(buffer, slot_index)) {
            return false;
        }

        generation.slot_indices[individual_index] = kInvalidSlabSlotIndex;
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm::genotype_slab
