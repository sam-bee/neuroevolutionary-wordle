#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genotype_buffer/buffer.hpp"

namespace neuroevolution::genetic_algorithm::genotype_buffer {

struct BufferGeneration {
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    std::unique_ptr<std::uint32_t[]> slot_indices{};
    std::unique_ptr<float[]> fitness{};
    std::unique_ptr<std::uint32_t[]> evaluation_counts{};
    std::unique_ptr<std::uint8_t[]> has_fitness{};
};

struct BufferGenerationView {
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    std::uint32_t *slot_indices = nullptr;
    float *fitness = nullptr;
    std::uint32_t *evaluation_counts = nullptr;
    std::uint8_t *has_fitness = nullptr;
};

struct ConstBufferGenerationView {
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    const std::uint32_t *slot_indices = nullptr;
    const float *fitness = nullptr;
    const std::uint32_t *evaluation_counts = nullptr;
    const std::uint8_t *has_fitness = nullptr;
};

inline BufferGenerationView MakeBufferGenerationView(BufferGeneration &generation) noexcept {
    return {
        .generation_index = generation.generation_index,
        .active_individual_count = generation.active_individual_count,
        .slot_indices = generation.slot_indices.get(),
        .fitness = generation.fitness.get(),
        .evaluation_counts = generation.evaluation_counts.get(),
        .has_fitness = generation.has_fitness.get(),
    };
}

inline ConstBufferGenerationView MakeConstBufferGenerationView(const BufferGeneration &generation) noexcept {
    return {
        .generation_index = generation.generation_index,
        .active_individual_count = generation.active_individual_count,
        .slot_indices = generation.slot_indices.get(),
        .fitness = generation.fitness.get(),
        .evaluation_counts = generation.evaluation_counts.get(),
        .has_fitness = generation.has_fitness.get(),
    };
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidBufferGenerationView(const BufferGenerationView &generation) noexcept {
    return (generation.active_individual_count > 0) && (generation.slot_indices != nullptr) &&
           (generation.fitness != nullptr) && (generation.evaluation_counts != nullptr) &&
           (generation.has_fitness != nullptr);
}

inline NEUROEVOLUTION_HOST_DEVICE bool
IsValidBufferGenerationView(const ConstBufferGenerationView &generation) noexcept {
    return (generation.active_individual_count > 0) && (generation.slot_indices != nullptr) &&
           (generation.fitness != nullptr) && (generation.evaluation_counts != nullptr) &&
           (generation.has_fitness != nullptr);
}

inline bool IsValidBufferGeneration(const BufferGeneration &generation) noexcept {
    return IsValidBufferGenerationView(MakeConstBufferGenerationView(generation));
}

inline NEUROEVOLUTION_HOST_DEVICE void ClearBufferGenerationFitness(const BufferGenerationView generation) noexcept {
    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        generation.fitness[individual_index] = 0.0f;
        generation.evaluation_counts[individual_index] = 0;
        generation.has_fitness[individual_index] = 0;
    }
}

inline void ClearBufferGenerationFitness(BufferGeneration &generation) noexcept {
    ClearBufferGenerationFitness(MakeBufferGenerationView(generation));
}

inline NEUROEVOLUTION_HOST_DEVICE bool TrySetBufferGenerationSlot(const BufferGenerationView generation,
                                                                  const std::size_t individual_index,
                                                                  const std::uint32_t slot_index) noexcept {
    if (!IsValidBufferGenerationView(generation) || (individual_index >= generation.active_individual_count) ||
        (slot_index == kInvalidBufferSlotIndex)) {
        return false;
    }

    generation.slot_indices[individual_index] = slot_index;
    return true;
}

inline bool TrySetBufferGenerationSlot(BufferGeneration &generation, const std::size_t individual_index,
                                       const std::uint32_t slot_index) noexcept {
    return TrySetBufferGenerationSlot(MakeBufferGenerationView(generation), individual_index, slot_index);
}

inline NEUROEVOLUTION_HOST_DEVICE void ClearBufferGenerationSlot(const BufferGenerationView generation,
                                                                 const std::size_t individual_index) noexcept {
    if (!IsValidBufferGenerationView(generation) || (individual_index >= generation.active_individual_count)) {
        return;
    }

    generation.slot_indices[individual_index] = kInvalidBufferSlotIndex;
}

inline void ClearBufferGenerationSlot(BufferGeneration &generation, const std::size_t individual_index) noexcept {
    ClearBufferGenerationSlot(MakeBufferGenerationView(generation), individual_index);
}

inline bool TryCreateBufferGeneration(BufferGeneration &generation, const std::size_t active_individual_count,
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
    if (!IsValidBufferGeneration(generation)) {
        generation = {};
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        generation.slot_indices[individual_index] = kInvalidBufferSlotIndex;
    }
    ClearBufferGenerationFitness(generation);
    return true;
}

inline bool TryCloneBufferSlotIntoGeneration(HostGenotypeBuffer &buffer, BufferGeneration &generation,
                                             const std::size_t individual_index, const std::uint32_t source_slot_index,
                                             std::uint32_t &cloned_slot_index) {
    if (!IsValidHostGenotypeBuffer(buffer) || !IsValidBufferGeneration(generation) ||
        (individual_index >= generation.active_individual_count) ||
        (generation.slot_indices[individual_index] != kInvalidBufferSlotIndex)) {
        return false;
    }

    if (!TryCloneBufferSlot(buffer, source_slot_index, cloned_slot_index)) {
        return false;
    }

    if (!TrySetBufferGenerationSlot(generation, individual_index, cloned_slot_index)) {
        (void)TryReleaseBufferSlot(buffer, cloned_slot_index);
        return false;
    }

    return true;
}

inline bool TryReleaseBufferGenerationSlots(HostGenotypeBuffer &buffer, BufferGeneration &generation) {
    if (!IsValidHostGenotypeBuffer(buffer) || !IsValidBufferGeneration(generation)) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        const std::uint32_t slot_index = generation.slot_indices[individual_index];
        if (slot_index == kInvalidBufferSlotIndex) {
            continue;
        }

        if (!TryReleaseBufferSlot(buffer, slot_index)) {
            return false;
        }

        generation.slot_indices[individual_index] = kInvalidBufferSlotIndex;
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm::genotype_buffer
