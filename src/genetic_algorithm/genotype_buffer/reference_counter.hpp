#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genotype_buffer/assembly.hpp"

namespace neuroevolution::genetic_algorithm::genotype_buffer {

namespace detail {

inline NEUROEVOLUTION_HOST_DEVICE std::uint32_t AtomicLoadReferenceCount(const std::uint32_t *counter) noexcept {
    if (counter == nullptr) {
        return 0;
    }

#if defined(__CUDA_ARCH__)
    return *counter;
#else
    return __atomic_load_n(counter, __ATOMIC_ACQUIRE);
#endif
}

inline NEUROEVOLUTION_HOST_DEVICE bool AtomicTryIncrementReferenceCount(std::uint32_t *counter) noexcept {
    if (counter == nullptr) {
        return false;
    }

#if defined(__CUDA_ARCH__)
    std::uint32_t observed = *counter;
    while (observed != kMaxBufferSlotReferenceCount) {
        const std::uint32_t previous = atomicCAS(counter, observed, observed + 1);
        if (previous == observed) {
            return true;
        }

        observed = previous;
    }

    return false;
#else
    std::uint32_t observed = __atomic_load_n(counter, __ATOMIC_RELAXED);
    while (observed != kMaxBufferSlotReferenceCount) {
        const std::uint32_t next = observed + 1;
        if (__atomic_compare_exchange_n(counter, &observed, next, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            return true;
        }
    }

    return false;
#endif
}

inline NEUROEVOLUTION_HOST_DEVICE bool AtomicTryDecrementReferenceCount(std::uint32_t *counter,
                                                                        std::uint32_t &previous_count) noexcept {
    previous_count = 0;
    if (counter == nullptr) {
        return false;
    }

#if defined(__CUDA_ARCH__)
    std::uint32_t observed = *counter;
    while (observed != 0) {
        const std::uint32_t previous = atomicCAS(counter, observed, observed - 1);
        if (previous == observed) {
            previous_count = observed;
            return true;
        }

        observed = previous;
    }

    return false;
#else
    std::uint32_t observed = __atomic_load_n(counter, __ATOMIC_RELAXED);
    while (observed != 0) {
        const std::uint32_t next = observed - 1;
        if (__atomic_compare_exchange_n(counter, &observed, next, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            previous_count = observed;
            return true;
        }
    }

    return false;
#endif
}

} // namespace detail

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidBufferParentIndex(const std::uint32_t *slot_indices,
                                                                const std::size_t active_individual_count,
                                                                const std::uint32_t parent_index) noexcept {
    return (slot_indices != nullptr) && (parent_index < active_individual_count) &&
           (slot_indices[parent_index] != kInvalidBufferSlotIndex);
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidBufferParentIndex(const BufferGenerationView generation,
                                                                const std::uint32_t parent_index) noexcept {
    return IsValidBufferGenerationView(generation) &&
           IsValidBufferParentIndex(generation.slot_indices, generation.active_individual_count, parent_index);
}

inline NEUROEVOLUTION_HOST_DEVICE void ClearParentReferenceCounts(std::uint32_t *parent_reference_counts,
                                                                  const std::size_t active_individual_count) noexcept {
    if (parent_reference_counts == nullptr) {
        return;
    }

    for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
        parent_reference_counts[parent_index] = 0;
    }
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryIncrementParentReferenceCount(std::uint32_t *parent_reference_counts,
                                                                        const std::uint32_t parent_index) noexcept {
    if (parent_reference_counts == nullptr) {
        return false;
    }

    return detail::AtomicTryIncrementReferenceCount(&parent_reference_counts[parent_index]);
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryDecrementParentReferenceCount(std::uint32_t *parent_reference_counts,
                                                                        const std::uint32_t parent_index,
                                                                        std::uint32_t &previous_count) noexcept {
    previous_count = 0;
    if (parent_reference_counts == nullptr) {
        return false;
    }

    return detail::AtomicTryDecrementReferenceCount(&parent_reference_counts[parent_index], previous_count);
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryBuildParentReferenceCounts(const std::uint32_t *slot_indices,
                                                                     const std::size_t active_individual_count,
                                                                     const BufferParentPair *parent_pairs,
                                                                     const std::size_t child_count,
                                                                     std::uint32_t *parent_reference_counts) noexcept {
    if ((slot_indices == nullptr) || (active_individual_count == 0) || (parent_pairs == nullptr) ||
        (child_count == 0) || (parent_reference_counts == nullptr)) {
        return false;
    }

    ClearParentReferenceCounts(parent_reference_counts, active_individual_count);
    for (std::size_t child_index = 0; child_index < child_count; ++child_index) {
        const BufferParentPair &parent_pair = parent_pairs[child_index];
        if (!IsValidBufferParentIndex(slot_indices, active_individual_count, parent_pair.first_parent_index) ||
            !IsValidBufferParentIndex(slot_indices, active_individual_count, parent_pair.second_parent_index)) {
            return false;
        }

        if (!TryIncrementParentReferenceCount(parent_reference_counts, parent_pair.first_parent_index) ||
            !TryIncrementParentReferenceCount(parent_reference_counts, parent_pair.second_parent_index)) {
            return false;
        }
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryBuildParentReferenceCounts(const BufferGenerationView generation,
                                                                     const BufferParentPair *parent_pairs,
                                                                     const std::size_t child_count,
                                                                     std::uint32_t *parent_reference_counts) noexcept {
    if (!IsValidBufferGenerationView(generation)) {
        return false;
    }

    return TryBuildParentReferenceCounts(generation.slot_indices, generation.active_individual_count, parent_pairs,
                                         child_count, parent_reference_counts);
}

inline NEUROEVOLUTION_HOST_DEVICE bool
TryCollectZeroReferenceParents(const GenotypeBufferView buffer, const BufferGenerationView current_generation,
                               const std::uint32_t *parent_reference_counts) noexcept {
    if (!IsValidGenotypeBufferView(buffer) || !IsValidBufferGenerationView(current_generation) ||
        (parent_reference_counts == nullptr)) {
        return false;
    }

    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        if ((detail::AtomicLoadReferenceCount(&parent_reference_counts[parent_index]) != 0) ||
            (current_generation.slot_indices[parent_index] == kInvalidBufferSlotIndex)) {
            continue;
        }

        if (!TryReleaseBufferSlot(buffer, current_generation.slot_indices[parent_index])) {
            return false;
        }

        ClearBufferGenerationSlot(current_generation, parent_index);
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryReleaseParentReference(const GenotypeBufferView buffer,
                                                                 const BufferGenerationView current_generation,
                                                                 std::uint32_t *parent_reference_counts,
                                                                 const std::uint32_t parent_index) noexcept {
    if (!IsValidBufferParentIndex(current_generation, parent_index) || (parent_reference_counts == nullptr)) {
        return false;
    }

    std::uint32_t previous_count = 0;
    if (!TryDecrementParentReferenceCount(parent_reference_counts, parent_index, previous_count)) {
        return false;
    }

    if (previous_count == 1) {
        if (!TryReleaseBufferSlot(buffer, current_generation.slot_indices[parent_index])) {
            return false;
        }

        ClearBufferGenerationSlot(current_generation, parent_index);
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm::genotype_buffer
