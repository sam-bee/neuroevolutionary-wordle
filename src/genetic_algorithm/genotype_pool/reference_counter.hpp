#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genotype_pool/assembly.hpp"

namespace neuroevolution::genetic_algorithm::genotype_pool {

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidPoolParentIndex(const std::uint32_t *slot_indices,
                                                              const std::size_t active_individual_count,
                                                              const std::uint32_t parent_index) noexcept {
    return (slot_indices != nullptr) && (parent_index < active_individual_count) &&
           (slot_indices[parent_index] != kInvalidPoolSlotIndex);
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidPoolParentIndex(const PoolGenerationView generation,
                                                              const std::uint32_t parent_index) noexcept {
    return IsValidPoolGenerationView(generation) &&
           IsValidPoolParentIndex(generation.slot_indices, generation.active_individual_count, parent_index);
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
    if ((parent_reference_counts == nullptr) || (parent_reference_counts[parent_index] == kMaxPoolSlotReferenceCount)) {
        return false;
    }

    ++parent_reference_counts[parent_index];
    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryBuildParentReferenceCounts(const std::uint32_t *slot_indices,
                                                                     const std::size_t active_individual_count,
                                                                     const PoolParentPair *parent_pairs,
                                                                     const std::size_t child_count,
                                                                     std::uint32_t *parent_reference_counts) noexcept {
    if ((slot_indices == nullptr) || (active_individual_count == 0) || (parent_pairs == nullptr) ||
        (child_count == 0) || (parent_reference_counts == nullptr)) {
        return false;
    }

    ClearParentReferenceCounts(parent_reference_counts, active_individual_count);
    for (std::size_t child_index = 0; child_index < child_count; ++child_index) {
        const PoolParentPair &parent_pair = parent_pairs[child_index];
        if (!IsValidPoolParentIndex(slot_indices, active_individual_count, parent_pair.first_parent_index) ||
            !IsValidPoolParentIndex(slot_indices, active_individual_count, parent_pair.second_parent_index)) {
            return false;
        }

        if (!TryIncrementParentReferenceCount(parent_reference_counts, parent_pair.first_parent_index) ||
            !TryIncrementParentReferenceCount(parent_reference_counts, parent_pair.second_parent_index)) {
            return false;
        }
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryBuildParentReferenceCounts(const PoolGenerationView generation,
                                                                     const PoolParentPair *parent_pairs,
                                                                     const std::size_t child_count,
                                                                     std::uint32_t *parent_reference_counts) noexcept {
    if (!IsValidPoolGenerationView(generation)) {
        return false;
    }

    return TryBuildParentReferenceCounts(generation.slot_indices, generation.active_individual_count, parent_pairs,
                                         child_count, parent_reference_counts);
}

inline NEUROEVOLUTION_HOST_DEVICE bool
TryCollectZeroReferenceParents(const GenotypePoolView pool, const PoolGenerationView current_generation,
                               const std::uint32_t *parent_reference_counts) noexcept {
    if (!IsValidGenotypePoolView(pool) || !IsValidPoolGenerationView(current_generation) ||
        (parent_reference_counts == nullptr)) {
        return false;
    }

    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        if ((parent_reference_counts[parent_index] != 0) ||
            (current_generation.slot_indices[parent_index] == kInvalidPoolSlotIndex)) {
            continue;
        }

        if (!TryReleasePoolSlot(pool, current_generation.slot_indices[parent_index])) {
            return false;
        }

        ClearPoolGenerationSlot(current_generation, parent_index);
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryReleaseParentReference(const GenotypePoolView pool,
                                                                 const PoolGenerationView current_generation,
                                                                 std::uint32_t *parent_reference_counts,
                                                                 const std::uint32_t parent_index) noexcept {
    if (!IsValidPoolParentIndex(current_generation, parent_index) || (parent_reference_counts == nullptr) ||
        (parent_reference_counts[parent_index] == 0)) {
        return false;
    }

    --parent_reference_counts[parent_index];
    if (parent_reference_counts[parent_index] == 0) {
        if (!TryReleasePoolSlot(pool, current_generation.slot_indices[parent_index])) {
            return false;
        }

        ClearPoolGenerationSlot(current_generation, parent_index);
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm::genotype_pool
