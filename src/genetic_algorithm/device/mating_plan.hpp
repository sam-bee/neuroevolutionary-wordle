#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/selection.hpp"

namespace neuroevolution::genetic_algorithm::dynamic_device {

enum class PlannedGenerationMemberOperation : std::uint8_t {
    kEliteCopy = 0,
    kBreedChild = 1,
};

struct PlannedGenerationMember {
    PlannedGenerationMemberOperation operation = PlannedGenerationMemberOperation::kEliteCopy;
    ParentPair parent_pair{};
};

NEUROEVOLUTION_HOST_DEVICE constexpr PlannedGenerationMember MakeInvalidPlannedGenerationMember() noexcept {
    PlannedGenerationMember member{};
    member.operation = static_cast<PlannedGenerationMemberOperation>(0xFF);
    member.parent_pair.first_parent_index = kNoIndividualIndex;
    member.parent_pair.second_parent_index = kNoIndividualIndex;
    return member;
}

NEUROEVOLUTION_HOST_DEVICE constexpr PlannedGenerationMember
MakeEliteCopyPlannedGenerationMember(const std::size_t elite_parent_index) noexcept {
    PlannedGenerationMember member{};
    member.operation = PlannedGenerationMemberOperation::kEliteCopy;
    member.parent_pair.first_parent_index = elite_parent_index;
    member.parent_pair.second_parent_index = kNoIndividualIndex;
    return member;
}

NEUROEVOLUTION_HOST_DEVICE constexpr PlannedGenerationMember
MakeBreedingPlannedGenerationMember(const ParentPair &parent_pair) noexcept {
    PlannedGenerationMember member{};
    member.operation = PlannedGenerationMemberOperation::kBreedChild;
    member.parent_pair = parent_pair;
    return member;
}

NEUROEVOLUTION_HOST_DEVICE constexpr bool
IsValidPlannedGenerationMember(const PlannedGenerationMember &member, const std::size_t active_population_size) noexcept {
    if (member.parent_pair.first_parent_index >= active_population_size) {
        return false;
    }

    switch (member.operation) {
    case PlannedGenerationMemberOperation::kEliteCopy:
        return member.parent_pair.second_parent_index == kNoIndividualIndex;
    case PlannedGenerationMemberOperation::kBreedChild:
        return member.parent_pair.second_parent_index < active_population_size;
    }

    return false;
}

inline bool TryCountPlannedParentUses(const PlannedGenerationMember *planned_generation_members,
                                      const std::size_t planned_generation_member_count,
                                      const std::size_t active_population_size,
                                      std::uint32_t *parent_use_counts,
                                      const std::size_t parent_use_count_capacity) noexcept {
    if ((planned_generation_members == nullptr) || (parent_use_counts == nullptr) ||
        (parent_use_count_capacity < active_population_size)) {
        return false;
    }

    for (std::size_t parent_index = 0; parent_index < active_population_size; ++parent_index) {
        parent_use_counts[parent_index] = 0;
    }

    for (std::size_t member_index = 0; member_index < planned_generation_member_count; ++member_index) {
        const PlannedGenerationMember &member = planned_generation_members[member_index];
        if (!IsValidPlannedGenerationMember(member, active_population_size)) {
            return false;
        }

        ++parent_use_counts[member.parent_pair.first_parent_index];
        if (member.operation == PlannedGenerationMemberOperation::kBreedChild) {
            ++parent_use_counts[member.parent_pair.second_parent_index];
        }
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm::dynamic_device
