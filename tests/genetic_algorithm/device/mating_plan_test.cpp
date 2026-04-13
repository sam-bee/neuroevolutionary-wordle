#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "genetic_algorithm/device/mating_plan.hpp"

namespace {

using neuroevolution::genetic_algorithm::ParentPair;
using neuroevolution::genetic_algorithm::dynamic_device::IsValidPlannedGenerationMember;
using neuroevolution::genetic_algorithm::dynamic_device::MakeBreedingPlannedGenerationMember;
using neuroevolution::genetic_algorithm::dynamic_device::MakeEliteCopyPlannedGenerationMember;
using neuroevolution::genetic_algorithm::dynamic_device::MakeInvalidPlannedGenerationMember;
using neuroevolution::genetic_algorithm::dynamic_device::PlannedGenerationMember;
using neuroevolution::genetic_algorithm::dynamic_device::TryCountPlannedParentUses;

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

bool TestCountPlannedParentUsesIncludesEliteCopiesAndBreedingPairs() {
    const PlannedGenerationMember plan[]{
        MakeEliteCopyPlannedGenerationMember(2),
        MakeBreedingPlannedGenerationMember(ParentPair{1, 2}),
        MakeBreedingPlannedGenerationMember(ParentPair{2, 1}),
        MakeEliteCopyPlannedGenerationMember(0),
    };
    std::uint32_t use_counts[4]{};

    bool ok = true;
    ok &= ExpectTrue(TryCountPlannedParentUses(plan, 4, 4, use_counts, 4),
                     "Expected valid mating plan to produce per-parent use counts");
    ok &= ExpectTrue(use_counts[0] == 1, "Expected parent zero to have one elite-copy use");
    ok &= ExpectTrue(use_counts[1] == 2, "Expected parent one to be used in two breeding roles");
    ok &= ExpectTrue(use_counts[2] == 3,
                     "Expected parent two to include one elite-copy use plus two breeding uses");
    ok &= ExpectTrue(use_counts[3] == 0, "Expected unused parent three to keep a zero use count");
    return ok;
}

bool TestCountPlannedParentUsesCountsSelfParentingTwice() {
    const PlannedGenerationMember plan[]{MakeBreedingPlannedGenerationMember(ParentPair{1, 1})};
    std::uint32_t use_counts[2]{};

    bool ok = true;
    ok &= ExpectTrue(TryCountPlannedParentUses(plan, 1, 2, use_counts, 2),
                     "Expected self-parenting plan entry to remain valid");
    ok &= ExpectTrue(use_counts[0] == 0, "Expected untouched parent zero to keep zero uses");
    ok &= ExpectTrue(use_counts[1] == 2, "Expected self-parenting to count as two parent uses");
    return ok;
}

bool TestCountPlannedParentUsesRejectsInvalidEntries() {
    PlannedGenerationMember invalid_member = MakeInvalidPlannedGenerationMember();
    invalid_member = MakeBreedingPlannedGenerationMember(ParentPair{0, 5});
    std::uint32_t use_counts[2]{};

    bool ok = true;
    ok &= ExpectTrue(!IsValidPlannedGenerationMember(invalid_member, 2),
                     "Expected out-of-range breeding parent to be invalid");
    ok &= ExpectTrue(!TryCountPlannedParentUses(&invalid_member, 1, 2, use_counts, 2),
                     "Expected invalid mating plan entry to be rejected");
    return ok;
}

} // namespace

int main() {
    if (!TestCountPlannedParentUsesIncludesEliteCopiesAndBreedingPairs()) {
        return 1;
    }

    if (!TestCountPlannedParentUsesCountsSelfParentingTwice()) {
        return 1;
    }

    if (!TestCountPlannedParentUsesRejectsInvalidEntries()) {
        return 1;
    }

    std::cout << "PASS: mating_plan_test\n";
    return 0;
}
