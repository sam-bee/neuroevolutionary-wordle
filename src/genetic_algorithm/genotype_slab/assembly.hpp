#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include "genetic_algorithm/genotype_slab/generation.hpp"

namespace neuroevolution::genetic_algorithm::genotype_slab {

struct SlabParentPair {
    std::uint32_t first_parent_index = 0;
    std::uint32_t second_parent_index = 0;
};

struct SlabAssemblyPlan {
    std::size_t child_count = 0;
    std::unique_ptr<SlabParentPair[]> parent_pairs{};
};

struct ConstSlabAssemblyPlanView {
    std::size_t child_count = 0;
    const SlabParentPair *parent_pairs = nullptr;
};

inline ConstSlabAssemblyPlanView MakeConstSlabAssemblyPlanView(const SlabAssemblyPlan &plan) noexcept {
    return {.child_count = plan.child_count, .parent_pairs = plan.parent_pairs.get()};
}

inline bool IsValidSlabAssemblyPlanView(const ConstSlabAssemblyPlanView &plan) noexcept {
    return (plan.child_count > 0) && (plan.parent_pairs != nullptr);
}

inline bool IsValidSlabAssemblyPlan(const SlabAssemblyPlan &plan) noexcept {
    return IsValidSlabAssemblyPlanView(MakeConstSlabAssemblyPlanView(plan));
}

inline bool TryCreateSlabAssemblyPlan(SlabAssemblyPlan &plan, const std::size_t child_count) {
    plan = {};
    if (child_count == 0) {
        return false;
    }

    plan.child_count = child_count;
    plan.parent_pairs.reset(new (std::nothrow) SlabParentPair[child_count]());
    if (!IsValidSlabAssemblyPlan(plan)) {
        plan = {};
        return false;
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm::genotype_slab
