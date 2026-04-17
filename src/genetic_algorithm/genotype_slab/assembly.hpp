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

struct SlabAssemblyPlanView {
    std::size_t child_count = 0;
    SlabParentPair *parent_pairs = nullptr;
};

struct ConstSlabAssemblyPlanView {
    std::size_t child_count = 0;
    const SlabParentPair *parent_pairs = nullptr;
};

using AssembleSlabChildGenomeFunction = bool (*)(const std::uint8_t *first_parent_genome_bytes,
                                                 const std::uint8_t *second_parent_genome_bytes,
                                                 std::size_t action_count, std::uint8_t *child_genome_bytes,
                                                 void *user_data);

struct SlabAssemblyCallbacks {
    AssembleSlabChildGenomeFunction assemble_child_genome = nullptr;
    void *user_data = nullptr;
};

inline SlabAssemblyPlanView MakeSlabAssemblyPlanView(SlabAssemblyPlan &plan) noexcept {
    return {.child_count = plan.child_count, .parent_pairs = plan.parent_pairs.get()};
}

inline ConstSlabAssemblyPlanView MakeConstSlabAssemblyPlanView(const SlabAssemblyPlan &plan) noexcept {
    return {.child_count = plan.child_count, .parent_pairs = plan.parent_pairs.get()};
}

inline bool IsValidSlabAssemblyPlanView(const SlabAssemblyPlanView &plan) noexcept {
    return (plan.child_count > 0) && (plan.parent_pairs != nullptr);
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

bool TryAssembleNextGeneration(HostGenotypeSlab &buffer, SlabGeneration &current_generation,
                               const SlabAssemblyPlan &plan, SlabGeneration &next_generation,
                               const SlabAssemblyCallbacks &callbacks);

} // namespace neuroevolution::genetic_algorithm::genotype_slab
