#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include "genetic_algorithm/genotype_pool/generation.hpp"

namespace neuroevolution::genetic_algorithm::genotype_pool {

struct PoolParentPair {
    std::uint32_t first_parent_index = 0;
    std::uint32_t second_parent_index = 0;
};

struct PoolAssemblyPlan {
    std::size_t child_count = 0;
    std::unique_ptr<PoolParentPair[]> parent_pairs{};
};

struct PoolAssemblyPlanView {
    std::size_t child_count = 0;
    PoolParentPair *parent_pairs = nullptr;
};

struct ConstPoolAssemblyPlanView {
    std::size_t child_count = 0;
    const PoolParentPair *parent_pairs = nullptr;
};

using AssemblePoolChildGenomeFunction = bool (*)(const std::uint8_t *first_parent_genome_bytes,
                                                 const std::uint8_t *second_parent_genome_bytes,
                                                 std::size_t action_count, std::uint8_t *child_genome_bytes,
                                                 void *user_data);

struct PoolAssemblyCallbacks {
    AssemblePoolChildGenomeFunction assemble_child_genome = nullptr;
    void *user_data = nullptr;
};

inline PoolAssemblyPlanView MakePoolAssemblyPlanView(PoolAssemblyPlan &plan) noexcept {
    return {.child_count = plan.child_count, .parent_pairs = plan.parent_pairs.get()};
}

inline ConstPoolAssemblyPlanView MakeConstPoolAssemblyPlanView(const PoolAssemblyPlan &plan) noexcept {
    return {.child_count = plan.child_count, .parent_pairs = plan.parent_pairs.get()};
}

inline bool IsValidPoolAssemblyPlanView(const PoolAssemblyPlanView &plan) noexcept {
    return (plan.child_count > 0) && (plan.parent_pairs != nullptr);
}

inline bool IsValidPoolAssemblyPlanView(const ConstPoolAssemblyPlanView &plan) noexcept {
    return (plan.child_count > 0) && (plan.parent_pairs != nullptr);
}

inline bool IsValidPoolAssemblyPlan(const PoolAssemblyPlan &plan) noexcept {
    return IsValidPoolAssemblyPlanView(MakeConstPoolAssemblyPlanView(plan));
}

inline bool TryCreatePoolAssemblyPlan(PoolAssemblyPlan &plan, const std::size_t child_count) {
    plan = {};
    if (child_count == 0) {
        return false;
    }

    plan.child_count = child_count;
    plan.parent_pairs.reset(new (std::nothrow) PoolParentPair[child_count]());
    if (!IsValidPoolAssemblyPlan(plan)) {
        plan = {};
        return false;
    }

    return true;
}

bool TryAssembleNextGeneration(HostGenotypePool &pool, PoolGeneration &current_generation, const PoolAssemblyPlan &plan,
                               PoolGeneration &next_generation, const PoolAssemblyCallbacks &callbacks);

} // namespace neuroevolution::genetic_algorithm::genotype_pool
