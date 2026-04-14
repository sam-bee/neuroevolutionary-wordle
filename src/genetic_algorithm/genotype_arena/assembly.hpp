#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include "genetic_algorithm/genotype_arena/generation.hpp"

namespace neuroevolution::genetic_algorithm::genotype_arena {

struct ArenaParentPair {
    std::uint32_t first_parent_index = 0;
    std::uint32_t second_parent_index = 0;
};

struct ArenaAssemblyPlan {
    std::size_t child_count = 0;
    std::unique_ptr<ArenaParentPair[]> parent_pairs{};
};

struct ArenaAssemblyPlanView {
    std::size_t child_count = 0;
    ArenaParentPair *parent_pairs = nullptr;
};

struct ConstArenaAssemblyPlanView {
    std::size_t child_count = 0;
    const ArenaParentPair *parent_pairs = nullptr;
};

using AssembleArenaChildGenomeFunction = bool (*)(const std::uint8_t *first_parent_genome_bytes,
                                                  const std::uint8_t *second_parent_genome_bytes,
                                                  std::size_t action_count, std::uint8_t *child_genome_bytes,
                                                  void *user_data);

struct ArenaAssemblyCallbacks {
    AssembleArenaChildGenomeFunction assemble_child_genome = nullptr;
    void *user_data = nullptr;
};

inline ArenaAssemblyPlanView MakeArenaAssemblyPlanView(ArenaAssemblyPlan &plan) noexcept {
    return {.child_count = plan.child_count, .parent_pairs = plan.parent_pairs.get()};
}

inline ConstArenaAssemblyPlanView MakeConstArenaAssemblyPlanView(const ArenaAssemblyPlan &plan) noexcept {
    return {.child_count = plan.child_count, .parent_pairs = plan.parent_pairs.get()};
}

inline bool IsValidArenaAssemblyPlanView(const ArenaAssemblyPlanView &plan) noexcept {
    return (plan.child_count > 0) && (plan.parent_pairs != nullptr);
}

inline bool IsValidArenaAssemblyPlanView(const ConstArenaAssemblyPlanView &plan) noexcept {
    return (plan.child_count > 0) && (plan.parent_pairs != nullptr);
}

inline bool IsValidArenaAssemblyPlan(const ArenaAssemblyPlan &plan) noexcept {
    return IsValidArenaAssemblyPlanView(MakeConstArenaAssemblyPlanView(plan));
}

inline bool TryCreateArenaAssemblyPlan(ArenaAssemblyPlan &plan, const std::size_t child_count) {
    plan = {};
    if (child_count == 0) {
        return false;
    }

    plan.child_count = child_count;
    plan.parent_pairs.reset(new (std::nothrow) ArenaParentPair[child_count]());
    if (!IsValidArenaAssemblyPlan(plan)) {
        plan = {};
        return false;
    }

    return true;
}

bool TryAssembleNextGenerationWithoutElitism(HostGenotypeArena &arena, ArenaGeneration &current_generation,
                                             const ArenaAssemblyPlan &plan, ArenaGeneration &next_generation,
                                             const ArenaAssemblyCallbacks &callbacks);

} // namespace neuroevolution::genetic_algorithm::genotype_arena
