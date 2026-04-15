#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include "genetic_algorithm/genotype_buffer/generation.hpp"

namespace neuroevolution::genetic_algorithm::genotype_buffer {

struct BufferParentPair {
    std::uint32_t first_parent_index = 0;
    std::uint32_t second_parent_index = 0;
};

struct BufferAssemblyPlan {
    std::size_t child_count = 0;
    std::unique_ptr<BufferParentPair[]> parent_pairs{};
};

struct BufferAssemblyPlanView {
    std::size_t child_count = 0;
    BufferParentPair *parent_pairs = nullptr;
};

struct ConstBufferAssemblyPlanView {
    std::size_t child_count = 0;
    const BufferParentPair *parent_pairs = nullptr;
};

using AssembleBufferChildGenomeFunction = bool (*)(const std::uint8_t *first_parent_genome_bytes,
                                                   const std::uint8_t *second_parent_genome_bytes,
                                                   std::size_t action_count, std::uint8_t *child_genome_bytes,
                                                   void *user_data);

struct BufferAssemblyCallbacks {
    AssembleBufferChildGenomeFunction assemble_child_genome = nullptr;
    void *user_data = nullptr;
};

inline BufferAssemblyPlanView MakeBufferAssemblyPlanView(BufferAssemblyPlan &plan) noexcept {
    return {.child_count = plan.child_count, .parent_pairs = plan.parent_pairs.get()};
}

inline ConstBufferAssemblyPlanView MakeConstBufferAssemblyPlanView(const BufferAssemblyPlan &plan) noexcept {
    return {.child_count = plan.child_count, .parent_pairs = plan.parent_pairs.get()};
}

inline bool IsValidBufferAssemblyPlanView(const BufferAssemblyPlanView &plan) noexcept {
    return (plan.child_count > 0) && (plan.parent_pairs != nullptr);
}

inline bool IsValidBufferAssemblyPlanView(const ConstBufferAssemblyPlanView &plan) noexcept {
    return (plan.child_count > 0) && (plan.parent_pairs != nullptr);
}

inline bool IsValidBufferAssemblyPlan(const BufferAssemblyPlan &plan) noexcept {
    return IsValidBufferAssemblyPlanView(MakeConstBufferAssemblyPlanView(plan));
}

inline bool TryCreateBufferAssemblyPlan(BufferAssemblyPlan &plan, const std::size_t child_count) {
    plan = {};
    if (child_count == 0) {
        return false;
    }

    plan.child_count = child_count;
    plan.parent_pairs.reset(new (std::nothrow) BufferParentPair[child_count]());
    if (!IsValidBufferAssemblyPlan(plan)) {
        plan = {};
        return false;
    }

    return true;
}

bool TryAssembleNextGeneration(HostGenotypeBuffer &buffer, BufferGeneration &current_generation,
                               const BufferAssemblyPlan &plan, BufferGeneration &next_generation,
                               const BufferAssemblyCallbacks &callbacks);

} // namespace neuroevolution::genetic_algorithm::genotype_buffer
