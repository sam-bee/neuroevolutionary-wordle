#pragma once

#include <cstddef>
#include <cstdint>

#include "genetic_algorithm/device/genome_ops.cuh"
#include "genetic_algorithm/selection.hpp"

namespace neuroevolution::genetic_algorithm::device_selection_ops {

using device_genome_ops::DeviceRandomState;
using device_genome_ops::SampleIndex;

__device__ inline bool IsBetterFitness(const float candidate_fitness, const std::size_t candidate_index,
                                       const float reference_fitness, const std::size_t reference_index) {
    return (candidate_fitness > reference_fitness) ||
           ((candidate_fitness == reference_fitness) && (candidate_index < reference_index));
}

__device__ inline bool TrySampleSelectableIndex(const std::uint8_t *has_fitness_flags,
                                                const std::size_t active_population_size,
                                                DeviceRandomState &random_state, const std::size_t excluded_index,
                                                std::size_t &selected_index) {
    if (active_population_size == 0) {
        return false;
    }

    const std::size_t start_index = SampleIndex(random_state, active_population_size);
    for (std::size_t offset = 0; offset < active_population_size; ++offset) {
        const std::size_t candidate_index = (start_index + offset) % active_population_size;
        if ((candidate_index == excluded_index) || (has_fitness_flags[candidate_index] == 0)) {
            continue;
        }

        selected_index = candidate_index;
        return true;
    }

    return false;
}

__device__ inline bool TrySelectParentIndexDevice(const float *fitness_values, const std::uint8_t *has_fitness_flags,
                                                  const std::size_t active_population_size,
                                                  DeviceRandomState &random_state, const ParentSelectionConfig &config,
                                                  std::size_t &selected_parent_index,
                                                  const std::size_t excluded_index = kNoIndividualIndex) {
    std::size_t selectable_count = 0;
    for (std::size_t individual_index = 0; individual_index < active_population_size; ++individual_index) {
        if ((individual_index != excluded_index) && (has_fitness_flags[individual_index] != 0)) {
            ++selectable_count;
        }
    }

    if (selectable_count == 0) {
        return false;
    }

    const std::size_t tournament_size =
        (config.tournament_size < selectable_count) ? config.tournament_size : selectable_count;
    if (tournament_size == 0) {
        return false;
    }

    bool found_parent = false;
    float best_fitness = 0.0f;

    for (std::size_t sample_index = 0; sample_index < tournament_size; ++sample_index) {
        std::size_t candidate_index = 0;
        if (!TrySampleSelectableIndex(has_fitness_flags, active_population_size, random_state, excluded_index,
                                      candidate_index)) {
            return false;
        }

        const float candidate_fitness = fitness_values[candidate_index];
        if (!found_parent || IsBetterFitness(candidate_fitness, candidate_index, best_fitness, selected_parent_index)) {
            found_parent = true;
            selected_parent_index = candidate_index;
            best_fitness = candidate_fitness;
        }
    }

    return found_parent;
}

__device__ inline bool TrySelectParentPairDevice(const float *fitness_values, const std::uint8_t *has_fitness_flags,
                                                 const std::size_t active_population_size,
                                                 DeviceRandomState &random_state, const ParentSelectionConfig &config,
                                                 ParentPair &parent_pair) {
    if (!TrySelectParentIndexDevice(fitness_values, has_fitness_flags, active_population_size, random_state, config,
                                    parent_pair.first_parent_index)) {
        return false;
    }

    const std::size_t excluded_index =
        config.allow_self_parenting ? kNoIndividualIndex : parent_pair.first_parent_index;
    return TrySelectParentIndexDevice(fitness_values, has_fitness_flags, active_population_size, random_state, config,
                                      parent_pair.second_parent_index, excluded_index);
}

} // namespace neuroevolution::genetic_algorithm::device_selection_ops
