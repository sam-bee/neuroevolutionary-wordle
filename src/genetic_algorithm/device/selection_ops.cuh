#pragma once

#include <cstddef>
#include <cstdint>

#include "genetic_algorithm/device/genome_ops.cuh"
#include "genetic_algorithm/selection.hpp"
#include "genetic_algorithm/spatial/grid.hpp"

namespace neuroevolution::genetic_algorithm::device_selection_ops {

using device_genome_ops::DeviceRandomState;
using device_genome_ops::NextUniform01;
using device_genome_ops::SampleIndex;
using spatial::CellularGridShape;
using spatial::CellularNeighborList;
using spatial::TryCollectCellularSecondParentCandidates;
using spatial::TryProjectCellIndexBetweenSquareGrids;

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

__device__ inline bool
TrySelectRouletteIndexFromCandidates(const float *fitness_values, const std::uint8_t *has_fitness_flags,
                                     const CellularNeighborList &candidate_indices, DeviceRandomState &random_state,
                                     std::size_t &selected_parent_index) {
    float total_fitness = 0.0f;
    bool found_candidate = false;
    std::size_t fallback_index = 0;

    for (std::size_t candidate_offset = 0; candidate_offset < candidate_indices.count; ++candidate_offset) {
        const std::size_t candidate_index = candidate_indices.indices[candidate_offset];
        if (has_fitness_flags[candidate_index] == 0) {
            continue;
        }

        total_fitness += fitness_values[candidate_index];
        fallback_index = candidate_index;
        found_candidate = true;
    }

    if (!found_candidate || (total_fitness <= 0.0f)) {
        return false;
    }

    const float threshold = NextUniform01(random_state) * total_fitness;
    float cumulative_fitness = 0.0f;
    for (std::size_t candidate_offset = 0; candidate_offset < candidate_indices.count; ++candidate_offset) {
        const std::size_t candidate_index = candidate_indices.indices[candidate_offset];
        if (has_fitness_flags[candidate_index] == 0) {
            continue;
        }

        cumulative_fitness += fitness_values[candidate_index];
        if (threshold <= cumulative_fitness) {
            selected_parent_index = candidate_index;
            return true;
        }
    }

    selected_parent_index = fallback_index;
    return true;
}

__device__ inline bool TrySelectCellularParentPairDevice(
    const float *fitness_values, const std::uint8_t *has_fitness_flags, const CellularGridShape &current_grid_shape,
    const CellularGridShape &next_grid_shape, const std::size_t child_index, DeviceRandomState &random_state,
    const ParentSelectionConfig &config, ParentPair &parent_pair) {
    if ((fitness_values == nullptr) || (has_fitness_flags == nullptr) || (child_index >= next_grid_shape.cell_count)) {
        return false;
    }

    if (!TryProjectCellIndexBetweenSquareGrids(current_grid_shape, next_grid_shape, child_index,
                                               parent_pair.first_parent_index) ||
        (has_fitness_flags[parent_pair.first_parent_index] == 0)) {
        return false;
    }

    CellularNeighborList candidate_indices{};
    if (!TryCollectCellularSecondParentCandidates(current_grid_shape, parent_pair.first_parent_index,
                                                  candidate_indices)) {
        return false;
    }

    if (config.allow_self_parenting) {
        if (candidate_indices.count >= spatial::kMaxCellularSecondParentCandidateCount) {
            return false;
        }

        candidate_indices.indices[candidate_indices.count] = parent_pair.first_parent_index;
        ++candidate_indices.count;
    }

    return TrySelectRouletteIndexFromCandidates(fitness_values, has_fitness_flags, candidate_indices, random_state,
                                                parent_pair.second_parent_index);
}

} // namespace neuroevolution::genetic_algorithm::device_selection_ops
