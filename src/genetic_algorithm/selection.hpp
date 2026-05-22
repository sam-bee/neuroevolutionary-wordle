#pragma once

#include <cstddef>
#include <limits>
#include <random>

#include "common/fixed_buffer.hpp"
#include "genetic_algorithm/population.hpp"
#include "genetic_algorithm/spatial/grid.hpp"

namespace neuroevolution::genetic_algorithm {

using SelectionRandomEngine = std::mt19937;

constexpr std::size_t kNoIndividualIndex = std::numeric_limits<std::size_t>::max();

struct ParentSelectionConfig {
    std::size_t tournament_size = 3;
    bool allow_self_parenting = false;
    std::size_t cellular_breeding_radius = spatial::kCellularBreedingRadius;
};

struct ParentPair {
    std::size_t first_parent_index = 0;
    std::size_t second_parent_index = 0;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidParentSelectionConfig(const ParentSelectionConfig &config) noexcept {
    return (config.tournament_size > 0) && (config.cellular_breeding_radius > 0);
}

namespace detail {

template <typename Genome, std::size_t PopulationSize>
inline std::size_t
CollectSelectableIndividualIndices(const Population<Genome, PopulationSize> &population,
                                   common::FixedBuffer<std::size_t, PopulationSize> &selectable_indices,
                                   const std::size_t excluded_index) noexcept {
    std::size_t selectable_count = 0;

    for (std::size_t individual_index = 0; individual_index < PopulationSize; ++individual_index) {
        const Individual<Genome> &individual = population.individuals[individual_index];
        if (!individual.has_fitness || (individual_index == excluded_index)) {
            continue;
        }

        selectable_indices[selectable_count] = individual_index;
        ++selectable_count;
    }

    return selectable_count;
}

template <std::size_t PopulationSize>
inline void ShufflePrefix(common::FixedBuffer<std::size_t, PopulationSize> &indices, const std::size_t selectable_count,
                          const std::size_t prefix_size, SelectionRandomEngine &random_engine) {
    for (std::size_t prefix_index = 0; prefix_index < prefix_size; ++prefix_index) {
        std::uniform_int_distribution<std::size_t> distribution(prefix_index, selectable_count - 1);
        const std::size_t swap_index = distribution(random_engine);

        const std::size_t temporary = indices[prefix_index];
        indices[prefix_index] = indices[swap_index];
        indices[swap_index] = temporary;
    }
}

} // namespace detail

template <typename Genome, std::size_t PopulationSize>
inline bool TrySelectParentIndex(const Population<Genome, PopulationSize> &population,
                                 SelectionRandomEngine &random_engine, std::size_t &selected_parent_index,
                                 const ParentSelectionConfig &config = {},
                                 const std::size_t excluded_index = kNoIndividualIndex) {
    if (!IsValidParentSelectionConfig(config)) {
        return false;
    }

    common::FixedBuffer<std::size_t, PopulationSize> selectable_indices{};
    const std::size_t selectable_count =
        detail::CollectSelectableIndividualIndices(population, selectable_indices, excluded_index);
    if (selectable_count == 0) {
        return false;
    }

    const std::size_t tournament_size =
        (config.tournament_size < selectable_count) ? config.tournament_size : selectable_count;

    detail::ShufflePrefix(selectable_indices, selectable_count, tournament_size, random_engine);

    selected_parent_index = selectable_indices[0];
    float best_fitness = population.individuals[selected_parent_index].fitness;

    for (std::size_t candidate_index = 1; candidate_index < tournament_size; ++candidate_index) {
        const std::size_t individual_index = selectable_indices[candidate_index];
        const float candidate_fitness = population.individuals[individual_index].fitness;
        if (candidate_fitness > best_fitness) {
            selected_parent_index = individual_index;
            best_fitness = candidate_fitness;
        }
    }

    return true;
}

template <typename Genome, std::size_t PopulationSize>
inline bool TrySelectParentPair(const Population<Genome, PopulationSize> &population,
                                SelectionRandomEngine &random_engine, ParentPair &parent_pair,
                                const ParentSelectionConfig &config = {}) {
    if (!TrySelectParentIndex(population, random_engine, parent_pair.first_parent_index, config)) {
        return false;
    }

    const std::size_t excluded_index =
        config.allow_self_parenting ? kNoIndividualIndex : parent_pair.first_parent_index;
    return TrySelectParentIndex(population, random_engine, parent_pair.second_parent_index, config, excluded_index);
}

} // namespace neuroevolution::genetic_algorithm
