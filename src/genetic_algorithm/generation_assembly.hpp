#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>

#include "common/fixed_buffer.hpp"
#include "genetic_algorithm/breeding.hpp"
#include "genetic_algorithm/mutation.hpp"
#include "genetic_algorithm/selection.hpp"

namespace neuroevolution::genetic_algorithm {

using GenerationAssemblyRandomEngine = std::mt19937;

struct GenerationAssemblyConfig {
    GeneticAlgorithmConfig genetic_algorithm{};
    ParentSelectionConfig parent_selection{};
    BreedingConfig breeding{};
    MutationConfig mutation{};
};

template <std::size_t PopulationSize>
constexpr bool IsValidGenerationAssemblyConfig(const GenerationAssemblyConfig &config) noexcept {
    return IsValidGeneticAlgorithmConfig<PopulationSize>(config.genetic_algorithm) &&
           IsValidParentSelectionConfig(config.parent_selection) && IsValidBreedingConfig(config.breeding) &&
           IsValidMutationConfig(config.mutation);
}

namespace detail {

template <typename Genome>
inline void WriteGenomeToUnevaluatedIndividual(const Genome &genome, Individual<Genome> &individual) {
    individual.genome = genome;
    individual.fitness = 0.0f;
    individual.evaluation_count = 0;
    individual.has_fitness = false;
}

template <typename Genome, std::size_t PopulationSize>
inline bool TryCollectTopIndividualIndices(const Population<Genome, PopulationSize> &population,
                                           const std::size_t top_count,
                                           common::FixedBuffer<std::size_t, PopulationSize> &top_indices) {
    common::FixedBuffer<bool, PopulationSize> selected_flags{};

    for (std::size_t top_index = 0; top_index < top_count; ++top_index) {
        bool found_candidate = false;
        std::size_t best_individual_index = 0;
        float best_fitness = 0.0f;

        for (std::size_t individual_index = 0; individual_index < PopulationSize; ++individual_index) {
            const Individual<Genome> &individual = population.individuals[individual_index];
            if (!individual.has_fitness || selected_flags[individual_index]) {
                continue;
            }

            if (!found_candidate || (individual.fitness > best_fitness)) {
                found_candidate = true;
                best_individual_index = individual_index;
                best_fitness = individual.fitness;
            }
        }

        if (!found_candidate) {
            return false;
        }

        selected_flags[best_individual_index] = true;
        top_indices[top_index] = best_individual_index;
    }

    return true;
}

} // namespace detail

template <std::size_t ActionCount, std::size_t PopulationSize>
inline bool TryAssembleNextGeneration(const Population<ModelGenome<ActionCount>, PopulationSize> &current_population,
                                      Population<ModelGenome<ActionCount>, PopulationSize> &next_population,
                                      GenerationAssemblyRandomEngine &random_engine,
                                      const GenerationAssemblyConfig &config = {}) {
    if (!IsValidGenerationAssemblyConfig<PopulationSize>(config)) {
        return false;
    }

    common::FixedBuffer<std::size_t, PopulationSize> elite_indices{};
    if (!detail::TryCollectTopIndividualIndices(current_population, config.genetic_algorithm.elite_count,
                                                elite_indices)) {
        return false;
    }

    next_population = {};
    next_population.generation_index = current_population.generation_index + 1;

    for (std::size_t elite_slot = 0; elite_slot < config.genetic_algorithm.elite_count; ++elite_slot) {
        const std::size_t elite_index = elite_indices[elite_slot];
        detail::WriteGenomeToUnevaluatedIndividual(current_population.individuals[elite_index].genome,
                                                   next_population.individuals[elite_slot]);
    }

    for (std::size_t slot_index = config.genetic_algorithm.elite_count; slot_index < PopulationSize; ++slot_index) {
        ParentPair parent_pair{};
        if (!TrySelectParentPair(current_population, random_engine, parent_pair, config.parent_selection)) {
            return false;
        }

        ModelGenome<ActionCount> child_genome{};
        if (!TryBreedChildGenomeFromPopulation(current_population, parent_pair, child_genome, random_engine,
                                               config.breeding)) {
            return false;
        }

        if (!TryMutateGenome(child_genome, random_engine, config.mutation)) {
            return false;
        }

        detail::WriteGenomeToUnevaluatedIndividual(child_genome, next_population.individuals[slot_index]);
    }

    return true;
}

template <std::size_t ActionCount, std::size_t PopulationSize>
inline Population<ModelGenome<ActionCount>, PopulationSize>
AssembleNextGeneration(const Population<ModelGenome<ActionCount>, PopulationSize> &current_population,
                       GenerationAssemblyRandomEngine &random_engine, const GenerationAssemblyConfig &config = {}) {
    Population<ModelGenome<ActionCount>, PopulationSize> next_population{};
    if (!TryAssembleNextGeneration(current_population, next_population, random_engine, config)) {
        throw std::invalid_argument("Next-generation assembly requires valid configs and enough fitted parents.");
    }

    return next_population;
}

} // namespace neuroevolution::genetic_algorithm
