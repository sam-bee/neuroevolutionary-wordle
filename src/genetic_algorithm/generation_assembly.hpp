#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>

#include "genetic_algorithm/breeding.hpp"
#include "genetic_algorithm/mutation.hpp"
#include "genetic_algorithm/selection.hpp"

namespace neuroevolution::genetic_algorithm {

using GenerationAssemblyRandomEngine = std::mt19937;

struct GenerationAssemblyConfig {
    ParentSelectionConfig parent_selection{};
    BreedingConfig breeding{};
    MutationConfig mutation{};
};

constexpr bool IsValidGenerationAssemblyConfig(const GenerationAssemblyConfig &config) noexcept {
    return IsValidParentSelectionConfig(config.parent_selection) && IsValidBreedingConfig(config.breeding) &&
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

} // namespace detail

template <std::size_t ActionCount, std::size_t PopulationSize>
inline bool TryAssembleNextGeneration(const Population<ModelGenome<ActionCount>, PopulationSize> &current_population,
                                      Population<ModelGenome<ActionCount>, PopulationSize> &next_population,
                                      GenerationAssemblyRandomEngine &random_engine,
                                      const GenerationAssemblyConfig &config = {}) {
    if (!IsValidGenerationAssemblyConfig(config)) {
        return false;
    }

    next_population = {};
    next_population.generation_index = current_population.generation_index + 1;

    for (std::size_t slot_index = 0; slot_index < PopulationSize; ++slot_index) {
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
