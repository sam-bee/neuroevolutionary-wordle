#pragma once

#include <random>
#include <stdexcept>

#include "genetic_algorithm/genome.hpp"
#include "genetic_algorithm/population.hpp"
#include "model/initialization/parameter_initialization.hpp"

namespace neuroevolution::genetic_algorithm {

using PopulationInitializationRandomEngine = std::mt19937;

struct PopulationInitializationConfig {
    model::initialization::ParameterInitializationConfig parameter_initialization{};
};

constexpr bool IsValidPopulationInitializationConfig(const PopulationInitializationConfig &config) noexcept {
    return model::initialization::IsValidParameterInitializationConfig(config.parameter_initialization);
}

template <std::size_t ActionCount>
inline bool TryInitializeRandomGenome(ModelGenome<ActionCount> &genome,
                                      PopulationInitializationRandomEngine &random_engine,
                                      const PopulationInitializationConfig &config = {}) {
    if (!IsValidPopulationInitializationConfig(config)) {
        return false;
    }

    model::initialization::InitializeRandomPolicyModelParameters(genome.policy_model, random_engine,
                                                                 config.parameter_initialization);
    model::initialization::InitializeRandomOutputEmbeddingTrainableTails(
        genome.output_embedding.trainable_tails, random_engine, config.parameter_initialization);
    return true;
}

template <std::size_t ActionCount, std::size_t PopulationSize>
inline bool TryInitializePopulation(Population<ModelGenome<ActionCount>, PopulationSize> &population,
                                    PopulationInitializationRandomEngine &random_engine,
                                    const PopulationInitializationConfig &config = {}) {
    if (!IsValidPopulationInitializationConfig(config)) {
        return false;
    }

    population = {};

    for (std::size_t individual_index = 0; individual_index < PopulationSize; ++individual_index) {
        if (!TryInitializeRandomGenome(population.individuals[individual_index].genome, random_engine, config)) {
            return false;
        }
    }

    return true;
}

template <std::size_t ActionCount, std::size_t PopulationSize>
inline Population<ModelGenome<ActionCount>, PopulationSize>
InitializePopulation(PopulationInitializationRandomEngine &random_engine,
                     const PopulationInitializationConfig &config = {}) {
    Population<ModelGenome<ActionCount>, PopulationSize> population{};
    if (!TryInitializePopulation(population, random_engine, config)) {
        throw std::invalid_argument("Population initialization requires a valid parameter-initialization config.");
    }

    return population;
}

} // namespace neuroevolution::genetic_algorithm
