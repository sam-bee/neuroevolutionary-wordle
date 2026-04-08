#pragma once

#include <random>

#include "genetic_algorithm/population.hpp"

namespace neuroevolution::genetic_algorithm {

using FitnessRandomEngine = std::mt19937;

struct FitnessEvaluationConfig {
    float minimum_fitness = 0.0f;
    float maximum_fitness = 1.0f;
};

constexpr bool IsValidFitnessEvaluationConfig(const FitnessEvaluationConfig &config) noexcept {
    return config.minimum_fitness <= config.maximum_fitness;
}

inline float SamplePlaceholderFitness(FitnessRandomEngine &random_engine, const FitnessEvaluationConfig &config = {}) {
    std::uniform_real_distribution<float> distribution(config.minimum_fitness, config.maximum_fitness);
    return distribution(random_engine);
}

template <typename Genome>
inline void EvaluateIndividualFitness(Individual<Genome> &individual, FitnessRandomEngine &random_engine,
                                      const FitnessEvaluationConfig &config = {}) {
    individual.fitness = SamplePlaceholderFitness(random_engine, config);
    ++individual.evaluation_count;
    individual.has_fitness = true;
}

template <typename Genome, std::size_t PopulationSize>
inline void EvaluatePopulationFitness(Population<Genome, PopulationSize> &population,
                                      FitnessRandomEngine &random_engine, const FitnessEvaluationConfig &config = {}) {
    for (std::size_t individual_index = 0; individual_index < PopulationSize; ++individual_index) {
        EvaluateIndividualFitness(population.individuals[individual_index], random_engine, config);
    }
}

} // namespace neuroevolution::genetic_algorithm
