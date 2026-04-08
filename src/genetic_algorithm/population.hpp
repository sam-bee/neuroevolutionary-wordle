#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"

namespace neuroevolution::genetic_algorithm {

struct GeneticAlgorithmConfig {
    std::size_t elite_count = 1;
};

template <typename Genome> struct Individual {
    Genome genome{};
    float fitness = 0.0f;
    std::uint32_t evaluation_count = 0;
    bool has_fitness = false;
};

template <typename Genome, std::size_t PopulationSize> struct Population {
    static_assert(PopulationSize > 0, "Population size must be non-zero.");

    common::FixedBuffer<Individual<Genome>, PopulationSize> individuals{};
    std::size_t generation_index = 0;
};

template <std::size_t PopulationSize>
constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidGeneticAlgorithmConfig(const GeneticAlgorithmConfig &config) noexcept {
    return (config.elite_count > 0) && (config.elite_count <= PopulationSize);
}

template <typename Genome, std::size_t PopulationSize>
inline NEUROEVOLUTION_HOST_DEVICE void ClearPopulationFitness(Population<Genome, PopulationSize> &population) noexcept {
    for (std::size_t individual_index = 0; individual_index < PopulationSize; ++individual_index) {
        population.individuals[individual_index].fitness = 0.0f;
        population.individuals[individual_index].evaluation_count = 0;
        population.individuals[individual_index].has_fitness = false;
    }
}

template <typename Genome, std::size_t PopulationSize>
inline NEUROEVOLUTION_HOST_DEVICE bool TryFindBestIndividualIndex(const Population<Genome, PopulationSize> &population,
                                                                  std::size_t &best_index) noexcept {
    bool found_best = false;
    float best_fitness = 0.0f;

    for (std::size_t individual_index = 0; individual_index < PopulationSize; ++individual_index) {
        const Individual<Genome> &individual = population.individuals[individual_index];
        if (!individual.has_fitness) {
            continue;
        }

        if (!found_best || (individual.fitness > best_fitness)) {
            found_best = true;
            best_fitness = individual.fitness;
            best_index = individual_index;
        }
    }

    return found_best;
}

template <typename Genome, std::size_t PopulationSize>
inline NEUROEVOLUTION_HOST_DEVICE void BeginNextGeneration(Population<Genome, PopulationSize> &population) noexcept {
    ++population.generation_index;
    ClearPopulationFitness(population);
}

} // namespace neuroevolution::genetic_algorithm
