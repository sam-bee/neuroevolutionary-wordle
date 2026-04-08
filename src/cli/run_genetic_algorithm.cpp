#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

#include "genetic_algorithm/genetic_algorithm.hpp"

namespace {

using neuroevolution::genetic_algorithm::EvaluatePopulationFitness;
using neuroevolution::genetic_algorithm::FitnessEvaluationConfig;
using neuroevolution::genetic_algorithm::FitnessRandomEngine;
using neuroevolution::genetic_algorithm::GenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::GenerationAssemblyRandomEngine;
using neuroevolution::genetic_algorithm::InitializePopulation;
using neuroevolution::genetic_algorithm::ModelGenome;
using neuroevolution::genetic_algorithm::Population;
using neuroevolution::genetic_algorithm::PopulationInitializationRandomEngine;
using neuroevolution::genetic_algorithm::TryAssembleNextGeneration;
using neuroevolution::genetic_algorithm::TryFindBestIndividualIndex;

constexpr std::size_t kDemoActionCount = 8;
constexpr std::size_t kDemoPopulationSize = 6;
constexpr std::size_t kDefaultGenerationCount = 3;
constexpr std::uint32_t kDefaultSeed = 12345;

struct CliConfig {
    std::size_t generation_count = kDefaultGenerationCount;
    std::uint32_t seed = kDefaultSeed;
};

enum class ArgumentParseResult {
    kSuccess,
    kHelpRequested,
    kFailure,
};

void PrintUsage() { std::cout << "Usage: run_genetic_algorithm [--seed N] [--generations N]\n"; }

bool TryParseUnsigned(const char *text, std::uint64_t &value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }

    value = 0;
    for (const char *cursor = text; *cursor != '\0'; ++cursor) {
        if ((*cursor < '0') || (*cursor > '9')) {
            return false;
        }

        value = (value * 10) + static_cast<std::uint64_t>(*cursor - '0');
    }

    return true;
}

ArgumentParseResult TryParseArguments(const int argc, char **argv, CliConfig &config) {
    for (int arg_index = 1; arg_index < argc; ++arg_index) {
        const std::string_view argument = argv[arg_index];

        if (argument == "--help") {
            PrintUsage();
            return ArgumentParseResult::kHelpRequested;
        }

        if ((argument == "--seed") || (argument == "--generations")) {
            if ((arg_index + 1) >= argc) {
                std::cerr << "Missing value for " << argument << '\n';
                return ArgumentParseResult::kFailure;
            }

            std::uint64_t parsed_value = 0;
            if (!TryParseUnsigned(argv[arg_index + 1], parsed_value)) {
                std::cerr << "Invalid numeric value for " << argument << '\n';
                return ArgumentParseResult::kFailure;
            }

            if (argument == "--seed") {
                if (parsed_value > std::numeric_limits<std::uint32_t>::max()) {
                    std::cerr << "Seed is out of range for uint32.\n";
                    return ArgumentParseResult::kFailure;
                }

                config.seed = static_cast<std::uint32_t>(parsed_value);
            } else {
                if (parsed_value == 0) {
                    std::cerr << "Generation count must be at least 1.\n";
                    return ArgumentParseResult::kFailure;
                }

                config.generation_count = static_cast<std::size_t>(parsed_value);
            }

            ++arg_index;
            continue;
        }

        std::cerr << "Unknown argument: " << argument << '\n';
        return ArgumentParseResult::kFailure;
    }

    return ArgumentParseResult::kSuccess;
}

float ComputeAverageFitness(const Population<ModelGenome<kDemoActionCount>, kDemoPopulationSize> &population) {
    float sum = 0.0f;
    for (std::size_t individual_index = 0; individual_index < kDemoPopulationSize; ++individual_index) {
        sum += population.individuals[individual_index].fitness;
    }

    return sum / static_cast<float>(kDemoPopulationSize);
}

} // namespace

int main(int argc, char **argv) {
    try {
        CliConfig cli_config{};
        const ArgumentParseResult parse_result = TryParseArguments(argc, argv, cli_config);
        if (parse_result == ArgumentParseResult::kHelpRequested) {
            return 0;
        }

        if (parse_result == ArgumentParseResult::kFailure) {
            return 1;
        }

        PopulationInitializationRandomEngine initialization_random_engine(cli_config.seed);
        FitnessRandomEngine fitness_random_engine(cli_config.seed + 1U);
        GenerationAssemblyRandomEngine assembly_random_engine(cli_config.seed + 2U);

        auto population = InitializePopulation<kDemoActionCount, kDemoPopulationSize>(initialization_random_engine);

        FitnessEvaluationConfig fitness_config{};
        fitness_config.minimum_fitness = 0.0f;
        fitness_config.maximum_fitness = 1.0f;

        GenerationAssemblyConfig assembly_config{};
        assembly_config.genetic_algorithm.elite_count = 1;
        assembly_config.parent_selection.tournament_size = 3;
        assembly_config.parent_selection.allow_self_parenting = false;
        assembly_config.breeding.first_parent_probability = 0.5f;
        assembly_config.mutation.mutation_probability = 0.02f;
        assembly_config.mutation.mutation_sigma = 0.05f;

        std::cout << "Running placeholder GA demo with population=" << kDemoPopulationSize
                  << ", action_count=" << kDemoActionCount << ", generations=" << cli_config.generation_count
                  << ", seed=" << cli_config.seed << '\n';
        std::cout << std::fixed << std::setprecision(4);

        for (std::size_t generation_step = 0; generation_step < cli_config.generation_count; ++generation_step) {
            EvaluatePopulationFitness(population, fitness_random_engine, fitness_config);

            std::size_t best_index = 0;
            if (!TryFindBestIndividualIndex(population, best_index)) {
                std::cerr << "Failed to find a best individual after evaluation.\n";
                return 1;
            }

            const float best_fitness = population.individuals[best_index].fitness;
            const float average_fitness = ComputeAverageFitness(population);

            std::cout << "Generation " << population.generation_index << ": best=" << best_fitness
                      << ", average=" << average_fitness << ", best_index=" << best_index << '\n';

            if ((generation_step + 1) == cli_config.generation_count) {
                break;
            }

            Population<ModelGenome<kDemoActionCount>, kDemoPopulationSize> next_population{};
            if (!TryAssembleNextGeneration(population, next_population, assembly_random_engine, assembly_config)) {
                std::cerr << "Failed to assemble next generation.\n";
                return 1;
            }

            population = next_population;
        }

        std::cout << "GA demo finished after " << cli_config.generation_count << " generations.\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "run_genetic_algorithm failed: " << exception.what() << '\n';
        return 1;
    }
}
