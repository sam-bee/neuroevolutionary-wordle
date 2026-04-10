#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

#include "genetic_algorithm/device/device_runtime.hpp"
#include "genetic_algorithm/genetic_algorithm.hpp"
#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::genetic_algorithm::GenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::InitializePopulation;
using neuroevolution::genetic_algorithm::PopulationInitializationRandomEngine;
using neuroevolution::genetic_algorithm::device::DestroyDeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::device::DeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::device::DeviceRuntimeStatusCode;
using neuroevolution::genetic_algorithm::device::DeviceRuntimeStatusCodeString;
using neuroevolution::genetic_algorithm::device::PopulationFitnessSummary;
using neuroevolution::genetic_algorithm::device::SwapDevicePopulationBuffers;
using neuroevolution::genetic_algorithm::device::TryAssembleNextGenerationOnDevice;
using neuroevolution::genetic_algorithm::device::TryCreateDeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::device::TryEvaluatePopulationFitnessOnDevice;
using neuroevolution::genetic_algorithm::device::TryReadDeviceRuntimeStatus;
using neuroevolution::genetic_algorithm::device::TryReadPopulationFitnessSummaryFromDevice;
using neuroevolution::genetic_algorithm::device::TryUploadCurrentPopulationToDevice;
using neuroevolution::training_folder::DefaultActionSpacePath;
using neuroevolution::training_folder::LoadInitialTrainingDataShardFromActionSpace;
using neuroevolution::training_folder::UploadTrainingDataShardToDeviceConstantMemory;

constexpr std::size_t kDefaultGenerationCount = 3;
constexpr std::uint32_t kDefaultSeed = 12345;
constexpr int kSelectedVisibleDeviceIndex = 0;

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

bool CheckCuda(const cudaError_t error, const std::string_view action) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA failure during " << action << ": " << cudaGetErrorString(error) << '\n';
        return false;
    }

    return true;
}

bool SelectVisibleCudaDevice() {
    int device_count = 0;
    if (!CheckCuda(cudaGetDeviceCount(&device_count), "querying visible CUDA device count")) {
        return false;
    }

    if (device_count <= kSelectedVisibleDeviceIndex) {
        std::cerr << "No visible CUDA device is available at logical index " << kSelectedVisibleDeviceIndex << '\n';
        return false;
    }

    return CheckCuda(cudaSetDevice(kSelectedVisibleDeviceIndex), "selecting visible CUDA device");
}

bool TryParseUnsigned(const char *text, std::uint64_t &value) {
    if ((text == nullptr) || (*text == '\0')) {
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

bool ReportDeviceRuntimeFailure(const DeviceRuntimeBuffers &buffers, const std::string_view action) {
    DeviceRuntimeStatusCode status_code = DeviceRuntimeStatusCode::kCudaFailure;
    if (TryReadDeviceRuntimeStatus(buffers, status_code)) {
        std::cerr << action << " failed: " << DeviceRuntimeStatusCodeString(status_code) << '\n';
    } else {
        std::cerr << action << " failed and the device status could not be read.\n";
    }

    return false;
}

GenerationAssemblyConfig MakeAssemblyConfig() {
    GenerationAssemblyConfig config{};
    config.genetic_algorithm.elite_count = 1;
    config.parent_selection.tournament_size = 3;
    config.parent_selection.allow_self_parenting = false;
    config.breeding.first_parent_probability = 0.5f;
    config.mutation.mutation_probability = 0.02f;
    config.mutation.mutation_sigma = 0.05f;
    return config;
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

        if (!SelectVisibleCudaDevice()) {
            return 1;
        }

        const auto training_data_path = DefaultActionSpacePath();
        const auto training_shard = LoadInitialTrainingDataShardFromActionSpace(training_data_path);
        if (!UploadTrainingDataShardToDeviceConstantMemory(training_shard)) {
            std::cerr << "Could not upload the training-data shard to device constant memory.\n";
            return 1;
        }

        PopulationInitializationRandomEngine initialization_random_engine(cli_config.seed);
        auto population = InitializePopulation<neuroevolution::genetic_algorithm::device::kDeviceActionCount,
                                               neuroevolution::genetic_algorithm::device::kDevicePopulationSize>(
            initialization_random_engine);

        DeviceRuntimeBuffers buffers{};
        if (!TryCreateDeviceRuntimeBuffers(buffers)) {
            std::cerr << "Could not allocate device-runtime buffers.\n";
            return 1;
        }

        const GenerationAssemblyConfig assembly_config = MakeAssemblyConfig();

        if (!TryUploadCurrentPopulationToDevice(population, buffers)) {
            std::cerr << "Could not upload the initial population to device memory.\n";
            DestroyDeviceRuntimeBuffers(buffers);
            return 1;
        }

        std::cout << "Running device GA demo with population="
                  << neuroevolution::genetic_algorithm::device::kDevicePopulationSize
                  << ", action_count=" << neuroevolution::genetic_algorithm::device::kDeviceActionCount
                  << ", generations=" << cli_config.generation_count << ", seed=" << cli_config.seed
                  << ", training_shard_entries=" << training_shard.entry_count
                  << ", training_source=" << training_data_path.filename().string()
                  << ", training_storage=constant_memory\n";
        std::cout << std::fixed << std::setprecision(4);

        for (std::size_t generation_step = 0; generation_step < cli_config.generation_count; ++generation_step) {
            if (!TryEvaluatePopulationFitnessOnDevice(buffers)) {
                (void)ReportDeviceRuntimeFailure(buffers, "Population fitness evaluation");
                DestroyDeviceRuntimeBuffers(buffers);
                return 1;
            }

            PopulationFitnessSummary summary{};
            if (!TryReadPopulationFitnessSummaryFromDevice(buffers, summary)) {
                std::cerr << "Could not read the population fitness summary back from device memory.\n";
                DestroyDeviceRuntimeBuffers(buffers);
                return 1;
            }

            std::cout << "Generation " << summary.generation_index << ": best=" << summary.best_fitness
                      << ", average=" << summary.average_fitness << ", best_index=" << summary.best_index << '\n';

            if ((generation_step + 1) == cli_config.generation_count) {
                break;
            }

            const std::uint32_t generation_seed = cli_config.seed + 2U + static_cast<std::uint32_t>(generation_step);
            if (!TryAssembleNextGenerationOnDevice(buffers, generation_seed, assembly_config)) {
                (void)ReportDeviceRuntimeFailure(buffers, "Next-generation assembly");
                DestroyDeviceRuntimeBuffers(buffers);
                return 1;
            }

            SwapDevicePopulationBuffers(buffers);
        }

        DestroyDeviceRuntimeBuffers(buffers);
        std::cout << "GA demo finished after " << cli_config.generation_count << " generations.\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "run_genetic_algorithm failed: " << exception.what() << '\n';
        return 1;
    }
}
