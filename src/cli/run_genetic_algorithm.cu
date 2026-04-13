#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

#include "genetic_algorithm/device/dynamic_runtime.hpp"
#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::genetic_algorithm::GenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::dynamic_device::ComputeDynamicGenomeStrideBytes;
using neuroevolution::genetic_algorithm::dynamic_device::DestroyDeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::dynamic_device::DeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::dynamic_device::DeviceRuntimeConfig;
using neuroevolution::genetic_algorithm::dynamic_device::DeviceRuntimeStatusCode;
using neuroevolution::genetic_algorithm::dynamic_device::DeviceRuntimeStatusCodeString;
using neuroevolution::genetic_algorithm::dynamic_device::HostPopulation;
using neuroevolution::genetic_algorithm::dynamic_device::PendingOutputEmbeddingInjection;
using neuroevolution::genetic_algorithm::dynamic_device::PopulationFitnessSummary;
using neuroevolution::genetic_algorithm::dynamic_device::PopulationSizeForGenotypeBudgetBytes;
using neuroevolution::genetic_algorithm::dynamic_device::RuntimeWordCounts;
using neuroevolution::genetic_algorithm::dynamic_device::SwapDevicePopulationBuffers;
using neuroevolution::genetic_algorithm::dynamic_device::TryAssembleNextGenerationOnDevice;
using neuroevolution::genetic_algorithm::dynamic_device::TryCreateDeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::dynamic_device::TryEvaluatePopulationFitnessOnDevice;
using neuroevolution::genetic_algorithm::dynamic_device::TryInitializeRandomHostPopulation;
using neuroevolution::genetic_algorithm::dynamic_device::TryReadDeviceRuntimeStatus;
using neuroevolution::genetic_algorithm::dynamic_device::TryReadPopulationFitnessSummaryFromDevice;
using neuroevolution::genetic_algorithm::dynamic_device::TryUploadCurrentPopulationToDevice;
using neuroevolution::training_folder::DefaultActionSpacePath;
using neuroevolution::training_folder::IsValidWordCountSchedule;
using neuroevolution::training_folder::LoadTrainingWordCatalogFromActionSpace;
using neuroevolution::training_folder::ScheduledWordCountForGeneration;
using neuroevolution::training_folder::UploadTrainingWordCatalogToDeviceConstantMemory;
using neuroevolution::training_folder::WordCountSchedule;

constexpr std::size_t kDefaultGenerationCount = 3;
constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr double kBytesPerVramGiB = 1024.0 * 1024.0 * 1024.0;

struct CliConfig {
    std::size_t generation_count = kDefaultGenerationCount;
    std::size_t population_size_ceiling = 0;
    bool population_size_was_provided = false;
    std::size_t initial_word_count = neuroevolution::training_folder::kDefaultInitialActiveWordCount;
    std::size_t word_count_step = 1;
    std::size_t word_count_step_period_generations = 1;
    double genotype_vram_gb = 0.0;
    bool genotype_vram_gb_was_provided = false;
    std::uint32_t seed = 0;
    bool seed_was_provided = false;
};

enum class ArgumentParseResult {
    kSuccess,
    kHelpRequested,
    kFailure,
};

void PrintUsage() {
    std::cout << "Usage: run_genetic_algorithm [--seed N] [--generations N] [--population-size N] "
                 "[--genotype-vram-gb F] [--initial-word-count N] [--word-count-step N] "
                 "[--word-count-step-period N]\n"
              << "If --population-size is omitted, the program does not apply an extra population ceiling.\n"
              << "If both --population-size and --genotype-vram-gb are omitted, the program uses "
              << neuroevolution::genetic_algorithm::dynamic_device::kDefaultPopulationSizeCeiling
              << " as the default starting-population target when deriving the initial genotype byte budget.\n"
              << "The shared training/action schedule defaults to initial_word_count="
              << neuroevolution::training_folder::kDefaultInitialActiveWordCount
              << ", word_count_step=1, word_count_step_period_generations=1.\n"
              << "If --genotype-vram-gb is omitted, the program uses exactly enough genotype bytes to fit the "
                 "requested initial population at the starting action count.\n"
              << "The VRAM budget flag is interpreted in binary GiB-style units (" << kBytesPerVramGiB
              << " bytes per unit).\n"
              << "If --seed is omitted, the program uses the current time in microseconds.\n";
}

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

bool TryParsePositiveReal(const char *text, double &value) {
    if ((text == nullptr) || (*text == '\0')) {
        return false;
    }

    char *end = nullptr;
    value = std::strtod(text, &end);
    return (end != nullptr) && (*end == '\0') && std::isfinite(value) && (value > 0.0);
}

std::uint32_t MakeSeedFromCurrentTimeMicroseconds() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    const std::uint64_t microsecond_count = static_cast<std::uint64_t>(microseconds);
    const std::uint32_t lower_bits = static_cast<std::uint32_t>(microsecond_count);
    const std::uint32_t upper_bits = static_cast<std::uint32_t>(microsecond_count >> 32U);
    return lower_bits ^ upper_bits;
}

ArgumentParseResult TryParseArguments(const int argc, char **argv, CliConfig &config) {
    for (int arg_index = 1; arg_index < argc; ++arg_index) {
        const std::string_view argument = argv[arg_index];

        if (argument == "--help") {
            PrintUsage();
            return ArgumentParseResult::kHelpRequested;
        }

        if ((argument == "--seed") || (argument == "--generations") || (argument == "--population-size") ||
            (argument == "--genotype-vram-gb") || (argument == "--initial-word-count") ||
            (argument == "--word-count-step") || (argument == "--word-count-step-period")) {
            if ((arg_index + 1) >= argc) {
                std::cerr << "Missing value for " << argument << '\n';
                return ArgumentParseResult::kFailure;
            }

            if (argument == "--genotype-vram-gb") {
                double parsed_value = 0.0;
                if (!TryParsePositiveReal(argv[arg_index + 1], parsed_value)) {
                    std::cerr << "Invalid numeric value for " << argument << '\n';
                    return ArgumentParseResult::kFailure;
                }

                config.genotype_vram_gb = parsed_value;
                config.genotype_vram_gb_was_provided = true;
            } else {
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
                    config.seed_was_provided = true;
                } else if (argument == "--generations") {
                    if (parsed_value == 0) {
                        std::cerr << "Generation count must be at least 1.\n";
                        return ArgumentParseResult::kFailure;
                    }

                    config.generation_count = static_cast<std::size_t>(parsed_value);
                } else if (argument == "--population-size") {
                    if (parsed_value == 0) {
                        std::cerr << "Population size must be at least 1.\n";
                        return ArgumentParseResult::kFailure;
                    }

                    config.population_size_ceiling = static_cast<std::size_t>(parsed_value);
                    config.population_size_was_provided = true;
                } else if (argument == "--initial-word-count") {
                    if (parsed_value == 0) {
                        std::cerr << "Initial word count must be at least 1.\n";
                        return ArgumentParseResult::kFailure;
                    }

                    config.initial_word_count = static_cast<std::size_t>(parsed_value);
                } else if (argument == "--word-count-step") {
                    if (parsed_value == 0) {
                        std::cerr << "Word-count step must be at least 1.\n";
                        return ArgumentParseResult::kFailure;
                    }

                    config.word_count_step = static_cast<std::size_t>(parsed_value);
                } else {
                    if (parsed_value == 0) {
                        std::cerr << "Word-count step period must be at least 1.\n";
                        return ArgumentParseResult::kFailure;
                    }

                    config.word_count_step_period_generations = static_cast<std::size_t>(parsed_value);
                }
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

PendingOutputEmbeddingInjection MakePendingOutputEmbeddingInjection(const RuntimeWordCounts &runtime_word_counts,
                                                                   const std::size_t next_scheduled_word_count) {
    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
    if (runtime_word_counts.action_space_word_count < next_scheduled_word_count) {
        pending_output_embedding_injection.enabled = true;
        pending_output_embedding_injection.first_catalog_word_index = runtime_word_counts.action_space_word_count;
        pending_output_embedding_injection.injection_count =
            next_scheduled_word_count - runtime_word_counts.action_space_word_count;
    }

    return pending_output_embedding_injection;
}

bool TryComputeGenotypeBudgetBytes(const CliConfig &cli_config, const std::size_t initial_action_count,
                                   std::size_t &budget_bytes_out) {
    if (initial_action_count == 0) {
        return false;
    }

    if (!cli_config.genotype_vram_gb_was_provided) {
        const std::size_t starting_population_target =
            cli_config.population_size_was_provided
                ? cli_config.population_size_ceiling
                : neuroevolution::genetic_algorithm::dynamic_device::kDefaultPopulationSizeCeiling;
        const std::size_t genome_stride_bytes = ComputeDynamicGenomeStrideBytes(initial_action_count);
        if ((genome_stride_bytes == 0) || (starting_population_target > (std::numeric_limits<std::size_t>::max() /
                                                                         genome_stride_bytes))) {
            return false;
        }

        budget_bytes_out = starting_population_target * genome_stride_bytes;
        return true;
    }

    const double budget_bytes = cli_config.genotype_vram_gb * kBytesPerVramGiB;
    if ((budget_bytes < 1.0) || (budget_bytes > static_cast<double>(std::numeric_limits<std::size_t>::max()))) {
        return false;
    }

    budget_bytes_out = static_cast<std::size_t>(budget_bytes);
    return budget_bytes_out > 0;
}

std::string PopulationCeilingLabel(const std::size_t population_size_ceiling) {
    return (population_size_ceiling == 0) ? "none" : std::to_string(population_size_ceiling);
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

        if (!cli_config.seed_was_provided) {
            cli_config.seed = MakeSeedFromCurrentTimeMicroseconds();
        }

        if (!SelectVisibleCudaDevice()) {
            return 1;
        }

        const auto training_data_path = DefaultActionSpacePath();
        const auto training_word_catalog = LoadTrainingWordCatalogFromActionSpace(training_data_path);
        if (!UploadTrainingWordCatalogToDeviceConstantMemory(training_word_catalog)) {
            std::cerr << "Could not upload the training-word catalog to device constant memory.\n";
            return 1;
        }

        const WordCountSchedule word_count_schedule{
            .initial_word_count = cli_config.initial_word_count,
            .word_count_step = cli_config.word_count_step,
            .word_count_step_period_generations = cli_config.word_count_step_period_generations,
        };
        if (!IsValidWordCountSchedule(word_count_schedule)) {
            std::cerr << "The configured word-count schedule is invalid.\n";
            return 1;
        }

        const std::size_t initial_active_word_count =
            ScheduledWordCountForGeneration(word_count_schedule, training_word_catalog.word_count, 0);
        if (initial_active_word_count == 0) {
            std::cerr << "The configured word-count schedule does not activate any catalog words.\n";
            return 1;
        }

        RuntimeWordCounts runtime_word_counts{};
        runtime_word_counts.training_word_count = initial_active_word_count;
        runtime_word_counts.action_space_word_count = initial_active_word_count;

        std::size_t genotype_memory_budget_bytes = 0;
        if (!TryComputeGenotypeBudgetBytes(cli_config, runtime_word_counts.action_space_word_count,
                                           genotype_memory_budget_bytes)) {
            std::cerr << "Could not derive a valid genotype VRAM budget from the command line.\n";
            return 1;
        }

        const std::size_t initial_population_size = PopulationSizeForGenotypeBudgetBytes(
            genotype_memory_budget_bytes, runtime_word_counts.action_space_word_count, cli_config.population_size_ceiling);
        if (initial_population_size == 0) {
            std::cerr << "The requested genotype VRAM budget is too small for even one starting genome.\n";
            return 1;
        }

        HostPopulation population{};
        if (!TryInitializeRandomHostPopulation(population, initial_population_size,
                                               runtime_word_counts.action_space_word_count, cli_config.seed)) {
            std::cerr << "Could not initialize the starting population.\n";
            return 1;
        }

        DeviceRuntimeConfig runtime_config{};
        runtime_config.genotype_memory_budget_bytes = genotype_memory_budget_bytes;
        runtime_config.population_size_ceiling = cli_config.population_size_ceiling;
        runtime_config.initial_action_count = runtime_word_counts.action_space_word_count;

        DeviceRuntimeBuffers buffers{};
        if (!TryCreateDeviceRuntimeBuffers(buffers, runtime_config)) {
            std::cerr << "Could not allocate device-runtime buffers.\n";
            return 1;
        }

        const GenerationAssemblyConfig assembly_config = MakeAssemblyConfig();

        if (!TryUploadCurrentPopulationToDevice(population, buffers)) {
            std::cerr << "Could not upload the initial population to device memory.\n";
            DestroyDeviceRuntimeBuffers(buffers);
            return 1;
        }

        std::cout << "Running device GA demo with initial_population=" << initial_population_size
                  << ", population_ceiling=" << PopulationCeilingLabel(runtime_config.population_size_ceiling)
                  << ", action_count=" << runtime_word_counts.action_space_word_count
                  << ", genome_stride_bytes=" << buffers.current_layout.genome_stride_bytes
                  << ", genotype_vram_budget_bytes=" << genotype_memory_budget_bytes
                  << ", genotype_vram_budget_gib="
                  << (static_cast<double>(genotype_memory_budget_bytes) / kBytesPerVramGiB)
                  << ", generations=" << cli_config.generation_count << ", seed=" << cli_config.seed
                  << ", training_word_catalog_entries=" << training_word_catalog.word_count
                  << ", configured_training_word_count=" << runtime_word_counts.training_word_count
                  << ", configured_action_space_word_count=" << runtime_word_counts.action_space_word_count
                  << ", schedule_initial_word_count=" << word_count_schedule.initial_word_count
                  << ", schedule_word_count_step=" << word_count_schedule.word_count_step
                  << ", schedule_word_count_step_period_generations="
                  << word_count_schedule.word_count_step_period_generations
                  << ", training_source=" << training_data_path.filename().string()
                  << ", training_storage=constant_memory\n";
        std::cout << std::fixed << std::setprecision(4);

        for (std::size_t generation_step = 0; generation_step < cli_config.generation_count; ++generation_step) {
            if (!TryEvaluatePopulationFitnessOnDevice(buffers, runtime_word_counts)) {
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
                      << ", average=" << summary.average_fitness << ", best_index=" << summary.best_index
                      << ", population=" << summary.population_size << ", action_count=" << summary.action_count
                      << ", genome_stride_bytes=" << buffers.current_layout.genome_stride_bytes << '\n';

            if ((generation_step + 1) == cli_config.generation_count) {
                break;
            }

            const std::uint32_t generation_seed = cli_config.seed + 2U + static_cast<std::uint32_t>(generation_step);
            const std::size_t next_scheduled_word_count = ScheduledWordCountForGeneration(
                word_count_schedule, training_word_catalog.word_count, summary.generation_index + 1);
            const PendingOutputEmbeddingInjection pending_output_embedding_injection =
                MakePendingOutputEmbeddingInjection(runtime_word_counts, next_scheduled_word_count);

            if (!TryAssembleNextGenerationOnDevice(buffers, generation_seed, assembly_config,
                                                   pending_output_embedding_injection)) {
                (void)ReportDeviceRuntimeFailure(buffers, "Next-generation assembly");
                DestroyDeviceRuntimeBuffers(buffers);
                return 1;
            }

            if (pending_output_embedding_injection.enabled) {
                std::cout << "Injecting catalog word range ["
                          << pending_output_embedding_injection.first_catalog_word_index << ", "
                          << (pending_output_embedding_injection.first_catalog_word_index +
                              pending_output_embedding_injection.injection_count - 1)
                          << "] into next generation: population=" << buffers.next_layout.active_individual_count
                          << ", action_count=" << buffers.next_layout.action_count
                          << ", genome_stride_bytes=" << buffers.next_layout.genome_stride_bytes << '\n';
            }

            SwapDevicePopulationBuffers(buffers);
            runtime_word_counts.training_word_count = buffers.current_layout.action_count;
            runtime_word_counts.action_space_word_count = buffers.current_layout.action_count;
        }

        DestroyDeviceRuntimeBuffers(buffers);
        std::cout << "GA demo finished after " << cli_config.generation_count << " generations.\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "run_genetic_algorithm failed: " << exception.what() << '\n';
        return 1;
    }
}
