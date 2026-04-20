#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

#include "common/progress_log.hpp"
#include "genetic_algorithm/device/slab_runtime.hpp"
#include "genetic_algorithm/genotype_slab/slab_allocator.hpp"
#include "genetic_algorithm/spatial/grid.hpp"
#include "model_artifact/winner_artifact.hpp"
#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::common::FormatCurrentLocalTimestamp;
using neuroevolution::genetic_algorithm::GenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::genotype_slab::ComputeSlabSlotStrideBytes;
using neuroevolution::genetic_algorithm::genotype_slab::SlabSlotCountForByteBudget;
using neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers;
using neuroevolution::genetic_algorithm::slab_device::DeviceSlabBootstrapConfig;
using neuroevolution::genetic_algorithm::slab_device::DeviceSlabGARuntimeBuffers;
using neuroevolution::genetic_algorithm::slab_device::DeviceSlabGARuntimeConfig;
using neuroevolution::genetic_algorithm::slab_device::DeviceSlabGARuntimeStatusCode;
using neuroevolution::genetic_algorithm::slab_device::DeviceSlabGARuntimeStatusCodeString;
using neuroevolution::genetic_algorithm::slab_device::PendingOutputEmbeddingInjection;
using neuroevolution::genetic_algorithm::slab_device::PopulationFitnessSummary;
using neuroevolution::genetic_algorithm::slab_device::RuntimeWordCounts;
using neuroevolution::genetic_algorithm::slab_device::TryAdvanceGenerationOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryBootstrapRandomCurrentGenerationOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryCreateDeviceSlabGARuntimeBuffers;
using neuroevolution::genetic_algorithm::slab_device::TryDownloadSlabSlotBytesFromDevice;
using neuroevolution::genetic_algorithm::slab_device::TryEvaluateCurrentGenerationFitnessOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryReadDeviceSlabGARuntimeStatus;
using neuroevolution::genetic_algorithm::slab_device::TryReadPopulationFitnessSummaryFromDevice;
using neuroevolution::genetic_algorithm::spatial::CellularGridShape;
using neuroevolution::genetic_algorithm::spatial::FloorSquarePopulationSize;
using neuroevolution::genetic_algorithm::spatial::TryMakeCellularGridShape;
using neuroevolution::model_artifact::TryWriteWinnerArtifact;
using neuroevolution::model_artifact::WinnerArtifactMetadata;
using neuroevolution::model_artifact::WinnerArtifactPaths;
using neuroevolution::training_folder::DefaultActionSpacePath;
using neuroevolution::training_folder::IsValidWordCountSchedule;
using neuroevolution::training_folder::LoadTrainingWordCatalogFromActionSpace;
using neuroevolution::training_folder::ScheduledWordCountForGeneration;
using neuroevolution::training_folder::UploadTrainingWordCatalogToDeviceConstantMemory;
using neuroevolution::training_folder::WordCountSchedule;

constexpr std::size_t kDefaultGenerationCount = 3;
constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr double kBytesPerVramGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kDefaultGenotypeSlabBudgetGiB = 6.0;
constexpr double kDefaultBufferToGenerationRatio = 1.5;

struct CliConfig {
    std::size_t generation_count = kDefaultGenerationCount;
    std::size_t population_size_ceiling = 0;
    bool population_size_was_provided = false;
    std::size_t initial_word_count = neuroevolution::training_folder::kDefaultInitialActiveWordCount;
    std::size_t word_count_step = 0;
    std::size_t word_count_step_period_generations = 1;
    std::size_t shard_radius_growth_period_generations =
        neuroevolution::training_folder::kDefaultShardRadiusGrowthPeriodGenerations;
    double genotype_vram_gb = 0.0;
    bool genotype_vram_gb_was_provided = false;
    double generation_vram_gb = 0.0;
    bool generation_vram_gb_was_provided = false;
    std::uint32_t seed = 0;
    bool seed_was_provided = false;
    bool verbose = false;
};

enum class ArgumentParseResult {
    kSuccess,
    kHelpRequested,
    kFailure,
};

void PrintUsage() {
    std::cout << "Usage: run_genetic_algorithm [--verbose] [--seed N] [--generations N] [--population-size N] "
                 "[--genotype-vram-gb F] [--generation-vram-gb F] [--initial-word-count N] [--word-count-step N] "
                 "[--word-count-step-period N] [--shard-radius-growth-period N]\n"
              << "If --population-size is omitted, the program does not apply an extra population ceiling.\n"
              << "The shared training/action schedule defaults to initial_word_count="
              << neuroevolution::training_folder::kDefaultInitialActiveWordCount
              << ", word_count_step=0, word_count_step_period_generations=1.\n"
              << "Spatial training-data shards grow their evaluation radius every "
              << neuroevolution::training_folder::kDefaultShardRadiusGrowthPeriodGenerations
              << " generations by default.\n"
              << "Positive word-count growth is handled by slab compaction/repacking, so later generations may "
                 "shrink population size as the output embedding grows.\n"
              << "The startup population is floored to the largest square number that fits the configured budget so "
                 "it can be laid out as a square cellular grid.\n"
              << "If --genotype-vram-gb is omitted, the program uses a default whole-slab budget of "
              << kDefaultGenotypeSlabBudgetGiB << " GiB.\n"
              << "If --generation-vram-gb is omitted, the program derives a single-generation budget from the whole "
                 "slab budget using a default slab-to-generation ratio of "
              << kDefaultBufferToGenerationRatio
              << ". If --population-size is provided without --generation-vram-gb, the initial generation budget is "
                 "sized to that population at the starting action count.\n"
              << "The VRAM budget flag is interpreted in binary GiB-style units (" << kBytesPerVramGiB
              << " bytes per unit).\n"
              << "If --seed is omitted, the program uses the current time in microseconds.\n"
              << "--verbose prints timestamped stage progress during generation advancement, including "
                 "slot-growth compaction/repacking.\n";
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

        if (argument == "--verbose") {
            config.verbose = true;
            continue;
        }

        if ((argument == "--seed") || (argument == "--generations") || (argument == "--population-size") ||
            (argument == "--genotype-vram-gb") || (argument == "--generation-vram-gb") ||
            (argument == "--initial-word-count") || (argument == "--word-count-step") ||
            (argument == "--word-count-step-period") || (argument == "--shard-radius-growth-period")) {
            if ((arg_index + 1) >= argc) {
                std::cerr << "Missing value for " << argument << '\n';
                return ArgumentParseResult::kFailure;
            }

            if ((argument == "--genotype-vram-gb") || (argument == "--generation-vram-gb")) {
                double parsed_value = 0.0;
                if (!TryParsePositiveReal(argv[arg_index + 1], parsed_value)) {
                    std::cerr << "Invalid numeric value for " << argument << '\n';
                    return ArgumentParseResult::kFailure;
                }

                if (argument == "--genotype-vram-gb") {
                    config.genotype_vram_gb = parsed_value;
                    config.genotype_vram_gb_was_provided = true;
                } else {
                    config.generation_vram_gb = parsed_value;
                    config.generation_vram_gb_was_provided = true;
                }
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
                    config.word_count_step = static_cast<std::size_t>(parsed_value);
                } else {
                    if (parsed_value == 0) {
                        std::cerr << ((argument == "--word-count-step-period")
                                          ? "Word-count step period must be at least 1.\n"
                                          : "Shard radius growth period must be at least 1.\n");
                        return ArgumentParseResult::kFailure;
                    }

                    if (argument == "--word-count-step-period") {
                        config.word_count_step_period_generations = static_cast<std::size_t>(parsed_value);
                    } else {
                        config.shard_radius_growth_period_generations = static_cast<std::size_t>(parsed_value);
                    }
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

bool ReportDeviceSlabRuntimeFailure(const DeviceSlabGARuntimeBuffers &buffers, const std::string_view action) {
    DeviceSlabGARuntimeStatusCode status_code = DeviceSlabGARuntimeStatusCode::kCudaFailure;
    if (TryReadDeviceSlabGARuntimeStatus(buffers, status_code)) {
        std::cerr << action << " failed: " << DeviceSlabGARuntimeStatusCodeString(status_code) << '\n';
    } else {
        std::cerr << action << " failed and the device status could not be read.\n";
    }

    return false;
}

GenerationAssemblyConfig MakeAssemblyConfig() {
    GenerationAssemblyConfig config{};
    config.parent_selection.tournament_size = 3;
    config.parent_selection.allow_self_parenting = false;
    config.breeding.first_parent_probability = 0.5f;
    config.mutation.mutation_probability = 0.02f;
    config.mutation.mutation_sigma = 0.05f;
    return config;
}

bool TryComputeVramBudgetBytesFromGiB(const double gib_value, std::size_t &budget_bytes_out) {
    budget_bytes_out = 0;
    const double budget_bytes = gib_value * kBytesPerVramGiB;
    if ((budget_bytes < 1.0) || (budget_bytes > static_cast<double>(std::numeric_limits<std::size_t>::max()))) {
        return false;
    }

    budget_bytes_out = static_cast<std::size_t>(budget_bytes);
    return budget_bytes_out > 0;
}

bool TryComputeTotalGenotypeBudgetBytes(const CliConfig &cli_config, std::size_t &budget_bytes_out) {
    const double configured_budget_gib =
        cli_config.genotype_vram_gb_was_provided ? cli_config.genotype_vram_gb : kDefaultGenotypeSlabBudgetGiB;
    return TryComputeVramBudgetBytesFromGiB(configured_budget_gib, budget_bytes_out);
}

bool TryComputeGenerationBudgetBytes(const CliConfig &cli_config, const std::size_t initial_action_count,
                                     const std::size_t total_budget_bytes, std::size_t &budget_bytes_out) {
    budget_bytes_out = 0;
    if ((initial_action_count == 0) || (total_budget_bytes == 0)) {
        return false;
    }

    if (cli_config.generation_vram_gb_was_provided) {
        return TryComputeVramBudgetBytesFromGiB(cli_config.generation_vram_gb, budget_bytes_out);
    }

    if (cli_config.population_size_was_provided) {
        const std::size_t slot_stride_bytes = ComputeSlabSlotStrideBytes(initial_action_count);
        if ((slot_stride_bytes == 0) ||
            (cli_config.population_size_ceiling > (std::numeric_limits<std::size_t>::max() / slot_stride_bytes))) {
            return false;
        }

        budget_bytes_out = cli_config.population_size_ceiling * slot_stride_bytes;
        return budget_bytes_out > 0;
    }

    const double generation_budget_bytes = static_cast<double>(total_budget_bytes) / kDefaultBufferToGenerationRatio;
    if ((generation_budget_bytes < 1.0) ||
        (generation_budget_bytes > static_cast<double>(std::numeric_limits<std::size_t>::max()))) {
        return false;
    }

    budget_bytes_out = static_cast<std::size_t>(generation_budget_bytes);
    return budget_bytes_out > 0;
}

std::string PopulationCeilingLabel(const std::size_t population_size_ceiling) {
    return (population_size_ceiling == 0) ? "none" : std::to_string(population_size_ceiling);
}

PendingOutputEmbeddingInjection MakePendingOutputEmbeddingInjection(const std::size_t current_action_count,
                                                                    const std::size_t next_scheduled_word_count) {
    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
    if (current_action_count < next_scheduled_word_count) {
        pending_output_embedding_injection.enabled = true;
        pending_output_embedding_injection.first_catalog_word_index = current_action_count;
        pending_output_embedding_injection.injection_count = next_scheduled_word_count - current_action_count;
    }

    return pending_output_embedding_injection;
}

bool TryPersistWinningGenome(const DeviceSlabGARuntimeBuffers &buffers, const PopulationFitnessSummary &summary,
                             const std::uint32_t seed,
                             const neuroevolution::training_folder::TrainingWordCatalog &action_space_words,
                             const std::filesystem::path &action_space_path, WinnerArtifactPaths &artifact_paths_out) {
    std::unique_ptr<std::uint8_t[]> genome_bytes{};
    std::size_t genome_byte_count = 0;
    if (!TryDownloadSlabSlotBytesFromDevice(buffers, summary.best_slot_index, genome_bytes, genome_byte_count)) {
        return false;
    }

    WinnerArtifactMetadata metadata{};
    metadata.best_fitness = summary.best_fitness;
    metadata.generation_index = summary.generation_index;
    metadata.best_index = summary.best_index;
    metadata.best_slot_index = summary.best_slot_index;
    metadata.action_count = summary.action_count;
    metadata.genome_byte_count = genome_byte_count;
    metadata.seed = seed;
    metadata.action_space_path = action_space_path;
    return TryWriteWinnerArtifact("models", genome_bytes.get(), action_space_words, metadata, artifact_paths_out);
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
        runtime_word_counts.training_word_schedule = word_count_schedule;
        runtime_word_counts.shard_radius_growth_period_generations = cli_config.shard_radius_growth_period_generations;

        std::size_t genotype_memory_budget_bytes = 0;
        if (!TryComputeTotalGenotypeBudgetBytes(cli_config, genotype_memory_budget_bytes)) {
            std::cerr << "Could not derive a valid total genotype VRAM budget from the command line.\n";
            return 1;
        }

        std::size_t generation_memory_budget_bytes = 0;
        if (!TryComputeGenerationBudgetBytes(cli_config, runtime_word_counts.action_space_word_count,
                                             genotype_memory_budget_bytes, generation_memory_budget_bytes)) {
            std::cerr << "Could not derive a valid generation VRAM budget from the command line.\n";
            return 1;
        }

        if (generation_memory_budget_bytes > genotype_memory_budget_bytes) {
            std::cerr << "The requested generation VRAM budget exceeds the total genotype VRAM budget.\n";
            return 1;
        }

        const std::size_t slab_slot_count =
            SlabSlotCountForByteBudget(genotype_memory_budget_bytes, runtime_word_counts.action_space_word_count);
        if (slab_slot_count < 2) {
            std::cerr << "The requested genotype VRAM budget is too small for a fixed-width slab generation.\n";
            return 1;
        }

        const std::size_t generation_population_capacity =
            SlabSlotCountForByteBudget(generation_memory_budget_bytes, runtime_word_counts.action_space_word_count);
        if (generation_population_capacity == 0) {
            std::cerr << "The requested generation VRAM budget is too small for a fixed-width generation.\n";
            return 1;
        }

        if (slab_slot_count <= generation_population_capacity) {
            std::cerr
                << "The requested total genotype VRAM budget does not leave any slab slack beyond one "
                   "generation. Increase --genotype-vram-gb or reduce --generation-vram-gb / --population-size.\n";
            return 1;
        }

        const std::size_t requested_initial_population_size = cli_config.population_size_was_provided
                                                                  ? cli_config.population_size_ceiling
                                                                  : generation_population_capacity;
        if (requested_initial_population_size > generation_population_capacity) {
            std::cerr << "The requested generation VRAM budget is too small to guarantee a population of "
                      << requested_initial_population_size
                      << " individuals at the starting action count. Increase --generation-vram-gb or lower "
                         "--population-size.\n";
            return 1;
        }

        const std::size_t initial_population_size = FloorSquarePopulationSize(requested_initial_population_size);
        if (initial_population_size == 0) {
            std::cerr << "The requested startup population is too small to form a square cellular grid.\n";
            return 1;
        }

        CellularGridShape initial_grid_shape{};
        if (!TryMakeCellularGridShape(initial_population_size, initial_grid_shape)) {
            std::cerr << "Could not derive a valid square cellular grid from the startup population size.\n";
            return 1;
        }

        DeviceSlabGARuntimeConfig runtime_config{};
        runtime_config.genotype_slab_byte_budget_bytes = genotype_memory_budget_bytes;
        runtime_config.generation_byte_budget_bytes = generation_memory_budget_bytes;
        runtime_config.host_spillover_byte_budget_bytes = generation_memory_budget_bytes / 2;
        runtime_config.action_count = runtime_word_counts.action_space_word_count;
        runtime_config.population_size_ceiling = initial_population_size;

        DeviceSlabGARuntimeBuffers buffers{};
        if (!TryCreateDeviceSlabGARuntimeBuffers(buffers, runtime_config)) {
            std::cerr << "Could not allocate slab-backed device-runtime buffers.\n";
            return 1;
        }

        const GenerationAssemblyConfig assembly_config = MakeAssemblyConfig();
        const DeviceSlabBootstrapConfig bootstrap_config{};

        if (!TryBootstrapRandomCurrentGenerationOnDevice(buffers, initial_population_size, cli_config.seed, 0,
                                                         bootstrap_config)) {
            std::cerr << "Could not bootstrap the initial device slab population.\n";
            DestroyDeviceSlabGARuntimeBuffers(buffers);
            return 1;
        }

        std::cout << '[' << FormatCurrentLocalTimestamp() << "] Running device GA with:\n"
                  << "  requested_initial_population=" << requested_initial_population_size << '\n'
                  << "  initial_population=" << initial_population_size << '\n'
                  << "  initial_grid_side_length=" << initial_grid_shape.side_length << '\n'
                  << "  population_ceiling=" << PopulationCeilingLabel(cli_config.population_size_ceiling) << '\n'
                  << "  slab_slot_count=" << slab_slot_count << '\n'
                  << "  generation_population_capacity=" << generation_population_capacity << '\n'
                  << "  action_count=" << runtime_word_counts.action_space_word_count << '\n'
                  << "  genome_stride_bytes=" << buffers.genotype_slab.slab_layout.slot_stride_bytes << '\n'
                  << "  generation_vram_budget_bytes=" << generation_memory_budget_bytes << '\n'
                  << "  generation_vram_budget_gib="
                  << (static_cast<double>(generation_memory_budget_bytes) / kBytesPerVramGiB) << '\n'
                  << "  genotype_vram_budget_bytes=" << genotype_memory_budget_bytes << '\n'
                  << "  genotype_vram_budget_gib="
                  << (static_cast<double>(genotype_memory_budget_bytes) / kBytesPerVramGiB) << '\n'
                  << "  generations=" << cli_config.generation_count << '\n'
                  << "  seed=" << cli_config.seed << '\n'
                  << "  training_word_catalog_entries=" << training_word_catalog.word_count << '\n'
                  << "  configured_training_word_count=" << runtime_word_counts.training_word_count << '\n'
                  << "  configured_action_space_word_count=" << runtime_word_counts.action_space_word_count << '\n'
                  << "  schedule_initial_word_count=" << word_count_schedule.initial_word_count << '\n'
                  << "  schedule_word_count_step=" << word_count_schedule.word_count_step << '\n'
                  << "  schedule_word_count_step_period_generations="
                  << word_count_schedule.word_count_step_period_generations << '\n'
                  << "  shard_radius_growth_period_generations="
                  << runtime_word_counts.shard_radius_growth_period_generations << '\n'
                  << "  training_source=" << training_data_path.filename().string() << '\n'
                  << "  training_storage=constant_memory\n";
        std::cout << std::fixed << std::setprecision(4);

        PopulationFitnessSummary final_summary{};
        for (std::size_t generation_step = 0; generation_step < cli_config.generation_count; ++generation_step) {
            const bool is_last_generation = ((generation_step + 1) == cli_config.generation_count);
            if (is_last_generation) {
                if (!TryEvaluateCurrentGenerationFitnessOnDevice(buffers, runtime_word_counts)) {
                    (void)ReportDeviceSlabRuntimeFailure(buffers, "Population fitness evaluation");
                    DestroyDeviceSlabGARuntimeBuffers(buffers);
                    return 1;
                }
            } else {
                const std::size_t next_scheduled_word_count = ScheduledWordCountForGeneration(
                    word_count_schedule, training_word_catalog.word_count, generation_step + 1);
                const PendingOutputEmbeddingInjection pending_output_embedding_injection =
                    MakePendingOutputEmbeddingInjection(runtime_word_counts.action_space_word_count,
                                                        next_scheduled_word_count);
                const std::uint32_t generation_seed =
                    cli_config.seed + 2U + static_cast<std::uint32_t>(generation_step);
                if (!TryAdvanceGenerationOnDevice(buffers, generation_seed, runtime_word_counts, assembly_config,
                                                  pending_output_embedding_injection, &training_word_catalog,
                                                  cli_config.verbose)) {
                    (void)ReportDeviceSlabRuntimeFailure(buffers, "Next-generation assembly");
                    DestroyDeviceSlabGARuntimeBuffers(buffers);
                    return 1;
                }

                if (buffers.last_generation_used_host_spillover) {
                    std::cerr << "WARNING: genotype slab overflowed its device budget during generation "
                              << (generation_step + 1)
                              << "; spilled assembled children to host-side slab storage. Consider increasing "
                                 "--genotype-vram-gb or --generation-vram-gb if this becomes frequent.\n";
                }

                runtime_word_counts.training_word_count = next_scheduled_word_count;
                runtime_word_counts.action_space_word_count = next_scheduled_word_count;
            }

            PopulationFitnessSummary summary{};
            if (!TryReadPopulationFitnessSummaryFromDevice(buffers, summary)) {
                std::cerr << "Could not read the population fitness summary back from device memory.\n";
                DestroyDeviceSlabGARuntimeBuffers(buffers);
                return 1;
            }
            final_summary = summary;

            std::cout << '[' << FormatCurrentLocalTimestamp() << "] Generation " << summary.generation_index
                      << ": best=" << summary.best_fitness << ", average=" << summary.average_fitness
                      << ", best_index=" << summary.best_index << ", population=" << summary.population_size
                      << ", action_count=" << summary.action_count
                      << ", genome_stride_bytes=" << ComputeSlabSlotStrideBytes(summary.action_count) << '\n';
        }

        WinnerArtifactPaths artifact_paths{};
        if (!TryPersistWinningGenome(buffers, final_summary, cli_config.seed, training_word_catalog, training_data_path,
                                     artifact_paths)) {
            std::cerr << "Could not persist the final-generation winner to models/.\n";
            DestroyDeviceSlabGARuntimeBuffers(buffers);
            return 1;
        }

        DestroyDeviceSlabGARuntimeBuffers(buffers);
        std::cout << "Saved final-generation winner to " << artifact_paths.binary_path.string() << " with metadata "
                  << artifact_paths.metadata_path.string() << '\n';
        std::cout << "GA demo finished after " << cli_config.generation_count << " generations.\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "run_genetic_algorithm failed: " << exception.what() << '\n';
        return 1;
    }
}
