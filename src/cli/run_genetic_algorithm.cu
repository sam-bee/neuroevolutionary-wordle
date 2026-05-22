#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "common/progress_log.hpp"
#include "genetic_algorithm/device/slab_runtime.hpp"
#include "genetic_algorithm/genotype_slab/slab_allocator.hpp"
#include "genetic_algorithm/spatial/grid.hpp"
#include "model_artifact/winner_artifact.hpp"
#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::common::FormatCurrentLocalTimestamp;
using neuroevolution::common::PrintTimestampedProgressDuration;
using neuroevolution::common::PrintTimestampedProgressLine;
using neuroevolution::common::ProgressClock;
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
using neuroevolution::genetic_algorithm::slab_device::RuntimeCheckpoint;
using neuroevolution::genetic_algorithm::slab_device::RuntimeCheckpointAsyncWriter;
using neuroevolution::genetic_algorithm::slab_device::RuntimeWordCounts;
using neuroevolution::genetic_algorithm::slab_device::TryAdvanceGenerationOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryBootstrapRandomCurrentGenerationOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryCreateDeviceSlabGARuntimeBuffers;
using neuroevolution::genetic_algorithm::slab_device::TryCreatePrebreedingCheckpointOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryDownloadSlabSlotBytesFromDevice;
using neuroevolution::genetic_algorithm::slab_device::TryEvaluateCurrentGenerationFitnessOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryReadDeviceSlabGARuntimeStatus;
using neuroevolution::genetic_algorithm::slab_device::TryReadPopulationFitnessSummaryFromDevice;
using neuroevolution::genetic_algorithm::slab_device::TryReadRuntimeCheckpoint;
using neuroevolution::genetic_algorithm::slab_device::TryRestorePrebreedingCheckpointToDevice;
using neuroevolution::genetic_algorithm::slab_device::TryResumeGenerationFromCheckpointOnDevice;
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
    std::size_t breeding_radius = neuroevolution::genetic_algorithm::spatial::kCellularBreedingRadius;
    std::size_t shard_initial_radius = neuroevolution::training_folder::kDefaultTrainingShardInitialRadius;
    std::size_t shard_radius_growth_period_generations =
        neuroevolution::training_folder::kDefaultShardRadiusGrowthPeriodGenerations;
    double genotype_vram_gb = 0.0;
    bool genotype_vram_gb_was_provided = false;
    double generation_vram_gb = 0.0;
    bool generation_vram_gb_was_provided = false;
    std::uint32_t seed = 0;
    bool seed_was_provided = false;
    bool verbose = false;
    std::filesystem::path checkpoint_path{};
    bool checkpoint_path_was_provided = false;
    std::size_t checkpoint_every_generations = 0;
    bool checkpoint_every_was_provided = false;
    std::filesystem::path resume_checkpoint_path{};
    bool resume_checkpoint_path_was_provided = false;
};

enum class ArgumentParseResult {
    kSuccess,
    kHelpRequested,
    kFailure,
};

void PrintUsage() {
    std::cout << "Usage: run_genetic_algorithm [--verbose] [--seed N] [--generations N] [--population-size N] "
                 "[--genotype-vram-gb F] [--generation-vram-gb F] [--initial-word-count N] [--word-count-step N] "
                 "[--word-count-step-period N] [--breeding-radius N] [--shard-initial-radius N] "
                 "[--shard-initial-radius-infinite] [--shard-radius-growth-period N] "
                 "[--checkpoint-path PATH] [--checkpoint-every N] [--resume-from-checkpoint PATH]\n"
              << "If --population-size is omitted, the program does not apply an extra population ceiling.\n"
              << "The shared training/action schedule defaults to initial_word_count="
              << neuroevolution::training_folder::kDefaultInitialActiveWordCount
              << ", word_count_step=0, word_count_step_period_generations=1.\n"
              << "Spatial training-data shards grow their evaluation radius every "
              << neuroevolution::training_folder::kDefaultShardRadiusGrowthPeriodGenerations
              << " generations by default.\n"
              << "--breeding-radius controls the toroidal Moore/Chebyshev parent-selection radius and defaults to "
              << neuroevolution::genetic_algorithm::spatial::kCellularBreedingRadius << ".\n"
              << "--shard-initial-radius-infinite starts every newly introduced non-foundation shard at a radius "
                 "large enough to cover the whole current population grid.\n"
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
                 "slot-growth compaction/repacking and checkpoint copy/write/resume stages.\n"
              << "--checkpoint-path enables runtime checkpointing. If --checkpoint-every is omitted, checkpoints are "
                 "written at every inter-generation boundary. If a previous async checkpoint write is still running, "
                 "the next checkpoint is skipped.\n"
              << "--resume-from-checkpoint loads a pre-recombination checkpoint, restores the saved assembly plan, "
                 "and resumes without rerunning the completed generation's fitness evaluation or selection.\n";
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

        if (argument == "--shard-initial-radius-infinite") {
            config.shard_initial_radius = neuroevolution::training_folder::kEffectivelyInfiniteTrainingShardRadius;
            continue;
        }

        if ((argument == "--checkpoint-path") || (argument == "--resume-from-checkpoint")) {
            if ((arg_index + 1) >= argc) {
                std::cerr << "Missing value for " << argument << '\n';
                return ArgumentParseResult::kFailure;
            }

            if (argv[arg_index + 1][0] == '\0') {
                std::cerr << "Checkpoint path for " << argument << " must not be empty.\n";
                return ArgumentParseResult::kFailure;
            }

            if (argument == "--checkpoint-path") {
                config.checkpoint_path = argv[arg_index + 1];
                config.checkpoint_path_was_provided = true;
            } else {
                config.resume_checkpoint_path = argv[arg_index + 1];
                config.resume_checkpoint_path_was_provided = true;
            }

            ++arg_index;
            continue;
        }

        if ((argument == "--seed") || (argument == "--generations") || (argument == "--population-size") ||
            (argument == "--genotype-vram-gb") || (argument == "--generation-vram-gb") ||
            (argument == "--initial-word-count") || (argument == "--word-count-step") ||
            (argument == "--word-count-step-period") || (argument == "--breeding-radius") ||
            (argument == "--shard-initial-radius") ||
            (argument == "--shard-radius-growth-period") || (argument == "--checkpoint-every")) {
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
                } else if (argument == "--breeding-radius") {
                    if (parsed_value == 0) {
                        std::cerr << "Breeding radius must be at least 1.\n";
                        return ArgumentParseResult::kFailure;
                    }

                    config.breeding_radius = static_cast<std::size_t>(parsed_value);
                } else if (argument == "--shard-initial-radius") {
                    config.shard_initial_radius = static_cast<std::size_t>(parsed_value);
                } else if (argument == "--checkpoint-every") {
                    if (parsed_value == 0) {
                        std::cerr << "Checkpoint period must be at least 1.\n";
                        return ArgumentParseResult::kFailure;
                    }

                    config.checkpoint_every_generations = static_cast<std::size_t>(parsed_value);
                    config.checkpoint_every_was_provided = true;
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

    if (config.checkpoint_every_was_provided && !config.checkpoint_path_was_provided) {
        std::cerr << "--checkpoint-every requires --checkpoint-path.\n";
        return ArgumentParseResult::kFailure;
    }
    if (config.checkpoint_path_was_provided && !config.checkpoint_every_was_provided) {
        config.checkpoint_every_generations = 1;
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

GenerationAssemblyConfig MakeAssemblyConfig(const CliConfig &cli_config) {
    GenerationAssemblyConfig config{};
    config.parent_selection.tournament_size = 3;
    config.parent_selection.allow_self_parenting = false;
    config.parent_selection.cellular_breeding_radius = cli_config.breeding_radius;
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

std::string TrainingShardRadiusLabel(const std::size_t radius) {
    return (radius == neuroevolution::training_folder::kEffectivelyInfiniteTrainingShardRadius)
               ? "infinite"
               : std::to_string(radius);
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

std::size_t CheckpointPayloadByteCount(const RuntimeCheckpoint &checkpoint) {
    std::size_t byte_count =
        checkpoint.assembly_plan.child_count * sizeof(neuroevolution::genetic_algorithm::genotype_slab::SlabParentPair);
    byte_count += checkpoint.current_generation.active_individual_count *
                  (sizeof(std::uint32_t) + sizeof(float) + sizeof(std::uint32_t) + sizeof(std::uint8_t));
    for (const auto &record : checkpoint.live_genotypes) {
        byte_count += record.genome_bytes.size();
    }
    return byte_count;
}

bool TryCloneRuntimeCheckpoint(const RuntimeCheckpoint &source, RuntimeCheckpoint &clone_out) {
    clone_out = {};
    clone_out.schema_version = source.schema_version;
    clone_out.genome_layout_version = source.genome_layout_version;
    clone_out.checksum = source.checksum;
    clone_out.training_data_identity_hash = source.training_data_identity_hash;
    clone_out.resume_phase = source.resume_phase;
    clone_out.generation_seed = source.generation_seed;
    clone_out.runtime_word_counts = source.runtime_word_counts;
    clone_out.assembly_config = source.assembly_config;
    clone_out.pending_output_embedding_injection = source.pending_output_embedding_injection;
    clone_out.runtime_config = source.runtime_config;
    clone_out.slab_layout = source.slab_layout;
    clone_out.current_grid_shape = source.current_grid_shape;
    clone_out.next_grid_shape = source.next_grid_shape;
    clone_out.epicenter_grid_shape = source.epicenter_grid_shape;
    if (!neuroevolution::genetic_algorithm::genotype_slab::TryCreateSlabGeneration(
            clone_out.current_generation, source.current_generation.active_individual_count,
            source.current_generation.generation_index)) {
        clone_out = {};
        return false;
    }
    for (std::size_t individual_index = 0; individual_index < source.current_generation.active_individual_count;
         ++individual_index) {
        clone_out.current_generation.slot_indices[individual_index] =
            source.current_generation.slot_indices[individual_index];
        clone_out.current_generation.fitness[individual_index] = source.current_generation.fitness[individual_index];
        clone_out.current_generation.evaluation_counts[individual_index] =
            source.current_generation.evaluation_counts[individual_index];
        clone_out.current_generation.has_fitness[individual_index] =
            source.current_generation.has_fitness[individual_index];
    }
    if (!neuroevolution::genetic_algorithm::genotype_slab::TryCreateSlabAssemblyPlan(
            clone_out.assembly_plan, source.assembly_plan.child_count)) {
        clone_out = {};
        return false;
    }
    for (std::size_t child_index = 0; child_index < source.assembly_plan.child_count; ++child_index) {
        clone_out.assembly_plan.parent_pairs[child_index] = source.assembly_plan.parent_pairs[child_index];
    }
    clone_out.live_genotypes = source.live_genotypes;
    return true;
}

RuntimeWordCounts RuntimeWordCountsAfterCheckpointResume(const RuntimeCheckpoint &checkpoint) {
    RuntimeWordCounts runtime_word_counts = checkpoint.runtime_word_counts;
    if (checkpoint.pending_output_embedding_injection.enabled) {
        const std::size_t next_word_count = checkpoint.pending_output_embedding_injection.first_catalog_word_index +
                                            checkpoint.pending_output_embedding_injection.injection_count;
        runtime_word_counts.training_word_count = next_word_count;
        runtime_word_counts.action_space_word_count = next_word_count;
    }
    return runtime_word_counts;
}

bool PrintCurrentPopulationSummary(const DeviceSlabGARuntimeBuffers &buffers, PopulationFitnessSummary &summary_out) {
    if (!TryReadPopulationFitnessSummaryFromDevice(buffers, summary_out)) {
        std::cerr << "Could not read the population fitness summary back from device memory.\n";
        return false;
    }

    std::cout << '[' << FormatCurrentLocalTimestamp() << "] Generation " << summary_out.generation_index
              << ": best=" << summary_out.best_fitness << ", average=" << summary_out.average_fitness
              << ", best_index=" << summary_out.best_index << ", population=" << summary_out.population_size
              << ", action_count=" << summary_out.action_count
              << ", genome_stride_bytes=" << ComputeSlabSlotStrideBytes(summary_out.action_count) << '\n';
    return true;
}

bool CollectFinishedCheckpointWrite(RuntimeCheckpointAsyncWriter &checkpoint_writer, const bool verbose) {
    bool write_finished = false;
    bool write_succeeded = true;
    const auto collect_start_time = ProgressClock::now();
    if (!checkpoint_writer.TryCollectFinishedWrite(write_finished, write_succeeded)) {
        std::cerr << "Checkpoint write failed.\n";
        return false;
    }
    if (verbose && write_finished) {
        PrintTimestampedProgressDuration(std::cout, "Finished async checkpoint write", collect_start_time);
    }
    return true;
}

bool WaitForCheckpointWrite(RuntimeCheckpointAsyncWriter &checkpoint_writer, const bool verbose) {
    if (!checkpoint_writer.IsWriteInProgress()) {
        return CollectFinishedCheckpointWrite(checkpoint_writer, verbose);
    }

    const auto wait_start_time = ProgressClock::now();
    if (verbose) {
        PrintTimestampedProgressLine(std::cout, "Waiting for async checkpoint write to finish");
    }
    if (!checkpoint_writer.TryWaitForWrite()) {
        std::cerr << "Checkpoint write failed.\n";
        return false;
    }
    if (verbose) {
        PrintTimestampedProgressDuration(std::cout, "Async checkpoint write finished", wait_start_time);
    }
    return true;
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

        DeviceSlabGARuntimeBuffers buffers{};
        const GenerationAssemblyConfig assembly_config = MakeAssemblyConfig(cli_config);
        RuntimeWordCounts runtime_word_counts{};

        if (cli_config.resume_checkpoint_path_was_provided) {
            RuntimeCheckpoint checkpoint{};
            const auto read_checkpoint_start_time = ProgressClock::now();
            if (cli_config.verbose) {
                PrintTimestampedProgressLine(std::cout,
                                             "Reading checkpoint from " + cli_config.resume_checkpoint_path.string());
            }
            if (!TryReadRuntimeCheckpoint(cli_config.resume_checkpoint_path, checkpoint)) {
                std::cerr << "Could not read or validate checkpoint " << cli_config.resume_checkpoint_path.string()
                          << ".\n";
                return 1;
            }
            if (cli_config.verbose) {
                PrintTimestampedProgressDuration(std::cout, "Checkpoint read and validated",
                                                 read_checkpoint_start_time);
            }

            const auto restore_checkpoint_start_time = ProgressClock::now();
            if (cli_config.verbose) {
                PrintTimestampedProgressLine(std::cout, "Restoring checkpoint to device runtime");
            }
            if (!TryRestorePrebreedingCheckpointToDevice(checkpoint, buffers)) {
                std::cerr << "Could not restore checkpoint into device runtime buffers.\n";
                return 1;
            }
            if (cli_config.verbose) {
                PrintTimestampedProgressDuration(std::cout, "Checkpoint restored to device runtime",
                                                 restore_checkpoint_start_time);
            }

            const auto resume_checkpoint_start_time = ProgressClock::now();
            if (cli_config.verbose) {
                PrintTimestampedProgressLine(
                    std::cout, "Generation " + std::to_string(checkpoint.current_generation.generation_index + 1) +
                                   ": resuming saved checkpoint assembly plan");
            }
            if (!TryResumeGenerationFromCheckpointOnDevice(buffers, checkpoint, cli_config.verbose)) {
                (void)ReportDeviceSlabRuntimeFailure(buffers, "Checkpoint resume assembly");
                DestroyDeviceSlabGARuntimeBuffers(buffers);
                return 1;
            }
            if (cli_config.verbose) {
                PrintTimestampedProgressDuration(
                    std::cout,
                    "Generation " + std::to_string(checkpoint.current_generation.generation_index + 1) +
                        ": saved checkpoint assembly plan resumed",
                    resume_checkpoint_start_time);
            }
            runtime_word_counts = RuntimeWordCountsAfterCheckpointResume(checkpoint);

            std::cout << '[' << FormatCurrentLocalTimestamp() << "] Resumed device GA from checkpoint:\n"
                      << "  checkpoint_path=" << cli_config.resume_checkpoint_path.string() << '\n'
                      << "  checkpoint_generation=" << checkpoint.current_generation.generation_index << '\n'
                      << "  resumed_generation=" << buffers.genotype_slab.current_generation_index << '\n'
                      << "  resumed_population=" << buffers.genotype_slab.current_generation_size << '\n'
                      << "  checkpoint_child_count=" << checkpoint.assembly_plan.child_count << '\n'
                      << "  checkpoint_live_genotypes=" << checkpoint.live_genotypes.size() << '\n'
                      << "  checkpoint_payload_bytes=" << CheckpointPayloadByteCount(checkpoint) << '\n'
                      << "  action_count=" << runtime_word_counts.action_space_word_count << '\n'
                      << "  genome_stride_bytes=" << buffers.genotype_slab.slab_layout.slot_stride_bytes << '\n'
                      << "  generations=" << cli_config.generation_count << '\n'
                      << "  seed=" << cli_config.seed << '\n'
                      << "  training_source=" << training_data_path.filename().string() << '\n'
                      << "  training_storage=constant_memory\n";
        } else {
            runtime_word_counts.training_word_count = initial_active_word_count;
            runtime_word_counts.action_space_word_count = initial_active_word_count;
            runtime_word_counts.training_word_schedule = word_count_schedule;
            runtime_word_counts.shard_initial_radius = cli_config.shard_initial_radius;
            runtime_word_counts.shard_radius_growth_period_generations =
                cli_config.shard_radius_growth_period_generations;

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
            runtime_config.grid_column_count = initial_grid_shape.column_count;

            if (!TryCreateDeviceSlabGARuntimeBuffers(buffers, runtime_config)) {
                std::cerr << "Could not allocate slab-backed device-runtime buffers.\n";
                return 1;
            }

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
                      << "  initial_grid_rows=" << initial_grid_shape.row_count << '\n'
                      << "  initial_grid_columns=" << initial_grid_shape.column_count << '\n'
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
                      << "  breeding_radius=" << assembly_config.parent_selection.cellular_breeding_radius << '\n'
                      << "  shard_initial_radius=" << TrainingShardRadiusLabel(runtime_word_counts.shard_initial_radius)
                      << '\n'
                      << "  shard_radius_growth_period_generations="
                      << runtime_word_counts.shard_radius_growth_period_generations << '\n'
                      << "  training_source=" << training_data_path.filename().string() << '\n'
                      << "  training_storage=constant_memory\n";
        }
        if (cli_config.checkpoint_path_was_provided) {
            std::cout << "  checkpoint_path=" << cli_config.checkpoint_path.string() << '\n'
                      << "  checkpoint_every_generations=" << cli_config.checkpoint_every_generations << '\n';
        }
        std::cout << std::fixed << std::setprecision(4);

        if (buffers.genotype_slab.current_generation_index >= cli_config.generation_count) {
            std::cerr << "The checkpoint has already advanced beyond the requested --generations count.\n";
            DestroyDeviceSlabGARuntimeBuffers(buffers);
            return 1;
        }

        RuntimeCheckpointAsyncWriter checkpoint_writer{};
        PopulationFitnessSummary final_summary{};
        const std::size_t final_generation_index = cli_config.generation_count - 1;
        while (buffers.genotype_slab.current_generation_index < final_generation_index) {
            if (!CollectFinishedCheckpointWrite(checkpoint_writer, cli_config.verbose)) {
                DestroyDeviceSlabGARuntimeBuffers(buffers);
                return 1;
            }

            const std::size_t current_generation_index = buffers.genotype_slab.current_generation_index;
            const std::size_t next_generation_index = current_generation_index + 1;
            const std::size_t next_scheduled_word_count = ScheduledWordCountForGeneration(
                runtime_word_counts.training_word_schedule, training_word_catalog.word_count, next_generation_index);
            const PendingOutputEmbeddingInjection pending_output_embedding_injection =
                MakePendingOutputEmbeddingInjection(runtime_word_counts.action_space_word_count,
                                                    next_scheduled_word_count);
            const std::uint32_t generation_seed =
                cli_config.seed + 2U + static_cast<std::uint32_t>(current_generation_index);
            const bool should_checkpoint =
                cli_config.checkpoint_path_was_provided &&
                (((current_generation_index + 1) % cli_config.checkpoint_every_generations) == 0);

            if (should_checkpoint) {
                if (checkpoint_writer.IsWriteInProgress()) {
                    if (cli_config.verbose) {
                        PrintTimestampedProgressLine(
                            std::cout, "Generation " + std::to_string(current_generation_index) +
                                           ": skipping checkpoint because previous async write is still running");
                    }
                    if (!TryAdvanceGenerationOnDevice(buffers, generation_seed, runtime_word_counts, assembly_config,
                                                      pending_output_embedding_injection, &training_word_catalog,
                                                      cli_config.verbose)) {
                        (void)ReportDeviceSlabRuntimeFailure(buffers, "Next-generation assembly");
                        DestroyDeviceSlabGARuntimeBuffers(buffers);
                        return 1;
                    }
                } else {
                    RuntimeCheckpoint checkpoint{};
                    if (!TryCreatePrebreedingCheckpointOnDevice(buffers, generation_seed, runtime_word_counts,
                                                                assembly_config, pending_output_embedding_injection,
                                                                checkpoint, &training_word_catalog,
                                                                cli_config.verbose)) {
                        (void)ReportDeviceSlabRuntimeFailure(buffers, "Checkpoint creation");
                        DestroyDeviceSlabGARuntimeBuffers(buffers);
                        return 1;
                    }

                    if (cli_config.verbose) {
                        std::ostringstream stream;
                        stream << "Generation " << checkpoint.current_generation.generation_index
                               << ": starting async checkpoint write"
                               << " (path=" << cli_config.checkpoint_path.string()
                               << ", payload_bytes=" << CheckpointPayloadByteCount(checkpoint) << ')';
                        PrintTimestampedProgressLine(std::cout, stream.str());
                    }
                    RuntimeCheckpoint checkpoint_for_write{};
                    if (!TryCloneRuntimeCheckpoint(checkpoint, checkpoint_for_write)) {
                        std::cerr << "Could not clone checkpoint payload for async writing.\n";
                        DestroyDeviceSlabGARuntimeBuffers(buffers);
                        return 1;
                    }
                    if (!checkpoint_writer.TryStartWrite(std::move(checkpoint_for_write), cli_config.checkpoint_path)) {
                        if (cli_config.verbose) {
                            PrintTimestampedProgressLine(
                                std::cout, "Generation " + std::to_string(current_generation_index) +
                                               ": checkpoint write not started because another write is in progress");
                        }
                    }

                    if (!TryResumeGenerationFromCheckpointOnDevice(buffers, checkpoint, cli_config.verbose)) {
                        (void)ReportDeviceSlabRuntimeFailure(buffers, "Checkpointed next-generation assembly");
                        DestroyDeviceSlabGARuntimeBuffers(buffers);
                        return 1;
                    }
                }
            } else {
                if (!TryAdvanceGenerationOnDevice(buffers, generation_seed, runtime_word_counts, assembly_config,
                                                  pending_output_embedding_injection, &training_word_catalog,
                                                  cli_config.verbose)) {
                    (void)ReportDeviceSlabRuntimeFailure(buffers, "Next-generation assembly");
                    DestroyDeviceSlabGARuntimeBuffers(buffers);
                    return 1;
                }
            }

            if (buffers.last_generation_used_host_spillover) {
                std::cerr << "WARNING: genotype slab overflowed its device budget during generation "
                          << next_generation_index
                          << "; spilled assembled children to host-side slab storage. Consider increasing "
                             "--genotype-vram-gb or --generation-vram-gb if this becomes frequent.\n";
            }

            runtime_word_counts.training_word_count = next_scheduled_word_count;
            runtime_word_counts.action_space_word_count = next_scheduled_word_count;
            if (!PrintCurrentPopulationSummary(buffers, final_summary)) {
                DestroyDeviceSlabGARuntimeBuffers(buffers);
                return 1;
            }
        }

        const auto fitness_start_time = ProgressClock::now();
        if (cli_config.verbose) {
            PrintTimestampedProgressLine(std::cout, "Generation " +
                                                        std::to_string(buffers.genotype_slab.current_generation_index) +
                                                        ": evaluating current generation fitness");
        }
        if (!TryEvaluateCurrentGenerationFitnessOnDevice(buffers, runtime_word_counts)) {
            (void)ReportDeviceSlabRuntimeFailure(buffers, "Population fitness evaluation");
            DestroyDeviceSlabGARuntimeBuffers(buffers);
            return 1;
        }
        if (cli_config.verbose) {
            PrintTimestampedProgressDuration(std::cout,
                                             "Generation " +
                                                 std::to_string(buffers.genotype_slab.current_generation_index) +
                                                 ": current generation fitness evaluation finished",
                                             fitness_start_time);
        }
        if (!PrintCurrentPopulationSummary(buffers, final_summary)) {
            DestroyDeviceSlabGARuntimeBuffers(buffers);
            return 1;
        }

        if (!WaitForCheckpointWrite(checkpoint_writer, cli_config.verbose)) {
            DestroyDeviceSlabGARuntimeBuffers(buffers);
            return 1;
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
