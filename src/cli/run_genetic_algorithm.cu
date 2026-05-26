#include <cuda_runtime.h>
#include <sqlite3.h>

#include <algorithm>
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
#include <vector>

#include "common/progress_log.hpp"
#include "genetic_algorithm/breeding.hpp"
#include "genetic_algorithm/device/slab_runtime.hpp"
#include "genetic_algorithm/genetic_convergence.hpp"
#include "genetic_algorithm/genotype_slab/slab_allocator.hpp"
#include "genetic_algorithm/mutation.hpp"
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
using neuroevolution::genetic_algorithm::genotype_slab::SlabParentPair;
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
using neuroevolution::training_folder::TrainingDataShardCountForIntroducedWordCount;
using neuroevolution::training_folder::TrainingDataShardReleaseHistory;
using neuroevolution::training_folder::TryRecordTrainingDataShardRelease;
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
    std::size_t shard_release_min_gap_generations =
        neuroevolution::training_folder::kDefaultTrainingShardReleaseMinimumGapGenerations;
    float shard_release_centroid_distance_threshold =
        neuroevolution::training_folder::kDefaultTrainingShardReleaseCentroidDistanceThreshold;
    std::size_t breeding_radius = neuroevolution::genetic_algorithm::spatial::kCellularBreedingRadius;
    float parent_selection_rank_exponent = neuroevolution::genetic_algorithm::kDefaultParentSelectionRankExponent;
    float crossover_temperature_level1 = neuroevolution::genetic_algorithm::kDefaultCrossoverTemperatureLevel1;
    float crossover_temperature_level2 = neuroevolution::genetic_algorithm::kDefaultCrossoverTemperatureLevel2;
    float crossover_temperature_level3 = neuroevolution::genetic_algorithm::kDefaultCrossoverTemperatureLevel3;
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
    std::filesystem::path telemetry_path{};
    bool telemetry_path_was_provided = false;
    std::filesystem::path telemetry_dir{};
    bool telemetry_dir_was_provided = false;
    bool telemetry_genetic_convergence = false;
    std::size_t telemetry_genetic_convergence_interval =
        neuroevolution::genetic_algorithm::genetic_convergence::kDefaultIntervalGenerations;
    bool telemetry_genetic_convergence_interval_was_provided = false;
};

enum class ArgumentParseResult {
    kSuccess,
    kHelpRequested,
    kFailure,
};

void PrintUsage() {
    std::cout << "Usage: run_genetic_algorithm [--verbose] [--seed N] [--generations N] [--population-size N] "
                 "[--genotype-vram-gb F] [--generation-vram-gb F] [--initial-word-count N] [--word-count-step N] "
                 "[--shard-release-min-gap N] [--shard-release-centroid-threshold F] "
                 "[--breeding-radius N] [--parent-selection-rank-exponent F] "
                 "[--crossover-temperature-level1 F] [--crossover-temperature-level2 F] "
                 "[--crossover-temperature-level3 F] "
                 "[--shard-initial-radius N] [--shard-initial-radius-infinite] [--shard-radius-growth-period N] "
                 "[--checkpoint-path PATH] [--checkpoint-every N] [--resume-from-checkpoint PATH] "
                 "[--telemetry-path PATH] [--telemetry-dir DIR] [--telemetry-genetic-convergence] "
                 "[--telemetry-genetic-convergence-interval N]\n"
              << "If --population-size is omitted, the program does not apply an extra population ceiling.\n"
              << "The shared training/action curriculum defaults to initial_word_count="
              << neuroevolution::training_folder::kDefaultInitialActiveWordCount
              << ", word_count_step=0, shard_release_min_gap_generations="
              << neuroevolution::training_folder::kDefaultTrainingShardReleaseMinimumGapGenerations
              << ", shard_release_centroid_distance_threshold="
              << neuroevolution::training_folder::kDefaultTrainingShardReleaseCentroidDistanceThreshold << ".\n"
              << "After the foundation shard, a new shard is released only when the minimum gap has elapsed and "
                 "either centroid distance is at or below the threshold or p99 fitness beats the previous release "
                 "baseline.\n"
              << "Spatial training-data shards grow their evaluation radius every "
              << neuroevolution::training_folder::kDefaultShardRadiusGrowthPeriodGenerations
              << " generations by default.\n"
              << "--breeding-radius controls the toroidal Moore/Chebyshev parent-selection radius and defaults to "
              << neuroevolution::genetic_algorithm::spatial::kCellularBreedingRadius << ".\n"
              << "--parent-selection-rank-exponent controls rank-weighted local parent selection and defaults to "
              << neuroevolution::genetic_algorithm::kDefaultParentSelectionRankExponent << ".\n"
              << "--crossover-temperature-level1 controls whole-subsystem donor flips and defaults to "
              << neuroevolution::genetic_algorithm::kDefaultCrossoverTemperatureLevel1 << ".\n"
              << "--crossover-temperature-level2 controls per-layer and single-output-row donor flips and defaults to "
              << neuroevolution::genetic_algorithm::kDefaultCrossoverTemperatureLevel2 << ".\n"
              << "--crossover-temperature-level3 controls single-neuron flips and output-row splicing and defaults to "
              << neuroevolution::genetic_algorithm::kDefaultCrossoverTemperatureLevel3 << ".\n"
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
                 "and resumes without rerunning the completed generation's fitness evaluation or selection.\n"
              << "--telemetry-path writes per-generation fitness summaries to the given SQLite database. "
                 "--telemetry-dir creates a datetime-named SQLite database in the given directory. "
                 "--telemetry-genetic-convergence adds sampled genetic convergence telemetry at the configured "
                 "interval, defaulting to every 10 generations.\n";
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

bool TryParseNonNegativeReal(const char *text, double &value) {
    if ((text == nullptr) || (*text == '\0')) {
        return false;
    }

    char *end = nullptr;
    value = std::strtod(text, &end);
    return (end != nullptr) && (*end == '\0') && std::isfinite(value) && (value >= 0.0);
}

std::uint32_t MakeSeedFromCurrentTimeMicroseconds() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    const std::uint64_t microsecond_count = static_cast<std::uint64_t>(microseconds);
    const std::uint32_t lower_bits = static_cast<std::uint32_t>(microsecond_count);
    const std::uint32_t upper_bits = static_cast<std::uint32_t>(microsecond_count >> 32U);
    return lower_bits ^ upper_bits;
}

std::string MakeFilesystemTimestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local_time{};
    localtime_r(&now, &local_time);

    std::ostringstream stream;
    stream << std::put_time(&local_time, "%Y-%m-%dT%H-%M-%S");
    return stream.str();
}

std::filesystem::path MakeTelemetryPathFromDirectory(const std::filesystem::path &telemetry_dir) {
    return telemetry_dir / ("ga-telemetry-" + MakeFilesystemTimestamp() + ".sqlite");
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

        if (argument == "--telemetry-genetic-convergence") {
            config.telemetry_genetic_convergence = true;
            continue;
        }

        if (argument == "--shard-initial-radius-infinite") {
            config.shard_initial_radius = neuroevolution::training_folder::kEffectivelyInfiniteTrainingShardRadius;
            continue;
        }

        if (argument == "--shard-release-centroid-threshold") {
            if ((arg_index + 1) >= argc) {
                std::cerr << "Missing value for " << argument << '\n';
                return ArgumentParseResult::kFailure;
            }

            double parsed_value = 0.0;
            if (!TryParsePositiveReal(argv[arg_index + 1], parsed_value) ||
                (parsed_value > static_cast<double>(std::numeric_limits<float>::max()))) {
                std::cerr << "Shard release centroid-distance threshold must be a positive finite number.\n";
                return ArgumentParseResult::kFailure;
            }

            config.shard_release_centroid_distance_threshold = static_cast<float>(parsed_value);
            ++arg_index;
            continue;
        }

        if (argument == "--parent-selection-rank-exponent") {
            if ((arg_index + 1) >= argc) {
                std::cerr << "Missing value for " << argument << '\n';
                return ArgumentParseResult::kFailure;
            }

            double parsed_value = 0.0;
            if (!TryParseNonNegativeReal(argv[arg_index + 1], parsed_value) ||
                (parsed_value > neuroevolution::genetic_algorithm::kMaximumParentSelectionRankExponent)) {
                std::cerr << "Parent-selection rank exponent must be between 0 and "
                          << neuroevolution::genetic_algorithm::kMaximumParentSelectionRankExponent << ".\n";
                return ArgumentParseResult::kFailure;
            }

            config.parent_selection_rank_exponent = static_cast<float>(parsed_value);
            ++arg_index;
            continue;
        }

        if ((argument == "--crossover-temperature-level1") || (argument == "--crossover-temperature-level2") ||
            (argument == "--crossover-temperature-level3")) {
            if ((arg_index + 1) >= argc) {
                std::cerr << "Missing value for " << argument << '\n';
                return ArgumentParseResult::kFailure;
            }

            double parsed_value = 0.0;
            if (!TryParseNonNegativeReal(argv[arg_index + 1], parsed_value) || (parsed_value > 1.0)) {
                std::cerr << "Crossover temperatures must be between 0 and 1.\n";
                return ArgumentParseResult::kFailure;
            }

            if (argument == "--crossover-temperature-level1") {
                config.crossover_temperature_level1 = static_cast<float>(parsed_value);
            } else if (argument == "--crossover-temperature-level2") {
                config.crossover_temperature_level2 = static_cast<float>(parsed_value);
            } else {
                config.crossover_temperature_level3 = static_cast<float>(parsed_value);
            }

            ++arg_index;
            continue;
        }

        if ((argument == "--checkpoint-path") || (argument == "--resume-from-checkpoint") ||
            (argument == "--telemetry-path") || (argument == "--telemetry-dir")) {
            if ((arg_index + 1) >= argc) {
                std::cerr << "Missing value for " << argument << '\n';
                return ArgumentParseResult::kFailure;
            }

            if (argv[arg_index + 1][0] == '\0') {
                std::cerr << "Path for " << argument << " must not be empty.\n";
                return ArgumentParseResult::kFailure;
            }

            if (argument == "--checkpoint-path") {
                config.checkpoint_path = argv[arg_index + 1];
                config.checkpoint_path_was_provided = true;
            } else if (argument == "--resume-from-checkpoint") {
                config.resume_checkpoint_path = argv[arg_index + 1];
                config.resume_checkpoint_path_was_provided = true;
            } else if (argument == "--telemetry-path") {
                config.telemetry_path = argv[arg_index + 1];
                config.telemetry_path_was_provided = true;
            } else {
                config.telemetry_dir = argv[arg_index + 1];
                config.telemetry_dir_was_provided = true;
            }

            ++arg_index;
            continue;
        }

        if ((argument == "--seed") || (argument == "--generations") || (argument == "--population-size") ||
            (argument == "--genotype-vram-gb") || (argument == "--generation-vram-gb") ||
            (argument == "--initial-word-count") || (argument == "--word-count-step") ||
            (argument == "--word-count-step-period") || (argument == "--shard-release-min-gap") ||
            (argument == "--breeding-radius") || (argument == "--shard-initial-radius") ||
            (argument == "--shard-radius-growth-period") || (argument == "--checkpoint-every") ||
            (argument == "--telemetry-genetic-convergence-interval")) {
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
                } else if ((argument == "--shard-release-min-gap") || (argument == "--word-count-step-period")) {
                    if (parsed_value <
                        neuroevolution::training_folder::kDefaultTrainingShardReleaseMinimumGapGenerations) {
                        std::cerr << "Shard release minimum generation gap must be at least "
                                  << neuroevolution::training_folder::kDefaultTrainingShardReleaseMinimumGapGenerations
                                  << ".\n";
                        return ArgumentParseResult::kFailure;
                    }

                    config.shard_release_min_gap_generations = static_cast<std::size_t>(parsed_value);
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
                } else if (argument == "--telemetry-genetic-convergence-interval") {
                    if (parsed_value == 0) {
                        std::cerr << "Genetic convergence telemetry interval must be at least 1.\n";
                        return ArgumentParseResult::kFailure;
                    }

                    config.telemetry_genetic_convergence_interval = static_cast<std::size_t>(parsed_value);
                    config.telemetry_genetic_convergence_interval_was_provided = true;
                } else {
                    if (parsed_value == 0) {
                        std::cerr << "Shard radius growth period must be at least 1.\n";
                        return ArgumentParseResult::kFailure;
                    }

                    config.shard_radius_growth_period_generations = static_cast<std::size_t>(parsed_value);
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
    if (config.telemetry_path_was_provided && config.telemetry_dir_was_provided) {
        std::cerr << "Use either --telemetry-path or --telemetry-dir, not both.\n";
        return ArgumentParseResult::kFailure;
    }
    if (config.telemetry_genetic_convergence && !config.telemetry_path_was_provided &&
        !config.telemetry_dir_was_provided) {
        std::cerr << "--telemetry-genetic-convergence requires --telemetry-path or --telemetry-dir.\n";
        return ArgumentParseResult::kFailure;
    }
    if (config.telemetry_genetic_convergence_interval_was_provided && !config.telemetry_genetic_convergence) {
        std::cerr << "--telemetry-genetic-convergence-interval requires --telemetry-genetic-convergence.\n";
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

struct GenerationFitnessTelemetryRow {
    std::size_t generation = 0;
    std::size_t population_size = 0;
    std::size_t training_word_count = 0;
    float fitness_min = 0.0f;
    float fitness_mean = 0.0f;
    float fitness_median = 0.0f;
    float fitness_p90 = 0.0f;
    float fitness_p99 = 0.0f;
    float fitness_max = 0.0f;
    float fitness_stddev = 0.0f;
    std::size_t distinct_fitness_count = 0;
    bool has_breeding_metrics = false;
    std::size_t parent_childless_count = 0;
    std::size_t parent_one_child_count = 0;
    std::size_t parent_multiple_children_count = 0;
};

struct GeneticConvergenceTelemetryRow {
    std::size_t generation = 0;
    std::size_t sample_organisms = 0;
    std::size_t sample_weights = 0;
    std::size_t pair_count = 0;
    float centroid_distance_mean = 0.0f;
    float centroid_distance_min = 0.0f;
    float centroid_distance_max = 0.0f;
    float pairwise_distance_mean = 0.0f;
    float pairwise_distance_min = 0.0f;
    float pairwise_distance_max = 0.0f;
    float elapsed_ms = 0.0f;
};

class TelemetryWriter {
  public:
    TelemetryWriter() = default;

    ~TelemetryWriter() {
        if (db_ != nullptr) {
            sqlite3_close(db_);
        }
    }

    TelemetryWriter(const TelemetryWriter &) = delete;
    TelemetryWriter &operator=(const TelemetryWriter &) = delete;

    bool TryOpen(const std::filesystem::path &telemetry_path) {
        path_ = telemetry_path;
        const std::filesystem::path parent_path = telemetry_path.parent_path();
        if (!parent_path.empty()) {
            std::error_code error_code;
            std::filesystem::create_directories(parent_path, error_code);
            if (error_code) {
                std::cerr << "Could not create telemetry directory " << parent_path.string() << ": "
                          << error_code.message() << '\n';
                return false;
            }
        }

        if (sqlite3_open(telemetry_path.string().c_str(), &db_) != SQLITE_OK) {
            std::cerr << "Could not open telemetry SQLite database " << telemetry_path.string() << ": " << LastError()
                      << '\n';
            return false;
        }

        return TryExec("PRAGMA journal_mode=WAL;") && TryExec("PRAGMA synchronous=NORMAL;") &&
               TryExec("PRAGMA busy_timeout=5000;") &&
               TryExec("CREATE TABLE IF NOT EXISTS generation_fitness ("
                       "generation INTEGER PRIMARY KEY,"
                       "population_size INTEGER NOT NULL DEFAULT 0,"
                       "training_word_count INTEGER NOT NULL DEFAULT 0,"
                       "fitness_min REAL NOT NULL,"
                       "fitness_max REAL NOT NULL,"
                       "fitness_mean REAL NOT NULL,"
                       "fitness_median REAL NOT NULL,"
                       "fitness_p90 REAL NOT NULL DEFAULT 0,"
                       "fitness_p99 REAL NOT NULL DEFAULT 0,"
                       "fitness_stddev REAL NOT NULL DEFAULT 0,"
                       "distinct_fitness_count INTEGER NOT NULL DEFAULT 0,"
                       "parent_childless_count INTEGER,"
                       "parent_one_child_count INTEGER,"
                       "parent_multiple_children_count INTEGER,"
                       "logged_at TEXT NOT NULL"
                       ");") &&
               TryEnsureColumn("population_size", "INTEGER NOT NULL DEFAULT 0") &&
               TryEnsureColumn("training_word_count", "INTEGER NOT NULL DEFAULT 0") &&
               TryEnsureColumn("fitness_p90", "REAL NOT NULL DEFAULT 0") &&
               TryEnsureColumn("fitness_p99", "REAL NOT NULL DEFAULT 0") &&
               TryEnsureColumn("fitness_stddev", "REAL NOT NULL DEFAULT 0") &&
               TryEnsureColumn("distinct_fitness_count", "INTEGER NOT NULL DEFAULT 0") &&
               TryEnsureColumn("parent_childless_count", "INTEGER") &&
               TryEnsureColumn("parent_one_child_count", "INTEGER") &&
               TryEnsureColumn("parent_multiple_children_count", "INTEGER");
    }

    bool TryEnsureGeneticConvergenceTable() {
        return TryExec("CREATE TABLE IF NOT EXISTS genetic_convergence_telemetry ("
                       "generation INTEGER NOT NULL PRIMARY KEY,"
                       "sample_organisms INTEGER NOT NULL,"
                       "sample_weights INTEGER NOT NULL,"
                       "pair_count INTEGER NOT NULL,"
                       "centroid_distance_mean REAL NOT NULL,"
                       "centroid_distance_min REAL NOT NULL,"
                       "centroid_distance_max REAL NOT NULL,"
                       "pairwise_distance_mean REAL NOT NULL,"
                       "pairwise_distance_min REAL NOT NULL,"
                       "pairwise_distance_max REAL NOT NULL,"
                       "elapsed_ms REAL NOT NULL"
                       ");");
    }

    bool TryLogGenerationFitness(const GenerationFitnessTelemetryRow &row) {
        static constexpr const char *kInsertSql =
            "INSERT OR REPLACE INTO generation_fitness "
            "(generation, population_size, training_word_count, fitness_min, fitness_mean, fitness_median, "
            "fitness_p90, fitness_p99, fitness_max, fitness_stddev, distinct_fitness_count, "
            "parent_childless_count, parent_one_child_count, parent_multiple_children_count, logged_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'));";

        sqlite3_stmt *statement = nullptr;
        if (sqlite3_prepare_v2(db_, kInsertSql, -1, &statement, nullptr) != SQLITE_OK) {
            std::cerr << "Could not prepare telemetry insert for " << path_.string() << ": " << LastError() << '\n';
            return false;
        }

        bool ok = true;
        ok &= sqlite3_bind_int64(statement, 1, static_cast<sqlite3_int64>(row.generation)) == SQLITE_OK;
        ok &= sqlite3_bind_int64(statement, 2, static_cast<sqlite3_int64>(row.population_size)) == SQLITE_OK;
        ok &= sqlite3_bind_int64(statement, 3, static_cast<sqlite3_int64>(row.training_word_count)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 4, static_cast<double>(row.fitness_min)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 5, static_cast<double>(row.fitness_mean)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 6, static_cast<double>(row.fitness_median)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 7, static_cast<double>(row.fitness_p90)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 8, static_cast<double>(row.fitness_p99)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 9, static_cast<double>(row.fitness_max)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 10, static_cast<double>(row.fitness_stddev)) == SQLITE_OK;
        ok &= sqlite3_bind_int64(statement, 11, static_cast<sqlite3_int64>(row.distinct_fitness_count)) == SQLITE_OK;
        if (row.has_breeding_metrics) {
            ok &=
                sqlite3_bind_int64(statement, 12, static_cast<sqlite3_int64>(row.parent_childless_count)) == SQLITE_OK;
            ok &=
                sqlite3_bind_int64(statement, 13, static_cast<sqlite3_int64>(row.parent_one_child_count)) == SQLITE_OK;
            ok &= sqlite3_bind_int64(statement, 14, static_cast<sqlite3_int64>(row.parent_multiple_children_count)) ==
                  SQLITE_OK;
        } else {
            ok &= sqlite3_bind_null(statement, 12) == SQLITE_OK;
            ok &= sqlite3_bind_null(statement, 13) == SQLITE_OK;
            ok &= sqlite3_bind_null(statement, 14) == SQLITE_OK;
        }

        if (!ok) {
            std::cerr << "Could not bind telemetry row for " << path_.string() << ": " << LastError() << '\n';
            sqlite3_finalize(statement);
            return false;
        }

        const int step_result = sqlite3_step(statement);
        if (step_result != SQLITE_DONE) {
            std::cerr << "Could not write telemetry row for generation " << row.generation << " to " << path_.string()
                      << ": " << LastError() << '\n';
            sqlite3_finalize(statement);
            return false;
        }

        sqlite3_finalize(statement);
        return true;
    }

    bool TryLogGeneticConvergence(const GeneticConvergenceTelemetryRow &row) {
        static constexpr const char *kInsertSql =
            "INSERT OR REPLACE INTO genetic_convergence_telemetry "
            "(generation, sample_organisms, sample_weights, pair_count, centroid_distance_mean, "
            "centroid_distance_min, centroid_distance_max, pairwise_distance_mean, pairwise_distance_min, "
            "pairwise_distance_max, elapsed_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

        sqlite3_stmt *statement = nullptr;
        if (sqlite3_prepare_v2(db_, kInsertSql, -1, &statement, nullptr) != SQLITE_OK) {
            std::cerr << "Could not prepare genetic convergence telemetry insert for " << path_.string() << ": "
                      << LastError() << '\n';
            return false;
        }

        bool ok = true;
        ok &= sqlite3_bind_int64(statement, 1, static_cast<sqlite3_int64>(row.generation)) == SQLITE_OK;
        ok &= sqlite3_bind_int64(statement, 2, static_cast<sqlite3_int64>(row.sample_organisms)) == SQLITE_OK;
        ok &= sqlite3_bind_int64(statement, 3, static_cast<sqlite3_int64>(row.sample_weights)) == SQLITE_OK;
        ok &= sqlite3_bind_int64(statement, 4, static_cast<sqlite3_int64>(row.pair_count)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 5, static_cast<double>(row.centroid_distance_mean)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 6, static_cast<double>(row.centroid_distance_min)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 7, static_cast<double>(row.centroid_distance_max)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 8, static_cast<double>(row.pairwise_distance_mean)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 9, static_cast<double>(row.pairwise_distance_min)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 10, static_cast<double>(row.pairwise_distance_max)) == SQLITE_OK;
        ok &= sqlite3_bind_double(statement, 11, static_cast<double>(row.elapsed_ms)) == SQLITE_OK;
        if (!ok) {
            std::cerr << "Could not bind genetic convergence telemetry row for " << path_.string() << ": "
                      << LastError() << '\n';
            sqlite3_finalize(statement);
            return false;
        }

        const int step_result = sqlite3_step(statement);
        if (step_result != SQLITE_DONE) {
            std::cerr << "Could not write genetic convergence telemetry row for generation " << row.generation << " to "
                      << path_.string() << ": " << LastError() << '\n';
            sqlite3_finalize(statement);
            return false;
        }

        sqlite3_finalize(statement);
        return true;
    }

    const std::filesystem::path &path() const noexcept { return path_; }

  private:
    const char *LastError() const noexcept { return (db_ == nullptr) ? "unknown SQLite error" : sqlite3_errmsg(db_); }

    bool TryExec(const char *sql) {
        char *error_message = nullptr;
        if (sqlite3_exec(db_, sql, nullptr, nullptr, &error_message) != SQLITE_OK) {
            std::cerr << "Telemetry SQLite statement failed for " << path_.string() << ": "
                      << ((error_message == nullptr) ? LastError() : error_message) << '\n';
            sqlite3_free(error_message);
            return false;
        }

        return true;
    }

    bool TryColumnExists(const char *column_name, bool &exists_out) {
        exists_out = false;
        sqlite3_stmt *statement = nullptr;
        if (sqlite3_prepare_v2(db_, "PRAGMA table_info(generation_fitness);", -1, &statement, nullptr) != SQLITE_OK) {
            std::cerr << "Could not inspect telemetry schema for " << path_.string() << ": " << LastError() << '\n';
            return false;
        }

        while (sqlite3_step(statement) == SQLITE_ROW) {
            const unsigned char *name = sqlite3_column_text(statement, 1);
            if ((name != nullptr) && (std::string_view(reinterpret_cast<const char *>(name)) == column_name)) {
                exists_out = true;
                break;
            }
        }

        sqlite3_finalize(statement);
        return true;
    }

    bool TryEnsureColumn(const char *column_name, const char *column_definition) {
        bool column_exists = false;
        if (!TryColumnExists(column_name, column_exists)) {
            return false;
        }
        if (column_exists) {
            return true;
        }

        const std::string sql =
            std::string("ALTER TABLE generation_fitness ADD COLUMN ") + column_name + " " + column_definition + ";";
        return TryExec(sql.c_str());
    }

    sqlite3 *db_ = nullptr;
    std::filesystem::path path_{};
};

template <typename Value> class DeviceTelemetryBuffer {
  public:
    DeviceTelemetryBuffer() = default;
    ~DeviceTelemetryBuffer() { cudaFree(data_); }

    DeviceTelemetryBuffer(const DeviceTelemetryBuffer &) = delete;
    DeviceTelemetryBuffer &operator=(const DeviceTelemetryBuffer &) = delete;

    bool TryAllocate(const std::size_t count, const std::string_view label) {
        count_ = count;
        if (count == 0) {
            return true;
        }
        return CheckCuda(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(Value)), label);
    }

    Value *data() noexcept { return data_; }
    const Value *data() const noexcept { return data_; }
    std::size_t count() const noexcept { return count_; }

  private:
    Value *data_ = nullptr;
    std::size_t count_ = 0;
};

__device__ float GeneticConvergenceTrainableValueAt(const std::uint8_t *genome_bytes, const std::size_t weight_index,
                                                    const std::size_t action_count) {
    namespace common = neuroevolution::common;
    namespace genome = neuroevolution::genetic_algorithm::genome;
    namespace output_embedding = neuroevolution::model::output_embedding;

    constexpr std::size_t kPolicyScalarCount = sizeof(genome::PolicyModelParameters) / sizeof(common::Float16);
    if (weight_index < kPolicyScalarCount) {
        const auto *policy_values = reinterpret_cast<const common::Float16 *>(genome_bytes);
        return common::ToFloat(policy_values[weight_index]);
    }

    const std::size_t tail_index = weight_index - kPolicyScalarCount;
    const std::size_t tail_value_count = action_count * output_embedding::kTrainableFeatureDimension;
    if (tail_index >= tail_value_count) {
        return 0.0f;
    }

    const auto *tail_rows = genome::GenomeTailRows(genome_bytes);
    const std::size_t action_index = tail_index / output_embedding::kTrainableFeatureDimension;
    const std::size_t feature_index = tail_index % output_embedding::kTrainableFeatureDimension;
    return common::ToFloat(tail_rows[action_index][feature_index]);
}

__global__ void GatherGeneticConvergenceSamplesKernel(
    const std::uint8_t *slab_storage, const neuroevolution::genetic_algorithm::genotype_slab::GenotypeSlabLayout layout,
    const std::uint32_t *sampled_slot_indices, const std::size_t sample_organism_count,
    const std::uint32_t *sampled_weight_indices, const std::size_t sample_weight_count, float *sampled_values) {
    namespace genotype_slab = neuroevolution::genetic_algorithm::genotype_slab;

    const std::size_t flat_index =
        (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
    const std::size_t total_value_count = sample_organism_count * sample_weight_count;
    if (flat_index >= total_value_count) {
        return;
    }

    const std::size_t organism_sample_index = flat_index / sample_weight_count;
    const std::size_t weight_sample_index = flat_index % sample_weight_count;
    const std::uint32_t slot_index = sampled_slot_indices[organism_sample_index];
    if (slot_index >= layout.slot_count) {
        sampled_values[flat_index] = 0.0f;
        return;
    }

    const std::uint8_t *genome_bytes = genotype_slab::SlabSlotBytesAt(slab_storage, layout, slot_index);
    sampled_values[flat_index] = GeneticConvergenceTrainableValueAt(
        genome_bytes, sampled_weight_indices[weight_sample_index], layout.action_count);
}

bool ShouldLogGeneticConvergenceTelemetry(const CliConfig &config, const std::size_t generation) noexcept {
    return config.telemetry_genetic_convergence && ((generation % config.telemetry_genetic_convergence_interval) == 0);
}

bool TryCollectGeneticConvergenceTelemetry(const DeviceSlabGARuntimeBuffers &buffers, const std::uint32_t run_seed,
                                           GeneticConvergenceTelemetryRow &row_out) {
    namespace convergence = neuroevolution::genetic_algorithm::genetic_convergence;
    namespace genotype_slab = neuroevolution::genetic_algorithm::genotype_slab;

    row_out = {};
    const std::size_t population_size = buffers.genotype_slab.current_generation_size;
    if (population_size == 0) {
        std::cerr << "Cannot collect genetic convergence telemetry for an empty generation.\n";
        return false;
    }

    const std::size_t trainable_weight_count =
        convergence::TrainableScalarCountForActionCount(buffers.genotype_slab.slab_layout.action_count);
    const convergence::GeneticConvergenceSamplePlan sample_plan = convergence::MakeSamplePlan(
        population_size, trainable_weight_count, run_seed, buffers.genotype_slab.current_generation_index);
    if (sample_plan.organism_indices.empty() || sample_plan.weight_indices.empty()) {
        std::cerr << "Cannot collect genetic convergence telemetry without sampled organisms and weights.\n";
        return false;
    }

    std::vector<std::uint32_t> current_slot_indices(population_size);
    if (!CheckCuda(cudaMemcpy(current_slot_indices.data(), buffers.genotype_slab.current_slot_indices,
                              population_size * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
                   "copying current generation slot indices for genetic convergence telemetry")) {
        return false;
    }

    std::vector<std::uint32_t> sampled_slot_indices{};
    sampled_slot_indices.reserve(sample_plan.organism_indices.size());
    for (const std::size_t organism_index : sample_plan.organism_indices) {
        if (organism_index >= current_slot_indices.size()) {
            std::cerr << "Genetic convergence telemetry sampled an invalid organism index.\n";
            return false;
        }

        const std::uint32_t slot_index = current_slot_indices[organism_index];
        if (slot_index == genotype_slab::kInvalidSlabSlotIndex) {
            std::cerr << "Genetic convergence telemetry sampled an organism without a live slab slot.\n";
            return false;
        }
        sampled_slot_indices.push_back(slot_index);
    }

    std::vector<std::uint32_t> sampled_weight_indices{};
    sampled_weight_indices.reserve(sample_plan.weight_indices.size());
    for (const std::size_t weight_index : sample_plan.weight_indices) {
        if (weight_index > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            std::cerr << "Genetic convergence telemetry sampled a weight index outside uint32 range.\n";
            return false;
        }
        sampled_weight_indices.push_back(static_cast<std::uint32_t>(weight_index));
    }

    DeviceTelemetryBuffer<std::uint32_t> device_slot_indices{};
    DeviceTelemetryBuffer<std::uint32_t> device_weight_indices{};
    DeviceTelemetryBuffer<float> device_sampled_values{};
    const std::size_t sample_value_count = sampled_slot_indices.size() * sampled_weight_indices.size();
    if (!device_slot_indices.TryAllocate(sampled_slot_indices.size(),
                                         "allocating genetic convergence sampled slot indices") ||
        !device_weight_indices.TryAllocate(sampled_weight_indices.size(),
                                           "allocating genetic convergence sampled weight indices") ||
        !device_sampled_values.TryAllocate(sample_value_count, "allocating genetic convergence sampled values")) {
        return false;
    }

    if (!CheckCuda(cudaMemcpy(device_slot_indices.data(), sampled_slot_indices.data(),
                              sampled_slot_indices.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
                   "uploading genetic convergence sampled slot indices") ||
        !CheckCuda(cudaMemcpy(device_weight_indices.data(), sampled_weight_indices.data(),
                              sampled_weight_indices.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
                   "uploading genetic convergence sampled weight indices")) {
        return false;
    }

    constexpr int kThreadBlockSize = 256;
    const int block_count = static_cast<int>((sample_value_count + static_cast<std::size_t>(kThreadBlockSize) - 1U) /
                                             static_cast<std::size_t>(kThreadBlockSize));
    GatherGeneticConvergenceSamplesKernel<<<block_count, kThreadBlockSize>>>(
        buffers.genotype_slab.slab_storage, buffers.genotype_slab.slab_layout, device_slot_indices.data(),
        sampled_slot_indices.size(), device_weight_indices.data(), sampled_weight_indices.size(),
        device_sampled_values.data());
    if (!CheckCuda(cudaGetLastError(), "launching genetic convergence sample gather kernel") ||
        !CheckCuda(cudaDeviceSynchronize(), "gathering genetic convergence samples")) {
        return false;
    }

    std::vector<float> sampled_values(sample_value_count);
    if (!CheckCuda(cudaMemcpy(sampled_values.data(), device_sampled_values.data(), sample_value_count * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "copying genetic convergence sampled values")) {
        return false;
    }

    convergence::GeneticConvergenceMetrics metrics{};
    if (!convergence::TryComputeMetrics(sampled_values, sampled_slot_indices.size(), sampled_weight_indices.size(),
                                        sample_plan.pairs, metrics)) {
        std::cerr << "Could not compute genetic convergence telemetry metrics.\n";
        return false;
    }

    row_out.generation = buffers.genotype_slab.current_generation_index;
    row_out.sample_organisms = sampled_slot_indices.size();
    row_out.sample_weights = sampled_weight_indices.size();
    row_out.pair_count = sample_plan.pairs.size();
    row_out.centroid_distance_mean = metrics.centroid_distance_mean;
    row_out.centroid_distance_min = metrics.centroid_distance_min;
    row_out.centroid_distance_max = metrics.centroid_distance_max;
    row_out.pairwise_distance_mean = metrics.pairwise_distance_mean;
    row_out.pairwise_distance_min = metrics.pairwise_distance_min;
    row_out.pairwise_distance_max = metrics.pairwise_distance_max;
    return true;
}

bool TryLogGeneticConvergenceForCurrentGeneration(const DeviceSlabGARuntimeBuffers &buffers,
                                                  const CliConfig &cli_config, TelemetryWriter *telemetry_writer) {
    if ((telemetry_writer == nullptr) ||
        !ShouldLogGeneticConvergenceTelemetry(cli_config, buffers.genotype_slab.current_generation_index)) {
        return true;
    }

    GeneticConvergenceTelemetryRow row{};
    const auto start_time = ProgressClock::now();
    if (!TryCollectGeneticConvergenceTelemetry(buffers, cli_config.seed, row)) {
        return false;
    }
    row.elapsed_ms = std::chrono::duration<float, std::milli>(ProgressClock::now() - start_time).count();
    if (!telemetry_writer->TryLogGeneticConvergence(row)) {
        return false;
    }

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(1) << "[telemetry] genetic convergence generation=" << row.generation
           << " sample_organisms=" << row.sample_organisms << " sample_weights=" << row.sample_weights
           << " pair_count=" << row.pair_count << " elapsed_ms=" << row.elapsed_ms;
    std::cout << stream.str() << '\n';
    return true;
}

bool TryCollectGenerationFitnessTelemetry(const DeviceSlabGARuntimeBuffers &buffers,
                                          const RuntimeWordCounts &runtime_word_counts,
                                          const bool collect_breeding_metrics, GenerationFitnessTelemetryRow &row_out) {
    row_out = {};
    const std::size_t population_size = buffers.genotype_slab.current_generation_size;
    if (population_size == 0) {
        std::cerr << "Cannot collect telemetry for an empty generation.\n";
        return false;
    }

    std::vector<float> fitness_values(population_size);
    std::vector<std::uint8_t> has_fitness_flags(population_size);
    if (!CheckCuda(cudaMemcpy(fitness_values.data(), buffers.genotype_slab.current_fitness,
                              population_size * sizeof(float), cudaMemcpyDeviceToHost),
                   "copying fitness values for telemetry") ||
        !CheckCuda(cudaMemcpy(has_fitness_flags.data(), buffers.genotype_slab.current_has_fitness,
                              population_size * sizeof(std::uint8_t), cudaMemcpyDeviceToHost),
                   "copying fitness flags for telemetry")) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < population_size; ++individual_index) {
        if (has_fitness_flags[individual_index] == 0) {
            std::cerr << "Cannot collect telemetry before every organism has a fitness value.\n";
            return false;
        }
    }

    std::sort(fitness_values.begin(), fitness_values.end());
    double fitness_sum = 0.0;
    double squared_delta_sum = 0.0;
    for (const float fitness : fitness_values) {
        fitness_sum += static_cast<double>(fitness);
    }
    const double fitness_mean = fitness_sum / static_cast<double>(population_size);
    for (const float fitness : fitness_values) {
        const double delta = static_cast<double>(fitness) - fitness_mean;
        squared_delta_sum += delta * delta;
    }

    const std::size_t median_index = population_size / 2;
    const float median = (population_size % 2 == 0)
                             ? ((fitness_values[median_index - 1] + fitness_values[median_index]) * 0.5f)
                             : fitness_values[median_index];
    const auto nearest_rank_percentile = [&](const double percentile) {
        const std::size_t rank =
            static_cast<std::size_t>(std::ceil((percentile / 100.0) * static_cast<double>(population_size)));
        const std::size_t index = (rank == 0) ? 0 : (rank - 1);
        return fitness_values[(index < population_size) ? index : (population_size - 1)];
    };

    std::size_t distinct_fitness_count = 1;
    for (std::size_t index = 1; index < population_size; ++index) {
        if (fitness_values[index] != fitness_values[index - 1]) {
            ++distinct_fitness_count;
        }
    }

    row_out.generation = buffers.genotype_slab.current_generation_index;
    row_out.population_size = population_size;
    row_out.training_word_count = runtime_word_counts.training_word_count;
    row_out.fitness_min = fitness_values.front();
    row_out.fitness_mean = static_cast<float>(fitness_mean);
    row_out.fitness_median = median;
    row_out.fitness_p90 = nearest_rank_percentile(90.0);
    row_out.fitness_p99 = nearest_rank_percentile(99.0);
    row_out.fitness_max = fitness_values.back();
    row_out.fitness_stddev = static_cast<float>(std::sqrt(squared_delta_sum / static_cast<double>(population_size)));
    row_out.distinct_fitness_count = distinct_fitness_count;
    if (collect_breeding_metrics) {
        const std::size_t child_count = buffers.genotype_slab.planned_child_count;
        if (child_count == 0) {
            std::cerr << "Cannot collect breeding telemetry before the assembly plan is available.\n";
            return false;
        }

        std::vector<SlabParentPair> parent_pairs(child_count);
        if (!CheckCuda(cudaMemcpy(parent_pairs.data(), buffers.genotype_slab.assembly_parent_pairs,
                                  child_count * sizeof(SlabParentPair), cudaMemcpyDeviceToHost),
                       "copying assembly parent pairs for telemetry")) {
            return false;
        }

        std::vector<std::size_t> parent_child_counts(population_size, 0);
        for (const SlabParentPair &parent_pair : parent_pairs) {
            if ((parent_pair.first_parent_index >= population_size) ||
                (parent_pair.second_parent_index >= population_size)) {
                std::cerr << "Cannot collect breeding telemetry from an assembly plan with an invalid parent index.\n";
                return false;
            }
            ++parent_child_counts[parent_pair.first_parent_index];
            ++parent_child_counts[parent_pair.second_parent_index];
        }

        row_out.has_breeding_metrics = true;
        for (const std::size_t child_count_for_parent : parent_child_counts) {
            if (child_count_for_parent == 0) {
                ++row_out.parent_childless_count;
            } else if (child_count_for_parent == 1) {
                ++row_out.parent_one_child_count;
            } else {
                ++row_out.parent_multiple_children_count;
            }
        }
    }
    return true;
}

bool TryLogTelemetryForCurrentGeneration(const DeviceSlabGARuntimeBuffers &buffers,
                                         const RuntimeWordCounts &runtime_word_counts,
                                         const bool collect_breeding_metrics, TelemetryWriter *telemetry_writer) {
    if (telemetry_writer == nullptr) {
        return true;
    }

    GenerationFitnessTelemetryRow row{};
    return TryCollectGenerationFitnessTelemetry(buffers, runtime_word_counts, collect_breeding_metrics, row) &&
           telemetry_writer->TryLogGenerationFitness(row);
}

bool TryMaybeReleaseTrainingDataShard(const DeviceSlabGARuntimeBuffers &buffers,
                                      const RuntimeWordCounts &runtime_word_counts, const CliConfig &cli_config,
                                      const std::size_t catalog_word_count,
                                      TrainingDataShardReleaseHistory &training_shard_release_history,
                                      bool &release_baseline_ready,
                                      PendingOutputEmbeddingInjection &pending_output_embedding_injection,
                                      std::size_t &next_word_count_out) {
    pending_output_embedding_injection = {};
    next_word_count_out = runtime_word_counts.action_space_word_count;

    if (training_shard_release_history.release_count == 0) {
        std::cerr << "Cannot apply adaptive training-shard release without a foundation release record.\n";
        return false;
    }

    if ((runtime_word_counts.action_space_word_count >= catalog_word_count) ||
        (runtime_word_counts.training_word_schedule.word_count_step == 0)) {
        return true;
    }

    const std::size_t latest_release_index = training_shard_release_history.release_count - 1;
    if (!release_baseline_ready) {
        GenerationFitnessTelemetryRow foundation_fitness_row{};
        if (!TryCollectGenerationFitnessTelemetry(buffers, runtime_word_counts, false, foundation_fitness_row)) {
            return false;
        }

        training_shard_release_history.release_fitness_p99[latest_release_index] = foundation_fitness_row.fitness_p99;
        release_baseline_ready = true;
    }

    const std::size_t current_generation_index = buffers.genotype_slab.current_generation_index;
    const std::size_t release_generation = current_generation_index + 1;
    const std::size_t previous_release_generation =
        training_shard_release_history.release_generations[latest_release_index];
    const std::size_t generations_since_release =
        (release_generation >= previous_release_generation) ? (release_generation - previous_release_generation) : 0;
    if (generations_since_release < cli_config.shard_release_min_gap_generations) {
        return true;
    }

    GenerationFitnessTelemetryRow fitness_row{};
    if (!TryCollectGenerationFitnessTelemetry(buffers, runtime_word_counts, false, fitness_row)) {
        return false;
    }

    GeneticConvergenceTelemetryRow convergence_row{};
    const auto convergence_start_time = ProgressClock::now();
    if (!TryCollectGeneticConvergenceTelemetry(buffers, cli_config.seed, convergence_row)) {
        return false;
    }
    convergence_row.elapsed_ms =
        std::chrono::duration<float, std::milli>(ProgressClock::now() - convergence_start_time).count();

    const float previous_release_fitness_p99 = training_shard_release_history.release_fitness_p99[latest_release_index];
    const bool centroid_triggered =
        convergence_row.centroid_distance_mean <= cli_config.shard_release_centroid_distance_threshold;
    const bool fitness_triggered = fitness_row.fitness_p99 > previous_release_fitness_p99;
    if (!centroid_triggered && !fitness_triggered) {
        return true;
    }

    const std::size_t first_catalog_word_index = runtime_word_counts.action_space_word_count;
    const std::size_t remaining_catalog_words = catalog_word_count - first_catalog_word_index;
    const std::size_t injection_count =
        (remaining_catalog_words < runtime_word_counts.training_word_schedule.word_count_step)
            ? remaining_catalog_words
            : runtime_word_counts.training_word_schedule.word_count_step;
    if (injection_count == 0) {
        return true;
    }

    pending_output_embedding_injection.enabled = true;
    pending_output_embedding_injection.first_catalog_word_index = first_catalog_word_index;
    pending_output_embedding_injection.injection_count = injection_count;
    next_word_count_out = first_catalog_word_index + injection_count;

    if (!TryRecordTrainingDataShardRelease(training_shard_release_history, release_generation,
                                           fitness_row.fitness_p99)) {
        std::cerr << "Could not record adaptive training-shard release metadata.\n";
        return false;
    }

    std::ostringstream reason_stream;
    reason_stream << std::fixed << std::setprecision(4);
    if (centroid_triggered) {
        reason_stream << "centroid_distance_mean=" << convergence_row.centroid_distance_mean
                      << " <= threshold=" << cli_config.shard_release_centroid_distance_threshold;
    }
    if (fitness_triggered) {
        if (centroid_triggered) {
            reason_stream << "; ";
        }
        reason_stream << "fitness_p99=" << fitness_row.fitness_p99
                      << " > previous_release_fitness_p99=" << previous_release_fitness_p99;
    }

    std::cout << '[' << FormatCurrentLocalTimestamp() << "] Training data shard released for generation "
              << release_generation << ": first_catalog_word_index=" << first_catalog_word_index
              << ", word_count=" << injection_count << ", new_training_word_count=" << next_word_count_out
              << ", generations_since_previous_release=" << generations_since_release
              << ", reason=" << reason_stream.str() << '\n';
    return true;
}

GenerationAssemblyConfig MakeAssemblyConfig(const CliConfig &cli_config) {
    GenerationAssemblyConfig config{};
    config.parent_selection.tournament_size = 3;
    config.parent_selection.allow_self_parenting = false;
    config.parent_selection.cellular_breeding_radius = cli_config.breeding_radius;
    config.parent_selection.rank_exponent = cli_config.parent_selection_rank_exponent;
    config.breeding.crossover_temperature_level1 = cli_config.crossover_temperature_level1;
    config.breeding.crossover_temperature_level2 = cli_config.crossover_temperature_level2;
    config.breeding.crossover_temperature_level3 = cli_config.crossover_temperature_level3;
    config.mutation.mutation_probability = neuroevolution::genetic_algorithm::kDefaultMutationProbability;
    config.mutation.mutation_sigma = neuroevolution::genetic_algorithm::kDefaultMutationSigma;
    config.mutation.output_tail_row_scale_mutation_probability =
        neuroevolution::genetic_algorithm::kDefaultOutputTailRowScaleMutationProbability;
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

std::size_t CheckpointPayloadByteCount(const RuntimeCheckpoint &checkpoint) {
    std::size_t byte_count =
        checkpoint.assembly_plan.child_count * sizeof(neuroevolution::genetic_algorithm::genotype_slab::SlabParentPair);
    byte_count += sizeof(checkpoint.training_shard_release_history.release_count);
    byte_count += checkpoint.training_shard_release_history.release_count * (sizeof(std::size_t) + sizeof(float));
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
    clone_out.training_shard_release_history = source.training_shard_release_history;
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
        TrainingDataShardReleaseHistory training_shard_release_history{};
        bool training_shard_release_baseline_ready = false;

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
            training_shard_release_history = checkpoint.training_shard_release_history;
            training_shard_release_baseline_ready = training_shard_release_history.release_count > 0;

            std::cout << '[' << FormatCurrentLocalTimestamp() << "] Resumed device GA from checkpoint:\n"
                      << "  checkpoint_path=" << cli_config.resume_checkpoint_path.string() << '\n'
                      << "  checkpoint_generation=" << checkpoint.current_generation.generation_index << '\n'
                      << "  resumed_generation=" << buffers.genotype_slab.current_generation_index << '\n'
                      << "  resumed_population=" << buffers.genotype_slab.current_generation_size << '\n'
                      << "  checkpoint_child_count=" << checkpoint.assembly_plan.child_count << '\n'
                      << "  checkpoint_live_genotypes=" << checkpoint.live_genotypes.size() << '\n'
                      << "  checkpoint_payload_bytes=" << CheckpointPayloadByteCount(checkpoint) << '\n'
                      << "  action_count=" << runtime_word_counts.action_space_word_count << '\n'
                      << "  training_shard_release_count=" << training_shard_release_history.release_count << '\n'
                      << "  shard_release_min_gap_generations=" << cli_config.shard_release_min_gap_generations << '\n'
                      << "  shard_release_centroid_distance_threshold="
                      << cli_config.shard_release_centroid_distance_threshold << '\n'
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
            if (!TryRecordTrainingDataShardRelease(training_shard_release_history, 0, 0.0f)) {
                std::cerr << "Could not initialize the foundation training-shard release record.\n";
                return 1;
            }

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
                      << "  shard_release_min_gap_generations=" << cli_config.shard_release_min_gap_generations << '\n'
                      << "  shard_release_centroid_distance_threshold="
                      << cli_config.shard_release_centroid_distance_threshold << '\n'
                      << "  training_shard_release_count=" << training_shard_release_history.release_count << '\n'
                      << "  breeding_radius=" << assembly_config.parent_selection.cellular_breeding_radius << '\n'
                      << "  parent_selection_rank_exponent=" << assembly_config.parent_selection.rank_exponent << '\n'
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
        const std::size_t required_training_shard_release_count = TrainingDataShardCountForIntroducedWordCount(
            runtime_word_counts.training_word_schedule, runtime_word_counts.training_word_count);
        if ((required_training_shard_release_count == 0) ||
            (training_shard_release_history.release_count < required_training_shard_release_count)) {
            std::cerr << "The training-shard release history is not compatible with the active training-word count.\n";
            DestroyDeviceSlabGARuntimeBuffers(buffers);
            return 1;
        }

        TelemetryWriter telemetry_writer{};
        TelemetryWriter *active_telemetry_writer = nullptr;
        if (cli_config.telemetry_path_was_provided || cli_config.telemetry_dir_was_provided) {
            const std::filesystem::path telemetry_path = cli_config.telemetry_path_was_provided
                                                             ? cli_config.telemetry_path
                                                             : MakeTelemetryPathFromDirectory(cli_config.telemetry_dir);
            if (!telemetry_writer.TryOpen(telemetry_path)) {
                DestroyDeviceSlabGARuntimeBuffers(buffers);
                return 1;
            }
            if (cli_config.telemetry_genetic_convergence && !telemetry_writer.TryEnsureGeneticConvergenceTable()) {
                DestroyDeviceSlabGARuntimeBuffers(buffers);
                return 1;
            }

            active_telemetry_writer = &telemetry_writer;
            std::cout << "  telemetry_path=" << telemetry_writer.path().string() << '\n';
            if (cli_config.telemetry_genetic_convergence) {
                std::cout << "  telemetry_genetic_convergence_interval="
                          << cli_config.telemetry_genetic_convergence_interval << '\n';
            }
        }

        const auto log_telemetry_after_fitness = [&](const DeviceSlabGARuntimeBuffers &telemetry_buffers,
                                                     const RuntimeWordCounts &telemetry_word_counts) {
            return TryLogTelemetryForCurrentGeneration(telemetry_buffers, telemetry_word_counts, true,
                                                       active_telemetry_writer) &&
                   TryLogGeneticConvergenceForCurrentGeneration(telemetry_buffers, cli_config, active_telemetry_writer);
        };

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
            std::size_t next_word_count = runtime_word_counts.action_space_word_count;
            PendingOutputEmbeddingInjection pending_output_embedding_injection{};
            const auto release_training_shard_after_fitness =
                [&](const DeviceSlabGARuntimeBuffers &release_buffers, const RuntimeWordCounts &release_word_counts,
                    PendingOutputEmbeddingInjection &release_pending_output_embedding_injection) {
                    return TryMaybeReleaseTrainingDataShard(
                        release_buffers, release_word_counts, cli_config, training_word_catalog.word_count,
                        training_shard_release_history, training_shard_release_baseline_ready,
                        release_pending_output_embedding_injection, next_word_count);
                };
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
                                                      cli_config.verbose, log_telemetry_after_fitness,
                                                      release_training_shard_after_fitness,
                                                      &training_shard_release_history)) {
                        (void)ReportDeviceSlabRuntimeFailure(buffers, "Next-generation assembly");
                        DestroyDeviceSlabGARuntimeBuffers(buffers);
                        return 1;
                    }
                } else {
                    RuntimeCheckpoint checkpoint{};
                    if (!TryCreatePrebreedingCheckpointOnDevice(
                            buffers, generation_seed, runtime_word_counts, assembly_config,
                            pending_output_embedding_injection, checkpoint, &training_word_catalog, cli_config.verbose,
                            log_telemetry_after_fitness, release_training_shard_after_fitness,
                            &training_shard_release_history)) {
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
                                                  cli_config.verbose, log_telemetry_after_fitness,
                                                  release_training_shard_after_fitness,
                                                  &training_shard_release_history)) {
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

            runtime_word_counts.training_word_count = next_word_count;
            runtime_word_counts.action_space_word_count = next_word_count;
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
        if (!TryEvaluateCurrentGenerationFitnessOnDevice(buffers, runtime_word_counts,
                                                         &training_shard_release_history)) {
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
        if (!TryLogTelemetryForCurrentGeneration(buffers, runtime_word_counts, false, active_telemetry_writer) ||
            !TryLogGeneticConvergenceForCurrentGeneration(buffers, cli_config, active_telemetry_writer)) {
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
