#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <vector>

#include "genetic_algorithm/device/runtime_common.hpp"
#include "genetic_algorithm/generation_assembly.hpp"
#include "genetic_algorithm/genotype_slab/device_runtime.hpp"
#include "training_folder/training_data.hpp"

namespace neuroevolution::genetic_algorithm::slab_device {

using device_common::PopulationFitnessSummary;
using device_common::RuntimeWordCounts;

enum class DeviceSlabGARuntimeStatusCode : int {
    kOk = 0,
    kCudaFailure = 1,
    kInvalidRuntimeConfig = 2,
    kInvalidSlab = 3,
    kInvalidGeneration = 4,
    kInvalidTrainingShard = 5,
    kGuessAppendFailed = 6,
    kPolicyForwardFailed = 7,
    kActionSelectionFailed = 8,
    kPopulationNotEvaluated = 9,
    kInvalidAssemblyConfig = 10,
    kParentSelectionFailed = 11,
    kInvalidAssemblyPlan = 12,
    kInvalidParentIndex = 13,
    kSlabFull = 14,
    kOutputEmbeddingInjectionFailed = 15,
    kSlabRepackFailed = 16,
};

struct DeviceSlabGARuntimeConfig {
    std::size_t genotype_slab_byte_budget_bytes = 0;
    std::size_t generation_byte_budget_bytes = 0;
    std::size_t host_spillover_byte_budget_bytes = 0;
    std::size_t action_count = 0;
    std::size_t population_size_ceiling = 0;
    std::size_t grid_column_count = 0;
};

constexpr bool IsValidDeviceSlabGARuntimeConfig(const DeviceSlabGARuntimeConfig &config) noexcept {
    return (config.genotype_slab_byte_budget_bytes >= config.generation_byte_budget_bytes) &&
           (config.action_count > 0) && (config.grid_column_count > 0) &&
           ((config.population_size_ceiling == 0) ||
            ((config.population_size_ceiling % config.grid_column_count) == 0)) &&
           (genotype_slab::SlabSlotCountForByteBudget(config.genotype_slab_byte_budget_bytes, config.action_count) >
            0) &&
           ((genotype_slab::SlabSlotCountForByteBudget(config.generation_byte_budget_bytes, config.action_count,
                                                       config.population_size_ceiling) /
             config.grid_column_count) > 0);
}

struct DeviceSlabGARuntimeBuffers {
    genotype_slab::device::DeviceSlabRuntimeBuffers genotype_slab{};
    PopulationFitnessSummary *summary = nullptr;
    int *status = nullptr;
    training_folder::TrainingDataShardRuntime *active_training_shards = nullptr;
    std::uint32_t *current_local_training_word_counts = nullptr;
    float *fitness_partial_sums = nullptr;
    std::size_t max_generation_size = 0;
    std::size_t generation_byte_budget_bytes = 0;
    std::size_t host_spillover_byte_budget_bytes = 0;
    std::size_t grid_column_count = 0;
    ::neuroevolution::spatial::CellularGridShape epicenter_grid_shape{};
    std::size_t active_training_shard_capacity = 0;
    std::size_t active_training_shard_count = 0;
    std::size_t host_spillover_count = 0;
    bool last_generation_used_host_spillover = false;
};

using PendingOutputEmbeddingInjection = genotype_slab::device::PendingOutputEmbeddingInjection;
using DeviceSlabBootstrapConfig = genotype_slab::device::DeviceSlabBootstrapConfig;
using PostFitnessEvaluationCallback =
    std::function<bool(const DeviceSlabGARuntimeBuffers &, const RuntimeWordCounts &)>;
using TrainingShardReleaseCallback = std::function<bool(const DeviceSlabGARuntimeBuffers &, const RuntimeWordCounts &,
                                                        PendingOutputEmbeddingInjection &)>;

constexpr std::uint32_t kRuntimeCheckpointSchemaVersion = 3;
constexpr std::uint32_t kRuntimeCheckpointGenomeLayoutVersion = 1;

enum class RuntimeCheckpointResumePhase : std::uint32_t {
    kPreRecombinationPreMutation = 1,
};

struct RuntimeCheckpointGenotypeRecord {
    std::uint32_t organism_index = 0;
    std::vector<std::uint8_t> genome_bytes{};
};

struct RuntimeCheckpoint {
    std::uint32_t schema_version = kRuntimeCheckpointSchemaVersion;
    std::uint32_t genome_layout_version = kRuntimeCheckpointGenomeLayoutVersion;
    RuntimeCheckpointResumePhase resume_phase = RuntimeCheckpointResumePhase::kPreRecombinationPreMutation;
    std::uint64_t checksum = 0;
    std::uint64_t training_data_identity_hash = 0;
    std::uint32_t generation_seed = 0;
    RuntimeWordCounts runtime_word_counts{};
    training_folder::TrainingDataShardReleaseHistory training_shard_release_history{};
    GenerationAssemblyConfig assembly_config{};
    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
    DeviceSlabGARuntimeConfig runtime_config{};
    genotype_slab::GenotypeSlabLayout slab_layout{};
    ::neuroevolution::spatial::CellularGridShape current_grid_shape{};
    ::neuroevolution::spatial::CellularGridShape next_grid_shape{};
    ::neuroevolution::spatial::CellularGridShape epicenter_grid_shape{};
    genotype_slab::SlabGeneration current_generation{};
    genotype_slab::SlabAssemblyPlan assembly_plan{};
    std::vector<RuntimeCheckpointGenotypeRecord> live_genotypes{};
};

bool TryCreateDeviceSlabGARuntimeBuffers(DeviceSlabGARuntimeBuffers &buffers, const DeviceSlabGARuntimeConfig &config);

void DestroyDeviceSlabGARuntimeBuffers(DeviceSlabGARuntimeBuffers &buffers) noexcept;

bool TryBootstrapRandomCurrentGenerationOnDevice(DeviceSlabGARuntimeBuffers &buffers, std::size_t generation_size,
                                                 std::uint32_t generation_seed, std::size_t generation_index = 0,
                                                 const DeviceSlabBootstrapConfig &config = {});

bool TryDownloadSlabFromDevice(const DeviceSlabGARuntimeBuffers &buffers, genotype_slab::HostGenotypeSlab &host_buffer);

bool TryDownloadSlabSlotBytesFromDevice(const DeviceSlabGARuntimeBuffers &buffers, std::uint32_t slot_index,
                                        std::unique_ptr<std::uint8_t[]> &slot_bytes, std::size_t &slot_byte_count);

bool TryDownloadCurrentGenerationFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                                            genotype_slab::SlabGeneration &generation);

bool TryDownloadNextGenerationFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                                         genotype_slab::SlabGeneration &generation);

bool TryEvaluateCurrentGenerationFitnessOnDevice(
    DeviceSlabGARuntimeBuffers &buffers, const RuntimeWordCounts &runtime_word_counts,
    const training_folder::TrainingDataShardReleaseHistory *training_shard_release_history = nullptr);

bool TryReadPopulationFitnessSummaryFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                                               PopulationFitnessSummary &summary);

bool TryAdvanceGenerationOnDevice(
    DeviceSlabGARuntimeBuffers &buffers, std::uint32_t generation_seed, const RuntimeWordCounts &runtime_word_counts,
    const GenerationAssemblyConfig &config = {},
    const PendingOutputEmbeddingInjection &pending_output_embedding_injection = {},
    const training_folder::TrainingWordCatalog *host_training_word_catalog = nullptr, bool verbose = false,
    const PostFitnessEvaluationCallback &post_fitness_evaluation_callback = {},
    const TrainingShardReleaseCallback &training_shard_release_callback = {},
    const training_folder::TrainingDataShardReleaseHistory *training_shard_release_history = nullptr);

bool TryCreatePrebreedingCheckpointOnDevice(
    DeviceSlabGARuntimeBuffers &buffers, std::uint32_t generation_seed, const RuntimeWordCounts &runtime_word_counts,
    const GenerationAssemblyConfig &config, const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
    RuntimeCheckpoint &checkpoint_out, const training_folder::TrainingWordCatalog *host_training_word_catalog = nullptr,
    bool verbose = false, const PostFitnessEvaluationCallback &post_fitness_evaluation_callback = {},
    const TrainingShardReleaseCallback &training_shard_release_callback = {},
    const training_folder::TrainingDataShardReleaseHistory *training_shard_release_history = nullptr);

bool TryRestorePrebreedingCheckpointToDevice(const RuntimeCheckpoint &checkpoint, DeviceSlabGARuntimeBuffers &buffers);

bool TryResumeGenerationFromCheckpointOnDevice(DeviceSlabGARuntimeBuffers &buffers, const RuntimeCheckpoint &checkpoint,
                                               bool verbose = false);

bool TryWriteRuntimeCheckpointAtomically(const RuntimeCheckpoint &checkpoint,
                                         const std::filesystem::path &checkpoint_path);

bool TryReadRuntimeCheckpoint(const std::filesystem::path &checkpoint_path, RuntimeCheckpoint &checkpoint_out);

class RuntimeCheckpointAsyncWriter {
  public:
    bool TryStartWrite(RuntimeCheckpoint checkpoint, std::filesystem::path checkpoint_path);
    bool IsWriteInProgress();
    bool TryCollectFinishedWrite(bool &write_finished_out, bool &write_succeeded_out);
    bool TryWaitForWrite();

  private:
    std::future<bool> pending_write_{};
};

void SwapDeviceSlabGenerationBuffers(DeviceSlabGARuntimeBuffers &buffers) noexcept;

bool TryReadDeviceSlabGARuntimeStatus(const DeviceSlabGARuntimeBuffers &buffers,
                                      DeviceSlabGARuntimeStatusCode &status_code);

const char *DeviceSlabGARuntimeStatusCodeString(DeviceSlabGARuntimeStatusCode status_code) noexcept;

} // namespace neuroevolution::genetic_algorithm::slab_device
