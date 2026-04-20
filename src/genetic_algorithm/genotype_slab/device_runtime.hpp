#pragma once

#include <cstddef>
#include <cstdint>

#include "genetic_algorithm/breeding.hpp"
#include "genetic_algorithm/genotype_slab/assembly.hpp"
#include "genetic_algorithm/mutation.hpp"

namespace neuroevolution::genetic_algorithm::genotype_slab::device {

enum class DeviceSlabRuntimeStatusCode : int {
    kOk = 0,
    kCudaFailure = 1,
    kInvalidRuntimeConfig = 2,
    kInvalidSlab = 3,
    kInvalidGeneration = 4,
    kInvalidAssemblyPlan = 5,
    kInvalidAssemblyConfig = 6,
    kInvalidParentIndex = 7,
    kSlabFull = 8,
    kOutputEmbeddingInjectionFailed = 9,
    kSlabRepackFailed = 10,
};

struct PendingOutputEmbeddingInjection {
    bool enabled = false;
    std::size_t first_catalog_word_index = 0;
    std::size_t injection_count = 0;
};

struct SlabDeviceAssemblyConfig {
    BreedingConfig breeding{};
    MutationConfig mutation{};
    std::size_t parent_action_count = 0;
    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidSlabDeviceAssemblyConfig(const SlabDeviceAssemblyConfig &config) noexcept {
    return IsValidBreedingConfig(config.breeding) && IsValidMutationConfig(config.mutation);
}

struct DeviceSlabRuntimeConfig {
    std::size_t slot_count = 0;
    std::size_t action_count = 0;
    std::size_t max_generation_size = 0;
};

constexpr bool IsValidDeviceSlabRuntimeConfig(const DeviceSlabRuntimeConfig &config) noexcept {
    return (config.slot_count > 0) && (config.action_count > 0) && (config.max_generation_size > 0);
}

struct DeviceSlabBootstrapConfig {
    float dense_weight_gain = 1.0f;
    float output_embedding_tail_stddev = 0.05f;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidDeviceSlabBootstrapConfig(const DeviceSlabBootstrapConfig &config) noexcept {
    return (config.dense_weight_gain > 0.0f) && (config.output_embedding_tail_stddev >= 0.0f);
}

struct DeviceSlabRuntimeBuffers {
    std::uint8_t *slab_storage = nullptr;
    SlabSlotState *slot_states = nullptr;
    std::uint32_t *free_slot_stack = nullptr;
    std::uint32_t *free_slot_count = nullptr;
    std::uint32_t *free_slot_lock = nullptr;

    std::uint32_t *current_slot_indices = nullptr;
    float *current_fitness = nullptr;
    std::uint32_t *current_evaluation_counts = nullptr;
    std::uint8_t *current_has_fitness = nullptr;

    std::uint32_t *next_slot_indices = nullptr;
    float *next_fitness = nullptr;
    std::uint32_t *next_evaluation_counts = nullptr;
    std::uint8_t *next_has_fitness = nullptr;

    SlabParentPair *assembly_parent_pairs = nullptr;
    std::uint32_t *parent_reference_counts = nullptr;

    int *status = nullptr;

    GenotypeSlabLayout slab_layout{};
    std::size_t max_generation_size = 0;
    std::size_t current_generation_index = 0;
    std::size_t current_generation_size = 0;
    std::size_t next_generation_index = 0;
    std::size_t next_generation_size = 0;
    std::size_t planned_child_count = 0;
};

bool TryCreateDeviceSlabRuntimeBuffers(DeviceSlabRuntimeBuffers &buffers, const DeviceSlabRuntimeConfig &config);

void DestroyDeviceSlabRuntimeBuffers(DeviceSlabRuntimeBuffers &buffers) noexcept;

bool TryUploadSlabToDevice(const HostGenotypeSlab &host_buffer, DeviceSlabRuntimeBuffers &buffers);

bool TryDownloadSlabFromDevice(const DeviceSlabRuntimeBuffers &buffers, HostGenotypeSlab &host_buffer);

bool TryUploadCurrentGenerationToDevice(const SlabGeneration &generation, DeviceSlabRuntimeBuffers &buffers);

bool TryBootstrapRandomCurrentGenerationOnDevice(DeviceSlabRuntimeBuffers &buffers, std::size_t generation_size,
                                                 std::uint32_t generation_seed, std::size_t generation_index = 0,
                                                 const DeviceSlabBootstrapConfig &config = {});

bool TryDownloadCurrentGenerationFromDevice(const DeviceSlabRuntimeBuffers &buffers, SlabGeneration &generation);

bool TryDownloadNextGenerationFromDevice(const DeviceSlabRuntimeBuffers &buffers, SlabGeneration &generation);

bool TryUploadAssemblyPlanToDevice(const SlabAssemblyPlan &plan, DeviceSlabRuntimeBuffers &buffers);

bool TryApplyFinalChildPriorityToAssemblyPlanOnDevice(DeviceSlabRuntimeBuffers &buffers);

bool TryPrepareSlabForExpandedActionCountOnDevice(DeviceSlabRuntimeBuffers &buffers, std::size_t next_action_count,
                                                  bool verbose = false);

bool TryInitializeNextGenerationAssemblyOnDevice(DeviceSlabRuntimeBuffers &buffers,
                                                 const SlabDeviceAssemblyConfig &config = {});

bool TryContinueNextGenerationAssemblyOnDevice(DeviceSlabRuntimeBuffers &buffers, std::uint32_t generation_seed,
                                               const SlabDeviceAssemblyConfig &config = {},
                                               std::size_t child_offset = 0);

bool TryCleanupFailedAssemblyOnDevice(DeviceSlabRuntimeBuffers &buffers);

bool TryAssembleNextGenerationOnDevice(DeviceSlabRuntimeBuffers &buffers, std::uint32_t generation_seed,
                                       const SlabDeviceAssemblyConfig &config = {});

bool TryReadDeviceSlabRuntimeStatus(const DeviceSlabRuntimeBuffers &buffers, DeviceSlabRuntimeStatusCode &status_code);

const char *DeviceSlabRuntimeStatusCodeString(DeviceSlabRuntimeStatusCode status_code) noexcept;

} // namespace neuroevolution::genetic_algorithm::genotype_slab::device
