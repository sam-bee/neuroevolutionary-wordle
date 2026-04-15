#pragma once

#include <cstddef>
#include <cstdint>

#include "genetic_algorithm/breeding.hpp"
#include "genetic_algorithm/genotype_pool/assembly.hpp"
#include "genetic_algorithm/mutation.hpp"

namespace neuroevolution::genetic_algorithm::genotype_pool::device {

enum class DevicePoolRuntimeStatusCode : int {
    kOk = 0,
    kCudaFailure = 1,
    kInvalidRuntimeConfig = 2,
    kInvalidPool = 3,
    kInvalidGeneration = 4,
    kInvalidAssemblyPlan = 5,
    kInvalidAssemblyConfig = 6,
    kInvalidParentIndex = 7,
    kPoolFull = 8,
    kOutputEmbeddingInjectionFailed = 9,
    kPoolRepackFailed = 10,
};

struct PendingOutputEmbeddingInjection {
    bool enabled = false;
    std::size_t first_catalog_word_index = 0;
    std::size_t injection_count = 0;
};

struct PoolDeviceAssemblyConfig {
    BreedingConfig breeding{};
    MutationConfig mutation{};
    std::size_t parent_action_count = 0;
    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidPoolDeviceAssemblyConfig(const PoolDeviceAssemblyConfig &config) noexcept {
    return IsValidBreedingConfig(config.breeding) && IsValidMutationConfig(config.mutation);
}

struct DevicePoolRuntimeConfig {
    std::size_t slot_count = 0;
    std::size_t action_count = 0;
    std::size_t max_generation_size = 0;
};

constexpr bool IsValidDevicePoolRuntimeConfig(const DevicePoolRuntimeConfig &config) noexcept {
    return (config.slot_count > 0) && (config.action_count > 0) && (config.max_generation_size > 0);
}

struct DevicePoolRuntimeBuffers {
    std::uint8_t *pool_storage = nullptr;
    PoolSlotState *slot_states = nullptr;
    std::uint32_t *free_slot_stack = nullptr;
    std::size_t *free_slot_count = nullptr;

    std::uint32_t *current_slot_indices = nullptr;
    float *current_fitness = nullptr;
    std::uint32_t *current_evaluation_counts = nullptr;
    std::uint8_t *current_has_fitness = nullptr;

    std::uint32_t *next_slot_indices = nullptr;
    float *next_fitness = nullptr;
    std::uint32_t *next_evaluation_counts = nullptr;
    std::uint8_t *next_has_fitness = nullptr;

    PoolParentPair *assembly_parent_pairs = nullptr;
    std::uint32_t *parent_reference_counts = nullptr;

    int *status = nullptr;

    GenotypePoolLayout pool_layout{};
    std::size_t max_generation_size = 0;
    std::size_t current_generation_index = 0;
    std::size_t current_generation_size = 0;
    std::size_t next_generation_index = 0;
    std::size_t next_generation_size = 0;
    std::size_t planned_child_count = 0;
};

bool TryCreateDevicePoolRuntimeBuffers(DevicePoolRuntimeBuffers &buffers, const DevicePoolRuntimeConfig &config);

void DestroyDevicePoolRuntimeBuffers(DevicePoolRuntimeBuffers &buffers) noexcept;

bool TryUploadPoolToDevice(const HostGenotypePool &host_pool, DevicePoolRuntimeBuffers &buffers);

bool TryDownloadPoolFromDevice(const DevicePoolRuntimeBuffers &buffers, HostGenotypePool &host_pool);

bool TryUploadCurrentGenerationToDevice(const PoolGeneration &generation, DevicePoolRuntimeBuffers &buffers);

bool TryDownloadCurrentGenerationFromDevice(const DevicePoolRuntimeBuffers &buffers, PoolGeneration &generation);

bool TryDownloadNextGenerationFromDevice(const DevicePoolRuntimeBuffers &buffers, PoolGeneration &generation);

bool TryUploadAssemblyPlanToDevice(const PoolAssemblyPlan &plan, DevicePoolRuntimeBuffers &buffers);

bool TryPreparePoolForExpandedActionCountOnDevice(DevicePoolRuntimeBuffers &buffers, std::size_t next_action_count);

bool TryAssembleNextGenerationOnDevice(DevicePoolRuntimeBuffers &buffers, std::uint32_t generation_seed,
                                       const PoolDeviceAssemblyConfig &config = {});

bool TryReadDevicePoolRuntimeStatus(const DevicePoolRuntimeBuffers &buffers, DevicePoolRuntimeStatusCode &status_code);

const char *DevicePoolRuntimeStatusCodeString(DevicePoolRuntimeStatusCode status_code) noexcept;

} // namespace neuroevolution::genetic_algorithm::genotype_pool::device
