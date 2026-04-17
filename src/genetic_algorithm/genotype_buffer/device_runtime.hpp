#pragma once

#include <cstddef>
#include <cstdint>

#include "genetic_algorithm/breeding.hpp"
#include "genetic_algorithm/genotype_buffer/assembly.hpp"
#include "genetic_algorithm/mutation.hpp"

namespace neuroevolution::genetic_algorithm::genotype_buffer::device {

enum class DeviceBufferRuntimeStatusCode : int {
    kOk = 0,
    kCudaFailure = 1,
    kInvalidRuntimeConfig = 2,
    kInvalidBuffer = 3,
    kInvalidGeneration = 4,
    kInvalidAssemblyPlan = 5,
    kInvalidAssemblyConfig = 6,
    kInvalidParentIndex = 7,
    kBufferFull = 8,
    kOutputEmbeddingInjectionFailed = 9,
    kBufferRepackFailed = 10,
};

struct PendingOutputEmbeddingInjection {
    bool enabled = false;
    std::size_t first_catalog_word_index = 0;
    std::size_t injection_count = 0;
};

struct BufferDeviceAssemblyConfig {
    BreedingConfig breeding{};
    MutationConfig mutation{};
    std::size_t parent_action_count = 0;
    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidBufferDeviceAssemblyConfig(const BufferDeviceAssemblyConfig &config) noexcept {
    return IsValidBreedingConfig(config.breeding) && IsValidMutationConfig(config.mutation);
}

struct DeviceBufferRuntimeConfig {
    std::size_t slot_count = 0;
    std::size_t action_count = 0;
    std::size_t max_generation_size = 0;
};

constexpr bool IsValidDeviceBufferRuntimeConfig(const DeviceBufferRuntimeConfig &config) noexcept {
    return (config.slot_count > 0) && (config.action_count > 0) && (config.max_generation_size > 0);
}

struct DeviceBufferRuntimeBuffers {
    std::uint8_t *buffer_storage = nullptr;
    BufferSlotState *slot_states = nullptr;
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

    BufferParentPair *assembly_parent_pairs = nullptr;
    std::uint32_t *parent_reference_counts = nullptr;

    int *status = nullptr;

    GenotypeBufferLayout buffer_layout{};
    std::size_t max_generation_size = 0;
    std::size_t current_generation_index = 0;
    std::size_t current_generation_size = 0;
    std::size_t next_generation_index = 0;
    std::size_t next_generation_size = 0;
    std::size_t planned_child_count = 0;
};

bool TryCreateDeviceBufferRuntimeBuffers(DeviceBufferRuntimeBuffers &buffers, const DeviceBufferRuntimeConfig &config);

void DestroyDeviceBufferRuntimeBuffers(DeviceBufferRuntimeBuffers &buffers) noexcept;

bool TryUploadBufferToDevice(const HostGenotypeBuffer &host_buffer, DeviceBufferRuntimeBuffers &buffers);

bool TryDownloadBufferFromDevice(const DeviceBufferRuntimeBuffers &buffers, HostGenotypeBuffer &host_buffer);

bool TryUploadCurrentGenerationToDevice(const BufferGeneration &generation, DeviceBufferRuntimeBuffers &buffers);

bool TryDownloadCurrentGenerationFromDevice(const DeviceBufferRuntimeBuffers &buffers, BufferGeneration &generation);

bool TryDownloadNextGenerationFromDevice(const DeviceBufferRuntimeBuffers &buffers, BufferGeneration &generation);

bool TryUploadAssemblyPlanToDevice(const BufferAssemblyPlan &plan, DeviceBufferRuntimeBuffers &buffers);

bool TryPrioritizeAssemblyPlanForParentReleaseOnDevice(DeviceBufferRuntimeBuffers &buffers);

bool TryPrepareBufferForExpandedActionCountOnDevice(DeviceBufferRuntimeBuffers &buffers, std::size_t next_action_count);

bool TryAssembleNextGenerationOnDevice(DeviceBufferRuntimeBuffers &buffers, std::uint32_t generation_seed,
                                       const BufferDeviceAssemblyConfig &config = {});

bool TryReadDeviceBufferRuntimeStatus(const DeviceBufferRuntimeBuffers &buffers,
                                      DeviceBufferRuntimeStatusCode &status_code);

const char *DeviceBufferRuntimeStatusCodeString(DeviceBufferRuntimeStatusCode status_code) noexcept;

} // namespace neuroevolution::genetic_algorithm::genotype_buffer::device
