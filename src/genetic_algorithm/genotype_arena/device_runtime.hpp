#pragma once

#include <cstddef>
#include <cstdint>

#include "genetic_algorithm/breeding.hpp"
#include "genetic_algorithm/genotype_arena/assembly.hpp"
#include "genetic_algorithm/mutation.hpp"

namespace neuroevolution::genetic_algorithm::genotype_arena::device {

enum class DeviceArenaRuntimeStatusCode : int {
    kOk = 0,
    kCudaFailure = 1,
    kInvalidRuntimeConfig = 2,
    kInvalidArena = 3,
    kInvalidGeneration = 4,
    kInvalidAssemblyPlan = 5,
    kInvalidAssemblyConfig = 6,
    kInvalidParentIndex = 7,
    kArenaFull = 8,
};

struct ArenaDeviceAssemblyConfig {
    BreedingConfig breeding{};
    MutationConfig mutation{};
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidArenaDeviceAssemblyConfig(const ArenaDeviceAssemblyConfig &config) noexcept {
    return IsValidBreedingConfig(config.breeding) && IsValidMutationConfig(config.mutation);
}

struct DeviceArenaRuntimeConfig {
    std::size_t slot_count = 0;
    std::size_t action_count = 0;
    std::size_t max_generation_size = 0;
};

constexpr bool IsValidDeviceArenaRuntimeConfig(const DeviceArenaRuntimeConfig &config) noexcept {
    return (config.slot_count > 0) && (config.action_count > 0) && (config.max_generation_size > 0);
}

struct DeviceArenaRuntimeBuffers {
    std::uint8_t *arena_storage = nullptr;
    ArenaSlotState *slot_states = nullptr;
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

    ArenaParentPair *assembly_parent_pairs = nullptr;
    std::uint32_t *remaining_parent_duties = nullptr;

    int *status = nullptr;

    GenotypeArenaLayout arena_layout{};
    std::size_t max_generation_size = 0;
    std::size_t current_generation_index = 0;
    std::size_t current_generation_size = 0;
    std::size_t next_generation_index = 0;
    std::size_t next_generation_size = 0;
    std::size_t planned_child_count = 0;
};

bool TryCreateDeviceArenaRuntimeBuffers(DeviceArenaRuntimeBuffers &buffers, const DeviceArenaRuntimeConfig &config);

void DestroyDeviceArenaRuntimeBuffers(DeviceArenaRuntimeBuffers &buffers) noexcept;

bool TryUploadArenaToDevice(const HostGenotypeArena &host_arena, DeviceArenaRuntimeBuffers &buffers);

bool TryDownloadArenaFromDevice(const DeviceArenaRuntimeBuffers &buffers, HostGenotypeArena &host_arena);

bool TryUploadCurrentGenerationToDevice(const ArenaGeneration &generation, DeviceArenaRuntimeBuffers &buffers);

bool TryDownloadCurrentGenerationFromDevice(const DeviceArenaRuntimeBuffers &buffers, ArenaGeneration &generation);

bool TryDownloadNextGenerationFromDevice(const DeviceArenaRuntimeBuffers &buffers, ArenaGeneration &generation);

bool TryUploadAssemblyPlanToDevice(const ArenaAssemblyPlan &plan, DeviceArenaRuntimeBuffers &buffers);

bool TryAssembleNextGenerationWithoutElitismOnDevice(DeviceArenaRuntimeBuffers &buffers, std::uint32_t generation_seed,
                                                     const ArenaDeviceAssemblyConfig &config = {});

bool TryReadDeviceArenaRuntimeStatus(const DeviceArenaRuntimeBuffers &buffers,
                                     DeviceArenaRuntimeStatusCode &status_code);

const char *DeviceArenaRuntimeStatusCodeString(DeviceArenaRuntimeStatusCode status_code) noexcept;

} // namespace neuroevolution::genetic_algorithm::genotype_arena::device
