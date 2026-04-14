#pragma once

#include <cstddef>
#include <cstdint>

#include "genetic_algorithm/device/runtime_common.hpp"
#include "genetic_algorithm/generation_assembly.hpp"
#include "genetic_algorithm/genotype_arena/device_runtime.hpp"

namespace neuroevolution::genetic_algorithm::arena_device {

using device_common::PopulationFitnessSummary;
using device_common::RuntimeWordCounts;

enum class DeviceArenaGARuntimeStatusCode : int {
    kOk = 0,
    kCudaFailure = 1,
    kInvalidRuntimeConfig = 2,
    kInvalidArena = 3,
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
    kArenaFull = 14,
};

struct DeviceArenaGARuntimeConfig {
    std::size_t slot_count = 0;
    std::size_t action_count = 0;
    std::size_t generation_size = 0;
};

constexpr bool IsValidDeviceArenaGARuntimeConfig(const DeviceArenaGARuntimeConfig &config) noexcept {
    return (config.slot_count >= config.generation_size) && (config.action_count > 0) && (config.generation_size > 0);
}

struct DeviceArenaGARuntimeBuffers {
    genotype_arena::device::DeviceArenaRuntimeBuffers arena_buffers{};
    PopulationFitnessSummary *summary = nullptr;
    int *status = nullptr;
    std::size_t generation_size = 0;
};

bool TryCreateDeviceArenaGARuntimeBuffers(DeviceArenaGARuntimeBuffers &buffers,
                                          const DeviceArenaGARuntimeConfig &config);

void DestroyDeviceArenaGARuntimeBuffers(DeviceArenaGARuntimeBuffers &buffers) noexcept;

bool TryUploadCurrentArenaPopulationToDevice(const genotype_arena::HostGenotypeArena &host_arena,
                                             const genotype_arena::ArenaGeneration &current_generation,
                                             DeviceArenaGARuntimeBuffers &buffers);

bool TryDownloadArenaFromDevice(const DeviceArenaGARuntimeBuffers &buffers,
                                genotype_arena::HostGenotypeArena &host_arena);

bool TryDownloadCurrentGenerationFromDevice(const DeviceArenaGARuntimeBuffers &buffers,
                                            genotype_arena::ArenaGeneration &generation);

bool TryDownloadNextGenerationFromDevice(const DeviceArenaGARuntimeBuffers &buffers,
                                         genotype_arena::ArenaGeneration &generation);

bool TryEvaluateCurrentGenerationFitnessOnDevice(DeviceArenaGARuntimeBuffers &buffers,
                                                 const RuntimeWordCounts &runtime_word_counts);

bool TryReadPopulationFitnessSummaryFromDevice(const DeviceArenaGARuntimeBuffers &buffers,
                                               PopulationFitnessSummary &summary);

bool TryAdvanceGenerationOnDevice(DeviceArenaGARuntimeBuffers &buffers, std::uint32_t generation_seed,
                                  const RuntimeWordCounts &runtime_word_counts,
                                  const GenerationAssemblyConfig &config = {});

void SwapDeviceArenaGenerationBuffers(DeviceArenaGARuntimeBuffers &buffers) noexcept;

bool TryReadDeviceArenaGARuntimeStatus(const DeviceArenaGARuntimeBuffers &buffers,
                                       DeviceArenaGARuntimeStatusCode &status_code);

const char *DeviceArenaGARuntimeStatusCodeString(DeviceArenaGARuntimeStatusCode status_code) noexcept;

} // namespace neuroevolution::genetic_algorithm::arena_device
