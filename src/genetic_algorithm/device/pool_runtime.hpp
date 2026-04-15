#pragma once

#include <cstddef>
#include <cstdint>

#include "genetic_algorithm/device/runtime_common.hpp"
#include "genetic_algorithm/generation_assembly.hpp"
#include "genetic_algorithm/genotype_pool/device_runtime.hpp"

namespace neuroevolution::genetic_algorithm::pool_device {

using device_common::PopulationFitnessSummary;
using device_common::RuntimeWordCounts;

enum class DevicePoolGARuntimeStatusCode : int {
    kOk = 0,
    kCudaFailure = 1,
    kInvalidRuntimeConfig = 2,
    kInvalidPool = 3,
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
    kPoolFull = 14,
    kOutputEmbeddingInjectionFailed = 15,
    kPoolRepackFailed = 16,
};

struct DevicePoolGARuntimeConfig {
    std::size_t slot_count = 0;
    std::size_t action_count = 0;
    std::size_t max_generation_size = 0;
};

constexpr bool IsValidDevicePoolGARuntimeConfig(const DevicePoolGARuntimeConfig &config) noexcept {
    return (config.slot_count >= config.max_generation_size) && (config.action_count > 0) &&
           (config.max_generation_size > 0);
}

struct DevicePoolGARuntimeBuffers {
    genotype_pool::device::DevicePoolRuntimeBuffers pool_buffers{};
    PopulationFitnessSummary *summary = nullptr;
    int *status = nullptr;
    std::size_t max_generation_size = 0;
};

using PendingOutputEmbeddingInjection = genotype_pool::device::PendingOutputEmbeddingInjection;

bool TryCreateDevicePoolGARuntimeBuffers(DevicePoolGARuntimeBuffers &buffers, const DevicePoolGARuntimeConfig &config);

void DestroyDevicePoolGARuntimeBuffers(DevicePoolGARuntimeBuffers &buffers) noexcept;

bool TryUploadCurrentPoolPopulationToDevice(const genotype_pool::HostGenotypePool &host_pool,
                                            const genotype_pool::PoolGeneration &current_generation,
                                            DevicePoolGARuntimeBuffers &buffers);

bool TryDownloadPoolFromDevice(const DevicePoolGARuntimeBuffers &buffers, genotype_pool::HostGenotypePool &host_pool);

bool TryDownloadCurrentGenerationFromDevice(const DevicePoolGARuntimeBuffers &buffers,
                                            genotype_pool::PoolGeneration &generation);

bool TryDownloadNextGenerationFromDevice(const DevicePoolGARuntimeBuffers &buffers,
                                         genotype_pool::PoolGeneration &generation);

bool TryEvaluateCurrentGenerationFitnessOnDevice(DevicePoolGARuntimeBuffers &buffers,
                                                 const RuntimeWordCounts &runtime_word_counts);

bool TryReadPopulationFitnessSummaryFromDevice(const DevicePoolGARuntimeBuffers &buffers,
                                               PopulationFitnessSummary &summary);

bool TryAdvanceGenerationOnDevice(DevicePoolGARuntimeBuffers &buffers, std::uint32_t generation_seed,
                                  const RuntimeWordCounts &runtime_word_counts,
                                  const GenerationAssemblyConfig &config = {},
                                  const PendingOutputEmbeddingInjection &pending_output_embedding_injection = {});

void SwapDevicePoolGenerationBuffers(DevicePoolGARuntimeBuffers &buffers) noexcept;

bool TryReadDevicePoolGARuntimeStatus(const DevicePoolGARuntimeBuffers &buffers,
                                      DevicePoolGARuntimeStatusCode &status_code);

const char *DevicePoolGARuntimeStatusCodeString(DevicePoolGARuntimeStatusCode status_code) noexcept;

} // namespace neuroevolution::genetic_algorithm::pool_device
