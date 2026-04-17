#pragma once

#include <cstddef>
#include <cstdint>

#include "genetic_algorithm/device/runtime_common.hpp"
#include "genetic_algorithm/generation_assembly.hpp"
#include "genetic_algorithm/genotype_slab/device_runtime.hpp"

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
    std::size_t action_count = 0;
    std::size_t population_size_ceiling = 0;
};

constexpr bool IsValidDeviceSlabGARuntimeConfig(const DeviceSlabGARuntimeConfig &config) noexcept {
    return (config.genotype_slab_byte_budget_bytes >= config.generation_byte_budget_bytes) &&
           (config.action_count > 0) &&
           (genotype_slab::SlabSlotCountForByteBudget(config.genotype_slab_byte_budget_bytes, config.action_count) >
            0) &&
           (genotype_slab::SlabSlotCountForByteBudget(config.generation_byte_budget_bytes, config.action_count,
                                                      config.population_size_ceiling) > 0);
}

struct DeviceSlabGARuntimeBuffers {
    genotype_slab::device::DeviceSlabRuntimeBuffers genotype_slab{};
    PopulationFitnessSummary *summary = nullptr;
    int *status = nullptr;
    std::size_t max_generation_size = 0;
    std::size_t generation_byte_budget_bytes = 0;
};

using PendingOutputEmbeddingInjection = genotype_slab::device::PendingOutputEmbeddingInjection;

bool TryCreateDeviceSlabGARuntimeBuffers(DeviceSlabGARuntimeBuffers &buffers, const DeviceSlabGARuntimeConfig &config);

void DestroyDeviceSlabGARuntimeBuffers(DeviceSlabGARuntimeBuffers &buffers) noexcept;

bool TryUploadCurrentSlabPopulationToDevice(const genotype_slab::HostGenotypeSlab &host_buffer,
                                            const genotype_slab::SlabGeneration &current_generation,
                                            DeviceSlabGARuntimeBuffers &buffers);

bool TryDownloadSlabFromDevice(const DeviceSlabGARuntimeBuffers &buffers, genotype_slab::HostGenotypeSlab &host_buffer);

bool TryDownloadCurrentGenerationFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                                            genotype_slab::SlabGeneration &generation);

bool TryDownloadNextGenerationFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                                         genotype_slab::SlabGeneration &generation);

bool TryEvaluateCurrentGenerationFitnessOnDevice(DeviceSlabGARuntimeBuffers &buffers,
                                                 const RuntimeWordCounts &runtime_word_counts);

bool TryReadPopulationFitnessSummaryFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                                               PopulationFitnessSummary &summary);

bool TryAdvanceGenerationOnDevice(DeviceSlabGARuntimeBuffers &buffers, std::uint32_t generation_seed,
                                  const RuntimeWordCounts &runtime_word_counts,
                                  const GenerationAssemblyConfig &config = {},
                                  const PendingOutputEmbeddingInjection &pending_output_embedding_injection = {});

void SwapDeviceSlabGenerationBuffers(DeviceSlabGARuntimeBuffers &buffers) noexcept;

bool TryReadDeviceSlabGARuntimeStatus(const DeviceSlabGARuntimeBuffers &buffers,
                                      DeviceSlabGARuntimeStatusCode &status_code);

const char *DeviceSlabGARuntimeStatusCodeString(DeviceSlabGARuntimeStatusCode status_code) noexcept;

} // namespace neuroevolution::genetic_algorithm::slab_device
