#pragma once

#include <cstddef>
#include <cstdint>

#include "genetic_algorithm/device/runtime_common.hpp"
#include "genetic_algorithm/generation_assembly.hpp"
#include "genetic_algorithm/genotype_buffer/device_runtime.hpp"

namespace neuroevolution::genetic_algorithm::buffer_device {

using device_common::PopulationFitnessSummary;
using device_common::RuntimeWordCounts;

enum class DeviceBufferGARuntimeStatusCode : int {
    kOk = 0,
    kCudaFailure = 1,
    kInvalidRuntimeConfig = 2,
    kInvalidBuffer = 3,
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
    kBufferFull = 14,
    kOutputEmbeddingInjectionFailed = 15,
    kBufferRepackFailed = 16,
};

struct DeviceBufferGARuntimeConfig {
    std::size_t slot_count = 0;
    std::size_t action_count = 0;
    std::size_t max_generation_size = 0;
};

constexpr bool IsValidDeviceBufferGARuntimeConfig(const DeviceBufferGARuntimeConfig &config) noexcept {
    return (config.slot_count >= config.max_generation_size) && (config.action_count > 0) &&
           (config.max_generation_size > 0);
}

struct DeviceBufferGARuntimeBuffers {
    genotype_buffer::device::DeviceBufferRuntimeBuffers genotype_buffer{};
    PopulationFitnessSummary *summary = nullptr;
    int *status = nullptr;
    std::size_t max_generation_size = 0;
    std::size_t generation_byte_budget_bytes = 0;
};

using PendingOutputEmbeddingInjection = genotype_buffer::device::PendingOutputEmbeddingInjection;

bool TryCreateDeviceBufferGARuntimeBuffers(DeviceBufferGARuntimeBuffers &buffers,
                                           const DeviceBufferGARuntimeConfig &config);

void DestroyDeviceBufferGARuntimeBuffers(DeviceBufferGARuntimeBuffers &buffers) noexcept;

bool TryUploadCurrentBufferPopulationToDevice(const genotype_buffer::HostGenotypeBuffer &host_buffer,
                                              const genotype_buffer::BufferGeneration &current_generation,
                                              DeviceBufferGARuntimeBuffers &buffers);

bool TryDownloadBufferFromDevice(const DeviceBufferGARuntimeBuffers &buffers,
                                 genotype_buffer::HostGenotypeBuffer &host_buffer);

bool TryDownloadCurrentGenerationFromDevice(const DeviceBufferGARuntimeBuffers &buffers,
                                            genotype_buffer::BufferGeneration &generation);

bool TryDownloadNextGenerationFromDevice(const DeviceBufferGARuntimeBuffers &buffers,
                                         genotype_buffer::BufferGeneration &generation);

bool TryEvaluateCurrentGenerationFitnessOnDevice(DeviceBufferGARuntimeBuffers &buffers,
                                                 const RuntimeWordCounts &runtime_word_counts);

bool TryReadPopulationFitnessSummaryFromDevice(const DeviceBufferGARuntimeBuffers &buffers,
                                               PopulationFitnessSummary &summary);

bool TryAdvanceGenerationOnDevice(DeviceBufferGARuntimeBuffers &buffers, std::uint32_t generation_seed,
                                  const RuntimeWordCounts &runtime_word_counts,
                                  const GenerationAssemblyConfig &config = {},
                                  const PendingOutputEmbeddingInjection &pending_output_embedding_injection = {});

void SwapDeviceBufferGenerationBuffers(DeviceBufferGARuntimeBuffers &buffers) noexcept;

bool TryReadDeviceBufferGARuntimeStatus(const DeviceBufferGARuntimeBuffers &buffers,
                                        DeviceBufferGARuntimeStatusCode &status_code);

const char *DeviceBufferGARuntimeStatusCodeString(DeviceBufferGARuntimeStatusCode status_code) noexcept;

} // namespace neuroevolution::genetic_algorithm::buffer_device
