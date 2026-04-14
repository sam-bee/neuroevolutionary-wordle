#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/device/runtime_common.hpp"
#include "genetic_algorithm/generation_assembly.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "training_folder/training_data.hpp"

namespace neuroevolution::genetic_algorithm::dynamic_device {

constexpr std::size_t kInitialDynamicActionCount = training_folder::kDefaultInitialActiveWordCount;
constexpr std::size_t kDynamicThreadBlockSize = 256;
constexpr std::size_t kDefaultPopulationSizeCeiling = 100;

using device_common::PopulationFitnessSummary;
using device_common::RuntimeWordCounts;
using genome::ComputeDynamicGenomeStrideBytes;
using genome::DynamicPopulationLayout;
using genome::GenomeBytesAt;
using genome::GenomePolicyModelParameters;
using genome::GenomeTailRows;
using genome::HostGenomeBytesAt;
using genome::HostPopulation;
using genome::IsValidDynamicPopulationLayout;
using genome::PolicyModelParameters;
using genome::PopulationSizeForGenotypeBudgetBytes;
using genome::TrainableActionEmbeddingTail;
using genome::TryAllocateHostGenomeStorage;
using genome::TryInitializeRandomHostPopulation;

struct PendingOutputEmbeddingInjection {
    bool enabled = false;
    std::size_t first_catalog_word_index = 0;
    std::size_t injection_count = 0;
};

enum class DeviceRuntimeStatusCode : int {
    kOk = 0,
    kCudaFailure = 1,
    kInvalidTrainingShard = 2,
    kGuessAppendFailed = 3,
    kPolicyForwardFailed = 4,
    kActionSelectionFailed = 5,
    kPopulationNotEvaluated = 6,
    kInvalidAssemblyConfig = 7,
    kParentSelectionFailed = 8,
    kOutputEmbeddingInjectionFailed = 9,
    kInvalidPopulationLayout = 10,
};

struct DeviceRuntimeConfig {
    std::size_t genotype_memory_budget_bytes = 0;
    std::size_t population_size_ceiling = kDefaultPopulationSizeCeiling;
    std::size_t initial_action_count = kInitialDynamicActionCount;
};

struct DeviceRuntimeBuffers {
    std::uint8_t *current_genomes = nullptr;
    std::uint8_t *next_genomes = nullptr;

    float *current_fitness = nullptr;
    float *next_fitness = nullptr;

    std::uint32_t *current_evaluation_counts = nullptr;
    std::uint32_t *next_evaluation_counts = nullptr;

    std::uint8_t *current_has_fitness = nullptr;
    std::uint8_t *next_has_fitness = nullptr;

    PopulationFitnessSummary *summary = nullptr;
    int *status = nullptr;

    std::size_t genotype_memory_budget_bytes = 0;
    std::size_t population_size_ceiling = 0;
    std::size_t max_population_count = 0;

    DynamicPopulationLayout current_layout{};
    DynamicPopulationLayout next_layout{};
};

bool TryCreateDeviceRuntimeBuffers(DeviceRuntimeBuffers &buffers, const DeviceRuntimeConfig &config);

void DestroyDeviceRuntimeBuffers(DeviceRuntimeBuffers &buffers) noexcept;

bool TryUploadCurrentPopulationToDevice(const HostPopulation &host_population, DeviceRuntimeBuffers &buffers);

bool TryDownloadCurrentPopulationFromDevice(const DeviceRuntimeBuffers &buffers, HostPopulation &host_population);

bool TryEvaluatePopulationFitnessOnDevice(DeviceRuntimeBuffers &buffers, const RuntimeWordCounts &runtime_word_counts);

bool TryReadPopulationFitnessSummaryFromDevice(const DeviceRuntimeBuffers &buffers, PopulationFitnessSummary &summary);

bool TryReadDeviceRuntimeStatus(const DeviceRuntimeBuffers &buffers, DeviceRuntimeStatusCode &status_code);

bool TryAssembleNextGenerationOnDevice(DeviceRuntimeBuffers &buffers, std::uint32_t generation_seed,
                                       const GenerationAssemblyConfig &config = {},
                                       const PendingOutputEmbeddingInjection &pending_output_embedding_injection = {});

void SwapDevicePopulationBuffers(DeviceRuntimeBuffers &buffers) noexcept;

const char *DeviceRuntimeStatusCodeString(DeviceRuntimeStatusCode status_code) noexcept;

} // namespace neuroevolution::genetic_algorithm::dynamic_device
