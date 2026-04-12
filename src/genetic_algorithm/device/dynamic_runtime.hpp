#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/generation_assembly.hpp"
#include "genetic_algorithm/population_initialization.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "model/policy_model/policy_model.hpp"
#include "training_folder/training_data.hpp"

namespace neuroevolution::genetic_algorithm::dynamic_device {

constexpr std::size_t kInitialDynamicActionCount = training_folder::kTrainingDataCurriculumEntryCount;
constexpr std::size_t kDynamicThreadBlockSize = 256;
constexpr std::size_t kDefaultPopulationSizeCeiling = 100;

using TrainableActionEmbeddingTail = model::output_embedding::TrainableActionEmbeddingTail;
using PolicyModelParameters = model::policy_model::PolicyModelParameters;

struct RuntimeWordCounts {
    std::size_t training_word_count = training_folder::kTrainingDataCurriculumEntryCount;
    std::size_t action_space_word_count = training_folder::kTrainingDataCurriculumEntryCount;
};

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

struct PopulationFitnessSummary {
    float best_fitness = 0.0f;
    float average_fitness = 0.0f;
    std::size_t best_index = 0;
    std::size_t generation_index = 0;
    std::size_t action_count = 0;
    std::size_t population_size = 0;
};

struct DynamicPopulationLayout {
    std::size_t active_individual_count = 0;
    std::size_t generation_index = 0;
    std::size_t action_count = 0;
    std::size_t genome_stride_bytes = 0;
    std::size_t genotype_bytes = 0;
};

struct AlignedGenomeStorageDeleter {
    void operator()(std::uint8_t *pointer) const noexcept;
};

using AlignedGenomeStorage = std::unique_ptr<std::uint8_t[], AlignedGenomeStorageDeleter>;

struct HostPopulation {
    DynamicPopulationLayout layout{};
    AlignedGenomeStorage genomes{};
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

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t DynamicGenomeAlignment() noexcept {
    return (alignof(PolicyModelParameters) > alignof(TrainableActionEmbeddingTail))
               ? alignof(PolicyModelParameters)
               : alignof(TrainableActionEmbeddingTail);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t RoundUpBytes(const std::size_t value,
                                                              const std::size_t alignment) noexcept {
    return (alignment == 0) ? value : ((value + alignment - 1) / alignment) * alignment;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t DynamicGenomeTailOffsetBytes() noexcept {
    return RoundUpBytes(sizeof(PolicyModelParameters), alignof(TrainableActionEmbeddingTail));
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t ComputeDynamicGenomeStrideBytes(
    const std::size_t action_count) noexcept {
    return (action_count == 0)
               ? 0
               : RoundUpBytes(DynamicGenomeTailOffsetBytes() + (action_count * sizeof(TrainableActionEmbeddingTail)),
                              DynamicGenomeAlignment());
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
PopulationSizeForGenotypeBudgetBytes(const std::size_t genotype_memory_budget_bytes, const std::size_t action_count,
                                     const std::size_t population_size_ceiling = 0) noexcept {
    const std::size_t genome_stride_bytes = ComputeDynamicGenomeStrideBytes(action_count);
    if ((genome_stride_bytes == 0) || (genotype_memory_budget_bytes < genome_stride_bytes)) {
        return 0;
    }

    const std::size_t uncapped_population_size = genotype_memory_budget_bytes / genome_stride_bytes;
    if ((population_size_ceiling == 0) || (uncapped_population_size < population_size_ceiling)) {
        return uncapped_population_size;
    }

    return population_size_ceiling;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidDynamicPopulationLayout(
    const DynamicPopulationLayout &layout) noexcept {
    return (layout.active_individual_count > 0) && (layout.action_count > 0) &&
           (layout.genome_stride_bytes == ComputeDynamicGenomeStrideBytes(layout.action_count)) &&
           (layout.genotype_bytes == (layout.active_individual_count * layout.genome_stride_bytes));
}

inline NEUROEVOLUTION_HOST_DEVICE std::uint8_t *GenomeBytesAt(std::uint8_t *population_genomes,
                                                              const DynamicPopulationLayout &layout,
                                                              const std::size_t individual_index) noexcept {
    return population_genomes + (individual_index * layout.genome_stride_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE const std::uint8_t *GenomeBytesAt(const std::uint8_t *population_genomes,
                                                                    const DynamicPopulationLayout &layout,
                                                                    const std::size_t individual_index) noexcept {
    return population_genomes + (individual_index * layout.genome_stride_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE PolicyModelParameters &GenomePolicyModelParameters(
    std::uint8_t *genome_bytes) noexcept {
    return *reinterpret_cast<PolicyModelParameters *>(genome_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE const PolicyModelParameters &GenomePolicyModelParameters(
    const std::uint8_t *genome_bytes) noexcept {
    return *reinterpret_cast<const PolicyModelParameters *>(genome_bytes);
}

inline NEUROEVOLUTION_HOST_DEVICE TrainableActionEmbeddingTail *GenomeTailRows(std::uint8_t *genome_bytes) noexcept {
    return reinterpret_cast<TrainableActionEmbeddingTail *>(genome_bytes + DynamicGenomeTailOffsetBytes());
}

inline NEUROEVOLUTION_HOST_DEVICE const TrainableActionEmbeddingTail *GenomeTailRows(
    const std::uint8_t *genome_bytes) noexcept {
    return reinterpret_cast<const TrainableActionEmbeddingTail *>(genome_bytes + DynamicGenomeTailOffsetBytes());
}

inline std::uint8_t *HostGenomeBytesAt(HostPopulation &population, const std::size_t individual_index) noexcept {
    return GenomeBytesAt(population.genomes.get(), population.layout, individual_index);
}

inline const std::uint8_t *HostGenomeBytesAt(const HostPopulation &population,
                                             const std::size_t individual_index) noexcept {
    return GenomeBytesAt(population.genomes.get(), population.layout, individual_index);
}

bool TryInitializeRandomHostPopulation(
    HostPopulation &population, std::size_t population_size, std::size_t action_count, std::uint32_t seed,
    const PopulationInitializationConfig &config = {});

bool TryCreateDeviceRuntimeBuffers(DeviceRuntimeBuffers &buffers, const DeviceRuntimeConfig &config);

void DestroyDeviceRuntimeBuffers(DeviceRuntimeBuffers &buffers) noexcept;

bool TryUploadCurrentPopulationToDevice(const HostPopulation &host_population, DeviceRuntimeBuffers &buffers);

bool TryDownloadCurrentPopulationFromDevice(const DeviceRuntimeBuffers &buffers, HostPopulation &host_population);

bool TryEvaluatePopulationFitnessOnDevice(DeviceRuntimeBuffers &buffers, const RuntimeWordCounts &runtime_word_counts);

bool TryReadPopulationFitnessSummaryFromDevice(const DeviceRuntimeBuffers &buffers, PopulationFitnessSummary &summary);

bool TryReadDeviceRuntimeStatus(const DeviceRuntimeBuffers &buffers, DeviceRuntimeStatusCode &status_code);

bool TryAssembleNextGenerationOnDevice(
    DeviceRuntimeBuffers &buffers, std::uint32_t generation_seed, const GenerationAssemblyConfig &config = {},
    const PendingOutputEmbeddingInjection &pending_output_embedding_injection = {});

void SwapDevicePopulationBuffers(DeviceRuntimeBuffers &buffers) noexcept;

const char *DeviceRuntimeStatusCodeString(DeviceRuntimeStatusCode status_code) noexcept;

} // namespace neuroevolution::genetic_algorithm::dynamic_device
