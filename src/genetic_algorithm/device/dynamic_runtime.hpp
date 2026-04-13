#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/device/mating_plan.hpp"
#include "genetic_algorithm/genome/arena_slots.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "genetic_algorithm/generation_assembly.hpp"
#include "training_folder/training_data.hpp"

namespace neuroevolution::genetic_algorithm::dynamic_device {

constexpr std::size_t kInitialDynamicActionCount = training_folder::kDefaultInitialActiveWordCount;
constexpr std::size_t kDynamicThreadBlockSize = 256;
constexpr std::size_t kDefaultPopulationSizeCeiling = 100;
constexpr std::size_t kDefaultTailChunkActionCapacity = 1;

using genome::ComputeDynamicGenomeStrideBytes;
using genome::ConstDynamicGenomeView;
using genome::DynamicArenaSlotId;
using genome::DynamicGenomeView;
using genome::DynamicPopulationLayout;
using genome::DynamicTailChunkView;
using genome::DynamicTailRowLocation;
using genome::DynamicTailSchema;
using genome::DynamicTailSchemaForLayout;
using genome::GenomeBodyParameters;
using genome::GenomeBytesAt;
using genome::GenomePolicyModelParameters;
using genome::GenomeTailChunk;
using genome::GenomeTailRow;
using genome::GenomeTailRows;
using genome::GenomeView;
using genome::GenomeViewAt;
using genome::ArenaGenomeView;
using genome::HostGenomeViewAt;
using genome::HostGenomeBytesAt;
using genome::HostPopulation;
using genome::IsValidDynamicTailSchema;
using genome::IsValidDynamicPopulationLayout;
using genome::IsValidArenaSlotId;
using genome::MakeDynamicPopulationLayout;
using genome::PolicyModelParameters;
using genome::PopulationSizeForGenotypeBudgetBytes;
using genome::TailRowSlotIdsForIndividual;
using genome::TrainableActionEmbeddingTail;
using genome::TryAssignArenaGenomeSlotIds;
using genome::TryConsumeParentUseAndMaybeRecycleArenaGenomeSlotIds;
using genome::TryAllocateHostGenomeStorage;
using genome::TryInitializeRandomHostPopulation;
using genome::kInvalidDynamicArenaSlotId;

struct RuntimeWordCounts {
    std::size_t training_word_count = training_folder::kDefaultInitialActiveWordCount;
    std::size_t action_space_word_count = training_folder::kDefaultInitialActiveWordCount;
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
    kArenaExhausted = 11,
};

struct PopulationFitnessSummary {
    float best_fitness = 0.0f;
    float average_fitness = 0.0f;
    std::size_t best_index = 0;
    std::size_t generation_index = 0;
    std::size_t action_count = 0;
    std::size_t population_size = 0;
};

struct DeviceRuntimeConfig {
    std::size_t genotype_memory_budget_bytes = 0;
    std::size_t population_size_ceiling = kDefaultPopulationSizeCeiling;
    std::size_t initial_action_count = kInitialDynamicActionCount;
    std::size_t max_action_count = kInitialDynamicActionCount;
    std::size_t tail_chunk_action_capacity = kDefaultTailChunkActionCapacity;
};

struct DeviceRuntimeBuffers {
    PolicyModelParameters *body_slots = nullptr;
    TrainableActionEmbeddingTail *tail_row_slots = nullptr;

    DynamicArenaSlotId *current_body_slot_ids = nullptr;
    DynamicArenaSlotId *next_body_slot_ids = nullptr;

    DynamicArenaSlotId *current_tail_row_slot_ids = nullptr;
    DynamicArenaSlotId *next_tail_row_slot_ids = nullptr;

    DynamicArenaSlotId *body_free_slot_ids = nullptr;
    DynamicArenaSlotId *tail_row_free_slot_ids = nullptr;

    std::uint8_t *body_slot_live_flags = nullptr;
    std::uint8_t *tail_row_slot_live_flags = nullptr;

    std::uint32_t *body_free_slot_count = nullptr;
    std::uint32_t *tail_row_free_slot_count = nullptr;

    PlannedGenerationMember *next_generation_plan = nullptr;
    std::uint32_t *current_parent_remaining_use_counts = nullptr;

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
    std::size_t max_action_count = 0;
    std::size_t tail_chunk_action_capacity = 0;
    std::size_t body_slot_capacity = 0;
    std::size_t tail_row_slot_capacity = 0;

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

bool TryAssembleNextGenerationOnDevice(
    DeviceRuntimeBuffers &buffers, std::uint32_t generation_seed, const GenerationAssemblyConfig &config = {},
    const PendingOutputEmbeddingInjection &pending_output_embedding_injection = {});

void SwapDevicePopulationBuffers(DeviceRuntimeBuffers &buffers) noexcept;

const char *DeviceRuntimeStatusCodeString(DeviceRuntimeStatusCode status_code) noexcept;

} // namespace neuroevolution::genetic_algorithm::dynamic_device
