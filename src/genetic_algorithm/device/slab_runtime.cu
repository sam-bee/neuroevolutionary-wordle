#include "genetic_algorithm/device/slab_runtime.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <utility>

#include "genetic_algorithm/device/evaluation_ops.cuh"
#include "genetic_algorithm/device/genome_ops.cuh"
#include "genetic_algorithm/device/selection_ops.cuh"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "genetic_algorithm/genotype_slab/reference_counter.hpp"
#include "genetic_algorithm/output_embedding_injection.hpp"

namespace neuroevolution::genetic_algorithm::slab_device {

namespace {

using device_evaluation_ops::DeviceGenomeEvaluationStatusCode;
using device_evaluation_ops::TryEvaluateGenomeFitness;
using device_genome_ops::DeviceRandomState;
using device_genome_ops::MakeDeviceRandomState;
using device_selection_ops::IsBetterFitness;
using device_selection_ops::TrySelectParentPairDevice;

constexpr std::size_t kSlabGARuntimeThreadBlockSize = 256;
constexpr std::size_t kMaxSlabAssemblyConcurrentChildren = 128U * 32U;

NEUROEVOLUTION_HOST_DEVICE constexpr int DeviceStatusValue(const DeviceSlabGARuntimeStatusCode status_code) {
    return static_cast<int>(status_code);
}

inline bool CheckCuda(const cudaError_t error) { return error == cudaSuccess; }

inline bool ReadDeviceStatus(const DeviceSlabGARuntimeBuffers &buffers, int &status_value) {
    return CheckCuda(cudaMemcpy(&status_value, buffers.status, sizeof(int), cudaMemcpyDeviceToHost));
}

inline bool WriteDeviceStatus(const DeviceSlabGARuntimeBuffers &buffers,
                              const DeviceSlabGARuntimeStatusCode status_code) {
    const int status_value = DeviceStatusValue(status_code);
    return CheckCuda(cudaMemcpy(buffers.status, &status_value, sizeof(int), cudaMemcpyHostToDevice));
}

inline bool ResetDeviceStatus(const DeviceSlabGARuntimeBuffers &buffers) {
    return WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kOk);
}

inline bool KernelCompletedSuccessfully(const DeviceSlabGARuntimeBuffers &buffers) {
    int status_value = DeviceStatusValue(DeviceSlabGARuntimeStatusCode::kCudaFailure);
    return ReadDeviceStatus(buffers, status_value) &&
           (status_value == DeviceStatusValue(DeviceSlabGARuntimeStatusCode::kOk));
}

inline bool IsCurrentGenerationCompatible(const DeviceSlabGARuntimeBuffers &buffers) noexcept {
    return genotype_slab::IsValidGenotypeSlabLayout(buffers.genotype_slab.slab_layout) &&
           (buffers.max_generation_size > 0) && (buffers.genotype_slab.current_generation_size > 0) &&
           (buffers.genotype_slab.current_generation_size <= buffers.genotype_slab.max_generation_size);
}

NEUROEVOLUTION_HOST_DEVICE constexpr DeviceSlabGARuntimeStatusCode
MapEvaluationStatus(const DeviceGenomeEvaluationStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceGenomeEvaluationStatusCode::kOk:
        return DeviceSlabGARuntimeStatusCode::kOk;
    case DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard:
        return DeviceSlabGARuntimeStatusCode::kInvalidTrainingShard;
    case DeviceGenomeEvaluationStatusCode::kGuessAppendFailed:
        return DeviceSlabGARuntimeStatusCode::kGuessAppendFailed;
    case DeviceGenomeEvaluationStatusCode::kPolicyForwardFailed:
        return DeviceSlabGARuntimeStatusCode::kPolicyForwardFailed;
    case DeviceGenomeEvaluationStatusCode::kActionSelectionFailed:
        return DeviceSlabGARuntimeStatusCode::kActionSelectionFailed;
    }

    return DeviceSlabGARuntimeStatusCode::kCudaFailure;
}

inline DeviceSlabGARuntimeStatusCode
MapSlabRuntimeStatus(const genotype_slab::device::DeviceSlabRuntimeStatusCode status_code) noexcept {
    using genotype_slab::device::DeviceSlabRuntimeStatusCode;

    switch (status_code) {
    case DeviceSlabRuntimeStatusCode::kOk:
        return DeviceSlabGARuntimeStatusCode::kOk;
    case DeviceSlabRuntimeStatusCode::kCudaFailure:
        return DeviceSlabGARuntimeStatusCode::kCudaFailure;
    case DeviceSlabRuntimeStatusCode::kInvalidRuntimeConfig:
        return DeviceSlabGARuntimeStatusCode::kInvalidRuntimeConfig;
    case DeviceSlabRuntimeStatusCode::kInvalidSlab:
        return DeviceSlabGARuntimeStatusCode::kInvalidSlab;
    case DeviceSlabRuntimeStatusCode::kInvalidGeneration:
        return DeviceSlabGARuntimeStatusCode::kInvalidGeneration;
    case DeviceSlabRuntimeStatusCode::kInvalidAssemblyPlan:
        return DeviceSlabGARuntimeStatusCode::kInvalidAssemblyPlan;
    case DeviceSlabRuntimeStatusCode::kInvalidAssemblyConfig:
        return DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig;
    case DeviceSlabRuntimeStatusCode::kInvalidParentIndex:
        return DeviceSlabGARuntimeStatusCode::kInvalidParentIndex;
    case DeviceSlabRuntimeStatusCode::kSlabFull:
        return DeviceSlabGARuntimeStatusCode::kSlabFull;
    case DeviceSlabRuntimeStatusCode::kOutputEmbeddingInjectionFailed:
        return DeviceSlabGARuntimeStatusCode::kOutputEmbeddingInjectionFailed;
    case DeviceSlabRuntimeStatusCode::kSlabRepackFailed:
        return DeviceSlabGARuntimeStatusCode::kSlabRepackFailed;
    }

    return DeviceSlabGARuntimeStatusCode::kCudaFailure;
}

__device__ void SetFailureStatus(int *status, const DeviceSlabGARuntimeStatusCode status_code) {
    atomicCAS(status, DeviceStatusValue(DeviceSlabGARuntimeStatusCode::kOk), DeviceStatusValue(status_code));
}

__global__ void EvaluateSlabGenerationFitnessKernel(
    const std::uint8_t *slab_storage, const genotype_slab::SlabSlotState *slot_states,
    const genotype_slab::GenotypeSlabLayout slab_layout, const std::uint32_t *current_slot_indices,
    const std::size_t current_generation_size, float *current_fitness, std::uint32_t *current_evaluation_counts,
    std::uint8_t *current_has_fitness, const RuntimeWordCounts runtime_word_counts, int *status) {
    const std::size_t individual_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (!genotype_slab::IsValidGenotypeSlabLayout(slab_layout) || (slab_storage == nullptr) ||
        (slot_states == nullptr) || (current_slot_indices == nullptr) || (current_fitness == nullptr) ||
        (current_evaluation_counts == nullptr) || (current_has_fitness == nullptr) || (current_generation_size == 0)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidSlab);
        }
        return;
    }

    if (individual_index >= current_generation_size) {
        return;
    }

    const std::uint32_t slot_index = current_slot_indices[individual_index];
    if ((slot_index == genotype_slab::kInvalidSlabSlotIndex) || (slot_index >= slab_layout.slot_count)) {
        SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidGeneration);
        return;
    }

    if (!slot_states[slot_index].occupied || (slot_states[slot_index].reference_count == 0)) {
        SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidSlab);
        return;
    }

    float fitness = 0.0f;
    const DeviceGenomeEvaluationStatusCode evaluation_status =
        TryEvaluateGenomeFitness(genotype_slab::SlabSlotBytesAt(slab_storage, slab_layout, slot_index),
                                 slab_layout.action_count, runtime_word_counts, fitness);
    if (evaluation_status != DeviceGenomeEvaluationStatusCode::kOk) {
        SetFailureStatus(status, MapEvaluationStatus(evaluation_status));
        return;
    }

    current_fitness[individual_index] = fitness;
    ++current_evaluation_counts[individual_index];
    current_has_fitness[individual_index] = 1;
}

__global__ void SummarizeSlabGenerationKernel(const float *current_fitness, const std::uint8_t *current_has_fitness,
                                              const std::size_t current_generation_index,
                                              const std::size_t current_generation_size, const std::size_t action_count,
                                              PopulationFitnessSummary *summary, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    if ((current_fitness == nullptr) || (current_has_fitness == nullptr) || (summary == nullptr) ||
        (current_generation_size == 0) || (action_count == 0)) {
        SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidGeneration);
        return;
    }

    bool found_best = false;
    float best_fitness = 0.0f;
    float fitness_sum = 0.0f;
    std::size_t best_index = 0;

    for (std::size_t individual_index = 0; individual_index < current_generation_size; ++individual_index) {
        if (current_has_fitness[individual_index] == 0) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kPopulationNotEvaluated);
            return;
        }

        fitness_sum += current_fitness[individual_index];

        if (!found_best ||
            IsBetterFitness(current_fitness[individual_index], individual_index, best_fitness, best_index)) {
            found_best = true;
            best_fitness = current_fitness[individual_index];
            best_index = individual_index;
        }
    }

    summary->best_fitness = best_fitness;
    summary->average_fitness = fitness_sum / static_cast<float>(current_generation_size);
    summary->best_index = best_index;
    summary->generation_index = current_generation_index;
    summary->action_count = action_count;
    summary->population_size = current_generation_size;
}

__global__ void BuildSlabAssemblyPlanKernel(const float *current_fitness, const std::uint8_t *current_has_fitness,
                                            const std::size_t current_generation_size, const std::size_t child_count,
                                            const std::size_t current_generation_index,
                                            const ParentSelectionConfig config, const std::uint32_t planning_seed,
                                            genotype_slab::SlabParentPair *parent_pairs, int *status) {
    const std::size_t child_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ((current_fitness == nullptr) || (current_has_fitness == nullptr) || (parent_pairs == nullptr) ||
        (current_generation_size == 0) || !IsValidParentSelectionConfig(config)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig);
        }
        return;
    }

    if (child_index >= child_count) {
        return;
    }

    DeviceRandomState random_state = MakeDeviceRandomState(
        planning_seed, static_cast<std::uint32_t>(child_index + (current_generation_index * 8191U)));

    ParentPair parent_pair{};
    if (!TrySelectParentPairDevice(current_fitness, current_has_fitness, current_generation_size, random_state, config,
                                   parent_pair)) {
        SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kParentSelectionFailed);
        return;
    }

    parent_pairs[child_index].first_parent_index = static_cast<std::uint32_t>(parent_pair.first_parent_index);
    parent_pairs[child_index].second_parent_index = static_cast<std::uint32_t>(parent_pair.second_parent_index);
}

inline bool
IsValidPendingOutputEmbeddingInjection(const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
                                       const std::size_t current_action_count) noexcept {
    return !pending_output_embedding_injection.enabled ||
           ((pending_output_embedding_injection.injection_count > 0) &&
            (pending_output_embedding_injection.first_catalog_word_index == current_action_count));
}

inline bool TryComputeGenerationByteBudgetBytes(const std::size_t action_count, const std::size_t generation_size,
                                                std::size_t &generation_byte_budget_bytes_out) {
    generation_byte_budget_bytes_out = 0;

    const std::size_t slot_stride_bytes = genotype_slab::ComputeSlabSlotStrideBytes(action_count);
    if ((slot_stride_bytes == 0) || (generation_size > (std::numeric_limits<std::size_t>::max() / slot_stride_bytes))) {
        return false;
    }

    generation_byte_budget_bytes_out = generation_size * slot_stride_bytes;
    return generation_byte_budget_bytes_out > 0;
}

inline std::size_t RuntimeGenerationCapacity(const DeviceSlabGARuntimeConfig &config) noexcept {
    return genotype_slab::SlabSlotCountForByteBudget(config.generation_byte_budget_bytes, config.action_count,
                                                     config.population_size_ceiling);
}

inline std::size_t RuntimeSlabSlotCount(const DeviceSlabGARuntimeConfig &config) noexcept {
    return genotype_slab::SlabSlotCountForByteBudget(config.genotype_slab_byte_budget_bytes, config.action_count);
}

inline bool TryReadPopulationFitnessSummary(const DeviceSlabGARuntimeBuffers &buffers,
                                            PopulationFitnessSummary &summary) {
    return CheckCuda(cudaMemcpy(&summary, buffers.summary, sizeof(PopulationFitnessSummary), cudaMemcpyDeviceToHost));
}

inline bool WritePopulationFitnessSummary(const DeviceSlabGARuntimeBuffers &buffers,
                                          const PopulationFitnessSummary &summary) {
    return CheckCuda(cudaMemcpy(buffers.summary, &summary, sizeof(PopulationFitnessSummary), cudaMemcpyHostToDevice));
}

inline bool TryReadDeviceFreeSlotCount(const genotype_slab::device::DeviceSlabRuntimeBuffers &buffers,
                                       std::uint32_t &free_slot_count) {
    free_slot_count = 0;
    return CheckCuda(
        cudaMemcpy(&free_slot_count, buffers.free_slot_count, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
}

inline bool TryDownloadAssemblyPlanFromDevice(const genotype_slab::device::DeviceSlabRuntimeBuffers &buffers,
                                              genotype_slab::SlabAssemblyPlan &plan) {
    if (buffers.planned_child_count == 0) {
        return false;
    }

    if (!genotype_slab::TryCreateSlabAssemblyPlan(plan, buffers.planned_child_count)) {
        return false;
    }

    return CheckCuda(cudaMemcpy(plan.parent_pairs.get(), buffers.assembly_parent_pairs,
                                plan.child_count * sizeof(genotype_slab::SlabParentPair), cudaMemcpyDeviceToHost));
}

inline std::size_t CountReferencedParents(const std::uint32_t *parent_reference_counts,
                                          const std::size_t active_individual_count) noexcept {
    if (parent_reference_counts == nullptr) {
        return 0;
    }

    std::size_t survivor_count = 0;
    for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
        if (parent_reference_counts[parent_index] > 0) {
            ++survivor_count;
        }
    }

    return survivor_count;
}

inline bool TrySimulateDeviceAssemblyBatches(const genotype_slab::SlabGeneration &current_generation,
                                             const genotype_slab::SlabAssemblyPlan &plan,
                                             const std::uint32_t initial_free_slot_count, bool &can_fit_out) {
    can_fit_out = false;
    if (!genotype_slab::IsValidSlabGeneration(current_generation) || !genotype_slab::IsValidSlabAssemblyPlan(plan)) {
        return false;
    }

    std::unique_ptr<std::uint32_t[]> parent_reference_counts(
        new (std::nothrow) std::uint32_t[current_generation.active_individual_count]());
    std::unique_ptr<std::uint8_t[]> live_parent_slots(new (std::nothrow) std::uint8_t[current_generation.active_individual_count]());
    if ((parent_reference_counts == nullptr) || (live_parent_slots == nullptr) ||
        !genotype_slab::TryBuildParentReferenceCounts(current_generation.slot_indices.get(),
                                                      current_generation.active_individual_count, plan.parent_pairs.get(),
                                                      plan.child_count, parent_reference_counts.get())) {
        return false;
    }

    std::uint32_t free_slot_count = initial_free_slot_count;
    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        const bool parent_is_live =
            current_generation.slot_indices[parent_index] != genotype_slab::kInvalidSlabSlotIndex;
        live_parent_slots[parent_index] = parent_is_live ? 1U : 0U;
        if (parent_is_live && (parent_reference_counts[parent_index] == 0)) {
            live_parent_slots[parent_index] = 0U;
            ++free_slot_count;
        }
    }

    auto release_parent = [&](const std::uint32_t parent_index) {
        if ((parent_index >= current_generation.active_individual_count) || (parent_reference_counts[parent_index] == 0)) {
            return false;
        }

        --parent_reference_counts[parent_index];
        if ((parent_reference_counts[parent_index] == 0) && (live_parent_slots[parent_index] != 0U)) {
            live_parent_slots[parent_index] = 0U;
            ++free_slot_count;
        }

        return true;
    };

    std::size_t child_offset = 0;
    while (child_offset < plan.child_count) {
        if (free_slot_count == 0U) {
            return true;
        }

        std::size_t batch_child_count = plan.child_count - child_offset;
        batch_child_count = std::min(batch_child_count, static_cast<std::size_t>(free_slot_count));
        batch_child_count = std::min(batch_child_count, kMaxSlabAssemblyConcurrentChildren);
        if (batch_child_count == 0) {
            return true;
        }

        free_slot_count -= static_cast<std::uint32_t>(batch_child_count);
        for (std::size_t batch_child_index = 0; batch_child_index < batch_child_count; ++batch_child_index) {
            const genotype_slab::SlabParentPair &parent_pair = plan.parent_pairs[child_offset + batch_child_index];
            if ((parent_pair.first_parent_index >= current_generation.active_individual_count) ||
                (parent_pair.second_parent_index >= current_generation.active_individual_count) ||
                (live_parent_slots[parent_pair.first_parent_index] == 0U) ||
                (live_parent_slots[parent_pair.second_parent_index] == 0U)) {
                return false;
            }
        }

        for (std::size_t batch_child_index = 0; batch_child_index < batch_child_count; ++batch_child_index) {
            const genotype_slab::SlabParentPair &parent_pair = plan.parent_pairs[child_offset + batch_child_index];
            if (!release_parent(parent_pair.first_parent_index) || !release_parent(parent_pair.second_parent_index)) {
                return false;
            }
        }

        child_offset += batch_child_count;
    }

    can_fit_out = true;
    return true;
}

inline bool TryCopyGenomeBytesIntoExpandedSlot(const genotype_slab::HostGenotypeSlab &source_slab,
                                               const std::uint32_t source_slot_index,
                                               genotype_slab::HostGenotypeSlab &target_slab,
                                               const std::uint32_t target_slot_index) {
    if (!genotype_slab::IsValidHostGenotypeSlab(source_slab) || !genotype_slab::IsValidHostGenotypeSlab(target_slab) ||
        (source_slot_index >= source_slab.layout.slot_count) || (target_slot_index >= target_slab.layout.slot_count) ||
        (source_slab.layout.slot_stride_bytes > target_slab.layout.slot_stride_bytes)) {
        return false;
    }

    const std::uint8_t *source_bytes = genotype_slab::HostSlabSlotBytesAt(source_slab, source_slot_index);
    std::uint8_t *target_bytes = genotype_slab::HostSlabSlotBytesAt(target_slab, target_slot_index);
    if ((source_bytes == nullptr) || (target_bytes == nullptr) || !target_slab.slot_states[target_slot_index].occupied) {
        return false;
    }

    for (std::size_t byte_index = 0; byte_index < source_slab.layout.slot_stride_bytes; ++byte_index) {
        target_bytes[byte_index] = source_bytes[byte_index];
    }
    for (std::size_t byte_index = source_slab.layout.slot_stride_bytes; byte_index < target_slab.layout.slot_stride_bytes;
         ++byte_index) {
        target_bytes[byte_index] = 0;
    }

    return true;
}

struct HostSpillAssemblyContext {
    GenerationAssemblyConfig assembly_config{};
    std::size_t parent_action_count = 0;
    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
    const training_folder::TrainingWordCatalog *training_word_catalog = nullptr;
    std::uint32_t generation_seed = 0;
    std::size_t current_generation_index = 0;
    std::size_t next_child_index = 0;
    DeviceSlabGARuntimeStatusCode failure_status = DeviceSlabGARuntimeStatusCode::kOk;
};

bool AssembleSlabChildGenomeOnHost(const std::uint8_t *first_parent_genome_bytes,
                                   const std::uint8_t *second_parent_genome_bytes, const std::size_t action_count,
                                   std::uint8_t *child_genome_bytes, void *user_data) {
    auto *context = static_cast<HostSpillAssemblyContext *>(user_data);
    if ((context == nullptr) || (first_parent_genome_bytes == nullptr) || (second_parent_genome_bytes == nullptr) ||
        (child_genome_bytes == nullptr) || (action_count == 0)) {
        if (context != nullptr) {
            context->failure_status = DeviceSlabGARuntimeStatusCode::kCudaFailure;
        }
        return false;
    }

    const std::size_t parent_action_count =
        (context->parent_action_count == 0) ? action_count : context->parent_action_count;
    if (parent_action_count > action_count) {
        context->failure_status = DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig;
        return false;
    }

    std::mt19937 random_engine(static_cast<std::mt19937::result_type>(
        context->generation_seed + static_cast<std::uint32_t>(context->next_child_index +
                                                              (context->current_generation_index * 4099U))));

    genome::PolicyModelParameters &child_policy = genome::GenomePolicyModelParameters(child_genome_bytes);
    BreedPolicyModelParameters(genome::GenomePolicyModelParameters(first_parent_genome_bytes),
                               genome::GenomePolicyModelParameters(second_parent_genome_bytes), child_policy,
                               random_engine, context->assembly_config.breeding);
    if (!TryMutatePolicyModelParameters(child_policy, random_engine, context->assembly_config.mutation)) {
        context->failure_status = DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig;
        return false;
    }

    const genome::TrainableActionEmbeddingTail *first_parent_tails = genome::GenomeTailRows(first_parent_genome_bytes);
    const genome::TrainableActionEmbeddingTail *second_parent_tails = genome::GenomeTailRows(second_parent_genome_bytes);
    genome::TrainableActionEmbeddingTail *child_tails = genome::GenomeTailRows(child_genome_bytes);

    for (std::size_t action_index = 0; action_index < parent_action_count; ++action_index) {
        detail::BreedFixedBuffer(first_parent_tails[action_index], second_parent_tails[action_index],
                                 child_tails[action_index], random_engine, context->assembly_config.breeding);
        detail::MutateFixedBuffer(child_tails[action_index], random_engine, context->assembly_config.mutation);
    }
    for (std::size_t action_index = parent_action_count; action_index < action_count; ++action_index) {
        child_tails[action_index] = {};
    }

    if (context->pending_output_embedding_injection.enabled) {
        if ((context->training_word_catalog == nullptr) ||
            !training_folder::IsValidTrainingWordCatalog(*context->training_word_catalog) ||
            (context->pending_output_embedding_injection.first_catalog_word_index +
                 context->pending_output_embedding_injection.injection_count >
             context->training_word_catalog->word_count) ||
            (context->pending_output_embedding_injection.first_catalog_word_index +
                 context->pending_output_embedding_injection.injection_count >
             action_count)) {
            context->failure_status = DeviceSlabGARuntimeStatusCode::kOutputEmbeddingInjectionFailed;
            return false;
        }

        for (std::size_t injection_offset = 0;
             injection_offset < context->pending_output_embedding_injection.injection_count; ++injection_offset) {
            const std::size_t action_index =
                context->pending_output_embedding_injection.first_catalog_word_index + injection_offset;
            if (!TrySeedOutputEmbeddingTailFromHintGrids(
                    child_policy, context->training_word_catalog->words[action_index], child_tails[action_index])) {
                context->failure_status = DeviceSlabGARuntimeStatusCode::kOutputEmbeddingInjectionFailed;
                return false;
            }
        }
    }

    ++context->next_child_index;
    return true;
}

inline bool TryCopyReferencedParentsIntoHostSpillSlab(
    const genotype_slab::HostGenotypeSlab &source_slab, const genotype_slab::SlabGeneration &source_generation,
    const std::uint32_t *parent_reference_counts, const genotype_slab::GenotypeSlabLayout &spill_layout,
    genotype_slab::HostGenotypeSlab &spill_slab, genotype_slab::SlabGeneration &spill_generation) {
    if (!genotype_slab::IsValidHostGenotypeSlab(source_slab) || !genotype_slab::IsValidSlabGeneration(source_generation) ||
        (parent_reference_counts == nullptr) || !genotype_slab::TryCreateHostGenotypeSlab(spill_slab, spill_layout) ||
        !genotype_slab::TryCreateSlabGeneration(spill_generation, source_generation.active_individual_count,
                                                source_generation.generation_index)) {
        return false;
    }

    for (std::size_t parent_index = 0; parent_index < source_generation.active_individual_count; ++parent_index) {
        if (parent_reference_counts[parent_index] == 0) {
            continue;
        }

        const std::uint32_t source_slot_index = source_generation.slot_indices[parent_index];
        if (source_slot_index == genotype_slab::kInvalidSlabSlotIndex) {
            return false;
        }

        std::uint32_t spill_slot_index = genotype_slab::kInvalidSlabSlotIndex;
        if (!genotype_slab::TryAllocateSlabSlot(spill_slab, spill_slot_index) ||
            !TryCopyGenomeBytesIntoExpandedSlot(source_slab, source_slot_index, spill_slab, spill_slot_index) ||
            !genotype_slab::TrySetSlabGenerationSlot(spill_generation, parent_index, spill_slot_index)) {
            return false;
        }
    }

    return true;
}

inline bool TryPackSpillChildrenIntoTargetSlab(const genotype_slab::HostGenotypeSlab &spill_slab,
                                               const genotype_slab::SlabGeneration &spill_generation,
                                               const genotype_slab::GenotypeSlabLayout &target_layout,
                                               genotype_slab::HostGenotypeSlab &target_slab,
                                               genotype_slab::SlabGeneration &target_generation) {
    if (!genotype_slab::IsValidHostGenotypeSlab(spill_slab) || !genotype_slab::IsValidSlabGeneration(spill_generation) ||
        (spill_slab.layout.action_count != target_layout.action_count) ||
        (spill_slab.layout.slot_stride_bytes != target_layout.slot_stride_bytes) ||
        (spill_generation.active_individual_count > target_layout.slot_count) ||
        !genotype_slab::TryCreateHostGenotypeSlab(target_slab, target_layout) ||
        !genotype_slab::TryCreateSlabGeneration(target_generation, spill_generation.active_individual_count,
                                                spill_generation.generation_index)) {
        return false;
    }

    for (std::size_t child_index = 0; child_index < spill_generation.active_individual_count; ++child_index) {
        const std::uint32_t spill_slot_index = spill_generation.slot_indices[child_index];
        if (spill_slot_index == genotype_slab::kInvalidSlabSlotIndex) {
            return false;
        }

        std::uint32_t target_slot_index = genotype_slab::kInvalidSlabSlotIndex;
        if (!genotype_slab::TryAllocateSlabSlot(target_slab, target_slot_index) ||
            !genotype_slab::TryCopyGenomeBytesIntoSlabSlot(target_slab, target_slot_index,
                                                           genotype_slab::HostSlabSlotBytesAt(spill_slab, spill_slot_index),
                                                           spill_slab.layout.slot_stride_bytes) ||
            !genotype_slab::TrySetSlabGenerationSlot(target_generation, child_index, target_slot_index)) {
            return false;
        }
    }

    return true;
}

inline bool TryAdvanceGenerationByHostFailover(
    DeviceSlabGARuntimeBuffers &buffers, const std::uint32_t generation_seed, const GenerationAssemblyConfig &config,
    const std::size_t parent_action_count, const genotype_slab::GenotypeSlabLayout &target_layout,
    const genotype_slab::SlabAssemblyPlan &plan,
    const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
    const training_folder::TrainingWordCatalog *host_training_word_catalog) {
    PopulationFitnessSummary parent_summary{};
    if (!TryReadPopulationFitnessSummary(buffers, parent_summary)) {
        return false;
    }

    genotype_slab::HostGenotypeSlab source_slab{};
    genotype_slab::SlabGeneration source_generation{};
    if (!TryDownloadSlabFromDevice(buffers, source_slab) || !TryDownloadCurrentGenerationFromDevice(buffers, source_generation)) {
        return false;
    }

    std::unique_ptr<std::uint32_t[]> parent_reference_counts(
        new (std::nothrow) std::uint32_t[source_generation.active_individual_count]());
    if ((parent_reference_counts == nullptr) ||
        !genotype_slab::TryBuildParentReferenceCounts(source_generation.slot_indices.get(),
                                                      source_generation.active_individual_count, plan.parent_pairs.get(),
                                                      plan.child_count, parent_reference_counts.get())) {
        (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kInvalidAssemblyPlan);
        return false;
    }

    const std::size_t survivor_count =
        CountReferencedParents(parent_reference_counts.get(), source_generation.active_individual_count);
    if (survivor_count == 0) {
        (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kInvalidAssemblyPlan);
        return false;
    }

    const std::size_t spill_slot_count = survivor_count + plan.child_count;
    genotype_slab::GenotypeSlabLayout spill_layout{};
    spill_layout.action_count = target_layout.action_count;
    spill_layout.slot_stride_bytes = genotype_slab::ComputeSlabSlotStrideBytes(target_layout.action_count);
    spill_layout.slot_count = spill_slot_count;
    spill_layout.slab_bytes = spill_layout.slot_stride_bytes * spill_layout.slot_count;
    if (!genotype_slab::IsValidGenotypeSlabLayout(spill_layout)) {
        (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
        return false;
    }

    genotype_slab::HostGenotypeSlab spill_slab{};
    genotype_slab::SlabGeneration spill_generation{};
    if (!TryCopyReferencedParentsIntoHostSpillSlab(source_slab, source_generation, parent_reference_counts.get(),
                                                   spill_layout, spill_slab, spill_generation)) {
        (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
        return false;
    }

    HostSpillAssemblyContext context{};
    context.assembly_config = config;
    context.parent_action_count = parent_action_count;
    context.pending_output_embedding_injection = pending_output_embedding_injection;
    context.training_word_catalog = host_training_word_catalog;
    context.generation_seed = generation_seed;
    context.current_generation_index = spill_generation.generation_index;

    genotype_slab::SlabAssemblyCallbacks callbacks{};
    callbacks.assemble_child_genome = AssembleSlabChildGenomeOnHost;
    callbacks.user_data = &context;

    genotype_slab::SlabGeneration spill_children{};
    if (!genotype_slab::TryAssembleNextGeneration(spill_slab, spill_generation, plan, spill_children, callbacks)) {
        const DeviceSlabGARuntimeStatusCode failure_status =
            (context.failure_status == DeviceSlabGARuntimeStatusCode::kOk) ? DeviceSlabGARuntimeStatusCode::kCudaFailure
                                                                           : context.failure_status;
        (void)WriteDeviceStatus(buffers, failure_status);
        return false;
    }

    genotype_slab::HostGenotypeSlab target_slab{};
    genotype_slab::SlabGeneration target_generation{};
    if (!TryPackSpillChildrenIntoTargetSlab(spill_slab, spill_children, target_layout, target_slab, target_generation)) {
        (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
        return false;
    }

    const genotype_slab::GenotypeSlabLayout original_layout = buffers.genotype_slab.slab_layout;
    buffers.genotype_slab.slab_layout = target_layout;
    if (!TryUploadCurrentSlabPopulationToDevice(target_slab, target_generation, buffers) ||
        !WritePopulationFitnessSummary(buffers, parent_summary) || !ResetDeviceStatus(buffers)) {
        buffers.genotype_slab.slab_layout = original_layout;
        (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
        return false;
    }

    buffers.host_failover_count += 1;
    buffers.last_generation_used_host_failover = true;
    return true;
}

inline bool TryPlanNextGenerationShape(const DeviceSlabGARuntimeBuffers &buffers,
                                       const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
                                       std::size_t &next_action_count_out, std::size_t &next_generation_size_out) {
    next_action_count_out = buffers.genotype_slab.slab_layout.action_count;
    if (!IsValidPendingOutputEmbeddingInjection(pending_output_embedding_injection,
                                                buffers.genotype_slab.slab_layout.action_count)) {
        return false;
    }

    if (pending_output_embedding_injection.enabled) {
        next_action_count_out += pending_output_embedding_injection.injection_count;
    }

    next_generation_size_out = genotype_slab::SlabSlotCountForByteBudget(
        buffers.generation_byte_budget_bytes, next_action_count_out, buffers.max_generation_size);
    return next_generation_size_out > 0;
}

} // namespace

bool TryCreateDeviceSlabGARuntimeBuffers(DeviceSlabGARuntimeBuffers &buffers, const DeviceSlabGARuntimeConfig &config) {
    buffers = {};
    if (!IsValidDeviceSlabGARuntimeConfig(config)) {
        return false;
    }

    const std::size_t slot_count = RuntimeSlabSlotCount(config);
    const std::size_t max_generation_size = RuntimeGenerationCapacity(config);
    if ((slot_count < max_generation_size) ||
        !TryComputeGenerationByteBudgetBytes(config.action_count, max_generation_size,
                                             buffers.generation_byte_budget_bytes)) {
        return false;
    }

    genotype_slab::device::DeviceSlabRuntimeConfig slab_config{};
    slab_config.slot_count = slot_count;
    slab_config.action_count = config.action_count;
    slab_config.max_generation_size = max_generation_size;
    if (!genotype_slab::device::TryCreateDeviceSlabRuntimeBuffers(buffers.genotype_slab, slab_config)) {
        return false;
    }

    buffers.max_generation_size = max_generation_size;
    bool ok = true;
    ok &= CheckCuda(cudaMalloc(&buffers.summary, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMalloc(&buffers.status, sizeof(int)));
    if (!ok) {
        DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    ok &= CheckCuda(cudaMemset(buffers.summary, 0, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    if (!ok) {
        DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    return true;
}

void DestroyDeviceSlabGARuntimeBuffers(DeviceSlabGARuntimeBuffers &buffers) noexcept {
    genotype_slab::device::DestroyDeviceSlabRuntimeBuffers(buffers.genotype_slab);
    cudaFree(buffers.summary);
    cudaFree(buffers.status);
    buffers = {};
}

bool TryUploadCurrentSlabPopulationToDevice(const genotype_slab::HostGenotypeSlab &host_buffer,
                                            const genotype_slab::SlabGeneration &current_generation,
                                            DeviceSlabGARuntimeBuffers &buffers) {
    if ((current_generation.active_individual_count > buffers.max_generation_size) ||
        !genotype_slab::device::TryUploadSlabToDevice(host_buffer, buffers.genotype_slab) ||
        !genotype_slab::device::TryUploadCurrentGenerationToDevice(current_generation, buffers.genotype_slab)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemset(buffers.summary, 0, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    buffers.last_generation_used_host_failover = false;
    return ok;
}

bool TryDownloadSlabFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                               genotype_slab::HostGenotypeSlab &host_buffer) {
    return genotype_slab::device::TryDownloadSlabFromDevice(buffers.genotype_slab, host_buffer);
}

bool TryDownloadCurrentGenerationFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                                            genotype_slab::SlabGeneration &generation) {
    return genotype_slab::device::TryDownloadCurrentGenerationFromDevice(buffers.genotype_slab, generation);
}

bool TryDownloadNextGenerationFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                                         genotype_slab::SlabGeneration &generation) {
    return genotype_slab::device::TryDownloadNextGenerationFromDevice(buffers.genotype_slab, generation);
}

bool TryEvaluateCurrentGenerationFitnessOnDevice(DeviceSlabGARuntimeBuffers &buffers,
                                                 const RuntimeWordCounts &runtime_word_counts) {
    if (!IsCurrentGenerationCompatible(buffers) || !ResetDeviceStatus(buffers)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemset(buffers.genotype_slab.current_fitness, 0,
                               buffers.genotype_slab.max_generation_size * sizeof(float)));
    ok &= CheckCuda(cudaMemset(buffers.genotype_slab.current_evaluation_counts, 0,
                               buffers.genotype_slab.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.genotype_slab.current_has_fitness, 0,
                               buffers.genotype_slab.max_generation_size * sizeof(std::uint8_t)));
    if (!ok) {
        return false;
    }

    const std::size_t block_count =
        (buffers.genotype_slab.current_generation_size + kSlabGARuntimeThreadBlockSize - 1) /
        kSlabGARuntimeThreadBlockSize;
    EvaluateSlabGenerationFitnessKernel<<<block_count, kSlabGARuntimeThreadBlockSize>>>(
        buffers.genotype_slab.slab_storage, buffers.genotype_slab.slot_states, buffers.genotype_slab.slab_layout,
        buffers.genotype_slab.current_slot_indices, buffers.genotype_slab.current_generation_size,
        buffers.genotype_slab.current_fitness, buffers.genotype_slab.current_evaluation_counts,
        buffers.genotype_slab.current_has_fitness, runtime_word_counts, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize()) ||
        !KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    SummarizeSlabGenerationKernel<<<1, 1>>>(
        buffers.genotype_slab.current_fitness, buffers.genotype_slab.current_has_fitness,
        buffers.genotype_slab.current_generation_index, buffers.genotype_slab.current_generation_size,
        buffers.genotype_slab.slab_layout.action_count, buffers.summary, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    return KernelCompletedSuccessfully(buffers);
}

bool TryReadPopulationFitnessSummaryFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                                               PopulationFitnessSummary &summary) {
    return CheckCuda(cudaMemcpy(&summary, buffers.summary, sizeof(PopulationFitnessSummary), cudaMemcpyDeviceToHost));
}

bool TryAdvanceGenerationOnDevice(DeviceSlabGARuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                  const RuntimeWordCounts &runtime_word_counts, const GenerationAssemblyConfig &config,
                                  const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
                                  const training_folder::TrainingWordCatalog *host_training_word_catalog) {
    buffers.last_generation_used_host_failover = false;
    if (!IsValidGenerationAssemblyConfig(config) || !IsCurrentGenerationCompatible(buffers) ||
        !TryEvaluateCurrentGenerationFitnessOnDevice(buffers, runtime_word_counts) || !ResetDeviceStatus(buffers)) {
        if (!IsValidGenerationAssemblyConfig(config)) {
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig);
        }
        return false;
    }

    std::size_t next_action_count = buffers.genotype_slab.slab_layout.action_count;
    std::size_t next_generation_size = buffers.genotype_slab.current_generation_size;
    if (!TryPlanNextGenerationShape(buffers, pending_output_embedding_injection, next_action_count,
                                    next_generation_size)) {
        (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig);
        return false;
    }

    const std::size_t block_count =
        (next_generation_size + kSlabGARuntimeThreadBlockSize - 1) / kSlabGARuntimeThreadBlockSize;
    const std::uint32_t planning_seed = generation_seed ^ 0xA341316CU;
    BuildSlabAssemblyPlanKernel<<<block_count, kSlabGARuntimeThreadBlockSize>>>(
        buffers.genotype_slab.current_fitness, buffers.genotype_slab.current_has_fitness,
        buffers.genotype_slab.current_generation_size, next_generation_size,
        buffers.genotype_slab.current_generation_index, config.parent_selection, planning_seed,
        buffers.genotype_slab.assembly_parent_pairs, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize()) ||
        !KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    buffers.genotype_slab.planned_child_count = next_generation_size;
    if (!genotype_slab::device::TryApplyFinalChildPriorityToAssemblyPlanOnDevice(buffers.genotype_slab)) {
        genotype_slab::device::DeviceSlabRuntimeStatusCode slab_status =
            genotype_slab::device::DeviceSlabRuntimeStatusCode::kCudaFailure;
        if (genotype_slab::device::TryReadDeviceSlabRuntimeStatus(buffers.genotype_slab, slab_status)) {
            (void)WriteDeviceStatus(buffers, MapSlabRuntimeStatus(slab_status));
        } else {
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
        }
        return false;
    }

    genotype_slab::SlabAssemblyPlan host_plan{};
    if (!TryDownloadAssemblyPlanFromDevice(buffers.genotype_slab, host_plan)) {
        (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
        return false;
    }

    const std::size_t parent_action_count = buffers.genotype_slab.slab_layout.action_count;
    genotype_slab::GenotypeSlabLayout target_layout = buffers.genotype_slab.slab_layout;
    if (pending_output_embedding_injection.enabled &&
        !genotype_slab::device::TryPrepareSlabForExpandedActionCountOnDevice(buffers.genotype_slab,
                                                                             next_action_count)) {
        genotype_slab::device::DeviceSlabRuntimeStatusCode slab_status =
            genotype_slab::device::DeviceSlabRuntimeStatusCode::kCudaFailure;
        if (genotype_slab::device::TryReadDeviceSlabRuntimeStatus(buffers.genotype_slab, slab_status)) {
            if (slab_status == genotype_slab::device::DeviceSlabRuntimeStatusCode::kSlabRepackFailed) {
                if (!genotype_slab::TryCreateExpandedSlabLayout(target_layout, next_action_count, target_layout)) {
                    (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kSlabRepackFailed);
                    return false;
                }
                if (!TryAdvanceGenerationByHostFailover(buffers, generation_seed, config, parent_action_count,
                                                        target_layout, host_plan, pending_output_embedding_injection,
                                                        host_training_word_catalog)) {
                    return false;
                }
                return true;
            }

            (void)WriteDeviceStatus(buffers, MapSlabRuntimeStatus(slab_status));
        } else {
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
        }
        return false;
    }
    target_layout = buffers.genotype_slab.slab_layout;

    genotype_slab::SlabGeneration current_generation{};
    std::uint32_t free_slot_count = 0;
    bool can_fit_device_assembly = false;
    if (!TryDownloadCurrentGenerationFromDevice(buffers, current_generation) ||
        !TryReadDeviceFreeSlotCount(buffers.genotype_slab, free_slot_count) ||
        !TrySimulateDeviceAssemblyBatches(current_generation, host_plan, free_slot_count, can_fit_device_assembly)) {
        (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
        return false;
    }

    if (!can_fit_device_assembly) {
        return TryAdvanceGenerationByHostFailover(buffers, generation_seed, config, parent_action_count, target_layout,
                                                  host_plan, pending_output_embedding_injection,
                                                  host_training_word_catalog);
    }

    genotype_slab::device::SlabDeviceAssemblyConfig slab_assembly_config{};
    slab_assembly_config.breeding = config.breeding;
    slab_assembly_config.mutation = config.mutation;
    slab_assembly_config.parent_action_count = parent_action_count;
    slab_assembly_config.pending_output_embedding_injection = pending_output_embedding_injection;
    if (!genotype_slab::device::TryAssembleNextGenerationOnDevice(buffers.genotype_slab, generation_seed,
                                                                  slab_assembly_config)) {
        genotype_slab::device::DeviceSlabRuntimeStatusCode slab_status =
            genotype_slab::device::DeviceSlabRuntimeStatusCode::kCudaFailure;
        if (genotype_slab::device::TryReadDeviceSlabRuntimeStatus(buffers.genotype_slab, slab_status)) {
            (void)WriteDeviceStatus(buffers, MapSlabRuntimeStatus(slab_status));
        } else {
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
        }
        return false;
    }

    SwapDeviceSlabGenerationBuffers(buffers);
    return true;
}

void SwapDeviceSlabGenerationBuffers(DeviceSlabGARuntimeBuffers &buffers) noexcept {
    std::swap(buffers.genotype_slab.current_slot_indices, buffers.genotype_slab.next_slot_indices);
    std::swap(buffers.genotype_slab.current_fitness, buffers.genotype_slab.next_fitness);
    std::swap(buffers.genotype_slab.current_evaluation_counts, buffers.genotype_slab.next_evaluation_counts);
    std::swap(buffers.genotype_slab.current_has_fitness, buffers.genotype_slab.next_has_fitness);
    std::swap(buffers.genotype_slab.current_generation_index, buffers.genotype_slab.next_generation_index);
    std::swap(buffers.genotype_slab.current_generation_size, buffers.genotype_slab.next_generation_size);
}

bool TryReadDeviceSlabGARuntimeStatus(const DeviceSlabGARuntimeBuffers &buffers,
                                      DeviceSlabGARuntimeStatusCode &status_code) {
    int status_value = 0;
    if (!ReadDeviceStatus(buffers, status_value)) {
        return false;
    }

    status_code = static_cast<DeviceSlabGARuntimeStatusCode>(status_value);
    return true;
}

const char *DeviceSlabGARuntimeStatusCodeString(const DeviceSlabGARuntimeStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceSlabGARuntimeStatusCode::kOk:
        return "ok";
    case DeviceSlabGARuntimeStatusCode::kCudaFailure:
        return "cuda failure";
    case DeviceSlabGARuntimeStatusCode::kInvalidRuntimeConfig:
        return "invalid slab ga runtime config";
    case DeviceSlabGARuntimeStatusCode::kInvalidSlab:
        return "invalid slab state";
    case DeviceSlabGARuntimeStatusCode::kInvalidGeneration:
        return "invalid generation state";
    case DeviceSlabGARuntimeStatusCode::kInvalidTrainingShard:
        return "invalid training-data shard";
    case DeviceSlabGARuntimeStatusCode::kGuessAppendFailed:
        return "could not append a model-selected guess to the Wordle grid";
    case DeviceSlabGARuntimeStatusCode::kPolicyForwardFailed:
        return "policy model forward pass failed";
    case DeviceSlabGARuntimeStatusCode::kActionSelectionFailed:
        return "output-embedding action selection failed";
    case DeviceSlabGARuntimeStatusCode::kPopulationNotEvaluated:
        return "population fitness summary requested before the population was fully evaluated";
    case DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig:
        return "invalid next-generation assembly config";
    case DeviceSlabGARuntimeStatusCode::kParentSelectionFailed:
        return "device parent selection failed";
    case DeviceSlabGARuntimeStatusCode::kInvalidAssemblyPlan:
        return "invalid slab assembly plan";
    case DeviceSlabGARuntimeStatusCode::kInvalidParentIndex:
        return "invalid parent index in slab assembly plan";
    case DeviceSlabGARuntimeStatusCode::kSlabFull:
        return "slab is genuinely full";
    case DeviceSlabGARuntimeStatusCode::kOutputEmbeddingInjectionFailed:
        return "device output-embedding injection failed";
    case DeviceSlabGARuntimeStatusCode::kSlabRepackFailed:
        return "slab compaction/repacking failed";
    }

    return "unknown slab ga runtime status";
}

} // namespace neuroevolution::genetic_algorithm::slab_device
