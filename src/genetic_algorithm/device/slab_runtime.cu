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

inline bool TryCountAssembledChildPrefix(const DeviceSlabGARuntimeBuffers &buffers, std::size_t &assembled_child_count) {
    assembled_child_count = 0;

    genotype_slab::SlabGeneration next_generation{};
    if (!TryDownloadNextGenerationFromDevice(buffers, next_generation)) {
        return false;
    }

    while ((assembled_child_count < next_generation.active_individual_count) &&
           (next_generation.slot_indices[assembled_child_count] != genotype_slab::kInvalidSlabSlotIndex)) {
        ++assembled_child_count;
    }

    for (std::size_t child_index = assembled_child_count; child_index < next_generation.active_individual_count;
         ++child_index) {
        if (next_generation.slot_indices[child_index] != genotype_slab::kInvalidSlabSlotIndex) {
            return false;
        }
    }

    return true;
}

__global__ void ReleaseNextGenerationPrefixKernel(
    std::uint8_t *slab_storage, genotype_slab::SlabSlotState *slot_states, std::uint32_t *free_slot_stack,
    std::uint32_t *free_slot_count, std::uint32_t *free_slot_lock, const genotype_slab::GenotypeSlabLayout slab_layout,
    std::uint32_t *next_slot_indices, float *next_fitness, std::uint32_t *next_evaluation_counts,
    std::uint8_t *next_has_fitness, const std::size_t spilled_child_count, int *status) {
    genotype_slab::GenotypeSlabView slab{
        .layout = slab_layout,
        .storage = slab_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };

    const std::size_t worker_index =
        (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
    const std::size_t worker_count = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t child_index = worker_index; child_index < spilled_child_count; child_index += worker_count) {
        const std::uint32_t slot_index = next_slot_indices[child_index];
        if (slot_index == genotype_slab::kInvalidSlabSlotIndex) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidGeneration);
            return;
        }

        if (!genotype_slab::TryReleaseSlabSlot(slab, slot_index)) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidSlab);
            return;
        }

        next_slot_indices[child_index] = genotype_slab::kInvalidSlabSlotIndex;
        next_fitness[child_index] = 0.0f;
        next_evaluation_counts[child_index] = 0;
        next_has_fitness[child_index] = 0;
    }
}

__global__ void AllocateRestoredChildPrefixKernel(
    std::uint8_t *slab_storage, genotype_slab::SlabSlotState *slot_states, std::uint32_t *free_slot_stack,
    std::uint32_t *free_slot_count, std::uint32_t *free_slot_lock, const genotype_slab::GenotypeSlabLayout slab_layout,
    std::uint32_t *next_slot_indices, float *next_fitness, std::uint32_t *next_evaluation_counts,
    std::uint8_t *next_has_fitness, const std::size_t spilled_child_count, int *status) {
    genotype_slab::GenotypeSlabView slab{
        .layout = slab_layout,
        .storage = slab_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };

    const std::size_t worker_index =
        (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
    const std::size_t worker_count = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t child_index = worker_index; child_index < spilled_child_count; child_index += worker_count) {
        if (next_slot_indices[child_index] != genotype_slab::kInvalidSlabSlotIndex) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidGeneration);
            return;
        }

        std::uint32_t slot_index = genotype_slab::kInvalidSlabSlotIndex;
        if (!genotype_slab::TryAllocateSlabSlot(slab, slot_index)) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kSlabFull);
            return;
        }

        next_slot_indices[child_index] = slot_index;
        next_fitness[child_index] = 0.0f;
        next_evaluation_counts[child_index] = 0;
        next_has_fitness[child_index] = 0;
    }
}

inline bool TryCreateHostSpillStorage(const DeviceSlabGARuntimeBuffers &buffers, const std::size_t action_count,
                                      const std::size_t next_generation_size, const std::size_t next_generation_index,
                                      genotype_slab::HostGenotypeSlab &spill_slab,
                                      genotype_slab::SlabGeneration &spill_generation) {
    return genotype_slab::TryCreateHostGenotypeSlabForByteBudget(spill_slab, buffers.host_spillover_byte_budget_bytes,
                                                                 action_count) &&
           (spill_slab.layout.slot_count > 0) &&
           genotype_slab::TryCreateSlabGeneration(spill_generation, next_generation_size, next_generation_index);
}

inline bool TrySpillAssembledChildrenToHost(DeviceSlabGARuntimeBuffers &buffers, const std::size_t spilled_child_count,
                                            genotype_slab::HostGenotypeSlab &spill_slab,
                                            genotype_slab::SlabGeneration &spill_generation) {
    if ((spilled_child_count == 0) || !genotype_slab::IsValidHostGenotypeSlab(spill_slab) ||
        !genotype_slab::IsValidSlabGeneration(spill_generation) ||
        (spilled_child_count > spill_generation.active_individual_count) ||
        (spill_slab.layout.action_count != buffers.genotype_slab.slab_layout.action_count) ||
        (spill_slab.layout.slot_stride_bytes != buffers.genotype_slab.slab_layout.slot_stride_bytes)) {
        return false;
    }

    genotype_slab::SlabGeneration next_generation{};
    if (!TryDownloadNextGenerationFromDevice(buffers, next_generation)) {
        return false;
    }

    for (std::size_t child_index = 0; child_index < spilled_child_count; ++child_index) {
        const std::uint32_t device_slot_index = next_generation.slot_indices[child_index];
        if (device_slot_index == genotype_slab::kInvalidSlabSlotIndex) {
            return false;
        }

        std::uint32_t spill_slot_index = genotype_slab::kInvalidSlabSlotIndex;
        if (!genotype_slab::TryAllocateSlabSlot(spill_slab, spill_slot_index) ||
            !genotype_slab::TrySetSlabGenerationSlot(spill_generation, child_index, spill_slot_index) ||
            !CheckCuda(cudaMemcpy(genotype_slab::HostSlabSlotBytesAt(spill_slab, spill_slot_index),
                                  genotype_slab::SlabSlotBytesAt(buffers.genotype_slab.slab_storage,
                                                                 buffers.genotype_slab.slab_layout, device_slot_index),
                                  spill_slab.layout.slot_stride_bytes, cudaMemcpyDeviceToHost))) {
            return false;
        }
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    ReleaseNextGenerationPrefixKernel<<<(spilled_child_count + kSlabGARuntimeThreadBlockSize - 1) /
                                            kSlabGARuntimeThreadBlockSize,
                                        kSlabGARuntimeThreadBlockSize>>>(
        buffers.genotype_slab.slab_storage, buffers.genotype_slab.slot_states, buffers.genotype_slab.free_slot_stack,
        buffers.genotype_slab.free_slot_count, buffers.genotype_slab.free_slot_lock, buffers.genotype_slab.slab_layout,
        buffers.genotype_slab.next_slot_indices, buffers.genotype_slab.next_fitness,
        buffers.genotype_slab.next_evaluation_counts, buffers.genotype_slab.next_has_fitness, spilled_child_count,
        buffers.status);
    return CheckCuda(cudaGetLastError()) && CheckCuda(cudaDeviceSynchronize()) && KernelCompletedSuccessfully(buffers);
}

inline bool TryRestoreSpilledChildrenToDevice(DeviceSlabGARuntimeBuffers &buffers, const std::size_t spilled_child_count,
                                              const genotype_slab::HostGenotypeSlab &spill_slab,
                                              const genotype_slab::SlabGeneration &spill_generation) {
    if ((spilled_child_count == 0) || !genotype_slab::IsValidHostGenotypeSlab(spill_slab) ||
        !genotype_slab::IsValidSlabGeneration(spill_generation) ||
        (spilled_child_count > spill_generation.active_individual_count) ||
        (spill_slab.layout.action_count != buffers.genotype_slab.slab_layout.action_count) ||
        (spill_slab.layout.slot_stride_bytes != buffers.genotype_slab.slab_layout.slot_stride_bytes)) {
        return false;
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    AllocateRestoredChildPrefixKernel<<<(spilled_child_count + kSlabGARuntimeThreadBlockSize - 1) /
                                             kSlabGARuntimeThreadBlockSize,
                                         kSlabGARuntimeThreadBlockSize>>>(
        buffers.genotype_slab.slab_storage, buffers.genotype_slab.slot_states, buffers.genotype_slab.free_slot_stack,
        buffers.genotype_slab.free_slot_count, buffers.genotype_slab.free_slot_lock, buffers.genotype_slab.slab_layout,
        buffers.genotype_slab.next_slot_indices, buffers.genotype_slab.next_fitness,
        buffers.genotype_slab.next_evaluation_counts, buffers.genotype_slab.next_has_fitness, spilled_child_count,
        buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize()) || !KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    genotype_slab::SlabGeneration next_generation{};
    if (!TryDownloadNextGenerationFromDevice(buffers, next_generation)) {
        return false;
    }

    for (std::size_t child_index = 0; child_index < spilled_child_count; ++child_index) {
        const std::uint32_t spill_slot_index = spill_generation.slot_indices[child_index];
        const std::uint32_t device_slot_index = next_generation.slot_indices[child_index];
        if ((spill_slot_index == genotype_slab::kInvalidSlabSlotIndex) ||
            (device_slot_index == genotype_slab::kInvalidSlabSlotIndex)) {
            return false;
        }

        if (!CheckCuda(cudaMemcpy(genotype_slab::SlabSlotBytesAt(buffers.genotype_slab.slab_storage,
                                                                 buffers.genotype_slab.slab_layout, device_slot_index),
                                  genotype_slab::HostSlabSlotBytesAt(spill_slab, spill_slot_index),
                                  spill_slab.layout.slot_stride_bytes, cudaMemcpyHostToDevice))) {
            return false;
        }
    }

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
    buffers.host_spillover_byte_budget_bytes = config.host_spillover_byte_budget_bytes;
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
    buffers.last_generation_used_host_spillover = false;
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
    (void)host_training_word_catalog;
    buffers.last_generation_used_host_spillover = false;
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

    const std::size_t parent_action_count = buffers.genotype_slab.slab_layout.action_count;
    if (pending_output_embedding_injection.enabled &&
        !genotype_slab::device::TryPrepareSlabForExpandedActionCountOnDevice(buffers.genotype_slab,
                                                                             next_action_count)) {
        genotype_slab::device::DeviceSlabRuntimeStatusCode slab_status =
            genotype_slab::device::DeviceSlabRuntimeStatusCode::kCudaFailure;
        if (genotype_slab::device::TryReadDeviceSlabRuntimeStatus(buffers.genotype_slab, slab_status)) {
            (void)WriteDeviceStatus(buffers, MapSlabRuntimeStatus(slab_status));
        } else {
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
        }
        return false;
    }

    genotype_slab::device::SlabDeviceAssemblyConfig slab_assembly_config{};
    slab_assembly_config.breeding = config.breeding;
    slab_assembly_config.mutation = config.mutation;
    slab_assembly_config.parent_action_count = parent_action_count;
    slab_assembly_config.pending_output_embedding_injection = pending_output_embedding_injection;

    if (!genotype_slab::device::TryInitializeNextGenerationAssemblyOnDevice(buffers.genotype_slab,
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

    bool used_host_spillover = false;
    std::size_t spilled_child_count = 0;
    genotype_slab::HostGenotypeSlab spill_slab{};
    genotype_slab::SlabGeneration spill_generation{};
    std::size_t child_offset = 0;

    while (child_offset < buffers.genotype_slab.next_generation_size) {
        if (genotype_slab::device::TryContinueNextGenerationAssemblyOnDevice(buffers.genotype_slab, generation_seed,
                                                                             slab_assembly_config, child_offset)) {
            child_offset = buffers.genotype_slab.next_generation_size;
            break;
        }

        genotype_slab::device::DeviceSlabRuntimeStatusCode slab_status =
            genotype_slab::device::DeviceSlabRuntimeStatusCode::kCudaFailure;
        if (!genotype_slab::device::TryReadDeviceSlabRuntimeStatus(buffers.genotype_slab, slab_status)) {
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
            return false;
        }

        if (slab_status != genotype_slab::device::DeviceSlabRuntimeStatusCode::kSlabFull) {
            (void)WriteDeviceStatus(buffers, MapSlabRuntimeStatus(slab_status));
            return false;
        }

        if (used_host_spillover || (buffers.host_spillover_byte_budget_bytes == 0)) {
            (void)genotype_slab::device::TryCleanupFailedAssemblyOnDevice(buffers.genotype_slab);
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kSlabFull);
            return false;
        }

        std::size_t assembled_child_count = 0;
        if (!TryCountAssembledChildPrefix(buffers, assembled_child_count) || (assembled_child_count == 0)) {
            (void)genotype_slab::device::TryCleanupFailedAssemblyOnDevice(buffers.genotype_slab);
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kSlabFull);
            return false;
        }

        if (!TryCreateHostSpillStorage(buffers, buffers.genotype_slab.slab_layout.action_count,
                                       buffers.genotype_slab.next_generation_size,
                                       buffers.genotype_slab.next_generation_index, spill_slab, spill_generation) ||
            (assembled_child_count > spill_slab.layout.slot_count) ||
            !TrySpillAssembledChildrenToHost(buffers, assembled_child_count, spill_slab, spill_generation)) {
            (void)genotype_slab::device::TryCleanupFailedAssemblyOnDevice(buffers.genotype_slab);
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
            return false;
        }

        used_host_spillover = true;
        spilled_child_count = assembled_child_count;
        child_offset = assembled_child_count;
    }

    if (used_host_spillover &&
        !TryRestoreSpilledChildrenToDevice(buffers, spilled_child_count, spill_slab, spill_generation)) {
        (void)genotype_slab::device::TryCleanupFailedAssemblyOnDevice(buffers.genotype_slab);
        (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
        return false;
    }

    buffers.last_generation_used_host_spillover = used_host_spillover;
    if (used_host_spillover) {
        buffers.host_spillover_count += 1;
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
