#include "genetic_algorithm/device/pool_runtime.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "genetic_algorithm/device/evaluation_ops.cuh"
#include "genetic_algorithm/device/genome_ops.cuh"
#include "genetic_algorithm/device/selection_ops.cuh"

namespace neuroevolution::genetic_algorithm::pool_device {

namespace {

using device_evaluation_ops::DeviceGenomeEvaluationStatusCode;
using device_evaluation_ops::TryEvaluateGenomeFitness;
using device_genome_ops::DeviceRandomState;
using device_genome_ops::MakeDeviceRandomState;
using device_selection_ops::IsBetterFitness;
using device_selection_ops::TrySelectParentPairDevice;

constexpr std::size_t kPoolGARuntimeThreadBlockSize = 256;

NEUROEVOLUTION_HOST_DEVICE constexpr int DeviceStatusValue(const DevicePoolGARuntimeStatusCode status_code) {
    return static_cast<int>(status_code);
}

inline bool CheckCuda(const cudaError_t error) { return error == cudaSuccess; }

inline bool ReadDeviceStatus(const DevicePoolGARuntimeBuffers &buffers, int &status_value) {
    return CheckCuda(cudaMemcpy(&status_value, buffers.status, sizeof(int), cudaMemcpyDeviceToHost));
}

inline bool WriteDeviceStatus(const DevicePoolGARuntimeBuffers &buffers,
                              const DevicePoolGARuntimeStatusCode status_code) {
    const int status_value = DeviceStatusValue(status_code);
    return CheckCuda(cudaMemcpy(buffers.status, &status_value, sizeof(int), cudaMemcpyHostToDevice));
}

inline bool ResetDeviceStatus(const DevicePoolGARuntimeBuffers &buffers) {
    return WriteDeviceStatus(buffers, DevicePoolGARuntimeStatusCode::kOk);
}

inline bool KernelCompletedSuccessfully(const DevicePoolGARuntimeBuffers &buffers) {
    int status_value = DeviceStatusValue(DevicePoolGARuntimeStatusCode::kCudaFailure);
    return ReadDeviceStatus(buffers, status_value) &&
           (status_value == DeviceStatusValue(DevicePoolGARuntimeStatusCode::kOk));
}

inline bool IsCurrentGenerationCompatible(const DevicePoolGARuntimeBuffers &buffers) noexcept {
    return genotype_pool::IsValidGenotypePoolLayout(buffers.pool_buffers.pool_layout) &&
           (buffers.max_generation_size > 0) && (buffers.pool_buffers.current_generation_size > 0) &&
           (buffers.pool_buffers.current_generation_size <= buffers.pool_buffers.max_generation_size);
}

NEUROEVOLUTION_HOST_DEVICE constexpr DevicePoolGARuntimeStatusCode
MapEvaluationStatus(const DeviceGenomeEvaluationStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceGenomeEvaluationStatusCode::kOk:
        return DevicePoolGARuntimeStatusCode::kOk;
    case DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard:
        return DevicePoolGARuntimeStatusCode::kInvalidTrainingShard;
    case DeviceGenomeEvaluationStatusCode::kGuessAppendFailed:
        return DevicePoolGARuntimeStatusCode::kGuessAppendFailed;
    case DeviceGenomeEvaluationStatusCode::kPolicyForwardFailed:
        return DevicePoolGARuntimeStatusCode::kPolicyForwardFailed;
    case DeviceGenomeEvaluationStatusCode::kActionSelectionFailed:
        return DevicePoolGARuntimeStatusCode::kActionSelectionFailed;
    }

    return DevicePoolGARuntimeStatusCode::kCudaFailure;
}

inline DevicePoolGARuntimeStatusCode
MapPoolRuntimeStatus(const genotype_pool::device::DevicePoolRuntimeStatusCode status_code) noexcept {
    using genotype_pool::device::DevicePoolRuntimeStatusCode;

    switch (status_code) {
    case DevicePoolRuntimeStatusCode::kOk:
        return DevicePoolGARuntimeStatusCode::kOk;
    case DevicePoolRuntimeStatusCode::kCudaFailure:
        return DevicePoolGARuntimeStatusCode::kCudaFailure;
    case DevicePoolRuntimeStatusCode::kInvalidRuntimeConfig:
        return DevicePoolGARuntimeStatusCode::kInvalidRuntimeConfig;
    case DevicePoolRuntimeStatusCode::kInvalidPool:
        return DevicePoolGARuntimeStatusCode::kInvalidPool;
    case DevicePoolRuntimeStatusCode::kInvalidGeneration:
        return DevicePoolGARuntimeStatusCode::kInvalidGeneration;
    case DevicePoolRuntimeStatusCode::kInvalidAssemblyPlan:
        return DevicePoolGARuntimeStatusCode::kInvalidAssemblyPlan;
    case DevicePoolRuntimeStatusCode::kInvalidAssemblyConfig:
        return DevicePoolGARuntimeStatusCode::kInvalidAssemblyConfig;
    case DevicePoolRuntimeStatusCode::kInvalidParentIndex:
        return DevicePoolGARuntimeStatusCode::kInvalidParentIndex;
    case DevicePoolRuntimeStatusCode::kPoolFull:
        return DevicePoolGARuntimeStatusCode::kPoolFull;
    case DevicePoolRuntimeStatusCode::kOutputEmbeddingInjectionFailed:
        return DevicePoolGARuntimeStatusCode::kOutputEmbeddingInjectionFailed;
    case DevicePoolRuntimeStatusCode::kPoolRepackFailed:
        return DevicePoolGARuntimeStatusCode::kPoolRepackFailed;
    }

    return DevicePoolGARuntimeStatusCode::kCudaFailure;
}

__device__ void SetFailureStatus(int *status, const DevicePoolGARuntimeStatusCode status_code) {
    atomicCAS(status, DeviceStatusValue(DevicePoolGARuntimeStatusCode::kOk), DeviceStatusValue(status_code));
}

__global__ void EvaluatePoolGenerationFitnessKernel(
    const std::uint8_t *pool_storage, const genotype_pool::PoolSlotState *slot_states,
    const genotype_pool::GenotypePoolLayout pool_layout, const std::uint32_t *current_slot_indices,
    const std::size_t current_generation_size, float *current_fitness, std::uint32_t *current_evaluation_counts,
    std::uint8_t *current_has_fitness, const RuntimeWordCounts runtime_word_counts, int *status) {
    const std::size_t individual_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (!genotype_pool::IsValidGenotypePoolLayout(pool_layout) || (pool_storage == nullptr) ||
        (slot_states == nullptr) || (current_slot_indices == nullptr) || (current_fitness == nullptr) ||
        (current_evaluation_counts == nullptr) || (current_has_fitness == nullptr) || (current_generation_size == 0)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DevicePoolGARuntimeStatusCode::kInvalidPool);
        }
        return;
    }

    if (individual_index >= current_generation_size) {
        return;
    }

    const std::uint32_t slot_index = current_slot_indices[individual_index];
    if ((slot_index == genotype_pool::kInvalidPoolSlotIndex) || (slot_index >= pool_layout.slot_count)) {
        SetFailureStatus(status, DevicePoolGARuntimeStatusCode::kInvalidGeneration);
        return;
    }

    if (!slot_states[slot_index].occupied || (slot_states[slot_index].reference_count == 0)) {
        SetFailureStatus(status, DevicePoolGARuntimeStatusCode::kInvalidPool);
        return;
    }

    float fitness = 0.0f;
    const DeviceGenomeEvaluationStatusCode evaluation_status =
        TryEvaluateGenomeFitness(genotype_pool::PoolSlotBytesAt(pool_storage, pool_layout, slot_index),
                                 pool_layout.action_count, runtime_word_counts, fitness);
    if (evaluation_status != DeviceGenomeEvaluationStatusCode::kOk) {
        SetFailureStatus(status, MapEvaluationStatus(evaluation_status));
        return;
    }

    current_fitness[individual_index] = fitness;
    ++current_evaluation_counts[individual_index];
    current_has_fitness[individual_index] = 1;
}

__global__ void SummarizePoolGenerationKernel(const float *current_fitness, const std::uint8_t *current_has_fitness,
                                              const std::size_t current_generation_index,
                                              const std::size_t current_generation_size, const std::size_t action_count,
                                              PopulationFitnessSummary *summary, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    if ((current_fitness == nullptr) || (current_has_fitness == nullptr) || (summary == nullptr) ||
        (current_generation_size == 0) || (action_count == 0)) {
        SetFailureStatus(status, DevicePoolGARuntimeStatusCode::kInvalidGeneration);
        return;
    }

    bool found_best = false;
    float best_fitness = 0.0f;
    float fitness_sum = 0.0f;
    std::size_t best_index = 0;

    for (std::size_t individual_index = 0; individual_index < current_generation_size; ++individual_index) {
        if (current_has_fitness[individual_index] == 0) {
            SetFailureStatus(status, DevicePoolGARuntimeStatusCode::kPopulationNotEvaluated);
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

__global__ void BuildPoolAssemblyPlanKernel(const float *current_fitness, const std::uint8_t *current_has_fitness,
                                            const std::size_t current_generation_size, const std::size_t child_count,
                                            const std::size_t current_generation_index,
                                            const ParentSelectionConfig config, const std::uint32_t planning_seed,
                                            genotype_pool::PoolParentPair *parent_pairs, int *status) {
    const std::size_t child_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ((current_fitness == nullptr) || (current_has_fitness == nullptr) || (parent_pairs == nullptr) ||
        (current_generation_size == 0) || !IsValidParentSelectionConfig(config)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DevicePoolGARuntimeStatusCode::kInvalidAssemblyConfig);
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
        SetFailureStatus(status, DevicePoolGARuntimeStatusCode::kParentSelectionFailed);
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

inline bool TryPlanNextGenerationShape(const DevicePoolGARuntimeBuffers &buffers,
                                       const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
                                       std::size_t &next_action_count_out, std::size_t &next_generation_size_out) {
    next_action_count_out = buffers.pool_buffers.pool_layout.action_count;
    next_generation_size_out = buffers.pool_buffers.current_generation_size;
    if (!IsValidPendingOutputEmbeddingInjection(pending_output_embedding_injection,
                                                buffers.pool_buffers.pool_layout.action_count)) {
        return false;
    }

    if (!pending_output_embedding_injection.enabled) {
        return next_generation_size_out > 0;
    }

    next_action_count_out += pending_output_embedding_injection.injection_count;
    genotype_pool::GenotypePoolLayout next_layout{};
    if (!genotype_pool::TryCreateExpandedPoolLayout(buffers.pool_buffers.pool_layout, next_action_count_out,
                                                    next_layout) ||
        (next_layout.slot_count <= 1)) {
        return false;
    }

    next_generation_size_out = (next_layout.slot_count - 1) / 2;
    if (next_generation_size_out > buffers.max_generation_size) {
        next_generation_size_out = buffers.max_generation_size;
    }

    return next_generation_size_out > 0;
}

} // namespace

bool TryCreateDevicePoolGARuntimeBuffers(DevicePoolGARuntimeBuffers &buffers, const DevicePoolGARuntimeConfig &config) {
    buffers = {};
    if (!IsValidDevicePoolGARuntimeConfig(config)) {
        return false;
    }

    genotype_pool::device::DevicePoolRuntimeConfig pool_config{};
    pool_config.slot_count = config.slot_count;
    pool_config.action_count = config.action_count;
    pool_config.max_generation_size = config.max_generation_size;
    if (!genotype_pool::device::TryCreateDevicePoolRuntimeBuffers(buffers.pool_buffers, pool_config)) {
        return false;
    }

    buffers.max_generation_size = config.max_generation_size;
    bool ok = true;
    ok &= CheckCuda(cudaMalloc(&buffers.summary, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMalloc(&buffers.status, sizeof(int)));
    if (!ok) {
        DestroyDevicePoolGARuntimeBuffers(buffers);
        return false;
    }

    ok &= CheckCuda(cudaMemset(buffers.summary, 0, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    if (!ok) {
        DestroyDevicePoolGARuntimeBuffers(buffers);
        return false;
    }

    return true;
}

void DestroyDevicePoolGARuntimeBuffers(DevicePoolGARuntimeBuffers &buffers) noexcept {
    genotype_pool::device::DestroyDevicePoolRuntimeBuffers(buffers.pool_buffers);
    cudaFree(buffers.summary);
    cudaFree(buffers.status);
    buffers = {};
}

bool TryUploadCurrentPoolPopulationToDevice(const genotype_pool::HostGenotypePool &host_pool,
                                            const genotype_pool::PoolGeneration &current_generation,
                                            DevicePoolGARuntimeBuffers &buffers) {
    if ((current_generation.active_individual_count > buffers.max_generation_size) ||
        !genotype_pool::device::TryUploadPoolToDevice(host_pool, buffers.pool_buffers) ||
        !genotype_pool::device::TryUploadCurrentGenerationToDevice(current_generation, buffers.pool_buffers)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemset(buffers.summary, 0, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    return ok;
}

bool TryDownloadPoolFromDevice(const DevicePoolGARuntimeBuffers &buffers, genotype_pool::HostGenotypePool &host_pool) {
    return genotype_pool::device::TryDownloadPoolFromDevice(buffers.pool_buffers, host_pool);
}

bool TryDownloadCurrentGenerationFromDevice(const DevicePoolGARuntimeBuffers &buffers,
                                            genotype_pool::PoolGeneration &generation) {
    return genotype_pool::device::TryDownloadCurrentGenerationFromDevice(buffers.pool_buffers, generation);
}

bool TryDownloadNextGenerationFromDevice(const DevicePoolGARuntimeBuffers &buffers,
                                         genotype_pool::PoolGeneration &generation) {
    return genotype_pool::device::TryDownloadNextGenerationFromDevice(buffers.pool_buffers, generation);
}

bool TryEvaluateCurrentGenerationFitnessOnDevice(DevicePoolGARuntimeBuffers &buffers,
                                                 const RuntimeWordCounts &runtime_word_counts) {
    if (!IsCurrentGenerationCompatible(buffers) || !ResetDeviceStatus(buffers)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(
        cudaMemset(buffers.pool_buffers.current_fitness, 0, buffers.pool_buffers.max_generation_size * sizeof(float)));
    ok &= CheckCuda(cudaMemset(buffers.pool_buffers.current_evaluation_counts, 0,
                               buffers.pool_buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.pool_buffers.current_has_fitness, 0,
                               buffers.pool_buffers.max_generation_size * sizeof(std::uint8_t)));
    if (!ok) {
        return false;
    }

    const std::size_t block_count = (buffers.pool_buffers.current_generation_size + kPoolGARuntimeThreadBlockSize - 1) /
                                    kPoolGARuntimeThreadBlockSize;
    EvaluatePoolGenerationFitnessKernel<<<block_count, kPoolGARuntimeThreadBlockSize>>>(
        buffers.pool_buffers.pool_storage, buffers.pool_buffers.slot_states, buffers.pool_buffers.pool_layout,
        buffers.pool_buffers.current_slot_indices, buffers.pool_buffers.current_generation_size,
        buffers.pool_buffers.current_fitness, buffers.pool_buffers.current_evaluation_counts,
        buffers.pool_buffers.current_has_fitness, runtime_word_counts, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize()) ||
        !KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    SummarizePoolGenerationKernel<<<1, 1>>>(
        buffers.pool_buffers.current_fitness, buffers.pool_buffers.current_has_fitness,
        buffers.pool_buffers.current_generation_index, buffers.pool_buffers.current_generation_size,
        buffers.pool_buffers.pool_layout.action_count, buffers.summary, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    return KernelCompletedSuccessfully(buffers);
}

bool TryReadPopulationFitnessSummaryFromDevice(const DevicePoolGARuntimeBuffers &buffers,
                                               PopulationFitnessSummary &summary) {
    return CheckCuda(cudaMemcpy(&summary, buffers.summary, sizeof(PopulationFitnessSummary), cudaMemcpyDeviceToHost));
}

bool TryAdvanceGenerationOnDevice(DevicePoolGARuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                  const RuntimeWordCounts &runtime_word_counts, const GenerationAssemblyConfig &config,
                                  const PendingOutputEmbeddingInjection &pending_output_embedding_injection) {
    if (!IsValidGenerationAssemblyConfig(config) || !IsCurrentGenerationCompatible(buffers) ||
        !TryEvaluateCurrentGenerationFitnessOnDevice(buffers, runtime_word_counts) || !ResetDeviceStatus(buffers)) {
        if (!IsValidGenerationAssemblyConfig(config)) {
            (void)WriteDeviceStatus(buffers, DevicePoolGARuntimeStatusCode::kInvalidAssemblyConfig);
        }
        return false;
    }

    std::size_t next_action_count = buffers.pool_buffers.pool_layout.action_count;
    std::size_t next_generation_size = buffers.pool_buffers.current_generation_size;
    if (!TryPlanNextGenerationShape(buffers, pending_output_embedding_injection, next_action_count,
                                    next_generation_size)) {
        (void)WriteDeviceStatus(buffers, DevicePoolGARuntimeStatusCode::kInvalidAssemblyConfig);
        return false;
    }

    const std::size_t block_count =
        (next_generation_size + kPoolGARuntimeThreadBlockSize - 1) / kPoolGARuntimeThreadBlockSize;
    const std::uint32_t planning_seed = generation_seed ^ 0xA341316CU;
    BuildPoolAssemblyPlanKernel<<<block_count, kPoolGARuntimeThreadBlockSize>>>(
        buffers.pool_buffers.current_fitness, buffers.pool_buffers.current_has_fitness,
        buffers.pool_buffers.current_generation_size, next_generation_size,
        buffers.pool_buffers.current_generation_index, config.parent_selection, planning_seed,
        buffers.pool_buffers.assembly_parent_pairs, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize()) ||
        !KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    buffers.pool_buffers.planned_child_count = next_generation_size;
    const std::size_t parent_action_count = buffers.pool_buffers.pool_layout.action_count;
    if (pending_output_embedding_injection.enabled &&
        !genotype_pool::device::TryPreparePoolForExpandedActionCountOnDevice(buffers.pool_buffers, next_action_count)) {
        genotype_pool::device::DevicePoolRuntimeStatusCode pool_status =
            genotype_pool::device::DevicePoolRuntimeStatusCode::kCudaFailure;
        if (genotype_pool::device::TryReadDevicePoolRuntimeStatus(buffers.pool_buffers, pool_status)) {
            (void)WriteDeviceStatus(buffers, MapPoolRuntimeStatus(pool_status));
        } else {
            (void)WriteDeviceStatus(buffers, DevicePoolGARuntimeStatusCode::kCudaFailure);
        }
        return false;
    }

    genotype_pool::device::PoolDeviceAssemblyConfig pool_assembly_config{};
    pool_assembly_config.breeding = config.breeding;
    pool_assembly_config.mutation = config.mutation;
    pool_assembly_config.parent_action_count = parent_action_count;
    pool_assembly_config.pending_output_embedding_injection = pending_output_embedding_injection;
    if (!genotype_pool::device::TryAssembleNextGenerationOnDevice(buffers.pool_buffers, generation_seed,
                                                                  pool_assembly_config)) {
        genotype_pool::device::DevicePoolRuntimeStatusCode pool_status =
            genotype_pool::device::DevicePoolRuntimeStatusCode::kCudaFailure;
        if (genotype_pool::device::TryReadDevicePoolRuntimeStatus(buffers.pool_buffers, pool_status)) {
            (void)WriteDeviceStatus(buffers, MapPoolRuntimeStatus(pool_status));
        } else {
            (void)WriteDeviceStatus(buffers, DevicePoolGARuntimeStatusCode::kCudaFailure);
        }
        return false;
    }

    SwapDevicePoolGenerationBuffers(buffers);
    return true;
}

void SwapDevicePoolGenerationBuffers(DevicePoolGARuntimeBuffers &buffers) noexcept {
    std::swap(buffers.pool_buffers.current_slot_indices, buffers.pool_buffers.next_slot_indices);
    std::swap(buffers.pool_buffers.current_fitness, buffers.pool_buffers.next_fitness);
    std::swap(buffers.pool_buffers.current_evaluation_counts, buffers.pool_buffers.next_evaluation_counts);
    std::swap(buffers.pool_buffers.current_has_fitness, buffers.pool_buffers.next_has_fitness);
    std::swap(buffers.pool_buffers.current_generation_index, buffers.pool_buffers.next_generation_index);
    std::swap(buffers.pool_buffers.current_generation_size, buffers.pool_buffers.next_generation_size);
}

bool TryReadDevicePoolGARuntimeStatus(const DevicePoolGARuntimeBuffers &buffers,
                                      DevicePoolGARuntimeStatusCode &status_code) {
    int status_value = 0;
    if (!ReadDeviceStatus(buffers, status_value)) {
        return false;
    }

    status_code = static_cast<DevicePoolGARuntimeStatusCode>(status_value);
    return true;
}

const char *DevicePoolGARuntimeStatusCodeString(const DevicePoolGARuntimeStatusCode status_code) noexcept {
    switch (status_code) {
    case DevicePoolGARuntimeStatusCode::kOk:
        return "ok";
    case DevicePoolGARuntimeStatusCode::kCudaFailure:
        return "cuda failure";
    case DevicePoolGARuntimeStatusCode::kInvalidRuntimeConfig:
        return "invalid pool ga runtime config";
    case DevicePoolGARuntimeStatusCode::kInvalidPool:
        return "invalid pool state";
    case DevicePoolGARuntimeStatusCode::kInvalidGeneration:
        return "invalid generation state";
    case DevicePoolGARuntimeStatusCode::kInvalidTrainingShard:
        return "invalid training-data shard";
    case DevicePoolGARuntimeStatusCode::kGuessAppendFailed:
        return "could not append a model-selected guess to the Wordle grid";
    case DevicePoolGARuntimeStatusCode::kPolicyForwardFailed:
        return "policy model forward pass failed";
    case DevicePoolGARuntimeStatusCode::kActionSelectionFailed:
        return "output-embedding action selection failed";
    case DevicePoolGARuntimeStatusCode::kPopulationNotEvaluated:
        return "population fitness summary requested before the population was fully evaluated";
    case DevicePoolGARuntimeStatusCode::kInvalidAssemblyConfig:
        return "invalid next-generation assembly config";
    case DevicePoolGARuntimeStatusCode::kParentSelectionFailed:
        return "device parent selection failed";
    case DevicePoolGARuntimeStatusCode::kInvalidAssemblyPlan:
        return "invalid pool assembly plan";
    case DevicePoolGARuntimeStatusCode::kInvalidParentIndex:
        return "invalid parent index in pool assembly plan";
    case DevicePoolGARuntimeStatusCode::kPoolFull:
        return "pool is genuinely full";
    case DevicePoolGARuntimeStatusCode::kOutputEmbeddingInjectionFailed:
        return "device output-embedding injection failed";
    case DevicePoolGARuntimeStatusCode::kPoolRepackFailed:
        return "pool compaction/repacking failed";
    }

    return "unknown pool ga runtime status";
}

} // namespace neuroevolution::genetic_algorithm::pool_device
