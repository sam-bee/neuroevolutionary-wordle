#include "genetic_algorithm/device/buffer_runtime.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "genetic_algorithm/device/evaluation_ops.cuh"
#include "genetic_algorithm/device/genome_ops.cuh"
#include "genetic_algorithm/device/selection_ops.cuh"

namespace neuroevolution::genetic_algorithm::buffer_device {

namespace {

using device_evaluation_ops::DeviceGenomeEvaluationStatusCode;
using device_evaluation_ops::TryEvaluateGenomeFitness;
using device_genome_ops::DeviceRandomState;
using device_genome_ops::MakeDeviceRandomState;
using device_selection_ops::IsBetterFitness;
using device_selection_ops::TrySelectParentPairDevice;

constexpr std::size_t kBufferGARuntimeThreadBlockSize = 256;

NEUROEVOLUTION_HOST_DEVICE constexpr int DeviceStatusValue(const DeviceBufferGARuntimeStatusCode status_code) {
    return static_cast<int>(status_code);
}

inline bool CheckCuda(const cudaError_t error) { return error == cudaSuccess; }

inline bool ReadDeviceStatus(const DeviceBufferGARuntimeBuffers &buffers, int &status_value) {
    return CheckCuda(cudaMemcpy(&status_value, buffers.status, sizeof(int), cudaMemcpyDeviceToHost));
}

inline bool WriteDeviceStatus(const DeviceBufferGARuntimeBuffers &buffers,
                              const DeviceBufferGARuntimeStatusCode status_code) {
    const int status_value = DeviceStatusValue(status_code);
    return CheckCuda(cudaMemcpy(buffers.status, &status_value, sizeof(int), cudaMemcpyHostToDevice));
}

inline bool ResetDeviceStatus(const DeviceBufferGARuntimeBuffers &buffers) {
    return WriteDeviceStatus(buffers, DeviceBufferGARuntimeStatusCode::kOk);
}

inline bool KernelCompletedSuccessfully(const DeviceBufferGARuntimeBuffers &buffers) {
    int status_value = DeviceStatusValue(DeviceBufferGARuntimeStatusCode::kCudaFailure);
    return ReadDeviceStatus(buffers, status_value) &&
           (status_value == DeviceStatusValue(DeviceBufferGARuntimeStatusCode::kOk));
}

inline bool IsCurrentGenerationCompatible(const DeviceBufferGARuntimeBuffers &buffers) noexcept {
    return genotype_buffer::IsValidGenotypeBufferLayout(buffers.genotype_buffer.buffer_layout) &&
           (buffers.max_generation_size > 0) && (buffers.genotype_buffer.current_generation_size > 0) &&
           (buffers.genotype_buffer.current_generation_size <= buffers.genotype_buffer.max_generation_size);
}

NEUROEVOLUTION_HOST_DEVICE constexpr DeviceBufferGARuntimeStatusCode
MapEvaluationStatus(const DeviceGenomeEvaluationStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceGenomeEvaluationStatusCode::kOk:
        return DeviceBufferGARuntimeStatusCode::kOk;
    case DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard:
        return DeviceBufferGARuntimeStatusCode::kInvalidTrainingShard;
    case DeviceGenomeEvaluationStatusCode::kGuessAppendFailed:
        return DeviceBufferGARuntimeStatusCode::kGuessAppendFailed;
    case DeviceGenomeEvaluationStatusCode::kPolicyForwardFailed:
        return DeviceBufferGARuntimeStatusCode::kPolicyForwardFailed;
    case DeviceGenomeEvaluationStatusCode::kActionSelectionFailed:
        return DeviceBufferGARuntimeStatusCode::kActionSelectionFailed;
    }

    return DeviceBufferGARuntimeStatusCode::kCudaFailure;
}

inline DeviceBufferGARuntimeStatusCode
MapBufferRuntimeStatus(const genotype_buffer::device::DeviceBufferRuntimeStatusCode status_code) noexcept {
    using genotype_buffer::device::DeviceBufferRuntimeStatusCode;

    switch (status_code) {
    case DeviceBufferRuntimeStatusCode::kOk:
        return DeviceBufferGARuntimeStatusCode::kOk;
    case DeviceBufferRuntimeStatusCode::kCudaFailure:
        return DeviceBufferGARuntimeStatusCode::kCudaFailure;
    case DeviceBufferRuntimeStatusCode::kInvalidRuntimeConfig:
        return DeviceBufferGARuntimeStatusCode::kInvalidRuntimeConfig;
    case DeviceBufferRuntimeStatusCode::kInvalidBuffer:
        return DeviceBufferGARuntimeStatusCode::kInvalidBuffer;
    case DeviceBufferRuntimeStatusCode::kInvalidGeneration:
        return DeviceBufferGARuntimeStatusCode::kInvalidGeneration;
    case DeviceBufferRuntimeStatusCode::kInvalidAssemblyPlan:
        return DeviceBufferGARuntimeStatusCode::kInvalidAssemblyPlan;
    case DeviceBufferRuntimeStatusCode::kInvalidAssemblyConfig:
        return DeviceBufferGARuntimeStatusCode::kInvalidAssemblyConfig;
    case DeviceBufferRuntimeStatusCode::kInvalidParentIndex:
        return DeviceBufferGARuntimeStatusCode::kInvalidParentIndex;
    case DeviceBufferRuntimeStatusCode::kBufferFull:
        return DeviceBufferGARuntimeStatusCode::kBufferFull;
    case DeviceBufferRuntimeStatusCode::kOutputEmbeddingInjectionFailed:
        return DeviceBufferGARuntimeStatusCode::kOutputEmbeddingInjectionFailed;
    case DeviceBufferRuntimeStatusCode::kBufferRepackFailed:
        return DeviceBufferGARuntimeStatusCode::kBufferRepackFailed;
    }

    return DeviceBufferGARuntimeStatusCode::kCudaFailure;
}

__device__ void SetFailureStatus(int *status, const DeviceBufferGARuntimeStatusCode status_code) {
    atomicCAS(status, DeviceStatusValue(DeviceBufferGARuntimeStatusCode::kOk), DeviceStatusValue(status_code));
}

__global__ void EvaluateBufferGenerationFitnessKernel(
    const std::uint8_t *buffer_storage, const genotype_buffer::BufferSlotState *slot_states,
    const genotype_buffer::GenotypeBufferLayout buffer_layout, const std::uint32_t *current_slot_indices,
    const std::size_t current_generation_size, float *current_fitness, std::uint32_t *current_evaluation_counts,
    std::uint8_t *current_has_fitness, const RuntimeWordCounts runtime_word_counts, int *status) {
    const std::size_t individual_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (!genotype_buffer::IsValidGenotypeBufferLayout(buffer_layout) || (buffer_storage == nullptr) ||
        (slot_states == nullptr) || (current_slot_indices == nullptr) || (current_fitness == nullptr) ||
        (current_evaluation_counts == nullptr) || (current_has_fitness == nullptr) || (current_generation_size == 0)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceBufferGARuntimeStatusCode::kInvalidBuffer);
        }
        return;
    }

    if (individual_index >= current_generation_size) {
        return;
    }

    const std::uint32_t slot_index = current_slot_indices[individual_index];
    if ((slot_index == genotype_buffer::kInvalidBufferSlotIndex) || (slot_index >= buffer_layout.slot_count)) {
        SetFailureStatus(status, DeviceBufferGARuntimeStatusCode::kInvalidGeneration);
        return;
    }

    if (!slot_states[slot_index].occupied || (slot_states[slot_index].reference_count == 0)) {
        SetFailureStatus(status, DeviceBufferGARuntimeStatusCode::kInvalidBuffer);
        return;
    }

    float fitness = 0.0f;
    const DeviceGenomeEvaluationStatusCode evaluation_status =
        TryEvaluateGenomeFitness(genotype_buffer::BufferSlotBytesAt(buffer_storage, buffer_layout, slot_index),
                                 buffer_layout.action_count, runtime_word_counts, fitness);
    if (evaluation_status != DeviceGenomeEvaluationStatusCode::kOk) {
        SetFailureStatus(status, MapEvaluationStatus(evaluation_status));
        return;
    }

    current_fitness[individual_index] = fitness;
    ++current_evaluation_counts[individual_index];
    current_has_fitness[individual_index] = 1;
}

__global__ void SummarizeBufferGenerationKernel(const float *current_fitness, const std::uint8_t *current_has_fitness,
                                                const std::size_t current_generation_index,
                                                const std::size_t current_generation_size,
                                                const std::size_t action_count, PopulationFitnessSummary *summary,
                                                int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    if ((current_fitness == nullptr) || (current_has_fitness == nullptr) || (summary == nullptr) ||
        (current_generation_size == 0) || (action_count == 0)) {
        SetFailureStatus(status, DeviceBufferGARuntimeStatusCode::kInvalidGeneration);
        return;
    }

    bool found_best = false;
    float best_fitness = 0.0f;
    float fitness_sum = 0.0f;
    std::size_t best_index = 0;

    for (std::size_t individual_index = 0; individual_index < current_generation_size; ++individual_index) {
        if (current_has_fitness[individual_index] == 0) {
            SetFailureStatus(status, DeviceBufferGARuntimeStatusCode::kPopulationNotEvaluated);
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

__global__ void BuildBufferAssemblyPlanKernel(const float *current_fitness, const std::uint8_t *current_has_fitness,
                                              const std::size_t current_generation_size, const std::size_t child_count,
                                              const std::size_t current_generation_index,
                                              const ParentSelectionConfig config, const std::uint32_t planning_seed,
                                              genotype_buffer::BufferParentPair *parent_pairs, int *status) {
    const std::size_t child_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ((current_fitness == nullptr) || (current_has_fitness == nullptr) || (parent_pairs == nullptr) ||
        (current_generation_size == 0) || !IsValidParentSelectionConfig(config)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceBufferGARuntimeStatusCode::kInvalidAssemblyConfig);
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
        SetFailureStatus(status, DeviceBufferGARuntimeStatusCode::kParentSelectionFailed);
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

    const std::size_t slot_stride_bytes = genotype_buffer::ComputeBufferSlotStrideBytes(action_count);
    if ((slot_stride_bytes == 0) || (generation_size > (std::numeric_limits<std::size_t>::max() / slot_stride_bytes))) {
        return false;
    }

    generation_byte_budget_bytes_out = generation_size * slot_stride_bytes;
    return generation_byte_budget_bytes_out > 0;
}

inline bool TryPlanNextGenerationShape(const DeviceBufferGARuntimeBuffers &buffers,
                                       const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
                                       std::size_t &next_action_count_out, std::size_t &next_generation_size_out) {
    next_action_count_out = buffers.genotype_buffer.buffer_layout.action_count;
    if (!IsValidPendingOutputEmbeddingInjection(pending_output_embedding_injection,
                                                buffers.genotype_buffer.buffer_layout.action_count)) {
        return false;
    }

    if (pending_output_embedding_injection.enabled) {
        next_action_count_out += pending_output_embedding_injection.injection_count;
    }

    next_generation_size_out = genotype_buffer::BufferSlotCountForByteBudget(
        buffers.generation_byte_budget_bytes, next_action_count_out, buffers.max_generation_size);
    return next_generation_size_out > 0;
}

} // namespace

bool TryCreateDeviceBufferGARuntimeBuffers(DeviceBufferGARuntimeBuffers &buffers,
                                           const DeviceBufferGARuntimeConfig &config) {
    buffers = {};
    if (!IsValidDeviceBufferGARuntimeConfig(config)) {
        return false;
    }

    if (!TryComputeGenerationByteBudgetBytes(config.action_count, config.max_generation_size,
                                             buffers.generation_byte_budget_bytes)) {
        return false;
    }

    genotype_buffer::device::DeviceBufferRuntimeConfig buffer_config{};
    buffer_config.slot_count = config.slot_count;
    buffer_config.action_count = config.action_count;
    buffer_config.max_generation_size = config.max_generation_size;
    if (!genotype_buffer::device::TryCreateDeviceBufferRuntimeBuffers(buffers.genotype_buffer, buffer_config)) {
        return false;
    }

    buffers.max_generation_size = config.max_generation_size;
    bool ok = true;
    ok &= CheckCuda(cudaMalloc(&buffers.summary, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMalloc(&buffers.status, sizeof(int)));
    if (!ok) {
        DestroyDeviceBufferGARuntimeBuffers(buffers);
        return false;
    }

    ok &= CheckCuda(cudaMemset(buffers.summary, 0, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    if (!ok) {
        DestroyDeviceBufferGARuntimeBuffers(buffers);
        return false;
    }

    return true;
}

void DestroyDeviceBufferGARuntimeBuffers(DeviceBufferGARuntimeBuffers &buffers) noexcept {
    genotype_buffer::device::DestroyDeviceBufferRuntimeBuffers(buffers.genotype_buffer);
    cudaFree(buffers.summary);
    cudaFree(buffers.status);
    buffers = {};
}

bool TryUploadCurrentBufferPopulationToDevice(const genotype_buffer::HostGenotypeBuffer &host_buffer,
                                              const genotype_buffer::BufferGeneration &current_generation,
                                              DeviceBufferGARuntimeBuffers &buffers) {
    if ((current_generation.active_individual_count > buffers.max_generation_size) ||
        !genotype_buffer::device::TryUploadBufferToDevice(host_buffer, buffers.genotype_buffer) ||
        !genotype_buffer::device::TryUploadCurrentGenerationToDevice(current_generation, buffers.genotype_buffer)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemset(buffers.summary, 0, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    return ok;
}

bool TryDownloadBufferFromDevice(const DeviceBufferGARuntimeBuffers &buffers,
                                 genotype_buffer::HostGenotypeBuffer &host_buffer) {
    return genotype_buffer::device::TryDownloadBufferFromDevice(buffers.genotype_buffer, host_buffer);
}

bool TryDownloadCurrentGenerationFromDevice(const DeviceBufferGARuntimeBuffers &buffers,
                                            genotype_buffer::BufferGeneration &generation) {
    return genotype_buffer::device::TryDownloadCurrentGenerationFromDevice(buffers.genotype_buffer, generation);
}

bool TryDownloadNextGenerationFromDevice(const DeviceBufferGARuntimeBuffers &buffers,
                                         genotype_buffer::BufferGeneration &generation) {
    return genotype_buffer::device::TryDownloadNextGenerationFromDevice(buffers.genotype_buffer, generation);
}

bool TryEvaluateCurrentGenerationFitnessOnDevice(DeviceBufferGARuntimeBuffers &buffers,
                                                 const RuntimeWordCounts &runtime_word_counts) {
    if (!IsCurrentGenerationCompatible(buffers) || !ResetDeviceStatus(buffers)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemset(buffers.genotype_buffer.current_fitness, 0,
                               buffers.genotype_buffer.max_generation_size * sizeof(float)));
    ok &= CheckCuda(cudaMemset(buffers.genotype_buffer.current_evaluation_counts, 0,
                               buffers.genotype_buffer.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.genotype_buffer.current_has_fitness, 0,
                               buffers.genotype_buffer.max_generation_size * sizeof(std::uint8_t)));
    if (!ok) {
        return false;
    }

    const std::size_t block_count =
        (buffers.genotype_buffer.current_generation_size + kBufferGARuntimeThreadBlockSize - 1) /
        kBufferGARuntimeThreadBlockSize;
    EvaluateBufferGenerationFitnessKernel<<<block_count, kBufferGARuntimeThreadBlockSize>>>(
        buffers.genotype_buffer.buffer_storage, buffers.genotype_buffer.slot_states,
        buffers.genotype_buffer.buffer_layout, buffers.genotype_buffer.current_slot_indices,
        buffers.genotype_buffer.current_generation_size, buffers.genotype_buffer.current_fitness,
        buffers.genotype_buffer.current_evaluation_counts, buffers.genotype_buffer.current_has_fitness,
        runtime_word_counts, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize()) ||
        !KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    SummarizeBufferGenerationKernel<<<1, 1>>>(
        buffers.genotype_buffer.current_fitness, buffers.genotype_buffer.current_has_fitness,
        buffers.genotype_buffer.current_generation_index, buffers.genotype_buffer.current_generation_size,
        buffers.genotype_buffer.buffer_layout.action_count, buffers.summary, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    return KernelCompletedSuccessfully(buffers);
}

bool TryReadPopulationFitnessSummaryFromDevice(const DeviceBufferGARuntimeBuffers &buffers,
                                               PopulationFitnessSummary &summary) {
    return CheckCuda(cudaMemcpy(&summary, buffers.summary, sizeof(PopulationFitnessSummary), cudaMemcpyDeviceToHost));
}

bool TryAdvanceGenerationOnDevice(DeviceBufferGARuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                  const RuntimeWordCounts &runtime_word_counts, const GenerationAssemblyConfig &config,
                                  const PendingOutputEmbeddingInjection &pending_output_embedding_injection) {
    if (!IsValidGenerationAssemblyConfig(config) || !IsCurrentGenerationCompatible(buffers) ||
        !TryEvaluateCurrentGenerationFitnessOnDevice(buffers, runtime_word_counts) || !ResetDeviceStatus(buffers)) {
        if (!IsValidGenerationAssemblyConfig(config)) {
            (void)WriteDeviceStatus(buffers, DeviceBufferGARuntimeStatusCode::kInvalidAssemblyConfig);
        }
        return false;
    }

    std::size_t next_action_count = buffers.genotype_buffer.buffer_layout.action_count;
    std::size_t next_generation_size = buffers.genotype_buffer.current_generation_size;
    if (!TryPlanNextGenerationShape(buffers, pending_output_embedding_injection, next_action_count,
                                    next_generation_size)) {
        (void)WriteDeviceStatus(buffers, DeviceBufferGARuntimeStatusCode::kInvalidAssemblyConfig);
        return false;
    }

    const std::size_t block_count =
        (next_generation_size + kBufferGARuntimeThreadBlockSize - 1) / kBufferGARuntimeThreadBlockSize;
    const std::uint32_t planning_seed = generation_seed ^ 0xA341316CU;
    BuildBufferAssemblyPlanKernel<<<block_count, kBufferGARuntimeThreadBlockSize>>>(
        buffers.genotype_buffer.current_fitness, buffers.genotype_buffer.current_has_fitness,
        buffers.genotype_buffer.current_generation_size, next_generation_size,
        buffers.genotype_buffer.current_generation_index, config.parent_selection, planning_seed,
        buffers.genotype_buffer.assembly_parent_pairs, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize()) ||
        !KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    buffers.genotype_buffer.planned_child_count = next_generation_size;
    if (!genotype_buffer::device::TryPrioritizeAssemblyPlanForParentReleaseOnDevice(buffers.genotype_buffer)) {
        genotype_buffer::device::DeviceBufferRuntimeStatusCode buffer_status =
            genotype_buffer::device::DeviceBufferRuntimeStatusCode::kCudaFailure;
        if (genotype_buffer::device::TryReadDeviceBufferRuntimeStatus(buffers.genotype_buffer, buffer_status)) {
            (void)WriteDeviceStatus(buffers, MapBufferRuntimeStatus(buffer_status));
        } else {
            (void)WriteDeviceStatus(buffers, DeviceBufferGARuntimeStatusCode::kCudaFailure);
        }
        return false;
    }

    const std::size_t parent_action_count = buffers.genotype_buffer.buffer_layout.action_count;
    if (pending_output_embedding_injection.enabled &&
        !genotype_buffer::device::TryPrepareBufferForExpandedActionCountOnDevice(buffers.genotype_buffer,
                                                                                 next_action_count)) {
        genotype_buffer::device::DeviceBufferRuntimeStatusCode buffer_status =
            genotype_buffer::device::DeviceBufferRuntimeStatusCode::kCudaFailure;
        if (genotype_buffer::device::TryReadDeviceBufferRuntimeStatus(buffers.genotype_buffer, buffer_status)) {
            (void)WriteDeviceStatus(buffers, MapBufferRuntimeStatus(buffer_status));
        } else {
            (void)WriteDeviceStatus(buffers, DeviceBufferGARuntimeStatusCode::kCudaFailure);
        }
        return false;
    }

    genotype_buffer::device::BufferDeviceAssemblyConfig buffer_assembly_config{};
    buffer_assembly_config.breeding = config.breeding;
    buffer_assembly_config.mutation = config.mutation;
    buffer_assembly_config.parent_action_count = parent_action_count;
    buffer_assembly_config.pending_output_embedding_injection = pending_output_embedding_injection;
    if (!genotype_buffer::device::TryAssembleNextGenerationOnDevice(buffers.genotype_buffer, generation_seed,
                                                                    buffer_assembly_config)) {
        genotype_buffer::device::DeviceBufferRuntimeStatusCode buffer_status =
            genotype_buffer::device::DeviceBufferRuntimeStatusCode::kCudaFailure;
        if (genotype_buffer::device::TryReadDeviceBufferRuntimeStatus(buffers.genotype_buffer, buffer_status)) {
            (void)WriteDeviceStatus(buffers, MapBufferRuntimeStatus(buffer_status));
        } else {
            (void)WriteDeviceStatus(buffers, DeviceBufferGARuntimeStatusCode::kCudaFailure);
        }
        return false;
    }

    SwapDeviceBufferGenerationBuffers(buffers);
    return true;
}

void SwapDeviceBufferGenerationBuffers(DeviceBufferGARuntimeBuffers &buffers) noexcept {
    std::swap(buffers.genotype_buffer.current_slot_indices, buffers.genotype_buffer.next_slot_indices);
    std::swap(buffers.genotype_buffer.current_fitness, buffers.genotype_buffer.next_fitness);
    std::swap(buffers.genotype_buffer.current_evaluation_counts, buffers.genotype_buffer.next_evaluation_counts);
    std::swap(buffers.genotype_buffer.current_has_fitness, buffers.genotype_buffer.next_has_fitness);
    std::swap(buffers.genotype_buffer.current_generation_index, buffers.genotype_buffer.next_generation_index);
    std::swap(buffers.genotype_buffer.current_generation_size, buffers.genotype_buffer.next_generation_size);
}

bool TryReadDeviceBufferGARuntimeStatus(const DeviceBufferGARuntimeBuffers &buffers,
                                        DeviceBufferGARuntimeStatusCode &status_code) {
    int status_value = 0;
    if (!ReadDeviceStatus(buffers, status_value)) {
        return false;
    }

    status_code = static_cast<DeviceBufferGARuntimeStatusCode>(status_value);
    return true;
}

const char *DeviceBufferGARuntimeStatusCodeString(const DeviceBufferGARuntimeStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceBufferGARuntimeStatusCode::kOk:
        return "ok";
    case DeviceBufferGARuntimeStatusCode::kCudaFailure:
        return "cuda failure";
    case DeviceBufferGARuntimeStatusCode::kInvalidRuntimeConfig:
        return "invalid buffer ga runtime config";
    case DeviceBufferGARuntimeStatusCode::kInvalidBuffer:
        return "invalid buffer state";
    case DeviceBufferGARuntimeStatusCode::kInvalidGeneration:
        return "invalid generation state";
    case DeviceBufferGARuntimeStatusCode::kInvalidTrainingShard:
        return "invalid training-data shard";
    case DeviceBufferGARuntimeStatusCode::kGuessAppendFailed:
        return "could not append a model-selected guess to the Wordle grid";
    case DeviceBufferGARuntimeStatusCode::kPolicyForwardFailed:
        return "policy model forward pass failed";
    case DeviceBufferGARuntimeStatusCode::kActionSelectionFailed:
        return "output-embedding action selection failed";
    case DeviceBufferGARuntimeStatusCode::kPopulationNotEvaluated:
        return "population fitness summary requested before the population was fully evaluated";
    case DeviceBufferGARuntimeStatusCode::kInvalidAssemblyConfig:
        return "invalid next-generation assembly config";
    case DeviceBufferGARuntimeStatusCode::kParentSelectionFailed:
        return "device parent selection failed";
    case DeviceBufferGARuntimeStatusCode::kInvalidAssemblyPlan:
        return "invalid buffer assembly plan";
    case DeviceBufferGARuntimeStatusCode::kInvalidParentIndex:
        return "invalid parent index in buffer assembly plan";
    case DeviceBufferGARuntimeStatusCode::kBufferFull:
        return "buffer is genuinely full";
    case DeviceBufferGARuntimeStatusCode::kOutputEmbeddingInjectionFailed:
        return "device output-embedding injection failed";
    case DeviceBufferGARuntimeStatusCode::kBufferRepackFailed:
        return "buffer compaction/repacking failed";
    }

    return "unknown buffer ga runtime status";
}

} // namespace neuroevolution::genetic_algorithm::buffer_device
