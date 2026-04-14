#include "genetic_algorithm/device/arena_runtime.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "genetic_algorithm/device/evaluation_ops.cuh"
#include "genetic_algorithm/device/genome_ops.cuh"
#include "genetic_algorithm/device/selection_ops.cuh"

namespace neuroevolution::genetic_algorithm::arena_device {

namespace {

using device_evaluation_ops::DeviceGenomeEvaluationStatusCode;
using device_evaluation_ops::TryEvaluateGenomeFitness;
using device_genome_ops::DeviceRandomState;
using device_genome_ops::MakeDeviceRandomState;
using device_selection_ops::IsBetterFitness;
using device_selection_ops::TrySelectParentPairDevice;

constexpr std::size_t kArenaGARuntimeThreadBlockSize = 256;

NEUROEVOLUTION_HOST_DEVICE constexpr int DeviceStatusValue(const DeviceArenaGARuntimeStatusCode status_code) {
    return static_cast<int>(status_code);
}

inline bool CheckCuda(const cudaError_t error) { return error == cudaSuccess; }

inline bool ReadDeviceStatus(const DeviceArenaGARuntimeBuffers &buffers, int &status_value) {
    return CheckCuda(cudaMemcpy(&status_value, buffers.status, sizeof(int), cudaMemcpyDeviceToHost));
}

inline bool WriteDeviceStatus(const DeviceArenaGARuntimeBuffers &buffers,
                              const DeviceArenaGARuntimeStatusCode status_code) {
    const int status_value = DeviceStatusValue(status_code);
    return CheckCuda(cudaMemcpy(buffers.status, &status_value, sizeof(int), cudaMemcpyHostToDevice));
}

inline bool ResetDeviceStatus(const DeviceArenaGARuntimeBuffers &buffers) {
    return WriteDeviceStatus(buffers, DeviceArenaGARuntimeStatusCode::kOk);
}

inline bool KernelCompletedSuccessfully(const DeviceArenaGARuntimeBuffers &buffers) {
    int status_value = DeviceStatusValue(DeviceArenaGARuntimeStatusCode::kCudaFailure);
    return ReadDeviceStatus(buffers, status_value) &&
           (status_value == DeviceStatusValue(DeviceArenaGARuntimeStatusCode::kOk));
}

inline bool IsCurrentGenerationCompatible(const DeviceArenaGARuntimeBuffers &buffers) noexcept {
    return genotype_arena::IsValidGenotypeArenaLayout(buffers.arena_buffers.arena_layout) &&
           (buffers.generation_size > 0) &&
           (buffers.arena_buffers.current_generation_size == buffers.generation_size) &&
           (buffers.arena_buffers.current_generation_size <= buffers.arena_buffers.max_generation_size);
}

NEUROEVOLUTION_HOST_DEVICE constexpr DeviceArenaGARuntimeStatusCode
MapEvaluationStatus(const DeviceGenomeEvaluationStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceGenomeEvaluationStatusCode::kOk:
        return DeviceArenaGARuntimeStatusCode::kOk;
    case DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard:
        return DeviceArenaGARuntimeStatusCode::kInvalidTrainingShard;
    case DeviceGenomeEvaluationStatusCode::kGuessAppendFailed:
        return DeviceArenaGARuntimeStatusCode::kGuessAppendFailed;
    case DeviceGenomeEvaluationStatusCode::kPolicyForwardFailed:
        return DeviceArenaGARuntimeStatusCode::kPolicyForwardFailed;
    case DeviceGenomeEvaluationStatusCode::kActionSelectionFailed:
        return DeviceArenaGARuntimeStatusCode::kActionSelectionFailed;
    }

    return DeviceArenaGARuntimeStatusCode::kCudaFailure;
}

inline DeviceArenaGARuntimeStatusCode
MapArenaRuntimeStatus(const genotype_arena::device::DeviceArenaRuntimeStatusCode status_code) noexcept {
    using genotype_arena::device::DeviceArenaRuntimeStatusCode;

    switch (status_code) {
    case DeviceArenaRuntimeStatusCode::kOk:
        return DeviceArenaGARuntimeStatusCode::kOk;
    case DeviceArenaRuntimeStatusCode::kCudaFailure:
        return DeviceArenaGARuntimeStatusCode::kCudaFailure;
    case DeviceArenaRuntimeStatusCode::kInvalidRuntimeConfig:
        return DeviceArenaGARuntimeStatusCode::kInvalidRuntimeConfig;
    case DeviceArenaRuntimeStatusCode::kInvalidArena:
        return DeviceArenaGARuntimeStatusCode::kInvalidArena;
    case DeviceArenaRuntimeStatusCode::kInvalidGeneration:
        return DeviceArenaGARuntimeStatusCode::kInvalidGeneration;
    case DeviceArenaRuntimeStatusCode::kInvalidAssemblyPlan:
        return DeviceArenaGARuntimeStatusCode::kInvalidAssemblyPlan;
    case DeviceArenaRuntimeStatusCode::kInvalidAssemblyConfig:
        return DeviceArenaGARuntimeStatusCode::kInvalidAssemblyConfig;
    case DeviceArenaRuntimeStatusCode::kInvalidParentIndex:
        return DeviceArenaGARuntimeStatusCode::kInvalidParentIndex;
    case DeviceArenaRuntimeStatusCode::kArenaFull:
        return DeviceArenaGARuntimeStatusCode::kArenaFull;
    }

    return DeviceArenaGARuntimeStatusCode::kCudaFailure;
}

__device__ void SetFailureStatus(int *status, const DeviceArenaGARuntimeStatusCode status_code) {
    atomicCAS(status, DeviceStatusValue(DeviceArenaGARuntimeStatusCode::kOk), DeviceStatusValue(status_code));
}

__global__ void EvaluateArenaGenerationFitnessKernel(
    const std::uint8_t *arena_storage, const genotype_arena::ArenaSlotState *slot_states,
    const genotype_arena::GenotypeArenaLayout arena_layout, const std::uint32_t *current_slot_indices,
    const std::size_t current_generation_size, float *current_fitness, std::uint32_t *current_evaluation_counts,
    std::uint8_t *current_has_fitness, const RuntimeWordCounts runtime_word_counts, int *status) {
    const std::size_t individual_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (!genotype_arena::IsValidGenotypeArenaLayout(arena_layout) || (arena_storage == nullptr) ||
        (slot_states == nullptr) || (current_slot_indices == nullptr) || (current_fitness == nullptr) ||
        (current_evaluation_counts == nullptr) || (current_has_fitness == nullptr) || (current_generation_size == 0)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceArenaGARuntimeStatusCode::kInvalidArena);
        }
        return;
    }

    if (individual_index >= current_generation_size) {
        return;
    }

    const std::uint32_t slot_index = current_slot_indices[individual_index];
    if ((slot_index == genotype_arena::kInvalidArenaSlotIndex) || (slot_index >= arena_layout.slot_count)) {
        SetFailureStatus(status, DeviceArenaGARuntimeStatusCode::kInvalidGeneration);
        return;
    }

    if (!slot_states[slot_index].occupied || (slot_states[slot_index].reference_count == 0)) {
        SetFailureStatus(status, DeviceArenaGARuntimeStatusCode::kInvalidArena);
        return;
    }

    float fitness = 0.0f;
    const DeviceGenomeEvaluationStatusCode evaluation_status =
        TryEvaluateGenomeFitness(genotype_arena::ArenaSlotBytesAt(arena_storage, arena_layout, slot_index),
                                 arena_layout.action_count, runtime_word_counts, fitness);
    if (evaluation_status != DeviceGenomeEvaluationStatusCode::kOk) {
        SetFailureStatus(status, MapEvaluationStatus(evaluation_status));
        return;
    }

    current_fitness[individual_index] = fitness;
    ++current_evaluation_counts[individual_index];
    current_has_fitness[individual_index] = 1;
}

__global__ void SummarizeArenaGenerationKernel(const float *current_fitness, const std::uint8_t *current_has_fitness,
                                               const std::size_t current_generation_index,
                                               const std::size_t current_generation_size,
                                               const std::size_t action_count, PopulationFitnessSummary *summary,
                                               int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    if ((current_fitness == nullptr) || (current_has_fitness == nullptr) || (summary == nullptr) ||
        (current_generation_size == 0) || (action_count == 0)) {
        SetFailureStatus(status, DeviceArenaGARuntimeStatusCode::kInvalidGeneration);
        return;
    }

    bool found_best = false;
    float best_fitness = 0.0f;
    float fitness_sum = 0.0f;
    std::size_t best_index = 0;

    for (std::size_t individual_index = 0; individual_index < current_generation_size; ++individual_index) {
        if (current_has_fitness[individual_index] == 0) {
            SetFailureStatus(status, DeviceArenaGARuntimeStatusCode::kPopulationNotEvaluated);
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

__global__ void BuildArenaAssemblyPlanKernel(const float *current_fitness, const std::uint8_t *current_has_fitness,
                                             const std::size_t current_generation_size,
                                             const std::size_t current_generation_index,
                                             const ParentSelectionConfig config, const std::uint32_t planning_seed,
                                             genotype_arena::ArenaParentPair *parent_pairs, int *status) {
    const std::size_t child_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ((current_fitness == nullptr) || (current_has_fitness == nullptr) || (parent_pairs == nullptr) ||
        (current_generation_size == 0) || !IsValidParentSelectionConfig(config)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceArenaGARuntimeStatusCode::kInvalidAssemblyConfig);
        }
        return;
    }

    if (child_index >= current_generation_size) {
        return;
    }

    DeviceRandomState random_state = MakeDeviceRandomState(
        planning_seed, static_cast<std::uint32_t>(child_index + (current_generation_index * 8191U)));

    ParentPair parent_pair{};
    if (!TrySelectParentPairDevice(current_fitness, current_has_fitness, current_generation_size, random_state, config,
                                   parent_pair)) {
        SetFailureStatus(status, DeviceArenaGARuntimeStatusCode::kParentSelectionFailed);
        return;
    }

    parent_pairs[child_index].first_parent_index = static_cast<std::uint32_t>(parent_pair.first_parent_index);
    parent_pairs[child_index].second_parent_index = static_cast<std::uint32_t>(parent_pair.second_parent_index);
}

} // namespace

bool TryCreateDeviceArenaGARuntimeBuffers(DeviceArenaGARuntimeBuffers &buffers,
                                          const DeviceArenaGARuntimeConfig &config) {
    buffers = {};
    if (!IsValidDeviceArenaGARuntimeConfig(config)) {
        return false;
    }

    genotype_arena::device::DeviceArenaRuntimeConfig arena_config{};
    arena_config.slot_count = config.slot_count;
    arena_config.action_count = config.action_count;
    arena_config.max_generation_size = config.generation_size;
    if (!genotype_arena::device::TryCreateDeviceArenaRuntimeBuffers(buffers.arena_buffers, arena_config)) {
        return false;
    }

    buffers.generation_size = config.generation_size;
    bool ok = true;
    ok &= CheckCuda(cudaMalloc(&buffers.summary, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMalloc(&buffers.status, sizeof(int)));
    if (!ok) {
        DestroyDeviceArenaGARuntimeBuffers(buffers);
        return false;
    }

    ok &= CheckCuda(cudaMemset(buffers.summary, 0, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    if (!ok) {
        DestroyDeviceArenaGARuntimeBuffers(buffers);
        return false;
    }

    return true;
}

void DestroyDeviceArenaGARuntimeBuffers(DeviceArenaGARuntimeBuffers &buffers) noexcept {
    genotype_arena::device::DestroyDeviceArenaRuntimeBuffers(buffers.arena_buffers);
    cudaFree(buffers.summary);
    cudaFree(buffers.status);
    buffers = {};
}

bool TryUploadCurrentArenaPopulationToDevice(const genotype_arena::HostGenotypeArena &host_arena,
                                             const genotype_arena::ArenaGeneration &current_generation,
                                             DeviceArenaGARuntimeBuffers &buffers) {
    if ((current_generation.active_individual_count != buffers.generation_size) ||
        !genotype_arena::device::TryUploadArenaToDevice(host_arena, buffers.arena_buffers) ||
        !genotype_arena::device::TryUploadCurrentGenerationToDevice(current_generation, buffers.arena_buffers)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemset(buffers.summary, 0, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    return ok;
}

bool TryDownloadArenaFromDevice(const DeviceArenaGARuntimeBuffers &buffers,
                                genotype_arena::HostGenotypeArena &host_arena) {
    return genotype_arena::device::TryDownloadArenaFromDevice(buffers.arena_buffers, host_arena);
}

bool TryDownloadCurrentGenerationFromDevice(const DeviceArenaGARuntimeBuffers &buffers,
                                            genotype_arena::ArenaGeneration &generation) {
    return genotype_arena::device::TryDownloadCurrentGenerationFromDevice(buffers.arena_buffers, generation);
}

bool TryDownloadNextGenerationFromDevice(const DeviceArenaGARuntimeBuffers &buffers,
                                         genotype_arena::ArenaGeneration &generation) {
    return genotype_arena::device::TryDownloadNextGenerationFromDevice(buffers.arena_buffers, generation);
}

bool TryEvaluateCurrentGenerationFitnessOnDevice(DeviceArenaGARuntimeBuffers &buffers,
                                                 const RuntimeWordCounts &runtime_word_counts) {
    if (!IsCurrentGenerationCompatible(buffers) || !ResetDeviceStatus(buffers)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemset(buffers.arena_buffers.current_fitness, 0,
                               buffers.arena_buffers.max_generation_size * sizeof(float)));
    ok &= CheckCuda(cudaMemset(buffers.arena_buffers.current_evaluation_counts, 0,
                               buffers.arena_buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.arena_buffers.current_has_fitness, 0,
                               buffers.arena_buffers.max_generation_size * sizeof(std::uint8_t)));
    if (!ok) {
        return false;
    }

    const std::size_t block_count =
        (buffers.arena_buffers.current_generation_size + kArenaGARuntimeThreadBlockSize - 1) /
        kArenaGARuntimeThreadBlockSize;
    EvaluateArenaGenerationFitnessKernel<<<block_count, kArenaGARuntimeThreadBlockSize>>>(
        buffers.arena_buffers.arena_storage, buffers.arena_buffers.slot_states, buffers.arena_buffers.arena_layout,
        buffers.arena_buffers.current_slot_indices, buffers.arena_buffers.current_generation_size,
        buffers.arena_buffers.current_fitness, buffers.arena_buffers.current_evaluation_counts,
        buffers.arena_buffers.current_has_fitness, runtime_word_counts, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize()) ||
        !KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    SummarizeArenaGenerationKernel<<<1, 1>>>(
        buffers.arena_buffers.current_fitness, buffers.arena_buffers.current_has_fitness,
        buffers.arena_buffers.current_generation_index, buffers.arena_buffers.current_generation_size,
        buffers.arena_buffers.arena_layout.action_count, buffers.summary, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    return KernelCompletedSuccessfully(buffers);
}

bool TryReadPopulationFitnessSummaryFromDevice(const DeviceArenaGARuntimeBuffers &buffers,
                                               PopulationFitnessSummary &summary) {
    return CheckCuda(cudaMemcpy(&summary, buffers.summary, sizeof(PopulationFitnessSummary), cudaMemcpyDeviceToHost));
}

bool TryAdvanceGenerationOnDevice(DeviceArenaGARuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                  const RuntimeWordCounts &runtime_word_counts,
                                  const GenerationAssemblyConfig &config) {
    if (!IsValidGenerationAssemblyConfig(config) || !IsCurrentGenerationCompatible(buffers) ||
        !TryEvaluateCurrentGenerationFitnessOnDevice(buffers, runtime_word_counts) || !ResetDeviceStatus(buffers)) {
        if (!IsValidGenerationAssemblyConfig(config)) {
            (void)WriteDeviceStatus(buffers, DeviceArenaGARuntimeStatusCode::kInvalidAssemblyConfig);
        }
        return false;
    }

    const std::size_t block_count =
        (buffers.arena_buffers.current_generation_size + kArenaGARuntimeThreadBlockSize - 1) /
        kArenaGARuntimeThreadBlockSize;
    const std::uint32_t planning_seed = generation_seed ^ 0xA341316CU;
    BuildArenaAssemblyPlanKernel<<<block_count, kArenaGARuntimeThreadBlockSize>>>(
        buffers.arena_buffers.current_fitness, buffers.arena_buffers.current_has_fitness,
        buffers.arena_buffers.current_generation_size, buffers.arena_buffers.current_generation_index,
        config.parent_selection, planning_seed, buffers.arena_buffers.assembly_parent_pairs, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize()) ||
        !KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    buffers.arena_buffers.planned_child_count = buffers.arena_buffers.current_generation_size;
    genotype_arena::device::ArenaDeviceAssemblyConfig arena_assembly_config{};
    arena_assembly_config.breeding = config.breeding;
    arena_assembly_config.mutation = config.mutation;
    if (!genotype_arena::device::TryAssembleNextGenerationWithoutElitismOnDevice(buffers.arena_buffers, generation_seed,
                                                                                 arena_assembly_config)) {
        genotype_arena::device::DeviceArenaRuntimeStatusCode arena_status =
            genotype_arena::device::DeviceArenaRuntimeStatusCode::kCudaFailure;
        if (genotype_arena::device::TryReadDeviceArenaRuntimeStatus(buffers.arena_buffers, arena_status)) {
            (void)WriteDeviceStatus(buffers, MapArenaRuntimeStatus(arena_status));
        } else {
            (void)WriteDeviceStatus(buffers, DeviceArenaGARuntimeStatusCode::kCudaFailure);
        }
        return false;
    }

    SwapDeviceArenaGenerationBuffers(buffers);
    return true;
}

void SwapDeviceArenaGenerationBuffers(DeviceArenaGARuntimeBuffers &buffers) noexcept {
    std::swap(buffers.arena_buffers.current_slot_indices, buffers.arena_buffers.next_slot_indices);
    std::swap(buffers.arena_buffers.current_fitness, buffers.arena_buffers.next_fitness);
    std::swap(buffers.arena_buffers.current_evaluation_counts, buffers.arena_buffers.next_evaluation_counts);
    std::swap(buffers.arena_buffers.current_has_fitness, buffers.arena_buffers.next_has_fitness);
    std::swap(buffers.arena_buffers.current_generation_index, buffers.arena_buffers.next_generation_index);
    std::swap(buffers.arena_buffers.current_generation_size, buffers.arena_buffers.next_generation_size);
}

bool TryReadDeviceArenaGARuntimeStatus(const DeviceArenaGARuntimeBuffers &buffers,
                                       DeviceArenaGARuntimeStatusCode &status_code) {
    int status_value = 0;
    if (!ReadDeviceStatus(buffers, status_value)) {
        return false;
    }

    status_code = static_cast<DeviceArenaGARuntimeStatusCode>(status_value);
    return true;
}

const char *DeviceArenaGARuntimeStatusCodeString(const DeviceArenaGARuntimeStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceArenaGARuntimeStatusCode::kOk:
        return "ok";
    case DeviceArenaGARuntimeStatusCode::kCudaFailure:
        return "cuda failure";
    case DeviceArenaGARuntimeStatusCode::kInvalidRuntimeConfig:
        return "invalid arena ga runtime config";
    case DeviceArenaGARuntimeStatusCode::kInvalidArena:
        return "invalid arena state";
    case DeviceArenaGARuntimeStatusCode::kInvalidGeneration:
        return "invalid generation state";
    case DeviceArenaGARuntimeStatusCode::kInvalidTrainingShard:
        return "invalid training-data shard";
    case DeviceArenaGARuntimeStatusCode::kGuessAppendFailed:
        return "could not append a model-selected guess to the Wordle grid";
    case DeviceArenaGARuntimeStatusCode::kPolicyForwardFailed:
        return "policy model forward pass failed";
    case DeviceArenaGARuntimeStatusCode::kActionSelectionFailed:
        return "output-embedding action selection failed";
    case DeviceArenaGARuntimeStatusCode::kPopulationNotEvaluated:
        return "population fitness summary requested before the population was fully evaluated";
    case DeviceArenaGARuntimeStatusCode::kInvalidAssemblyConfig:
        return "invalid next-generation assembly config";
    case DeviceArenaGARuntimeStatusCode::kParentSelectionFailed:
        return "device parent selection failed";
    case DeviceArenaGARuntimeStatusCode::kInvalidAssemblyPlan:
        return "invalid arena assembly plan";
    case DeviceArenaGARuntimeStatusCode::kInvalidParentIndex:
        return "invalid parent index in arena assembly plan";
    case DeviceArenaGARuntimeStatusCode::kArenaFull:
        return "arena is genuinely full";
    }

    return "unknown arena ga runtime status";
}

} // namespace neuroevolution::genetic_algorithm::arena_device
