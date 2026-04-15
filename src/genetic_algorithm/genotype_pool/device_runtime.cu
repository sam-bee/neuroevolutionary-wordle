#include "genetic_algorithm/genotype_pool/device_runtime.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "genetic_algorithm/device/genome_ops.cuh"
#include "genetic_algorithm/device/injection_ops.cuh"
#include "genetic_algorithm/genotype_pool/reference_counter.hpp"
#include "genetic_algorithm/genotype_pool/repacking.hpp"

namespace neuroevolution::genetic_algorithm::genotype_pool::device {

namespace {

using device_genome_ops::BreedAndMutateGenome;
using device_genome_ops::DeviceRandomState;
using device_genome_ops::MakeDeviceRandomState;
using device_injection_ops::DeviceOutputEmbeddingInjectionStatusCode;
using device_injection_ops::TryInjectExpandedOutputEmbeddingTails;

constexpr std::size_t kPoolRuntimeThreadBlockSize = 1;

NEUROEVOLUTION_HOST_DEVICE constexpr int DeviceStatusValue(const DevicePoolRuntimeStatusCode status_code) {
    return static_cast<int>(status_code);
}

inline bool CheckCuda(const cudaError_t error) { return error == cudaSuccess; }

inline bool IsGenerationCompatibleWithPool(const PoolGeneration &generation,
                                           const DevicePoolRuntimeBuffers &buffers) noexcept {
    if (!IsValidPoolGeneration(generation) || (generation.active_individual_count > buffers.max_generation_size) ||
        !IsValidGenotypePoolLayout(buffers.pool_layout)) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        const std::uint32_t slot_index = generation.slot_indices[individual_index];
        if ((slot_index != kInvalidPoolSlotIndex) && (slot_index >= buffers.pool_layout.slot_count)) {
            return false;
        }
    }

    return true;
}

inline bool ReadDeviceStatus(const DevicePoolRuntimeBuffers &buffers, int &status_value) {
    return CheckCuda(cudaMemcpy(&status_value, buffers.status, sizeof(int), cudaMemcpyDeviceToHost));
}

inline bool WriteDeviceStatus(const DevicePoolRuntimeBuffers &buffers, const DevicePoolRuntimeStatusCode status_code) {
    const int status_value = DeviceStatusValue(status_code);
    return CheckCuda(cudaMemcpy(buffers.status, &status_value, sizeof(int), cudaMemcpyHostToDevice));
}

inline bool ResetDeviceStatus(const DevicePoolRuntimeBuffers &buffers) {
    return WriteDeviceStatus(buffers, DevicePoolRuntimeStatusCode::kOk);
}

inline bool ClearGenerationBuffers(std::uint32_t *slot_indices, float *fitness, std::uint32_t *evaluation_counts,
                                   std::uint8_t *has_fitness, const std::size_t element_count) {
    bool ok = true;
    ok &= CheckCuda(cudaMemset(slot_indices, 0xFF, element_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(fitness, 0, element_count * sizeof(float)));
    ok &= CheckCuda(cudaMemset(evaluation_counts, 0, element_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(has_fitness, 0, element_count * sizeof(std::uint8_t)));
    return ok;
}

inline NEUROEVOLUTION_HOST_DEVICE PoolGenerationView MakeDeviceGenerationView(
    const std::size_t generation_index, const std::size_t generation_size, std::uint32_t *slot_indices, float *fitness,
    std::uint32_t *evaluation_counts, std::uint8_t *has_fitness) {
    return {
        .generation_index = generation_index,
        .active_individual_count = generation_size,
        .slot_indices = slot_indices,
        .fitness = fitness,
        .evaluation_counts = evaluation_counts,
        .has_fitness = has_fitness,
    };
}

__device__ void SetFailureStatus(int *status, const DevicePoolRuntimeStatusCode status_code) {
    atomicCAS(status, DeviceStatusValue(DevicePoolRuntimeStatusCode::kOk), DeviceStatusValue(status_code));
}

NEUROEVOLUTION_HOST_DEVICE constexpr DevicePoolRuntimeStatusCode
MapInjectionStatus(const DeviceOutputEmbeddingInjectionStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceOutputEmbeddingInjectionStatusCode::kOk:
        return DevicePoolRuntimeStatusCode::kOk;
    case DeviceOutputEmbeddingInjectionStatusCode::kInvalidTrainingShard:
        return DevicePoolRuntimeStatusCode::kOutputEmbeddingInjectionFailed;
    case DeviceOutputEmbeddingInjectionStatusCode::kInjectionFailed:
        return DevicePoolRuntimeStatusCode::kOutputEmbeddingInjectionFailed;
    }

    return DevicePoolRuntimeStatusCode::kOutputEmbeddingInjectionFailed;
}

__global__ void AssembleNextGenerationKernel(
    std::uint8_t *pool_storage, PoolSlotState *slot_states, std::uint32_t *free_slot_stack,
    std::size_t *free_slot_count, const GenotypePoolLayout pool_layout, std::uint32_t *current_slot_indices,
    float *current_fitness, std::uint32_t *current_evaluation_counts, std::uint8_t *current_has_fitness,
    const std::size_t current_generation_index, const std::size_t current_generation_size,
    std::uint32_t *next_slot_indices, float *next_fitness, std::uint32_t *next_evaluation_counts,
    std::uint8_t *next_has_fitness, const std::size_t next_generation_index, const std::size_t next_generation_size,
    const PoolParentPair *parent_pairs, std::uint32_t *parent_reference_counts, const std::uint32_t generation_seed,
    const PoolDeviceAssemblyConfig config, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    GenotypePoolView pool{
        .layout = pool_layout,
        .storage = pool_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
    };
    PoolGenerationView current_generation =
        MakeDeviceGenerationView(current_generation_index, current_generation_size, current_slot_indices,
                                 current_fitness, current_evaluation_counts, current_has_fitness);
    PoolGenerationView next_generation =
        MakeDeviceGenerationView(next_generation_index, next_generation_size, next_slot_indices, next_fitness,
                                 next_evaluation_counts, next_has_fitness);

    if (!IsValidGenotypePoolView(pool)) {
        SetFailureStatus(status, DevicePoolRuntimeStatusCode::kInvalidPool);
        return;
    }

    if (!IsValidPoolGenerationView(current_generation) || !IsValidPoolGenerationView(next_generation)) {
        SetFailureStatus(status, DevicePoolRuntimeStatusCode::kInvalidGeneration);
        return;
    }

    if ((parent_pairs == nullptr) || (parent_reference_counts == nullptr) || (next_generation_size == 0)) {
        SetFailureStatus(status, DevicePoolRuntimeStatusCode::kInvalidAssemblyPlan);
        return;
    }

    if (!IsValidPoolDeviceAssemblyConfig(config)) {
        SetFailureStatus(status, DevicePoolRuntimeStatusCode::kInvalidAssemblyConfig);
        return;
    }

    const std::size_t parent_action_count =
        (config.parent_action_count == 0) ? pool.layout.action_count : config.parent_action_count;
    if ((parent_action_count == 0) || (parent_action_count > pool.layout.action_count)) {
        SetFailureStatus(status, DevicePoolRuntimeStatusCode::kInvalidAssemblyConfig);
        return;
    }

    for (std::size_t child_index = 0; child_index < next_generation_size; ++child_index) {
        next_generation.slot_indices[child_index] = kInvalidPoolSlotIndex;
        next_generation.fitness[child_index] = 0.0f;
        next_generation.evaluation_counts[child_index] = 0;
        next_generation.has_fitness[child_index] = 0;
    }

    if (!TryBuildParentReferenceCounts(current_generation, parent_pairs, next_generation_size,
                                       parent_reference_counts)) {
        SetFailureStatus(status, DevicePoolRuntimeStatusCode::kInvalidParentIndex);
        return;
    }

    if (!TryCollectZeroReferenceParents(pool, current_generation, parent_reference_counts)) {
        SetFailureStatus(status, DevicePoolRuntimeStatusCode::kInvalidPool);
        return;
    }

    for (std::size_t child_index = 0; child_index < next_generation_size; ++child_index) {
        const PoolParentPair &parent_pair = parent_pairs[child_index];
        const std::uint32_t first_parent_slot = current_generation.slot_indices[parent_pair.first_parent_index];
        const std::uint32_t second_parent_slot = current_generation.slot_indices[parent_pair.second_parent_index];
        if ((first_parent_slot == kInvalidPoolSlotIndex) || (second_parent_slot == kInvalidPoolSlotIndex)) {
            SetFailureStatus(status, DevicePoolRuntimeStatusCode::kInvalidParentIndex);
            return;
        }

        std::uint32_t child_slot = kInvalidPoolSlotIndex;
        if (!TryAllocatePoolSlot(pool, child_slot)) {
            SetFailureStatus(status, DevicePoolRuntimeStatusCode::kPoolFull);
            return;
        }

        DeviceRandomState random_state = MakeDeviceRandomState(
            generation_seed, static_cast<std::uint32_t>(child_index + (current_generation_index * 4099U)));
        BreedAndMutateGenome(PoolSlotBytesAt(pool.storage, pool.layout, first_parent_slot),
                             PoolSlotBytesAt(pool.storage, pool.layout, second_parent_slot), parent_action_count,
                             PoolSlotBytesAt(pool.storage, pool.layout, child_slot), random_state, config.breeding,
                             config.mutation);
        if (config.pending_output_embedding_injection.enabled) {
            const DeviceOutputEmbeddingInjectionStatusCode injection_status = TryInjectExpandedOutputEmbeddingTails(
                PoolSlotBytesAt(pool.storage, pool.layout, child_slot), parent_action_count,
                config.pending_output_embedding_injection.first_catalog_word_index,
                config.pending_output_embedding_injection.injection_count);
            if (injection_status != DeviceOutputEmbeddingInjectionStatusCode::kOk) {
                SetFailureStatus(status, MapInjectionStatus(injection_status));
                return;
            }
        }

        next_generation.slot_indices[child_index] = child_slot;

        if (!TryReleaseParentReference(pool, current_generation, parent_reference_counts,
                                       parent_pair.first_parent_index) ||
            !TryReleaseParentReference(pool, current_generation, parent_reference_counts,
                                       parent_pair.second_parent_index)) {
            SetFailureStatus(status, DevicePoolRuntimeStatusCode::kInvalidPool);
            return;
        }
    }
}

__global__ void PreparePoolForExpandedActionCountKernel(
    const GenotypePoolLayout current_layout, const GenotypePoolLayout next_layout, std::uint8_t *pool_storage,
    PoolSlotState *slot_states, std::uint32_t *free_slot_stack, std::size_t *free_slot_count,
    std::uint32_t *current_slot_indices, const std::size_t current_generation_size, const PoolParentPair *parent_pairs,
    const std::size_t planned_child_count, std::uint32_t *parent_reference_counts, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    if ((pool_storage == nullptr) || (slot_states == nullptr) || (free_slot_stack == nullptr) ||
        (free_slot_count == nullptr) || (current_slot_indices == nullptr) || (parent_pairs == nullptr) ||
        (parent_reference_counts == nullptr) || (planned_child_count == 0) || (current_generation_size == 0)) {
        SetFailureStatus(status, DevicePoolRuntimeStatusCode::kInvalidPool);
        return;
    }

    if (!IsValidGenotypePoolLayout(current_layout) || !IsValidGenotypePoolLayout(next_layout)) {
        SetFailureStatus(status, DevicePoolRuntimeStatusCode::kInvalidPool);
        return;
    }

    if (!TryBuildParentReferenceCounts(current_slot_indices, current_generation_size, parent_pairs, planned_child_count,
                                       parent_reference_counts)) {
        SetFailureStatus(status, DevicePoolRuntimeStatusCode::kInvalidParentIndex);
        return;
    }

    GenotypePoolLayout working_layout = current_layout;
    if (!TryCompactAndRepackPoolForExpandedActionCount(working_layout, pool_storage, slot_states, free_slot_stack,
                                                       *free_slot_count, current_slot_indices, current_generation_size,
                                                       parent_reference_counts, next_layout.action_count)) {
        SetFailureStatus(status, DevicePoolRuntimeStatusCode::kPoolRepackFailed);
    }
}

} // namespace

bool TryCreateDevicePoolRuntimeBuffers(DevicePoolRuntimeBuffers &buffers, const DevicePoolRuntimeConfig &config) {
    buffers = {};
    if (!IsValidDevicePoolRuntimeConfig(config)) {
        return false;
    }

    buffers.pool_layout.action_count = config.action_count;
    buffers.pool_layout.slot_stride_bytes = ComputePoolSlotStrideBytes(config.action_count);
    buffers.pool_layout.slot_count = config.slot_count;
    buffers.pool_layout.pool_bytes = buffers.pool_layout.slot_stride_bytes * buffers.pool_layout.slot_count;
    buffers.max_generation_size = config.max_generation_size;
    if (!IsValidGenotypePoolLayout(buffers.pool_layout)) {
        buffers = {};
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMalloc(&buffers.pool_storage, buffers.pool_layout.pool_bytes));
    ok &= CheckCuda(cudaMalloc(&buffers.slot_states, buffers.pool_layout.slot_count * sizeof(PoolSlotState)));
    ok &= CheckCuda(cudaMalloc(&buffers.free_slot_stack, buffers.pool_layout.slot_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.free_slot_count, sizeof(std::size_t)));

    ok &= CheckCuda(cudaMalloc(&buffers.current_slot_indices, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.current_fitness, buffers.max_generation_size * sizeof(float)));
    ok &=
        CheckCuda(cudaMalloc(&buffers.current_evaluation_counts, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.current_has_fitness, buffers.max_generation_size * sizeof(std::uint8_t)));

    ok &= CheckCuda(cudaMalloc(&buffers.next_slot_indices, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.next_fitness, buffers.max_generation_size * sizeof(float)));
    ok &= CheckCuda(cudaMalloc(&buffers.next_evaluation_counts, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.next_has_fitness, buffers.max_generation_size * sizeof(std::uint8_t)));

    ok &= CheckCuda(cudaMalloc(&buffers.assembly_parent_pairs, buffers.max_generation_size * sizeof(PoolParentPair)));
    ok &= CheckCuda(cudaMalloc(&buffers.parent_reference_counts, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.status, sizeof(int)));
    if (!ok) {
        DestroyDevicePoolRuntimeBuffers(buffers);
        return false;
    }

    ok &= CheckCuda(cudaMemset(buffers.pool_storage, 0, buffers.pool_layout.pool_bytes));
    ok &= CheckCuda(cudaMemset(buffers.slot_states, 0, buffers.pool_layout.slot_count * sizeof(PoolSlotState)));
    ok &= CheckCuda(cudaMemset(buffers.free_slot_stack, 0, buffers.pool_layout.slot_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.free_slot_count, 0, sizeof(std::size_t)));
    ok &=
        ClearGenerationBuffers(buffers.current_slot_indices, buffers.current_fitness, buffers.current_evaluation_counts,
                               buffers.current_has_fitness, buffers.max_generation_size);
    ok &= ClearGenerationBuffers(buffers.next_slot_indices, buffers.next_fitness, buffers.next_evaluation_counts,
                                 buffers.next_has_fitness, buffers.max_generation_size);
    ok &= CheckCuda(cudaMemset(buffers.assembly_parent_pairs, 0, buffers.max_generation_size * sizeof(PoolParentPair)));
    ok &=
        CheckCuda(cudaMemset(buffers.parent_reference_counts, 0, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    if (!ok) {
        DestroyDevicePoolRuntimeBuffers(buffers);
        return false;
    }

    return true;
}

void DestroyDevicePoolRuntimeBuffers(DevicePoolRuntimeBuffers &buffers) noexcept {
    cudaFree(buffers.pool_storage);
    cudaFree(buffers.slot_states);
    cudaFree(buffers.free_slot_stack);
    cudaFree(buffers.free_slot_count);
    cudaFree(buffers.current_slot_indices);
    cudaFree(buffers.current_fitness);
    cudaFree(buffers.current_evaluation_counts);
    cudaFree(buffers.current_has_fitness);
    cudaFree(buffers.next_slot_indices);
    cudaFree(buffers.next_fitness);
    cudaFree(buffers.next_evaluation_counts);
    cudaFree(buffers.next_has_fitness);
    cudaFree(buffers.assembly_parent_pairs);
    cudaFree(buffers.parent_reference_counts);
    cudaFree(buffers.status);
    buffers = {};
}

bool TryUploadPoolToDevice(const HostGenotypePool &host_pool, DevicePoolRuntimeBuffers &buffers) {
    if (!IsValidHostGenotypePool(host_pool) || (host_pool.layout.pool_bytes != buffers.pool_layout.pool_bytes) ||
        (host_pool.layout.action_count != buffers.pool_layout.action_count) ||
        (host_pool.layout.slot_count != buffers.pool_layout.slot_count) ||
        (host_pool.layout.slot_stride_bytes != buffers.pool_layout.slot_stride_bytes)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(
        cudaMemcpy(buffers.pool_storage, host_pool.storage.get(), host_pool.layout.pool_bytes, cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.slot_states, host_pool.slot_states.get(),
                               host_pool.layout.slot_count * sizeof(PoolSlotState), cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.free_slot_stack, host_pool.free_slot_stack.get(),
                               host_pool.layout.slot_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    ok &= CheckCuda(
        cudaMemcpy(buffers.free_slot_count, &host_pool.free_slot_count, sizeof(std::size_t), cudaMemcpyHostToDevice));
    return ok;
}

bool TryDownloadPoolFromDevice(const DevicePoolRuntimeBuffers &buffers, HostGenotypePool &host_pool) {
    if (!IsValidGenotypePoolLayout(buffers.pool_layout) || (buffers.pool_layout.slot_count == 0)) {
        return false;
    }

    if (!TryCreateHostGenotypePool(host_pool, buffers.pool_layout)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(
        cudaMemcpy(host_pool.storage.get(), buffers.pool_storage, host_pool.layout.pool_bytes, cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(host_pool.slot_states.get(), buffers.slot_states,
                               host_pool.layout.slot_count * sizeof(PoolSlotState), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(host_pool.free_slot_stack.get(), buffers.free_slot_stack,
                               host_pool.layout.slot_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(
        cudaMemcpy(&host_pool.free_slot_count, buffers.free_slot_count, sizeof(std::size_t), cudaMemcpyDeviceToHost));
    return ok;
}

bool TryUploadCurrentGenerationToDevice(const PoolGeneration &generation, DevicePoolRuntimeBuffers &buffers) {
    if (!IsGenerationCompatibleWithPool(generation, buffers)) {
        return false;
    }

    buffers.current_generation_index = generation.generation_index;
    buffers.current_generation_size = generation.active_individual_count;
    buffers.next_generation_index = 0;
    buffers.next_generation_size = 0;
    buffers.planned_child_count = 0;

    bool ok = true;
    ok &=
        ClearGenerationBuffers(buffers.current_slot_indices, buffers.current_fitness, buffers.current_evaluation_counts,
                               buffers.current_has_fitness, buffers.max_generation_size);
    ok &= ClearGenerationBuffers(buffers.next_slot_indices, buffers.next_fitness, buffers.next_evaluation_counts,
                                 buffers.next_has_fitness, buffers.max_generation_size);
    ok &= CheckCuda(cudaMemcpy(buffers.current_slot_indices, generation.slot_indices.get(),
                               generation.active_individual_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.current_fitness, generation.fitness.get(),
                               generation.active_individual_count * sizeof(float), cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.current_evaluation_counts, generation.evaluation_counts.get(),
                               generation.active_individual_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.current_has_fitness, generation.has_fitness.get(),
                               generation.active_individual_count * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
    return ok;
}

bool TryDownloadCurrentGenerationFromDevice(const DevicePoolRuntimeBuffers &buffers, PoolGeneration &generation) {
    if (buffers.current_generation_size == 0) {
        return false;
    }

    if (!TryCreatePoolGeneration(generation, buffers.current_generation_size, buffers.current_generation_index)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemcpy(generation.slot_indices.get(), buffers.current_slot_indices,
                               generation.active_individual_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(generation.fitness.get(), buffers.current_fitness,
                               generation.active_individual_count * sizeof(float), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(generation.evaluation_counts.get(), buffers.current_evaluation_counts,
                               generation.active_individual_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(generation.has_fitness.get(), buffers.current_has_fitness,
                               generation.active_individual_count * sizeof(std::uint8_t), cudaMemcpyDeviceToHost));
    return ok;
}

bool TryDownloadNextGenerationFromDevice(const DevicePoolRuntimeBuffers &buffers, PoolGeneration &generation) {
    if (buffers.next_generation_size == 0) {
        return false;
    }

    if (!TryCreatePoolGeneration(generation, buffers.next_generation_size, buffers.next_generation_index)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemcpy(generation.slot_indices.get(), buffers.next_slot_indices,
                               generation.active_individual_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(generation.fitness.get(), buffers.next_fitness,
                               generation.active_individual_count * sizeof(float), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(generation.evaluation_counts.get(), buffers.next_evaluation_counts,
                               generation.active_individual_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(generation.has_fitness.get(), buffers.next_has_fitness,
                               generation.active_individual_count * sizeof(std::uint8_t), cudaMemcpyDeviceToHost));
    return ok;
}

bool TryUploadAssemblyPlanToDevice(const PoolAssemblyPlan &plan, DevicePoolRuntimeBuffers &buffers) {
    if (!IsValidPoolAssemblyPlan(plan) || (plan.child_count > buffers.max_generation_size)) {
        return false;
    }

    buffers.planned_child_count = plan.child_count;
    return CheckCuda(cudaMemcpy(buffers.assembly_parent_pairs, plan.parent_pairs.get(),
                                plan.child_count * sizeof(PoolParentPair), cudaMemcpyHostToDevice));
}

bool TryPreparePoolForExpandedActionCountOnDevice(DevicePoolRuntimeBuffers &buffers,
                                                  const std::size_t next_action_count) {
    if (!IsValidGenotypePoolLayout(buffers.pool_layout) || (buffers.current_generation_size == 0) ||
        (buffers.planned_child_count == 0) || (next_action_count <= buffers.pool_layout.action_count)) {
        (void)WriteDeviceStatus(buffers, DevicePoolRuntimeStatusCode::kInvalidAssemblyConfig);
        return false;
    }

    GenotypePoolLayout next_layout{};
    if (!TryCreateExpandedPoolLayout(buffers.pool_layout, next_action_count, next_layout)) {
        (void)WriteDeviceStatus(buffers, DevicePoolRuntimeStatusCode::kPoolRepackFailed);
        return false;
    }

    bool ok = true;
    ok &= ResetDeviceStatus(buffers);
    ok &=
        CheckCuda(cudaMemset(buffers.parent_reference_counts, 0, buffers.max_generation_size * sizeof(std::uint32_t)));
    if (!ok) {
        return false;
    }

    PreparePoolForExpandedActionCountKernel<<<1, 1>>>(
        buffers.pool_layout, next_layout, buffers.pool_storage, buffers.slot_states, buffers.free_slot_stack,
        buffers.free_slot_count, buffers.current_slot_indices, buffers.current_generation_size,
        buffers.assembly_parent_pairs, buffers.planned_child_count, buffers.parent_reference_counts, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    int status_value = DeviceStatusValue(DevicePoolRuntimeStatusCode::kCudaFailure);
    if (!ReadDeviceStatus(buffers, status_value) ||
        (status_value != DeviceStatusValue(DevicePoolRuntimeStatusCode::kOk))) {
        return false;
    }

    buffers.pool_layout = next_layout;
    return true;
}

bool TryAssembleNextGenerationOnDevice(DevicePoolRuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                       const PoolDeviceAssemblyConfig &config) {
    if (!IsValidPoolDeviceAssemblyConfig(config) || !IsValidGenotypePoolLayout(buffers.pool_layout) ||
        (buffers.current_generation_size == 0) || (buffers.planned_child_count == 0)) {
        (void)WriteDeviceStatus(buffers, DevicePoolRuntimeStatusCode::kInvalidAssemblyConfig);
        return false;
    }

    buffers.next_generation_index = buffers.current_generation_index + 1;
    buffers.next_generation_size = buffers.planned_child_count;

    bool ok = true;
    ok &= ResetDeviceStatus(buffers);
    ok &= ClearGenerationBuffers(buffers.next_slot_indices, buffers.next_fitness, buffers.next_evaluation_counts,
                                 buffers.next_has_fitness, buffers.max_generation_size);
    ok &=
        CheckCuda(cudaMemset(buffers.parent_reference_counts, 0, buffers.max_generation_size * sizeof(std::uint32_t)));
    if (!ok) {
        return false;
    }

    AssembleNextGenerationKernel<<<1, kPoolRuntimeThreadBlockSize>>>(
        buffers.pool_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
        buffers.pool_layout, buffers.current_slot_indices, buffers.current_fitness, buffers.current_evaluation_counts,
        buffers.current_has_fitness, buffers.current_generation_index, buffers.current_generation_size,
        buffers.next_slot_indices, buffers.next_fitness, buffers.next_evaluation_counts, buffers.next_has_fitness,
        buffers.next_generation_index, buffers.next_generation_size, buffers.assembly_parent_pairs,
        buffers.parent_reference_counts, generation_seed, config, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    int status_value = DeviceStatusValue(DevicePoolRuntimeStatusCode::kCudaFailure);
    return ReadDeviceStatus(buffers, status_value) &&
           (status_value == DeviceStatusValue(DevicePoolRuntimeStatusCode::kOk));
}

bool TryReadDevicePoolRuntimeStatus(const DevicePoolRuntimeBuffers &buffers, DevicePoolRuntimeStatusCode &status_code) {
    int status_value = 0;
    if (!ReadDeviceStatus(buffers, status_value)) {
        return false;
    }

    status_code = static_cast<DevicePoolRuntimeStatusCode>(status_value);
    return true;
}

const char *DevicePoolRuntimeStatusCodeString(const DevicePoolRuntimeStatusCode status_code) noexcept {
    switch (status_code) {
    case DevicePoolRuntimeStatusCode::kOk:
        return "ok";
    case DevicePoolRuntimeStatusCode::kCudaFailure:
        return "cuda failure";
    case DevicePoolRuntimeStatusCode::kInvalidRuntimeConfig:
        return "invalid device pool runtime config";
    case DevicePoolRuntimeStatusCode::kInvalidPool:
        return "invalid genotype pool state";
    case DevicePoolRuntimeStatusCode::kInvalidGeneration:
        return "invalid pool generation state";
    case DevicePoolRuntimeStatusCode::kInvalidAssemblyPlan:
        return "invalid pool assembly plan";
    case DevicePoolRuntimeStatusCode::kInvalidAssemblyConfig:
        return "invalid pool assembly config";
    case DevicePoolRuntimeStatusCode::kInvalidParentIndex:
        return "invalid parent index in pool assembly plan";
    case DevicePoolRuntimeStatusCode::kPoolFull:
        return "pool is genuinely full";
    case DevicePoolRuntimeStatusCode::kOutputEmbeddingInjectionFailed:
        return "device output-embedding injection failed";
    case DevicePoolRuntimeStatusCode::kPoolRepackFailed:
        return "pool compaction/repacking failed";
    }

    return "unknown device pool runtime status";
}

} // namespace neuroevolution::genetic_algorithm::genotype_pool::device
