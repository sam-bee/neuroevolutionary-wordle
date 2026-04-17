#include "genetic_algorithm/genotype_buffer/device_runtime.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "genetic_algorithm/device/genome_ops.cuh"
#include "genetic_algorithm/device/injection_ops.cuh"
#include "genetic_algorithm/genotype_buffer/reference_counter.hpp"
#include "genetic_algorithm/genotype_buffer/repacking.hpp"

namespace neuroevolution::genetic_algorithm::genotype_buffer::device {

namespace {

using device_genome_ops::BreedAndMutateGenome;
using device_genome_ops::DeviceRandomState;
using device_genome_ops::MakeDeviceRandomState;
using device_injection_ops::DeviceOutputEmbeddingInjectionStatusCode;
using device_injection_ops::TryInjectExpandedOutputEmbeddingTails;

constexpr int kBufferAssemblyThreadBlockSize = 128;
constexpr int kMaxBufferAssemblyThreadBlocks = 32;
constexpr std::size_t kMaxBufferAssemblyConcurrentChildren =
    static_cast<std::size_t>(kBufferAssemblyThreadBlockSize) * static_cast<std::size_t>(kMaxBufferAssemblyThreadBlocks);

NEUROEVOLUTION_HOST_DEVICE constexpr int DeviceStatusValue(const DeviceBufferRuntimeStatusCode status_code) {
    return static_cast<int>(status_code);
}

inline bool CheckCuda(const cudaError_t error) { return error == cudaSuccess; }

inline bool IsGenerationCompatibleWithBuffer(const BufferGeneration &generation,
                                             const DeviceBufferRuntimeBuffers &buffers) noexcept {
    if (!IsValidBufferGeneration(generation) || (generation.active_individual_count > buffers.max_generation_size) ||
        !IsValidGenotypeBufferLayout(buffers.buffer_layout)) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        const std::uint32_t slot_index = generation.slot_indices[individual_index];
        if ((slot_index != kInvalidBufferSlotIndex) && (slot_index >= buffers.buffer_layout.slot_count)) {
            return false;
        }
    }

    return true;
}

inline bool ReadDeviceStatus(const DeviceBufferRuntimeBuffers &buffers, int &status_value) {
    return CheckCuda(cudaMemcpy(&status_value, buffers.status, sizeof(int), cudaMemcpyDeviceToHost));
}

inline bool WriteDeviceStatus(const DeviceBufferRuntimeBuffers &buffers,
                              const DeviceBufferRuntimeStatusCode status_code) {
    const int status_value = DeviceStatusValue(status_code);
    return CheckCuda(cudaMemcpy(buffers.status, &status_value, sizeof(int), cudaMemcpyHostToDevice));
}

inline bool ResetDeviceStatus(const DeviceBufferRuntimeBuffers &buffers) {
    return WriteDeviceStatus(buffers, DeviceBufferRuntimeStatusCode::kOk);
}

inline bool IsDeviceStatusOk(const DeviceBufferRuntimeBuffers &buffers) {
    int status_value = DeviceStatusValue(DeviceBufferRuntimeStatusCode::kCudaFailure);
    return ReadDeviceStatus(buffers, status_value) &&
           (status_value == DeviceStatusValue(DeviceBufferRuntimeStatusCode::kOk));
}

inline int BoundedAssemblyBlockCount(const std::size_t item_count) noexcept {
    if (item_count == 0) {
        return 1;
    }

    const std::size_t block_count = (item_count + static_cast<std::size_t>(kBufferAssemblyThreadBlockSize) - 1) /
                                    static_cast<std::size_t>(kBufferAssemblyThreadBlockSize);
    return static_cast<int>((block_count < static_cast<std::size_t>(kMaxBufferAssemblyThreadBlocks))
                                ? block_count
                                : static_cast<std::size_t>(kMaxBufferAssemblyThreadBlocks));
}

inline bool ReadDeviceFreeSlotCount(const DeviceBufferRuntimeBuffers &buffers, std::uint32_t &free_slot_count) {
    free_slot_count = 0;
    return CheckCuda(
        cudaMemcpy(&free_slot_count, buffers.free_slot_count, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
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

inline NEUROEVOLUTION_HOST_DEVICE BufferGenerationView MakeDeviceGenerationView(
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

__device__ void SetFailureStatus(int *status, const DeviceBufferRuntimeStatusCode status_code) {
    if (status == nullptr) {
        return;
    }

    atomicCAS(status, DeviceStatusValue(DeviceBufferRuntimeStatusCode::kOk), DeviceStatusValue(status_code));
}

__device__ bool IsDeviceStatusOk(int *status) {
    return (status != nullptr) && (atomicCAS(status, DeviceStatusValue(DeviceBufferRuntimeStatusCode::kOk),
                                             DeviceStatusValue(DeviceBufferRuntimeStatusCode::kOk)) ==
                                   DeviceStatusValue(DeviceBufferRuntimeStatusCode::kOk));
}

NEUROEVOLUTION_HOST_DEVICE constexpr DeviceBufferRuntimeStatusCode
MapInjectionStatus(const DeviceOutputEmbeddingInjectionStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceOutputEmbeddingInjectionStatusCode::kOk:
        return DeviceBufferRuntimeStatusCode::kOk;
    case DeviceOutputEmbeddingInjectionStatusCode::kInvalidTrainingShard:
        return DeviceBufferRuntimeStatusCode::kOutputEmbeddingInjectionFailed;
    case DeviceOutputEmbeddingInjectionStatusCode::kInjectionFailed:
        return DeviceBufferRuntimeStatusCode::kOutputEmbeddingInjectionFailed;
    }

    return DeviceBufferRuntimeStatusCode::kOutputEmbeddingInjectionFailed;
}

__global__ void ValidateAssemblyInputsKernel(
    std::uint8_t *buffer_storage, BufferSlotState *slot_states, std::uint32_t *free_slot_stack,
    std::uint32_t *free_slot_count, std::uint32_t *free_slot_lock, const GenotypeBufferLayout buffer_layout,
    std::uint32_t *current_slot_indices, float *current_fitness, std::uint32_t *current_evaluation_counts,
    std::uint8_t *current_has_fitness, const std::size_t current_generation_index,
    const std::size_t current_generation_size, std::uint32_t *next_slot_indices, float *next_fitness,
    std::uint32_t *next_evaluation_counts, std::uint8_t *next_has_fitness, const std::size_t next_generation_index,
    const std::size_t next_generation_size, const BufferParentPair *parent_pairs,
    std::uint32_t *parent_reference_counts, const BufferDeviceAssemblyConfig config, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    GenotypeBufferView buffer{
        .layout = buffer_layout,
        .storage = buffer_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };
    BufferGenerationView current_generation =
        MakeDeviceGenerationView(current_generation_index, current_generation_size, current_slot_indices,
                                 current_fitness, current_evaluation_counts, current_has_fitness);
    BufferGenerationView next_generation =
        MakeDeviceGenerationView(next_generation_index, next_generation_size, next_slot_indices, next_fitness,
                                 next_evaluation_counts, next_has_fitness);

    if (!IsValidGenotypeBufferView(buffer)) {
        SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidBuffer);
        return;
    }

    if (!IsValidBufferGenerationView(current_generation) || !IsValidBufferGenerationView(next_generation)) {
        SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidGeneration);
        return;
    }

    if ((parent_pairs == nullptr) || (parent_reference_counts == nullptr) || (next_generation_size == 0)) {
        SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidAssemblyPlan);
        return;
    }

    if (!IsValidBufferDeviceAssemblyConfig(config)) {
        SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidAssemblyConfig);
        return;
    }

    const std::size_t parent_action_count =
        (config.parent_action_count == 0) ? buffer.layout.action_count : config.parent_action_count;
    if ((parent_action_count == 0) || (parent_action_count > buffer.layout.action_count)) {
        SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidAssemblyConfig);
        return;
    }
}

__global__ void BuildParentReferenceCountsKernel(std::uint32_t *current_slot_indices,
                                                 const std::size_t current_generation_size,
                                                 const BufferParentPair *parent_pairs, const std::size_t child_count,
                                                 std::uint32_t *parent_reference_counts, int *status) {
    if (!IsDeviceStatusOk(status)) {
        return;
    }

    const std::size_t worker_index =
        (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
    const std::size_t worker_count = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t child_index = worker_index; child_index < child_count; child_index += worker_count) {
        if (!IsDeviceStatusOk(status)) {
            return;
        }

        const BufferParentPair &parent_pair = parent_pairs[child_index];
        if (!IsValidBufferParentIndex(current_slot_indices, current_generation_size, parent_pair.first_parent_index) ||
            !IsValidBufferParentIndex(current_slot_indices, current_generation_size, parent_pair.second_parent_index)) {
            SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidParentIndex);
            return;
        }

        if (!TryIncrementParentReferenceCount(parent_reference_counts, parent_pair.first_parent_index) ||
            !TryIncrementParentReferenceCount(parent_reference_counts, parent_pair.second_parent_index)) {
            SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidParentIndex);
            return;
        }
    }
}

__global__ void CollectZeroReferenceParentsKernel(
    std::uint8_t *buffer_storage, BufferSlotState *slot_states, std::uint32_t *free_slot_stack,
    std::uint32_t *free_slot_count, std::uint32_t *free_slot_lock, const GenotypeBufferLayout buffer_layout,
    std::uint32_t *current_slot_indices, float *current_fitness, std::uint32_t *current_evaluation_counts,
    std::uint8_t *current_has_fitness, const std::size_t current_generation_index,
    const std::size_t current_generation_size, std::uint32_t *parent_reference_counts, int *status) {
    if (!IsDeviceStatusOk(status)) {
        return;
    }

    GenotypeBufferView buffer{
        .layout = buffer_layout,
        .storage = buffer_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };
    BufferGenerationView current_generation =
        MakeDeviceGenerationView(current_generation_index, current_generation_size, current_slot_indices,
                                 current_fitness, current_evaluation_counts, current_has_fitness);

    const std::size_t worker_index =
        (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
    const std::size_t worker_count = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t parent_index = worker_index; parent_index < current_generation_size;
         parent_index += worker_count) {
        if (!IsDeviceStatusOk(status)) {
            return;
        }

        if ((detail::AtomicLoadReferenceCount(&parent_reference_counts[parent_index]) != 0) ||
            (current_generation.slot_indices[parent_index] == kInvalidBufferSlotIndex)) {
            continue;
        }

        if (!TryReleaseBufferSlot(buffer, current_generation.slot_indices[parent_index])) {
            SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidBuffer);
            return;
        }

        ClearBufferGenerationSlot(current_generation, parent_index);
    }
}

__global__ void AssembleChildBatchKernel(
    std::uint8_t *buffer_storage, BufferSlotState *slot_states, std::uint32_t *free_slot_stack,
    std::uint32_t *free_slot_count, std::uint32_t *free_slot_lock, const GenotypeBufferLayout buffer_layout,
    std::uint32_t *current_slot_indices, float *current_fitness, std::uint32_t *current_evaluation_counts,
    std::uint8_t *current_has_fitness, const std::size_t current_generation_index,
    const std::size_t current_generation_size, std::uint32_t *next_slot_indices, float *next_fitness,
    std::uint32_t *next_evaluation_counts, std::uint8_t *next_has_fitness, const std::size_t next_generation_index,
    const std::size_t next_generation_size, const BufferParentPair *parent_pairs,
    std::uint32_t *parent_reference_counts, const std::size_t child_offset, const std::size_t batch_child_count,
    const std::uint32_t generation_seed, const BufferDeviceAssemblyConfig config, int *status) {
    if (!IsDeviceStatusOk(status)) {
        return;
    }

    GenotypeBufferView buffer{
        .layout = buffer_layout,
        .storage = buffer_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };
    BufferGenerationView current_generation =
        MakeDeviceGenerationView(current_generation_index, current_generation_size, current_slot_indices,
                                 current_fitness, current_evaluation_counts, current_has_fitness);
    BufferGenerationView next_generation =
        MakeDeviceGenerationView(next_generation_index, next_generation_size, next_slot_indices, next_fitness,
                                 next_evaluation_counts, next_has_fitness);

    const std::size_t parent_action_count =
        (config.parent_action_count == 0) ? buffer.layout.action_count : config.parent_action_count;
    const std::size_t worker_index =
        (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
    const std::size_t worker_count = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t batch_child_index = worker_index; batch_child_index < batch_child_count;
         batch_child_index += worker_count) {
        if (!IsDeviceStatusOk(status)) {
            return;
        }

        const std::size_t child_index = child_offset + batch_child_index;
        if (child_index >= next_generation_size) {
            SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidAssemblyPlan);
            return;
        }

        const BufferParentPair &parent_pair = parent_pairs[child_index];
        const std::uint32_t first_parent_slot = current_generation.slot_indices[parent_pair.first_parent_index];
        const std::uint32_t second_parent_slot = current_generation.slot_indices[parent_pair.second_parent_index];
        if ((first_parent_slot == kInvalidBufferSlotIndex) || (second_parent_slot == kInvalidBufferSlotIndex)) {
            SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidParentIndex);
            return;
        }

        std::uint32_t child_slot = kInvalidBufferSlotIndex;
        if (!TryAllocateBufferSlot(buffer, child_slot)) {
            SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kBufferFull);
            return;
        }

        DeviceRandomState random_state = MakeDeviceRandomState(
            generation_seed, static_cast<std::uint32_t>(child_index + (current_generation_index * 4099U)));
        BreedAndMutateGenome(BufferSlotBytesAt(buffer.storage, buffer.layout, first_parent_slot),
                             BufferSlotBytesAt(buffer.storage, buffer.layout, second_parent_slot), parent_action_count,
                             BufferSlotBytesAt(buffer.storage, buffer.layout, child_slot), random_state,
                             config.breeding, config.mutation);
        if (config.pending_output_embedding_injection.enabled) {
            const DeviceOutputEmbeddingInjectionStatusCode injection_status = TryInjectExpandedOutputEmbeddingTails(
                BufferSlotBytesAt(buffer.storage, buffer.layout, child_slot), parent_action_count,
                config.pending_output_embedding_injection.first_catalog_word_index,
                config.pending_output_embedding_injection.injection_count);
            if (injection_status != DeviceOutputEmbeddingInjectionStatusCode::kOk) {
                (void)TryReleaseBufferSlot(buffer, child_slot);
                SetFailureStatus(status, MapInjectionStatus(injection_status));
                return;
            }
        }

        next_generation.slot_indices[child_index] = child_slot;
    }
}

__global__ void
ReleaseParentReferenceBatchKernel(std::uint8_t *buffer_storage, BufferSlotState *slot_states,
                                  std::uint32_t *free_slot_stack, std::uint32_t *free_slot_count,
                                  std::uint32_t *free_slot_lock, const GenotypeBufferLayout buffer_layout,
                                  std::uint32_t *current_slot_indices, float *current_fitness,
                                  std::uint32_t *current_evaluation_counts, std::uint8_t *current_has_fitness,
                                  const std::size_t current_generation_index, const std::size_t current_generation_size,
                                  const BufferParentPair *parent_pairs, std::uint32_t *parent_reference_counts,
                                  const std::size_t child_offset, const std::size_t batch_child_count, int *status) {
    if (!IsDeviceStatusOk(status)) {
        return;
    }

    GenotypeBufferView buffer{
        .layout = buffer_layout,
        .storage = buffer_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };
    BufferGenerationView current_generation =
        MakeDeviceGenerationView(current_generation_index, current_generation_size, current_slot_indices,
                                 current_fitness, current_evaluation_counts, current_has_fitness);

    const std::size_t worker_index =
        (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
    const std::size_t worker_count = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t batch_child_index = worker_index; batch_child_index < batch_child_count;
         batch_child_index += worker_count) {
        if (!IsDeviceStatusOk(status)) {
            return;
        }

        const std::size_t child_index = child_offset + batch_child_index;
        const BufferParentPair &parent_pair = parent_pairs[child_index];
        if (!TryReleaseParentReference(buffer, current_generation, parent_reference_counts,
                                       parent_pair.first_parent_index) ||
            !TryReleaseParentReference(buffer, current_generation, parent_reference_counts,
                                       parent_pair.second_parent_index)) {
            SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidBuffer);
            return;
        }
    }
}

__global__ void CleanupNextGenerationSlotsKernel(std::uint8_t *buffer_storage, BufferSlotState *slot_states,
                                                 std::uint32_t *free_slot_stack, std::uint32_t *free_slot_count,
                                                 std::uint32_t *free_slot_lock,
                                                 const GenotypeBufferLayout buffer_layout,
                                                 std::uint32_t *next_slot_indices, float *next_fitness,
                                                 std::uint32_t *next_evaluation_counts, std::uint8_t *next_has_fitness,
                                                 const std::size_t next_generation_size, int *status) {
    GenotypeBufferView buffer{
        .layout = buffer_layout,
        .storage = buffer_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };

    const std::size_t worker_index =
        (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
    const std::size_t worker_count = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t child_index = worker_index; child_index < next_generation_size; child_index += worker_count) {
        const std::uint32_t child_slot = next_slot_indices[child_index];
        if ((child_slot != kInvalidBufferSlotIndex) && !TryReleaseBufferSlot(buffer, child_slot)) {
            SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidBuffer);
        }

        next_slot_indices[child_index] = kInvalidBufferSlotIndex;
        next_fitness[child_index] = 0.0f;
        next_evaluation_counts[child_index] = 0;
        next_has_fitness[child_index] = 0;
    }
}

inline bool CleanupFailedAssemblyOnDevice(DeviceBufferRuntimeBuffers &buffers) {
    const std::size_t cleanup_generation_size = buffers.next_generation_size;
    int original_status = DeviceStatusValue(DeviceBufferRuntimeStatusCode::kCudaFailure);
    bool ok = ReadDeviceStatus(buffers, original_status);
    ok &= ResetDeviceStatus(buffers);

    if (cleanup_generation_size > 0) {
        CleanupNextGenerationSlotsKernel<<<BoundedAssemblyBlockCount(cleanup_generation_size),
                                           kBufferAssemblyThreadBlockSize>>>(
            buffers.buffer_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
            buffers.free_slot_lock, buffers.buffer_layout, buffers.next_slot_indices, buffers.next_fitness,
            buffers.next_evaluation_counts, buffers.next_has_fitness, cleanup_generation_size, buffers.status);
        ok &= CheckCuda(cudaGetLastError());
        ok &= CheckCuda(cudaDeviceSynchronize());
    }

    ok &= ClearGenerationBuffers(buffers.next_slot_indices, buffers.next_fitness, buffers.next_evaluation_counts,
                                 buffers.next_has_fitness, buffers.max_generation_size);

    int cleanup_status = DeviceStatusValue(DeviceBufferRuntimeStatusCode::kCudaFailure);
    ok &= ReadDeviceStatus(buffers, cleanup_status);
    if (original_status != DeviceStatusValue(DeviceBufferRuntimeStatusCode::kOk)) {
        ok &= WriteDeviceStatus(buffers, static_cast<DeviceBufferRuntimeStatusCode>(original_status));
    }

    buffers.next_generation_index = 0;
    buffers.next_generation_size = 0;
    return ok && (cleanup_status == DeviceStatusValue(DeviceBufferRuntimeStatusCode::kOk));
}

inline bool FinishAssemblyKernelOrCleanup(DeviceBufferRuntimeBuffers &buffers) {
    const bool launch_ok = CheckCuda(cudaGetLastError());
    const bool sync_ok = CheckCuda(cudaDeviceSynchronize());
    if (!launch_ok || !sync_ok) {
        (void)WriteDeviceStatus(buffers, DeviceBufferRuntimeStatusCode::kCudaFailure);
        (void)CleanupFailedAssemblyOnDevice(buffers);
        return false;
    }

    if (!IsDeviceStatusOk(buffers)) {
        (void)CleanupFailedAssemblyOnDevice(buffers);
        return false;
    }

    return true;
}

__global__ void PrepareBufferForExpandedActionCountKernel(
    const GenotypeBufferLayout current_layout, const GenotypeBufferLayout next_layout, std::uint8_t *buffer_storage,
    BufferSlotState *slot_states, std::uint32_t *free_slot_stack, std::uint32_t *free_slot_count,
    std::uint32_t *free_slot_lock, std::uint32_t *current_slot_indices, const std::size_t current_generation_size,
    const BufferParentPair *parent_pairs, const std::size_t planned_child_count, std::uint32_t *parent_reference_counts,
    int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    if ((buffer_storage == nullptr) || (slot_states == nullptr) || (free_slot_stack == nullptr) ||
        (free_slot_count == nullptr) || (free_slot_lock == nullptr) || (current_slot_indices == nullptr) ||
        (parent_pairs == nullptr) || (parent_reference_counts == nullptr) || (planned_child_count == 0) ||
        (current_generation_size == 0)) {
        SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidBuffer);
        return;
    }

    if (!IsValidGenotypeBufferLayout(current_layout) || !IsValidGenotypeBufferLayout(next_layout)) {
        SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidBuffer);
        return;
    }

    if (!TryBuildParentReferenceCounts(current_slot_indices, current_generation_size, parent_pairs, planned_child_count,
                                       parent_reference_counts)) {
        SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kInvalidParentIndex);
        return;
    }

    GenotypeBufferLayout working_layout = current_layout;
    if (!TryCompactAndRepackBufferForExpandedActionCount(
            working_layout, buffer_storage, slot_states, free_slot_stack, *free_slot_count, current_slot_indices,
            current_generation_size, parent_reference_counts, next_layout.action_count, planned_child_count)) {
        SetFailureStatus(status, DeviceBufferRuntimeStatusCode::kBufferRepackFailed);
    }
}

} // namespace

bool TryCreateDeviceBufferRuntimeBuffers(DeviceBufferRuntimeBuffers &buffers, const DeviceBufferRuntimeConfig &config) {
    buffers = {};
    if (!IsValidDeviceBufferRuntimeConfig(config)) {
        return false;
    }

    buffers.buffer_layout.action_count = config.action_count;
    buffers.buffer_layout.slot_stride_bytes = ComputeBufferSlotStrideBytes(config.action_count);
    buffers.buffer_layout.slot_count = config.slot_count;
    buffers.buffer_layout.buffer_bytes = buffers.buffer_layout.slot_stride_bytes * buffers.buffer_layout.slot_count;
    buffers.max_generation_size = config.max_generation_size;
    if (!IsValidGenotypeBufferLayout(buffers.buffer_layout)) {
        buffers = {};
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMalloc(&buffers.buffer_storage, buffers.buffer_layout.buffer_bytes));
    ok &= CheckCuda(cudaMalloc(&buffers.slot_states, buffers.buffer_layout.slot_count * sizeof(BufferSlotState)));
    ok &= CheckCuda(cudaMalloc(&buffers.free_slot_stack, buffers.buffer_layout.slot_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.free_slot_count, sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.free_slot_lock, sizeof(std::uint32_t)));

    ok &= CheckCuda(cudaMalloc(&buffers.current_slot_indices, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.current_fitness, buffers.max_generation_size * sizeof(float)));
    ok &=
        CheckCuda(cudaMalloc(&buffers.current_evaluation_counts, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.current_has_fitness, buffers.max_generation_size * sizeof(std::uint8_t)));

    ok &= CheckCuda(cudaMalloc(&buffers.next_slot_indices, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.next_fitness, buffers.max_generation_size * sizeof(float)));
    ok &= CheckCuda(cudaMalloc(&buffers.next_evaluation_counts, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.next_has_fitness, buffers.max_generation_size * sizeof(std::uint8_t)));

    ok &= CheckCuda(cudaMalloc(&buffers.assembly_parent_pairs, buffers.max_generation_size * sizeof(BufferParentPair)));
    ok &= CheckCuda(cudaMalloc(&buffers.parent_reference_counts, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.status, sizeof(int)));
    if (!ok) {
        DestroyDeviceBufferRuntimeBuffers(buffers);
        return false;
    }

    ok &= CheckCuda(cudaMemset(buffers.buffer_storage, 0, buffers.buffer_layout.buffer_bytes));
    ok &= CheckCuda(cudaMemset(buffers.slot_states, 0, buffers.buffer_layout.slot_count * sizeof(BufferSlotState)));
    ok &= CheckCuda(cudaMemset(buffers.free_slot_stack, 0, buffers.buffer_layout.slot_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.free_slot_count, 0, sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.free_slot_lock, 0, sizeof(std::uint32_t)));
    ok &=
        ClearGenerationBuffers(buffers.current_slot_indices, buffers.current_fitness, buffers.current_evaluation_counts,
                               buffers.current_has_fitness, buffers.max_generation_size);
    ok &= ClearGenerationBuffers(buffers.next_slot_indices, buffers.next_fitness, buffers.next_evaluation_counts,
                                 buffers.next_has_fitness, buffers.max_generation_size);
    ok &=
        CheckCuda(cudaMemset(buffers.assembly_parent_pairs, 0, buffers.max_generation_size * sizeof(BufferParentPair)));
    ok &=
        CheckCuda(cudaMemset(buffers.parent_reference_counts, 0, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    if (!ok) {
        DestroyDeviceBufferRuntimeBuffers(buffers);
        return false;
    }

    return true;
}

void DestroyDeviceBufferRuntimeBuffers(DeviceBufferRuntimeBuffers &buffers) noexcept {
    cudaFree(buffers.buffer_storage);
    cudaFree(buffers.slot_states);
    cudaFree(buffers.free_slot_stack);
    cudaFree(buffers.free_slot_count);
    cudaFree(buffers.free_slot_lock);
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

bool TryUploadBufferToDevice(const HostGenotypeBuffer &host_buffer, DeviceBufferRuntimeBuffers &buffers) {
    if (!IsValidHostGenotypeBuffer(host_buffer) ||
        (host_buffer.layout.buffer_bytes != buffers.buffer_layout.buffer_bytes) ||
        (host_buffer.layout.action_count != buffers.buffer_layout.action_count) ||
        (host_buffer.layout.slot_count != buffers.buffer_layout.slot_count) ||
        (host_buffer.layout.slot_stride_bytes != buffers.buffer_layout.slot_stride_bytes)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemcpy(buffers.buffer_storage, host_buffer.storage.get(), host_buffer.layout.buffer_bytes,
                               cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.slot_states, host_buffer.slot_states.get(),
                               host_buffer.layout.slot_count * sizeof(BufferSlotState), cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.free_slot_stack, host_buffer.free_slot_stack.get(),
                               host_buffer.layout.slot_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.free_slot_count, &host_buffer.free_slot_count, sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemset(buffers.free_slot_lock, 0, sizeof(std::uint32_t)));
    return ok;
}

bool TryDownloadBufferFromDevice(const DeviceBufferRuntimeBuffers &buffers, HostGenotypeBuffer &host_buffer) {
    if (!IsValidGenotypeBufferLayout(buffers.buffer_layout) || (buffers.buffer_layout.slot_count == 0)) {
        return false;
    }

    if (!TryCreateHostGenotypeBuffer(host_buffer, buffers.buffer_layout)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemcpy(host_buffer.storage.get(), buffers.buffer_storage, host_buffer.layout.buffer_bytes,
                               cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(host_buffer.slot_states.get(), buffers.slot_states,
                               host_buffer.layout.slot_count * sizeof(BufferSlotState), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(host_buffer.free_slot_stack.get(), buffers.free_slot_stack,
                               host_buffer.layout.slot_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(&host_buffer.free_slot_count, buffers.free_slot_count, sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost));
    host_buffer.free_slot_lock = 0;
    return ok;
}

bool TryUploadCurrentGenerationToDevice(const BufferGeneration &generation, DeviceBufferRuntimeBuffers &buffers) {
    if (!IsGenerationCompatibleWithBuffer(generation, buffers)) {
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

bool TryDownloadCurrentGenerationFromDevice(const DeviceBufferRuntimeBuffers &buffers, BufferGeneration &generation) {
    if (buffers.current_generation_size == 0) {
        return false;
    }

    if (!TryCreateBufferGeneration(generation, buffers.current_generation_size, buffers.current_generation_index)) {
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

bool TryDownloadNextGenerationFromDevice(const DeviceBufferRuntimeBuffers &buffers, BufferGeneration &generation) {
    if (buffers.next_generation_size == 0) {
        return false;
    }

    if (!TryCreateBufferGeneration(generation, buffers.next_generation_size, buffers.next_generation_index)) {
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

bool TryUploadAssemblyPlanToDevice(const BufferAssemblyPlan &plan, DeviceBufferRuntimeBuffers &buffers) {
    if (!IsValidBufferAssemblyPlan(plan) || (plan.child_count > buffers.max_generation_size)) {
        return false;
    }

    buffers.planned_child_count = plan.child_count;
    return CheckCuda(cudaMemcpy(buffers.assembly_parent_pairs, plan.parent_pairs.get(),
                                plan.child_count * sizeof(BufferParentPair), cudaMemcpyHostToDevice));
}

bool TryPrepareBufferForExpandedActionCountOnDevice(DeviceBufferRuntimeBuffers &buffers,
                                                    const std::size_t next_action_count) {
    if (!IsValidGenotypeBufferLayout(buffers.buffer_layout) || (buffers.current_generation_size == 0) ||
        (buffers.planned_child_count == 0) || (next_action_count <= buffers.buffer_layout.action_count)) {
        (void)WriteDeviceStatus(buffers, DeviceBufferRuntimeStatusCode::kInvalidAssemblyConfig);
        return false;
    }

    GenotypeBufferLayout next_layout{};
    if (!TryCreateExpandedBufferLayout(buffers.buffer_layout, next_action_count, next_layout)) {
        (void)WriteDeviceStatus(buffers, DeviceBufferRuntimeStatusCode::kBufferRepackFailed);
        return false;
    }

    bool ok = true;
    ok &= ResetDeviceStatus(buffers);
    ok &=
        CheckCuda(cudaMemset(buffers.parent_reference_counts, 0, buffers.max_generation_size * sizeof(std::uint32_t)));
    if (!ok) {
        return false;
    }

    PrepareBufferForExpandedActionCountKernel<<<1, 1>>>(
        buffers.buffer_layout, next_layout, buffers.buffer_storage, buffers.slot_states, buffers.free_slot_stack,
        buffers.free_slot_count, buffers.free_slot_lock, buffers.current_slot_indices, buffers.current_generation_size,
        buffers.assembly_parent_pairs, buffers.planned_child_count, buffers.parent_reference_counts, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    int status_value = DeviceStatusValue(DeviceBufferRuntimeStatusCode::kCudaFailure);
    if (!ReadDeviceStatus(buffers, status_value) ||
        (status_value != DeviceStatusValue(DeviceBufferRuntimeStatusCode::kOk))) {
        return false;
    }

    buffers.buffer_layout = next_layout;
    return true;
}

bool TryAssembleNextGenerationOnDevice(DeviceBufferRuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                       const BufferDeviceAssemblyConfig &config) {
    if (!IsValidBufferDeviceAssemblyConfig(config) || !IsValidGenotypeBufferLayout(buffers.buffer_layout) ||
        (buffers.current_generation_size == 0) || (buffers.planned_child_count == 0)) {
        (void)WriteDeviceStatus(buffers, DeviceBufferRuntimeStatusCode::kInvalidAssemblyConfig);
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

    ValidateAssemblyInputsKernel<<<1, 1>>>(
        buffers.buffer_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
        buffers.free_slot_lock, buffers.buffer_layout, buffers.current_slot_indices, buffers.current_fitness,
        buffers.current_evaluation_counts, buffers.current_has_fitness, buffers.current_generation_index,
        buffers.current_generation_size, buffers.next_slot_indices, buffers.next_fitness,
        buffers.next_evaluation_counts, buffers.next_has_fitness, buffers.next_generation_index,
        buffers.next_generation_size, buffers.assembly_parent_pairs, buffers.parent_reference_counts, config,
        buffers.status);
    if (!FinishAssemblyKernelOrCleanup(buffers)) {
        return false;
    }

    BuildParentReferenceCountsKernel<<<BoundedAssemblyBlockCount(buffers.next_generation_size),
                                       kBufferAssemblyThreadBlockSize>>>(
        buffers.current_slot_indices, buffers.current_generation_size, buffers.assembly_parent_pairs,
        buffers.next_generation_size, buffers.parent_reference_counts, buffers.status);
    if (!FinishAssemblyKernelOrCleanup(buffers)) {
        return false;
    }

    CollectZeroReferenceParentsKernel<<<BoundedAssemblyBlockCount(buffers.current_generation_size),
                                        kBufferAssemblyThreadBlockSize>>>(
        buffers.buffer_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
        buffers.free_slot_lock, buffers.buffer_layout, buffers.current_slot_indices, buffers.current_fitness,
        buffers.current_evaluation_counts, buffers.current_has_fitness, buffers.current_generation_index,
        buffers.current_generation_size, buffers.parent_reference_counts, buffers.status);
    if (!FinishAssemblyKernelOrCleanup(buffers)) {
        return false;
    }

    std::size_t child_offset = 0;
    while (child_offset < buffers.next_generation_size) {
        std::uint32_t free_slot_count = 0;
        if (!ReadDeviceFreeSlotCount(buffers, free_slot_count)) {
            (void)WriteDeviceStatus(buffers, DeviceBufferRuntimeStatusCode::kCudaFailure);
            (void)CleanupFailedAssemblyOnDevice(buffers);
            return false;
        }

        if (free_slot_count == 0) {
            (void)WriteDeviceStatus(buffers, DeviceBufferRuntimeStatusCode::kBufferFull);
            (void)CleanupFailedAssemblyOnDevice(buffers);
            return false;
        }

        std::size_t batch_child_count = buffers.next_generation_size - child_offset;
        if (batch_child_count > static_cast<std::size_t>(free_slot_count)) {
            batch_child_count = static_cast<std::size_t>(free_slot_count);
        }
        if (batch_child_count > kMaxBufferAssemblyConcurrentChildren) {
            batch_child_count = kMaxBufferAssemblyConcurrentChildren;
        }

        AssembleChildBatchKernel<<<BoundedAssemblyBlockCount(batch_child_count), kBufferAssemblyThreadBlockSize>>>(
            buffers.buffer_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
            buffers.free_slot_lock, buffers.buffer_layout, buffers.current_slot_indices, buffers.current_fitness,
            buffers.current_evaluation_counts, buffers.current_has_fitness, buffers.current_generation_index,
            buffers.current_generation_size, buffers.next_slot_indices, buffers.next_fitness,
            buffers.next_evaluation_counts, buffers.next_has_fitness, buffers.next_generation_index,
            buffers.next_generation_size, buffers.assembly_parent_pairs, buffers.parent_reference_counts, child_offset,
            batch_child_count, generation_seed, config, buffers.status);
        if (!FinishAssemblyKernelOrCleanup(buffers)) {
            return false;
        }

        ReleaseParentReferenceBatchKernel<<<BoundedAssemblyBlockCount(batch_child_count),
                                            kBufferAssemblyThreadBlockSize>>>(
            buffers.buffer_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
            buffers.free_slot_lock, buffers.buffer_layout, buffers.current_slot_indices, buffers.current_fitness,
            buffers.current_evaluation_counts, buffers.current_has_fitness, buffers.current_generation_index,
            buffers.current_generation_size, buffers.assembly_parent_pairs, buffers.parent_reference_counts,
            child_offset, batch_child_count, buffers.status);
        if (!FinishAssemblyKernelOrCleanup(buffers)) {
            return false;
        }

        child_offset += batch_child_count;
    }

    return true;
}

bool TryReadDeviceBufferRuntimeStatus(const DeviceBufferRuntimeBuffers &buffers,
                                      DeviceBufferRuntimeStatusCode &status_code) {
    int status_value = 0;
    if (!ReadDeviceStatus(buffers, status_value)) {
        return false;
    }

    status_code = static_cast<DeviceBufferRuntimeStatusCode>(status_value);
    return true;
}

const char *DeviceBufferRuntimeStatusCodeString(const DeviceBufferRuntimeStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceBufferRuntimeStatusCode::kOk:
        return "ok";
    case DeviceBufferRuntimeStatusCode::kCudaFailure:
        return "cuda failure";
    case DeviceBufferRuntimeStatusCode::kInvalidRuntimeConfig:
        return "invalid device buffer runtime config";
    case DeviceBufferRuntimeStatusCode::kInvalidBuffer:
        return "invalid genotype buffer state";
    case DeviceBufferRuntimeStatusCode::kInvalidGeneration:
        return "invalid buffer generation state";
    case DeviceBufferRuntimeStatusCode::kInvalidAssemblyPlan:
        return "invalid buffer assembly plan";
    case DeviceBufferRuntimeStatusCode::kInvalidAssemblyConfig:
        return "invalid buffer assembly config";
    case DeviceBufferRuntimeStatusCode::kInvalidParentIndex:
        return "invalid parent index in buffer assembly plan";
    case DeviceBufferRuntimeStatusCode::kBufferFull:
        return "buffer is genuinely full";
    case DeviceBufferRuntimeStatusCode::kOutputEmbeddingInjectionFailed:
        return "device output-embedding injection failed";
    case DeviceBufferRuntimeStatusCode::kBufferRepackFailed:
        return "buffer compaction/repacking failed";
    }

    return "unknown device buffer runtime status";
}

} // namespace neuroevolution::genetic_algorithm::genotype_buffer::device
