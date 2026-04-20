#include "genetic_algorithm/genotype_slab/device_runtime.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "genetic_algorithm/device/genome_ops.cuh"
#include "genetic_algorithm/device/injection_ops.cuh"
#include "genetic_algorithm/genotype_slab/reference_counter.hpp"
#include "genetic_algorithm/genotype_slab/repacking.hpp"

namespace neuroevolution::genetic_algorithm::genotype_slab::device {

namespace {

using device_genome_ops::BreedAndMutateGenome;
using device_genome_ops::DeviceRandomState;
using device_genome_ops::InitializeRandomGenome;
using device_genome_ops::IsValidRandomGenomeInitializationConfig;
using device_genome_ops::MakeDeviceRandomState;
using device_genome_ops::RandomGenomeInitializationConfig;
using device_injection_ops::DeviceOutputEmbeddingInjectionStatusCode;
using device_injection_ops::TryInjectExpandedOutputEmbeddingTails;

constexpr int kSlabAssemblyThreadBlockSize = 128;
constexpr int kMaxSlabAssemblyThreadBlocks = 32;
constexpr int kSlabBootstrapThreadBlockSize = 256;
constexpr int kSlabRepackThreadBlockSize = 256;
constexpr int kSlabRepackCompactionMoveGroupSize = 32;
constexpr int kSlabRepackConcurrentCompactionMoves = kSlabRepackThreadBlockSize / kSlabRepackCompactionMoveGroupSize;
constexpr std::size_t kMaxSlabAssemblyConcurrentChildren =
    static_cast<std::size_t>(kSlabAssemblyThreadBlockSize) * static_cast<std::size_t>(kMaxSlabAssemblyThreadBlocks);

NEUROEVOLUTION_HOST_DEVICE constexpr int DeviceStatusValue(const DeviceSlabRuntimeStatusCode status_code) {
    return static_cast<int>(status_code);
}

inline bool CheckCuda(const cudaError_t error) { return error == cudaSuccess; }

inline bool IsGenerationCompatibleWithSlab(const SlabGeneration &generation,
                                           const DeviceSlabRuntimeBuffers &buffers) noexcept {
    if (!IsValidSlabGeneration(generation) || (generation.active_individual_count > buffers.max_generation_size) ||
        !IsValidGenotypeSlabLayout(buffers.slab_layout)) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        const std::uint32_t slot_index = generation.slot_indices[individual_index];
        if ((slot_index != kInvalidSlabSlotIndex) && (slot_index >= buffers.slab_layout.slot_count)) {
            return false;
        }
    }

    return true;
}

inline bool ReadDeviceStatus(const DeviceSlabRuntimeBuffers &buffers, int &status_value) {
    return CheckCuda(cudaMemcpy(&status_value, buffers.status, sizeof(int), cudaMemcpyDeviceToHost));
}

inline bool WriteDeviceStatus(const DeviceSlabRuntimeBuffers &buffers, const DeviceSlabRuntimeStatusCode status_code) {
    const int status_value = DeviceStatusValue(status_code);
    return CheckCuda(cudaMemcpy(buffers.status, &status_value, sizeof(int), cudaMemcpyHostToDevice));
}

inline bool ResetDeviceStatus(const DeviceSlabRuntimeBuffers &buffers) {
    return WriteDeviceStatus(buffers, DeviceSlabRuntimeStatusCode::kOk);
}

inline bool IsDeviceStatusOk(const DeviceSlabRuntimeBuffers &buffers) {
    int status_value = DeviceStatusValue(DeviceSlabRuntimeStatusCode::kCudaFailure);
    return ReadDeviceStatus(buffers, status_value) &&
           (status_value == DeviceStatusValue(DeviceSlabRuntimeStatusCode::kOk));
}

inline int BoundedAssemblyBlockCount(const std::size_t item_count) noexcept {
    if (item_count == 0) {
        return 1;
    }

    const std::size_t block_count = (item_count + static_cast<std::size_t>(kSlabAssemblyThreadBlockSize) - 1) /
                                    static_cast<std::size_t>(kSlabAssemblyThreadBlockSize);
    return static_cast<int>((block_count < static_cast<std::size_t>(kMaxSlabAssemblyThreadBlocks))
                                ? block_count
                                : static_cast<std::size_t>(kMaxSlabAssemblyThreadBlocks));
}

inline bool ReadDeviceFreeSlotCount(const DeviceSlabRuntimeBuffers &buffers, std::uint32_t &free_slot_count) {
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

inline NEUROEVOLUTION_HOST_DEVICE SlabGenerationView MakeDeviceGenerationView(
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

__device__ void SetFailureStatus(int *status, const DeviceSlabRuntimeStatusCode status_code) {
    if (status == nullptr) {
        return;
    }

    atomicCAS(status, DeviceStatusValue(DeviceSlabRuntimeStatusCode::kOk), DeviceStatusValue(status_code));
}

__device__ bool IsDeviceStatusOk(int *status) {
    return (status != nullptr) && (atomicCAS(status, DeviceStatusValue(DeviceSlabRuntimeStatusCode::kOk),
                                             DeviceStatusValue(DeviceSlabRuntimeStatusCode::kOk)) ==
                                   DeviceStatusValue(DeviceSlabRuntimeStatusCode::kOk));
}

NEUROEVOLUTION_HOST_DEVICE constexpr DeviceSlabRuntimeStatusCode
MapInjectionStatus(const DeviceOutputEmbeddingInjectionStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceOutputEmbeddingInjectionStatusCode::kOk:
        return DeviceSlabRuntimeStatusCode::kOk;
    case DeviceOutputEmbeddingInjectionStatusCode::kInvalidTrainingShard:
        return DeviceSlabRuntimeStatusCode::kOutputEmbeddingInjectionFailed;
    case DeviceOutputEmbeddingInjectionStatusCode::kInjectionFailed:
        return DeviceSlabRuntimeStatusCode::kOutputEmbeddingInjectionFailed;
    }

    return DeviceSlabRuntimeStatusCode::kOutputEmbeddingInjectionFailed;
}

__device__ int FinalChildPriorityScore(const SlabParentPair &parent_pair,
                                       const std::uint32_t *remaining_parent_references) noexcept {
    if (remaining_parent_references == nullptr) {
        return -1;
    }

    const bool self_parenting = parent_pair.first_parent_index == parent_pair.second_parent_index;
    int priority = 0;
    if (remaining_parent_references[parent_pair.first_parent_index] == (self_parenting ? 2U : 1U)) {
        ++priority;
    }
    if (!self_parenting && (remaining_parent_references[parent_pair.second_parent_index] == 1U)) {
        ++priority;
    }

    return priority;
}

__global__ void ApplyFinalChildPriorityToAssemblyPlanKernel(SlabParentPair *parent_pairs,
                                                            const std::size_t current_generation_size,
                                                            const std::size_t child_count,
                                                            std::uint32_t *parent_reference_counts, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    if ((parent_pairs == nullptr) || (parent_reference_counts == nullptr) || (current_generation_size == 0) ||
        (child_count == 0)) {
        SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidAssemblyPlan);
        return;
    }

    for (std::size_t parent_index = 0; parent_index < current_generation_size; ++parent_index) {
        parent_reference_counts[parent_index] = 0U;
    }

    for (std::size_t child_index = 0; child_index < child_count; ++child_index) {
        const SlabParentPair &parent_pair = parent_pairs[child_index];
        if ((parent_pair.first_parent_index >= current_generation_size) ||
            (parent_pair.second_parent_index >= current_generation_size)) {
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidParentIndex);
            return;
        }

        ++parent_reference_counts[parent_pair.first_parent_index];
        ++parent_reference_counts[parent_pair.second_parent_index];
    }

    for (std::size_t ordered_child_index = 0; ordered_child_index < child_count; ++ordered_child_index) {
        std::size_t chosen_child_index = ordered_child_index;
        int best_priority = -1;
        for (std::size_t candidate_child_index = ordered_child_index; candidate_child_index < child_count;
             ++candidate_child_index) {
            const int priority =
                FinalChildPriorityScore(parent_pairs[candidate_child_index], parent_reference_counts);
            if (priority > best_priority) {
                best_priority = priority;
                chosen_child_index = candidate_child_index;
                if (priority == 2) {
                    break;
                }
            }
        }

        if (chosen_child_index != ordered_child_index) {
            const SlabParentPair temporary = parent_pairs[ordered_child_index];
            parent_pairs[ordered_child_index] = parent_pairs[chosen_child_index];
            parent_pairs[chosen_child_index] = temporary;
        }

        const SlabParentPair &selected_child = parent_pairs[ordered_child_index];
        --parent_reference_counts[selected_child.first_parent_index];
        --parent_reference_counts[selected_child.second_parent_index];
    }
}

__global__ void InitializeEmptySlabMetadataKernel(SlabSlotState *slot_states, std::uint32_t *free_slot_stack,
                                                  std::uint32_t *free_slot_count, std::uint32_t *free_slot_lock,
                                                  const GenotypeSlabLayout slab_layout, int *status) {
    if (!IsDeviceStatusOk(status)) {
        return;
    }

    if ((slot_states == nullptr) || (free_slot_stack == nullptr) || (free_slot_count == nullptr) ||
        (free_slot_lock == nullptr) || !IsValidGenotypeSlabLayout(slab_layout)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidSlab);
        }
        return;
    }

    const std::size_t slot_index =
        (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
    if (slot_index < slab_layout.slot_count) {
        slot_states[slot_index] = {};
        free_slot_stack[slot_index] = static_cast<std::uint32_t>((slab_layout.slot_count - 1U) - slot_index);
    }

    if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
        *free_slot_count = static_cast<std::uint32_t>(slab_layout.slot_count);
        *free_slot_lock = 0U;
    }
}

__global__ void BootstrapRandomGenerationKernel(
    std::uint8_t *slab_storage, SlabSlotState *slot_states, std::uint32_t *free_slot_stack,
    std::uint32_t *free_slot_count, std::uint32_t *free_slot_lock, const GenotypeSlabLayout slab_layout,
    std::uint32_t *current_slot_indices, const std::size_t generation_size, const std::size_t generation_index,
    const std::uint32_t generation_seed, const DeviceSlabBootstrapConfig bootstrap_config, int *status) {
    if (!IsDeviceStatusOk(status)) {
        return;
    }

    const std::size_t organism_index = blockIdx.x;
    if (organism_index >= generation_size) {
        return;
    }

    if ((slab_storage == nullptr) || (slot_states == nullptr) || (free_slot_stack == nullptr) ||
        (free_slot_count == nullptr) || (free_slot_lock == nullptr) || (current_slot_indices == nullptr) ||
        !IsValidGenotypeSlabLayout(slab_layout) || !IsValidDeviceSlabBootstrapConfig(bootstrap_config)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidRuntimeConfig);
        }
        return;
    }

    GenotypeSlabView slab{
        .layout = slab_layout,
        .storage = slab_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };

    __shared__ std::uint32_t slot_index;
    if (threadIdx.x == 0) {
        slot_index = kInvalidSlabSlotIndex;
        if (!TryAllocateSlabSlot(slab, slot_index)) {
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kSlabFull);
        } else {
            current_slot_indices[organism_index] = slot_index;
        }
    }
    __syncthreads();

    if (!IsDeviceStatusOk(status) || (slot_index == kInvalidSlabSlotIndex)) {
        return;
    }

    RandomGenomeInitializationConfig genome_init_config{};
    genome_init_config.dense_weight_gain = bootstrap_config.dense_weight_gain;
    genome_init_config.output_embedding_tail_stddev = bootstrap_config.output_embedding_tail_stddev;
    if (!IsValidRandomGenomeInitializationConfig(genome_init_config)) {
        SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidRuntimeConfig);
        return;
    }

    const std::uint64_t random_stream =
        (((static_cast<std::uint64_t>(generation_index) * static_cast<std::uint64_t>(generation_size)) +
          static_cast<std::uint64_t>(organism_index)) *
         static_cast<std::uint64_t>(blockDim.x)) +
        static_cast<std::uint64_t>(threadIdx.x);
    DeviceRandomState random_state = MakeDeviceRandomState(generation_seed, random_stream);
    InitializeRandomGenome(SlabSlotBytesAt(slab.storage, slab.layout, slot_index), slab.layout.action_count,
                           random_state, genome_init_config, threadIdx.x, blockDim.x);
}

__global__ void ValidateAssemblyInputsKernel(
    std::uint8_t *slab_storage, SlabSlotState *slot_states, std::uint32_t *free_slot_stack,
    std::uint32_t *free_slot_count, std::uint32_t *free_slot_lock, const GenotypeSlabLayout slab_layout,
    std::uint32_t *current_slot_indices, float *current_fitness, std::uint32_t *current_evaluation_counts,
    std::uint8_t *current_has_fitness, const std::size_t current_generation_index,
    const std::size_t current_generation_size, std::uint32_t *next_slot_indices, float *next_fitness,
    std::uint32_t *next_evaluation_counts, std::uint8_t *next_has_fitness, const std::size_t next_generation_index,
    const std::size_t next_generation_size, const SlabParentPair *parent_pairs, std::uint32_t *parent_reference_counts,
    const SlabDeviceAssemblyConfig config, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    GenotypeSlabView slab{
        .layout = slab_layout,
        .storage = slab_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };
    SlabGenerationView current_generation =
        MakeDeviceGenerationView(current_generation_index, current_generation_size, current_slot_indices,
                                 current_fitness, current_evaluation_counts, current_has_fitness);
    SlabGenerationView next_generation =
        MakeDeviceGenerationView(next_generation_index, next_generation_size, next_slot_indices, next_fitness,
                                 next_evaluation_counts, next_has_fitness);

    if (!IsValidGenotypeSlabView(slab)) {
        SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidSlab);
        return;
    }

    if (!IsValidSlabGenerationView(current_generation) || !IsValidSlabGenerationView(next_generation)) {
        SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidGeneration);
        return;
    }

    if ((parent_pairs == nullptr) || (parent_reference_counts == nullptr) || (next_generation_size == 0)) {
        SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidAssemblyPlan);
        return;
    }

    if (!IsValidSlabDeviceAssemblyConfig(config)) {
        SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidAssemblyConfig);
        return;
    }

    const std::size_t parent_action_count =
        (config.parent_action_count == 0) ? slab.layout.action_count : config.parent_action_count;
    if ((parent_action_count == 0) || (parent_action_count > slab.layout.action_count)) {
        SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidAssemblyConfig);
        return;
    }
}

__global__ void BuildParentReferenceCountsKernel(std::uint32_t *current_slot_indices,
                                                 const std::size_t current_generation_size,
                                                 const SlabParentPair *parent_pairs, const std::size_t child_count,
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

        const SlabParentPair &parent_pair = parent_pairs[child_index];
        if (!IsValidBufferParentIndex(current_slot_indices, current_generation_size, parent_pair.first_parent_index) ||
            !IsValidBufferParentIndex(current_slot_indices, current_generation_size, parent_pair.second_parent_index)) {
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidParentIndex);
            return;
        }

        if (!TryIncrementParentReferenceCount(parent_reference_counts, parent_pair.first_parent_index) ||
            !TryIncrementParentReferenceCount(parent_reference_counts, parent_pair.second_parent_index)) {
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidParentIndex);
            return;
        }
    }
}

__global__ void CollectZeroReferenceParentsKernel(
    std::uint8_t *slab_storage, SlabSlotState *slot_states, std::uint32_t *free_slot_stack,
    std::uint32_t *free_slot_count, std::uint32_t *free_slot_lock, const GenotypeSlabLayout slab_layout,
    std::uint32_t *current_slot_indices, float *current_fitness, std::uint32_t *current_evaluation_counts,
    std::uint8_t *current_has_fitness, const std::size_t current_generation_index,
    const std::size_t current_generation_size, std::uint32_t *parent_reference_counts, int *status) {
    if (!IsDeviceStatusOk(status)) {
        return;
    }

    GenotypeSlabView slab{
        .layout = slab_layout,
        .storage = slab_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };
    SlabGenerationView current_generation =
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
            (current_generation.slot_indices[parent_index] == kInvalidSlabSlotIndex)) {
            continue;
        }

        if (!TryReleaseSlabSlot(slab, current_generation.slot_indices[parent_index])) {
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidSlab);
            return;
        }

        ClearSlabGenerationSlot(current_generation, parent_index);
    }
}

__global__ void AssembleChildBatchKernel(
    std::uint8_t *slab_storage, SlabSlotState *slot_states, std::uint32_t *free_slot_stack,
    std::uint32_t *free_slot_count, std::uint32_t *free_slot_lock, const GenotypeSlabLayout slab_layout,
    std::uint32_t *current_slot_indices, float *current_fitness, std::uint32_t *current_evaluation_counts,
    std::uint8_t *current_has_fitness, const std::size_t current_generation_index,
    const std::size_t current_generation_size, std::uint32_t *next_slot_indices, float *next_fitness,
    std::uint32_t *next_evaluation_counts, std::uint8_t *next_has_fitness, const std::size_t next_generation_index,
    const std::size_t next_generation_size, const SlabParentPair *parent_pairs, std::uint32_t *parent_reference_counts,
    const std::size_t child_offset, const std::size_t batch_child_count, const std::uint32_t generation_seed,
    const SlabDeviceAssemblyConfig config, int *status) {
    if (!IsDeviceStatusOk(status)) {
        return;
    }

    GenotypeSlabView slab{
        .layout = slab_layout,
        .storage = slab_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };
    SlabGenerationView current_generation =
        MakeDeviceGenerationView(current_generation_index, current_generation_size, current_slot_indices,
                                 current_fitness, current_evaluation_counts, current_has_fitness);
    SlabGenerationView next_generation =
        MakeDeviceGenerationView(next_generation_index, next_generation_size, next_slot_indices, next_fitness,
                                 next_evaluation_counts, next_has_fitness);

    const std::size_t parent_action_count =
        (config.parent_action_count == 0) ? slab.layout.action_count : config.parent_action_count;
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
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidAssemblyPlan);
            return;
        }

        const SlabParentPair &parent_pair = parent_pairs[child_index];
        const std::uint32_t first_parent_slot = current_generation.slot_indices[parent_pair.first_parent_index];
        const std::uint32_t second_parent_slot = current_generation.slot_indices[parent_pair.second_parent_index];
        if ((first_parent_slot == kInvalidSlabSlotIndex) || (second_parent_slot == kInvalidSlabSlotIndex)) {
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidParentIndex);
            return;
        }

        std::uint32_t child_slot = kInvalidSlabSlotIndex;
        if (!TryAllocateSlabSlot(slab, child_slot)) {
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kSlabFull);
            return;
        }

        DeviceRandomState random_state = MakeDeviceRandomState(
            generation_seed, static_cast<std::uint32_t>(child_index + (current_generation_index * 4099U)));
        BreedAndMutateGenome(SlabSlotBytesAt(slab.storage, slab.layout, first_parent_slot),
                             SlabSlotBytesAt(slab.storage, slab.layout, second_parent_slot), parent_action_count,
                             SlabSlotBytesAt(slab.storage, slab.layout, child_slot), random_state, config.breeding,
                             config.mutation);
        if (config.pending_output_embedding_injection.enabled) {
            const DeviceOutputEmbeddingInjectionStatusCode injection_status = TryInjectExpandedOutputEmbeddingTails(
                SlabSlotBytesAt(slab.storage, slab.layout, child_slot), parent_action_count,
                config.pending_output_embedding_injection.first_catalog_word_index,
                config.pending_output_embedding_injection.injection_count);
            if (injection_status != DeviceOutputEmbeddingInjectionStatusCode::kOk) {
                (void)TryReleaseSlabSlot(slab, child_slot);
                SetFailureStatus(status, MapInjectionStatus(injection_status));
                return;
            }
        }

        next_generation.slot_indices[child_index] = child_slot;
    }
}

__global__ void
ReleaseParentReferenceBatchKernel(std::uint8_t *slab_storage, SlabSlotState *slot_states,
                                  std::uint32_t *free_slot_stack, std::uint32_t *free_slot_count,
                                  std::uint32_t *free_slot_lock, const GenotypeSlabLayout slab_layout,
                                  std::uint32_t *current_slot_indices, float *current_fitness,
                                  std::uint32_t *current_evaluation_counts, std::uint8_t *current_has_fitness,
                                  const std::size_t current_generation_index, const std::size_t current_generation_size,
                                  const SlabParentPair *parent_pairs, std::uint32_t *parent_reference_counts,
                                  const std::size_t child_offset, const std::size_t batch_child_count, int *status) {
    if (!IsDeviceStatusOk(status)) {
        return;
    }

    GenotypeSlabView slab{
        .layout = slab_layout,
        .storage = slab_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };
    SlabGenerationView current_generation =
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
        const SlabParentPair &parent_pair = parent_pairs[child_index];
        if (!TryReleaseParentReference(slab, current_generation, parent_reference_counts,
                                       parent_pair.first_parent_index) ||
            !TryReleaseParentReference(slab, current_generation, parent_reference_counts,
                                       parent_pair.second_parent_index)) {
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidSlab);
            return;
        }
    }
}

__global__ void CleanupNextGenerationSlotsKernel(std::uint8_t *slab_storage, SlabSlotState *slot_states,
                                                 std::uint32_t *free_slot_stack, std::uint32_t *free_slot_count,
                                                 std::uint32_t *free_slot_lock, const GenotypeSlabLayout slab_layout,
                                                 std::uint32_t *next_slot_indices, float *next_fitness,
                                                 std::uint32_t *next_evaluation_counts, std::uint8_t *next_has_fitness,
                                                 const std::size_t next_generation_size, int *status) {
    GenotypeSlabView slab{
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
    for (std::size_t child_index = worker_index; child_index < next_generation_size; child_index += worker_count) {
        const std::uint32_t child_slot = next_slot_indices[child_index];
        if ((child_slot != kInvalidSlabSlotIndex) && !TryReleaseSlabSlot(slab, child_slot)) {
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidSlab);
        }

        next_slot_indices[child_index] = kInvalidSlabSlotIndex;
        next_fitness[child_index] = 0.0f;
        next_evaluation_counts[child_index] = 0;
        next_has_fitness[child_index] = 0;
    }
}

bool CleanupFailedAssemblyOnDeviceInternal(DeviceSlabRuntimeBuffers &buffers) {
    const std::size_t cleanup_generation_size = buffers.next_generation_size;
    int original_status = DeviceStatusValue(DeviceSlabRuntimeStatusCode::kCudaFailure);
    bool ok = ReadDeviceStatus(buffers, original_status);
    ok &= ResetDeviceStatus(buffers);

    if (cleanup_generation_size > 0) {
        CleanupNextGenerationSlotsKernel<<<BoundedAssemblyBlockCount(cleanup_generation_size),
                                           kSlabAssemblyThreadBlockSize>>>(
            buffers.slab_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
            buffers.free_slot_lock, buffers.slab_layout, buffers.next_slot_indices, buffers.next_fitness,
            buffers.next_evaluation_counts, buffers.next_has_fitness, cleanup_generation_size, buffers.status);
        ok &= CheckCuda(cudaGetLastError());
        ok &= CheckCuda(cudaDeviceSynchronize());
    }

    ok &= ClearGenerationBuffers(buffers.next_slot_indices, buffers.next_fitness, buffers.next_evaluation_counts,
                                 buffers.next_has_fitness, buffers.max_generation_size);

    int cleanup_status = DeviceStatusValue(DeviceSlabRuntimeStatusCode::kCudaFailure);
    ok &= ReadDeviceStatus(buffers, cleanup_status);
    if (original_status != DeviceStatusValue(DeviceSlabRuntimeStatusCode::kOk)) {
        ok &= WriteDeviceStatus(buffers, static_cast<DeviceSlabRuntimeStatusCode>(original_status));
    }

    buffers.next_generation_index = 0;
    buffers.next_generation_size = 0;
    return ok && (cleanup_status == DeviceStatusValue(DeviceSlabRuntimeStatusCode::kOk));
}

inline bool FinishKernelWithoutCleanup(const DeviceSlabRuntimeBuffers &buffers) {
    return CheckCuda(cudaGetLastError()) && CheckCuda(cudaDeviceSynchronize()) && IsDeviceStatusOk(buffers);
}

inline bool ResetSlabMetadataOnDevice(DeviceSlabRuntimeBuffers &buffers) {
    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    const std::size_t block_count =
        (buffers.slab_layout.slot_count + static_cast<std::size_t>(kSlabAssemblyThreadBlockSize) - 1U) /
        static_cast<std::size_t>(kSlabAssemblyThreadBlockSize);
    InitializeEmptySlabMetadataKernel<<<static_cast<int>(block_count), kSlabAssemblyThreadBlockSize>>>(
        buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count, buffers.free_slot_lock,
        buffers.slab_layout, buffers.status);
    return FinishKernelWithoutCleanup(buffers);
}

inline bool FinishAssemblyKernelOrCleanup(DeviceSlabRuntimeBuffers &buffers) {
    const bool launch_ok = CheckCuda(cudaGetLastError());
    const bool sync_ok = CheckCuda(cudaDeviceSynchronize());
    if (!launch_ok || !sync_ok) {
        (void)WriteDeviceStatus(buffers, DeviceSlabRuntimeStatusCode::kCudaFailure);
        (void)CleanupFailedAssemblyOnDeviceInternal(buffers);
        return false;
    }

    if (!IsDeviceStatusOk(buffers)) {
        (void)CleanupFailedAssemblyOnDeviceInternal(buffers);
        return false;
    }

    return true;
}

__device__ bool TryCompactReferencedParentsIntoPrefixConcurrently(
    const GenotypeSlabLayout slab_layout, std::uint8_t *slab_storage, std::uint32_t *generation_slot_indices,
    const std::size_t active_individual_count, const std::uint32_t *parent_reference_counts,
    const std::size_t survivor_count, std::uint32_t *move_target_slots, std::uint32_t *move_source_slots,
    std::uint32_t *next_target_slot, std::uint32_t *next_source_slot, std::uint32_t *move_count, bool *failed) {
    if ((move_target_slots == nullptr) || (move_source_slots == nullptr) || (next_target_slot == nullptr) ||
        (next_source_slot == nullptr) || (move_count == nullptr) || (failed == nullptr)) {
        return false;
    }

    constexpr std::size_t kCopyWorkerCount = static_cast<std::size_t>(kSlabRepackCompactionMoveGroupSize);
    const std::size_t move_group_index = static_cast<std::size_t>(threadIdx.x / kSlabRepackCompactionMoveGroupSize);
    const std::size_t move_group_worker_index = static_cast<std::size_t>(threadIdx.x % kSlabRepackCompactionMoveGroupSize);

    if (threadIdx.x == 0) {
        *next_target_slot = 0;
        *next_source_slot = static_cast<std::uint32_t>(survivor_count);
        *move_count = 0;
        *failed = false;
    }
    __syncthreads();

    while (*next_target_slot < survivor_count) {
        if (threadIdx.x == 0) {
            *move_count = 0;
            std::uint32_t target_slot_index = *next_target_slot;
            while ((target_slot_index < survivor_count) &&
                   (*move_count < static_cast<std::uint32_t>(kSlabRepackConcurrentCompactionMoves))) {
                std::uint32_t parent_index = detail::FindReferencedParentIndexOwningSlot(
                    generation_slot_indices, active_individual_count, parent_reference_counts, target_slot_index);
                if (parent_index != kInvalidSlabSlotIndex) {
                    generation_slot_indices[parent_index] = target_slot_index;
                    ++target_slot_index;
                    continue;
                }

                while (*next_source_slot < slab_layout.slot_count) {
                    parent_index = detail::FindReferencedParentIndexOwningSlot(
                        generation_slot_indices, active_individual_count, parent_reference_counts, *next_source_slot);
                    if (parent_index != kInvalidSlabSlotIndex) {
                        break;
                    }
                    ++(*next_source_slot);
                }

                if ((*next_source_slot >= slab_layout.slot_count) || (parent_index == kInvalidSlabSlotIndex)) {
                    *failed = true;
                    break;
                }

                move_target_slots[*move_count] = target_slot_index;
                move_source_slots[*move_count] = *next_source_slot;
                generation_slot_indices[parent_index] = target_slot_index;
                ++(*move_count);
                ++(*next_source_slot);
                ++target_slot_index;
            }

            *next_target_slot = target_slot_index;
        }
        __syncthreads();

        if (*failed) {
            return false;
        }

        if (move_group_index < static_cast<std::size_t>(*move_count)) {
            const std::uint32_t source_slot_index = move_source_slots[move_group_index];
            const std::uint32_t target_slot_index = move_target_slots[move_group_index];
            detail::MoveSlabBytesOverlapping(SlabSlotBytesAt(slab_storage, slab_layout, source_slot_index),
                                             SlabSlotBytesAt(slab_storage, slab_layout, target_slot_index),
                                             slab_layout.slot_stride_bytes, move_group_worker_index,
                                             kCopyWorkerCount);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (std::size_t parent_index = 0; parent_index < active_individual_count; ++parent_index) {
            if (parent_reference_counts[parent_index] == 0) {
                generation_slot_indices[parent_index] = kInvalidSlabSlotIndex;
            }
        }

        for (std::uint32_t source_slot_index = static_cast<std::uint32_t>(survivor_count);
             source_slot_index < slab_layout.slot_count; ++source_slot_index) {
            if (detail::FindReferencedParentIndexOwningSlot(generation_slot_indices, active_individual_count,
                                                            parent_reference_counts, source_slot_index) !=
                kInvalidSlabSlotIndex) {
                *failed = true;
                break;
            }
        }
    }
    __syncthreads();

    return !(*failed);
}

__global__ void PrepareSlabForExpandedActionCountKernel(
    const GenotypeSlabLayout current_layout, const GenotypeSlabLayout next_layout, std::uint8_t *slab_storage,
    SlabSlotState *slot_states, std::uint32_t *free_slot_stack, std::uint32_t *free_slot_count,
    std::uint32_t *free_slot_lock, std::uint32_t *current_slot_indices, const std::size_t current_generation_size,
    std::uint32_t *parent_reference_counts, int *status) {
    if (blockIdx.x != 0) {
        return;
    }

    __shared__ bool inputs_valid;
    __shared__ bool repack_failed;
    __shared__ detail::SlabRepackPreflight preflight;
    __shared__ std::uint32_t move_target_slots[kSlabRepackConcurrentCompactionMoves];
    __shared__ std::uint32_t move_source_slots[kSlabRepackConcurrentCompactionMoves];
    __shared__ std::uint32_t next_target_slot;
    __shared__ std::uint32_t next_source_slot;
    __shared__ std::uint32_t move_count;
    if (threadIdx.x == 0) {
        inputs_valid = (slab_storage != nullptr) && (slot_states != nullptr) && (free_slot_stack != nullptr) &&
                       (free_slot_count != nullptr) && (free_slot_lock != nullptr) && (current_slot_indices != nullptr) &&
                       (parent_reference_counts != nullptr) && (current_generation_size > 0) &&
                       IsValidGenotypeSlabLayout(current_layout) && IsValidGenotypeSlabLayout(next_layout);
        repack_failed = false;
        if (!inputs_valid) {
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kInvalidSlab);
        } else if (!detail::TryPreflightCompactionAndRepackForExpandedActionCount(
                current_layout, slot_states, current_slot_indices, current_generation_size, parent_reference_counts,
                next_layout.action_count, preflight)) {
            inputs_valid = false;
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kSlabRepackFailed);
        }
    }
    __syncthreads();

    if (!inputs_valid) {
        return;
    }

    if (!TryCompactReferencedParentsIntoPrefixConcurrently(
            current_layout, slab_storage, current_slot_indices, current_generation_size, parent_reference_counts,
            preflight.survivor_count, move_target_slots, move_source_slots, &next_target_slot, &next_source_slot,
            &move_count, &repack_failed) ||
        !detail::TryRepackCompactedParentsForExpandedActionCount(
            current_layout, next_layout, slab_storage, slot_states, free_slot_stack, *free_slot_count,
            current_slot_indices, current_generation_size, preflight.survivor_count, preflight.destination_base_slot,
            static_cast<std::size_t>(threadIdx.x), static_cast<std::size_t>(blockDim.x))) {
        __syncthreads();
        if (threadIdx.x == 0) {
            SetFailureStatus(status, DeviceSlabRuntimeStatusCode::kSlabRepackFailed);
        }
    }
}

} // namespace

bool TryCleanupFailedAssemblyOnDevice(DeviceSlabRuntimeBuffers &buffers) {
    return CleanupFailedAssemblyOnDeviceInternal(buffers);
}

bool TryCreateDeviceSlabRuntimeBuffers(DeviceSlabRuntimeBuffers &buffers, const DeviceSlabRuntimeConfig &config) {
    buffers = {};
    if (!IsValidDeviceSlabRuntimeConfig(config)) {
        return false;
    }

    buffers.slab_layout.action_count = config.action_count;
    buffers.slab_layout.slot_stride_bytes = ComputeSlabSlotStrideBytes(config.action_count);
    buffers.slab_layout.slot_count = config.slot_count;
    buffers.slab_layout.slab_bytes = buffers.slab_layout.slot_stride_bytes * buffers.slab_layout.slot_count;
    buffers.max_generation_size = config.max_generation_size;
    if (!IsValidGenotypeSlabLayout(buffers.slab_layout)) {
        buffers = {};
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMalloc(&buffers.slab_storage, buffers.slab_layout.slab_bytes));
    ok &= CheckCuda(cudaMalloc(&buffers.slot_states, buffers.slab_layout.slot_count * sizeof(SlabSlotState)));
    ok &= CheckCuda(cudaMalloc(&buffers.free_slot_stack, buffers.slab_layout.slot_count * sizeof(std::uint32_t)));
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

    ok &= CheckCuda(cudaMalloc(&buffers.assembly_parent_pairs, buffers.max_generation_size * sizeof(SlabParentPair)));
    ok &= CheckCuda(cudaMalloc(&buffers.parent_reference_counts, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.status, sizeof(int)));
    if (!ok) {
        DestroyDeviceSlabRuntimeBuffers(buffers);
        return false;
    }

    ok &= CheckCuda(cudaMemset(buffers.slot_states, 0, buffers.slab_layout.slot_count * sizeof(SlabSlotState)));
    ok &= CheckCuda(cudaMemset(buffers.free_slot_stack, 0, buffers.slab_layout.slot_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.free_slot_count, 0, sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.free_slot_lock, 0, sizeof(std::uint32_t)));
    ok &=
        ClearGenerationBuffers(buffers.current_slot_indices, buffers.current_fitness, buffers.current_evaluation_counts,
                               buffers.current_has_fitness, buffers.max_generation_size);
    ok &= ClearGenerationBuffers(buffers.next_slot_indices, buffers.next_fitness, buffers.next_evaluation_counts,
                                 buffers.next_has_fitness, buffers.max_generation_size);
    ok &= CheckCuda(cudaMemset(buffers.assembly_parent_pairs, 0, buffers.max_generation_size * sizeof(SlabParentPair)));
    ok &=
        CheckCuda(cudaMemset(buffers.parent_reference_counts, 0, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    if (!ok) {
        DestroyDeviceSlabRuntimeBuffers(buffers);
        return false;
    }

    return true;
}

void DestroyDeviceSlabRuntimeBuffers(DeviceSlabRuntimeBuffers &buffers) noexcept {
    cudaFree(buffers.slab_storage);
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

bool TryUploadSlabToDevice(const HostGenotypeSlab &host_buffer, DeviceSlabRuntimeBuffers &buffers) {
    if (!IsValidHostGenotypeSlab(host_buffer) || (host_buffer.layout.slab_bytes != buffers.slab_layout.slab_bytes) ||
        (host_buffer.layout.action_count != buffers.slab_layout.action_count) ||
        (host_buffer.layout.slot_count != buffers.slab_layout.slot_count) ||
        (host_buffer.layout.slot_stride_bytes != buffers.slab_layout.slot_stride_bytes)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemcpy(buffers.slab_storage, host_buffer.storage.get(), host_buffer.layout.slab_bytes,
                               cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.slot_states, host_buffer.slot_states.get(),
                               host_buffer.layout.slot_count * sizeof(SlabSlotState), cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.free_slot_stack, host_buffer.free_slot_stack.get(),
                               host_buffer.layout.slot_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.free_slot_count, &host_buffer.free_slot_count, sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemset(buffers.free_slot_lock, 0, sizeof(std::uint32_t)));
    return ok;
}

bool TryDownloadSlabFromDevice(const DeviceSlabRuntimeBuffers &buffers, HostGenotypeSlab &host_buffer) {
    if (!IsValidGenotypeSlabLayout(buffers.slab_layout) || (buffers.slab_layout.slot_count == 0)) {
        return false;
    }

    if (!TryCreateHostGenotypeSlab(host_buffer, buffers.slab_layout)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemcpy(host_buffer.storage.get(), buffers.slab_storage, host_buffer.layout.slab_bytes,
                               cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(host_buffer.slot_states.get(), buffers.slot_states,
                               host_buffer.layout.slot_count * sizeof(SlabSlotState), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(host_buffer.free_slot_stack.get(), buffers.free_slot_stack,
                               host_buffer.layout.slot_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(&host_buffer.free_slot_count, buffers.free_slot_count, sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost));
    host_buffer.free_slot_lock = 0;
    return ok;
}

bool TryUploadCurrentGenerationToDevice(const SlabGeneration &generation, DeviceSlabRuntimeBuffers &buffers) {
    if (!IsGenerationCompatibleWithSlab(generation, buffers)) {
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

bool TryBootstrapRandomCurrentGenerationOnDevice(DeviceSlabRuntimeBuffers &buffers, const std::size_t generation_size,
                                                 const std::uint32_t generation_seed,
                                                 const std::size_t generation_index,
                                                 const DeviceSlabBootstrapConfig &config) {
    if ((generation_size == 0) || (generation_size > buffers.max_generation_size) ||
        !IsValidDeviceSlabBootstrapConfig(config) || !IsValidGenotypeSlabLayout(buffers.slab_layout)) {
        return false;
    }

    bool ok = true;
    ok &= ClearGenerationBuffers(buffers.current_slot_indices, buffers.current_fitness, buffers.current_evaluation_counts,
                                 buffers.current_has_fitness, buffers.max_generation_size);
    ok &= ClearGenerationBuffers(buffers.next_slot_indices, buffers.next_fitness, buffers.next_evaluation_counts,
                                 buffers.next_has_fitness, buffers.max_generation_size);
    ok &= CheckCuda(cudaMemset(buffers.assembly_parent_pairs, 0, buffers.max_generation_size * sizeof(SlabParentPair)));
    ok &= CheckCuda(cudaMemset(buffers.parent_reference_counts, 0,
                               buffers.max_generation_size * sizeof(std::uint32_t)));
    if (!ok || !ResetSlabMetadataOnDevice(buffers) || !ResetDeviceStatus(buffers)) {
        buffers.current_generation_index = 0;
        buffers.current_generation_size = 0;
        buffers.next_generation_index = 0;
        buffers.next_generation_size = 0;
        buffers.planned_child_count = 0;
        return false;
    }

    BootstrapRandomGenerationKernel<<<static_cast<int>(generation_size), kSlabBootstrapThreadBlockSize>>>(
        buffers.slab_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
        buffers.free_slot_lock, buffers.slab_layout, buffers.current_slot_indices, generation_size, generation_index,
        generation_seed, config, buffers.status);

    int original_status = DeviceStatusValue(DeviceSlabRuntimeStatusCode::kCudaFailure);
    const bool launch_ok = CheckCuda(cudaGetLastError());
    const bool sync_ok = CheckCuda(cudaDeviceSynchronize());
    if (!launch_ok || !sync_ok) {
        (void)WriteDeviceStatus(buffers, DeviceSlabRuntimeStatusCode::kCudaFailure);
    }
    if ((!launch_ok || !sync_ok) || !IsDeviceStatusOk(buffers)) {
        (void)ReadDeviceStatus(buffers, original_status);
        (void)ResetSlabMetadataOnDevice(buffers);
        (void)ClearGenerationBuffers(buffers.current_slot_indices, buffers.current_fitness,
                                     buffers.current_evaluation_counts, buffers.current_has_fitness,
                                     buffers.max_generation_size);
        (void)ClearGenerationBuffers(buffers.next_slot_indices, buffers.next_fitness,
                                     buffers.next_evaluation_counts, buffers.next_has_fitness,
                                     buffers.max_generation_size);
        if (original_status != DeviceStatusValue(DeviceSlabRuntimeStatusCode::kOk)) {
            (void)WriteDeviceStatus(buffers, static_cast<DeviceSlabRuntimeStatusCode>(original_status));
        }
        buffers.current_generation_index = 0;
        buffers.current_generation_size = 0;
        buffers.next_generation_index = 0;
        buffers.next_generation_size = 0;
        buffers.planned_child_count = 0;
        return false;
    }

    buffers.current_generation_index = generation_index;
    buffers.current_generation_size = generation_size;
    buffers.next_generation_index = 0;
    buffers.next_generation_size = 0;
    buffers.planned_child_count = 0;
    return true;
}

bool TryDownloadCurrentGenerationFromDevice(const DeviceSlabRuntimeBuffers &buffers, SlabGeneration &generation) {
    if (buffers.current_generation_size == 0) {
        return false;
    }

    if (!TryCreateSlabGeneration(generation, buffers.current_generation_size, buffers.current_generation_index)) {
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

bool TryDownloadNextGenerationFromDevice(const DeviceSlabRuntimeBuffers &buffers, SlabGeneration &generation) {
    if (buffers.next_generation_size == 0) {
        return false;
    }

    if (!TryCreateSlabGeneration(generation, buffers.next_generation_size, buffers.next_generation_index)) {
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

bool TryUploadAssemblyPlanToDevice(const SlabAssemblyPlan &plan, DeviceSlabRuntimeBuffers &buffers) {
    if (!IsValidSlabAssemblyPlan(plan) || (plan.child_count > buffers.max_generation_size)) {
        return false;
    }

    buffers.planned_child_count = plan.child_count;
    return CheckCuda(cudaMemcpy(buffers.assembly_parent_pairs, plan.parent_pairs.get(),
                                plan.child_count * sizeof(SlabParentPair), cudaMemcpyHostToDevice));
}

bool TryApplyFinalChildPriorityToAssemblyPlanOnDevice(DeviceSlabRuntimeBuffers &buffers) {
    if ((buffers.current_generation_size == 0) || (buffers.planned_child_count == 0) ||
        (buffers.current_generation_size > buffers.max_generation_size)) {
        (void)WriteDeviceStatus(buffers, DeviceSlabRuntimeStatusCode::kInvalidAssemblyPlan);
        return false;
    }

    bool ok = true;
    ok &= ResetDeviceStatus(buffers);
    ok &=
        CheckCuda(cudaMemset(buffers.parent_reference_counts, 0, buffers.max_generation_size * sizeof(std::uint32_t)));
    if (!ok) {
        return false;
    }

    ApplyFinalChildPriorityToAssemblyPlanKernel<<<1, 1>>>(buffers.assembly_parent_pairs,
                                                          buffers.current_generation_size, buffers.planned_child_count,
                                                          buffers.parent_reference_counts, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    int status_value = DeviceStatusValue(DeviceSlabRuntimeStatusCode::kCudaFailure);
    return ReadDeviceStatus(buffers, status_value) &&
           (status_value == DeviceStatusValue(DeviceSlabRuntimeStatusCode::kOk));
}

bool TryPrepareSlabForExpandedActionCountOnDevice(DeviceSlabRuntimeBuffers &buffers,
                                                  const std::size_t next_action_count) {
    if (!IsValidGenotypeSlabLayout(buffers.slab_layout) || (buffers.current_generation_size == 0) ||
        (buffers.planned_child_count == 0) || (next_action_count <= buffers.slab_layout.action_count)) {
        (void)WriteDeviceStatus(buffers, DeviceSlabRuntimeStatusCode::kInvalidAssemblyConfig);
        return false;
    }

    GenotypeSlabLayout next_layout{};
    if (!TryCreateExpandedSlabLayout(buffers.slab_layout, next_action_count, next_layout)) {
        (void)WriteDeviceStatus(buffers, DeviceSlabRuntimeStatusCode::kSlabRepackFailed);
        return false;
    }

    bool ok = true;
    ok &= ResetDeviceStatus(buffers);
    ok &=
        CheckCuda(cudaMemset(buffers.parent_reference_counts, 0, buffers.max_generation_size * sizeof(std::uint32_t)));
    if (!ok) {
        return false;
    }

    BuildParentReferenceCountsKernel<<<BoundedAssemblyBlockCount(buffers.planned_child_count),
                                       kSlabAssemblyThreadBlockSize>>>(
        buffers.current_slot_indices, buffers.current_generation_size, buffers.assembly_parent_pairs,
        buffers.planned_child_count, buffers.parent_reference_counts, buffers.status);
    if (!FinishKernelWithoutCleanup(buffers)) {
        return false;
    }

    CollectZeroReferenceParentsKernel<<<BoundedAssemblyBlockCount(buffers.current_generation_size),
                                        kSlabAssemblyThreadBlockSize>>>(
        buffers.slab_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
        buffers.free_slot_lock, buffers.slab_layout, buffers.current_slot_indices, buffers.current_fitness,
        buffers.current_evaluation_counts, buffers.current_has_fitness, buffers.current_generation_index,
        buffers.current_generation_size, buffers.parent_reference_counts, buffers.status);
    if (!FinishKernelWithoutCleanup(buffers)) {
        return false;
    }

    PrepareSlabForExpandedActionCountKernel<<<1, kSlabRepackThreadBlockSize>>>(
        buffers.slab_layout, next_layout, buffers.slab_storage, buffers.slot_states, buffers.free_slot_stack,
        buffers.free_slot_count, buffers.free_slot_lock, buffers.current_slot_indices, buffers.current_generation_size,
        buffers.parent_reference_counts, buffers.status);
    if (!FinishKernelWithoutCleanup(buffers)) {
        return false;
    }

    buffers.slab_layout = next_layout;
    return true;
}

bool TryInitializeNextGenerationAssemblyOnDevice(DeviceSlabRuntimeBuffers &buffers,
                                                 const SlabDeviceAssemblyConfig &config) {
    if (!IsValidSlabDeviceAssemblyConfig(config) || !IsValidGenotypeSlabLayout(buffers.slab_layout) ||
        (buffers.current_generation_size == 0) || (buffers.planned_child_count == 0)) {
        (void)WriteDeviceStatus(buffers, DeviceSlabRuntimeStatusCode::kInvalidAssemblyConfig);
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
        buffers.slab_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
        buffers.free_slot_lock, buffers.slab_layout, buffers.current_slot_indices, buffers.current_fitness,
        buffers.current_evaluation_counts, buffers.current_has_fitness, buffers.current_generation_index,
        buffers.current_generation_size, buffers.next_slot_indices, buffers.next_fitness,
        buffers.next_evaluation_counts, buffers.next_has_fitness, buffers.next_generation_index,
        buffers.next_generation_size, buffers.assembly_parent_pairs, buffers.parent_reference_counts, config,
        buffers.status);
    if (!FinishAssemblyKernelOrCleanup(buffers)) {
        return false;
    }

    BuildParentReferenceCountsKernel<<<BoundedAssemblyBlockCount(buffers.next_generation_size),
                                       kSlabAssemblyThreadBlockSize>>>(
        buffers.current_slot_indices, buffers.current_generation_size, buffers.assembly_parent_pairs,
        buffers.next_generation_size, buffers.parent_reference_counts, buffers.status);
    if (!FinishAssemblyKernelOrCleanup(buffers)) {
        return false;
    }

    CollectZeroReferenceParentsKernel<<<BoundedAssemblyBlockCount(buffers.current_generation_size),
                                        kSlabAssemblyThreadBlockSize>>>(
        buffers.slab_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
        buffers.free_slot_lock, buffers.slab_layout, buffers.current_slot_indices, buffers.current_fitness,
        buffers.current_evaluation_counts, buffers.current_has_fitness, buffers.current_generation_index,
        buffers.current_generation_size, buffers.parent_reference_counts, buffers.status);
    return FinishAssemblyKernelOrCleanup(buffers);
}

bool TryContinueNextGenerationAssemblyOnDevice(DeviceSlabRuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                               const SlabDeviceAssemblyConfig &config, std::size_t child_offset) {
    if (!IsValidSlabDeviceAssemblyConfig(config) || !IsValidGenotypeSlabLayout(buffers.slab_layout) ||
        (buffers.current_generation_size == 0) || (buffers.next_generation_size == 0) ||
        (child_offset > buffers.next_generation_size)) {
        (void)WriteDeviceStatus(buffers, DeviceSlabRuntimeStatusCode::kInvalidAssemblyConfig);
        return false;
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    while (child_offset < buffers.next_generation_size) {
        std::uint32_t free_slot_count = 0;
        if (!ReadDeviceFreeSlotCount(buffers, free_slot_count)) {
            (void)WriteDeviceStatus(buffers, DeviceSlabRuntimeStatusCode::kCudaFailure);
            (void)TryCleanupFailedAssemblyOnDevice(buffers);
            return false;
        }

        if (free_slot_count == 0) {
            (void)WriteDeviceStatus(buffers, DeviceSlabRuntimeStatusCode::kSlabFull);
            return false;
        }

        std::size_t batch_child_count = buffers.next_generation_size - child_offset;
        if (batch_child_count > static_cast<std::size_t>(free_slot_count)) {
            batch_child_count = static_cast<std::size_t>(free_slot_count);
        }
        if (batch_child_count > kMaxSlabAssemblyConcurrentChildren) {
            batch_child_count = kMaxSlabAssemblyConcurrentChildren;
        }

        AssembleChildBatchKernel<<<BoundedAssemblyBlockCount(batch_child_count), kSlabAssemblyThreadBlockSize>>>(
            buffers.slab_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
            buffers.free_slot_lock, buffers.slab_layout, buffers.current_slot_indices, buffers.current_fitness,
            buffers.current_evaluation_counts, buffers.current_has_fitness, buffers.current_generation_index,
            buffers.current_generation_size, buffers.next_slot_indices, buffers.next_fitness,
            buffers.next_evaluation_counts, buffers.next_has_fitness, buffers.next_generation_index,
            buffers.next_generation_size, buffers.assembly_parent_pairs, buffers.parent_reference_counts, child_offset,
            batch_child_count, generation_seed, config, buffers.status);
        if (!FinishAssemblyKernelOrCleanup(buffers)) {
            return false;
        }

        ReleaseParentReferenceBatchKernel<<<BoundedAssemblyBlockCount(batch_child_count),
                                            kSlabAssemblyThreadBlockSize>>>(
            buffers.slab_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
            buffers.free_slot_lock, buffers.slab_layout, buffers.current_slot_indices, buffers.current_fitness,
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

bool TryAssembleNextGenerationOnDevice(DeviceSlabRuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                       const SlabDeviceAssemblyConfig &config) {
    if (!TryInitializeNextGenerationAssemblyOnDevice(buffers, config)) {
        return false;
    }

    if (TryContinueNextGenerationAssemblyOnDevice(buffers, generation_seed, config, 0)) {
        return true;
    }

    int status_value = DeviceStatusValue(DeviceSlabRuntimeStatusCode::kCudaFailure);
    if (ReadDeviceStatus(buffers, status_value) &&
        (status_value == DeviceStatusValue(DeviceSlabRuntimeStatusCode::kSlabFull))) {
        (void)CleanupFailedAssemblyOnDeviceInternal(buffers);
    }

    return false;
}

bool TryReadDeviceSlabRuntimeStatus(const DeviceSlabRuntimeBuffers &buffers, DeviceSlabRuntimeStatusCode &status_code) {
    int status_value = 0;
    if (!ReadDeviceStatus(buffers, status_value)) {
        return false;
    }

    status_code = static_cast<DeviceSlabRuntimeStatusCode>(status_value);
    return true;
}

const char *DeviceSlabRuntimeStatusCodeString(const DeviceSlabRuntimeStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceSlabRuntimeStatusCode::kOk:
        return "ok";
    case DeviceSlabRuntimeStatusCode::kCudaFailure:
        return "cuda failure";
    case DeviceSlabRuntimeStatusCode::kInvalidRuntimeConfig:
        return "invalid device slab runtime config";
    case DeviceSlabRuntimeStatusCode::kInvalidSlab:
        return "invalid genotype slab state";
    case DeviceSlabRuntimeStatusCode::kInvalidGeneration:
        return "invalid slab generation state";
    case DeviceSlabRuntimeStatusCode::kInvalidAssemblyPlan:
        return "invalid slab assembly plan";
    case DeviceSlabRuntimeStatusCode::kInvalidAssemblyConfig:
        return "invalid slab assembly config";
    case DeviceSlabRuntimeStatusCode::kInvalidParentIndex:
        return "invalid parent index in slab assembly plan";
    case DeviceSlabRuntimeStatusCode::kSlabFull:
        return "slab is genuinely full";
    case DeviceSlabRuntimeStatusCode::kOutputEmbeddingInjectionFailed:
        return "device output-embedding injection failed";
    case DeviceSlabRuntimeStatusCode::kSlabRepackFailed:
        return "slab compaction/repacking failed";
    }

    return "unknown device slab runtime status";
}

} // namespace neuroevolution::genetic_algorithm::genotype_slab::device
