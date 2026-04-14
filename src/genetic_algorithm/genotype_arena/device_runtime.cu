#include "genetic_algorithm/genotype_arena/device_runtime.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "genetic_algorithm/device/genome_ops.cuh"

namespace neuroevolution::genetic_algorithm::genotype_arena::device {

namespace {

using device_genome_ops::BreedAndMutateGenome;
using device_genome_ops::DeviceRandomState;
using device_genome_ops::MakeDeviceRandomState;

constexpr std::size_t kArenaRuntimeThreadBlockSize = 1;

NEUROEVOLUTION_HOST_DEVICE constexpr int DeviceStatusValue(const DeviceArenaRuntimeStatusCode status_code) {
    return static_cast<int>(status_code);
}

inline bool CheckCuda(const cudaError_t error) { return error == cudaSuccess; }

inline bool IsGenerationCompatibleWithArena(const ArenaGeneration &generation,
                                            const DeviceArenaRuntimeBuffers &buffers) noexcept {
    if (!IsValidArenaGeneration(generation) || (generation.active_individual_count > buffers.max_generation_size) ||
        !IsValidGenotypeArenaLayout(buffers.arena_layout)) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        const std::uint32_t slot_index = generation.slot_indices[individual_index];
        if ((slot_index != kInvalidArenaSlotIndex) && (slot_index >= buffers.arena_layout.slot_count)) {
            return false;
        }
    }

    return true;
}

inline bool ReadDeviceStatus(const DeviceArenaRuntimeBuffers &buffers, int &status_value) {
    return CheckCuda(cudaMemcpy(&status_value, buffers.status, sizeof(int), cudaMemcpyDeviceToHost));
}

inline bool WriteDeviceStatus(const DeviceArenaRuntimeBuffers &buffers,
                              const DeviceArenaRuntimeStatusCode status_code) {
    const int status_value = DeviceStatusValue(status_code);
    return CheckCuda(cudaMemcpy(buffers.status, &status_value, sizeof(int), cudaMemcpyHostToDevice));
}

inline bool ResetDeviceStatus(const DeviceArenaRuntimeBuffers &buffers) {
    return WriteDeviceStatus(buffers, DeviceArenaRuntimeStatusCode::kOk);
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

inline NEUROEVOLUTION_HOST_DEVICE ArenaGenerationView MakeDeviceGenerationView(
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

__device__ void SetFailureStatus(int *status, const DeviceArenaRuntimeStatusCode status_code) {
    atomicCAS(status, DeviceStatusValue(DeviceArenaRuntimeStatusCode::kOk), DeviceStatusValue(status_code));
}

__device__ bool IsValidParentIndex(const ArenaGenerationView &generation, const std::uint32_t parent_index) {
    return IsValidArenaGenerationView(generation) && (parent_index < generation.active_individual_count) &&
           (generation.slot_indices[parent_index] != kInvalidArenaSlotIndex);
}

__device__ bool TryConsumeParentDuty(GenotypeArenaView arena, ArenaGenerationView current_generation,
                                     std::uint32_t *remaining_parent_duties, const std::uint32_t parent_index) {
    if (!IsValidParentIndex(current_generation, parent_index) || (remaining_parent_duties == nullptr) ||
        (remaining_parent_duties[parent_index] == 0)) {
        return false;
    }

    --remaining_parent_duties[parent_index];
    if (remaining_parent_duties[parent_index] == 0) {
        if (!TryReleaseArenaSlot(arena, current_generation.slot_indices[parent_index])) {
            return false;
        }

        ClearArenaGenerationSlot(current_generation, parent_index);
    }

    return true;
}

__global__ void AssembleNextGenerationWithoutElitismKernel(
    std::uint8_t *arena_storage, ArenaSlotState *slot_states, std::uint32_t *free_slot_stack,
    std::size_t *free_slot_count, const GenotypeArenaLayout arena_layout, std::uint32_t *current_slot_indices,
    float *current_fitness, std::uint32_t *current_evaluation_counts, std::uint8_t *current_has_fitness,
    const std::size_t current_generation_index, const std::size_t current_generation_size,
    std::uint32_t *next_slot_indices, float *next_fitness, std::uint32_t *next_evaluation_counts,
    std::uint8_t *next_has_fitness, const std::size_t next_generation_index, const std::size_t next_generation_size,
    const ArenaParentPair *parent_pairs, std::uint32_t *remaining_parent_duties, const std::uint32_t generation_seed,
    const ArenaDeviceAssemblyConfig config, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    GenotypeArenaView arena{
        .layout = arena_layout,
        .storage = arena_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
    };
    ArenaGenerationView current_generation =
        MakeDeviceGenerationView(current_generation_index, current_generation_size, current_slot_indices,
                                 current_fitness, current_evaluation_counts, current_has_fitness);
    ArenaGenerationView next_generation =
        MakeDeviceGenerationView(next_generation_index, next_generation_size, next_slot_indices, next_fitness,
                                 next_evaluation_counts, next_has_fitness);

    if (!IsValidGenotypeArenaView(arena)) {
        SetFailureStatus(status, DeviceArenaRuntimeStatusCode::kInvalidArena);
        return;
    }

    if (!IsValidArenaGenerationView(current_generation) || !IsValidArenaGenerationView(next_generation)) {
        SetFailureStatus(status, DeviceArenaRuntimeStatusCode::kInvalidGeneration);
        return;
    }

    if ((parent_pairs == nullptr) || (remaining_parent_duties == nullptr) || (next_generation_size == 0)) {
        SetFailureStatus(status, DeviceArenaRuntimeStatusCode::kInvalidAssemblyPlan);
        return;
    }

    if (!IsValidArenaDeviceAssemblyConfig(config)) {
        SetFailureStatus(status, DeviceArenaRuntimeStatusCode::kInvalidAssemblyConfig);
        return;
    }

    for (std::size_t child_index = 0; child_index < next_generation_size; ++child_index) {
        next_generation.slot_indices[child_index] = kInvalidArenaSlotIndex;
        next_generation.fitness[child_index] = 0.0f;
        next_generation.evaluation_counts[child_index] = 0;
        next_generation.has_fitness[child_index] = 0;
    }

    for (std::size_t parent_index = 0; parent_index < current_generation_size; ++parent_index) {
        remaining_parent_duties[parent_index] = 0;
    }

    for (std::size_t child_index = 0; child_index < next_generation_size; ++child_index) {
        const ArenaParentPair &parent_pair = parent_pairs[child_index];
        if (!IsValidParentIndex(current_generation, parent_pair.first_parent_index) ||
            !IsValidParentIndex(current_generation, parent_pair.second_parent_index)) {
            SetFailureStatus(status, DeviceArenaRuntimeStatusCode::kInvalidParentIndex);
            return;
        }

        ++remaining_parent_duties[parent_pair.first_parent_index];
        ++remaining_parent_duties[parent_pair.second_parent_index];
    }

    for (std::size_t parent_index = 0; parent_index < current_generation_size; ++parent_index) {
        if ((remaining_parent_duties[parent_index] == 0) &&
            (current_generation.slot_indices[parent_index] != kInvalidArenaSlotIndex)) {
            if (!TryReleaseArenaSlot(arena, current_generation.slot_indices[parent_index])) {
                SetFailureStatus(status, DeviceArenaRuntimeStatusCode::kInvalidArena);
                return;
            }

            ClearArenaGenerationSlot(current_generation, parent_index);
        }
    }

    for (std::size_t child_index = 0; child_index < next_generation_size; ++child_index) {
        const ArenaParentPair &parent_pair = parent_pairs[child_index];
        const std::uint32_t first_parent_slot = current_generation.slot_indices[parent_pair.first_parent_index];
        const std::uint32_t second_parent_slot = current_generation.slot_indices[parent_pair.second_parent_index];
        if ((first_parent_slot == kInvalidArenaSlotIndex) || (second_parent_slot == kInvalidArenaSlotIndex)) {
            SetFailureStatus(status, DeviceArenaRuntimeStatusCode::kInvalidParentIndex);
            return;
        }

        std::uint32_t child_slot = kInvalidArenaSlotIndex;
        if (!TryAllocateArenaSlot(arena, child_slot)) {
            SetFailureStatus(status, DeviceArenaRuntimeStatusCode::kArenaFull);
            return;
        }

        DeviceRandomState random_state = MakeDeviceRandomState(
            generation_seed, static_cast<std::uint32_t>(child_index + (current_generation_index * 4099U)));
        BreedAndMutateGenome(ArenaSlotBytesAt(arena.storage, arena.layout, first_parent_slot),
                             ArenaSlotBytesAt(arena.storage, arena.layout, second_parent_slot),
                             arena.layout.action_count, ArenaSlotBytesAt(arena.storage, arena.layout, child_slot),
                             random_state, config.breeding, config.mutation);
        next_generation.slot_indices[child_index] = child_slot;

        if (!TryConsumeParentDuty(arena, current_generation, remaining_parent_duties, parent_pair.first_parent_index) ||
            !TryConsumeParentDuty(arena, current_generation, remaining_parent_duties,
                                  parent_pair.second_parent_index)) {
            SetFailureStatus(status, DeviceArenaRuntimeStatusCode::kInvalidArena);
            return;
        }
    }
}

} // namespace

bool TryCreateDeviceArenaRuntimeBuffers(DeviceArenaRuntimeBuffers &buffers, const DeviceArenaRuntimeConfig &config) {
    buffers = {};
    if (!IsValidDeviceArenaRuntimeConfig(config)) {
        return false;
    }

    buffers.arena_layout.action_count = config.action_count;
    buffers.arena_layout.slot_stride_bytes = ComputeArenaSlotStrideBytes(config.action_count);
    buffers.arena_layout.slot_count = config.slot_count;
    buffers.arena_layout.arena_bytes = buffers.arena_layout.slot_stride_bytes * buffers.arena_layout.slot_count;
    buffers.max_generation_size = config.max_generation_size;
    if (!IsValidGenotypeArenaLayout(buffers.arena_layout)) {
        buffers = {};
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMalloc(&buffers.arena_storage, buffers.arena_layout.arena_bytes));
    ok &= CheckCuda(cudaMalloc(&buffers.slot_states, buffers.arena_layout.slot_count * sizeof(ArenaSlotState)));
    ok &= CheckCuda(cudaMalloc(&buffers.free_slot_stack, buffers.arena_layout.slot_count * sizeof(std::uint32_t)));
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

    ok &= CheckCuda(cudaMalloc(&buffers.assembly_parent_pairs, buffers.max_generation_size * sizeof(ArenaParentPair)));
    ok &= CheckCuda(cudaMalloc(&buffers.remaining_parent_duties, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.status, sizeof(int)));
    if (!ok) {
        DestroyDeviceArenaRuntimeBuffers(buffers);
        return false;
    }

    ok &= CheckCuda(cudaMemset(buffers.arena_storage, 0, buffers.arena_layout.arena_bytes));
    ok &= CheckCuda(cudaMemset(buffers.slot_states, 0, buffers.arena_layout.slot_count * sizeof(ArenaSlotState)));
    ok &= CheckCuda(cudaMemset(buffers.free_slot_stack, 0, buffers.arena_layout.slot_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.free_slot_count, 0, sizeof(std::size_t)));
    ok &=
        ClearGenerationBuffers(buffers.current_slot_indices, buffers.current_fitness, buffers.current_evaluation_counts,
                               buffers.current_has_fitness, buffers.max_generation_size);
    ok &= ClearGenerationBuffers(buffers.next_slot_indices, buffers.next_fitness, buffers.next_evaluation_counts,
                                 buffers.next_has_fitness, buffers.max_generation_size);
    ok &=
        CheckCuda(cudaMemset(buffers.assembly_parent_pairs, 0, buffers.max_generation_size * sizeof(ArenaParentPair)));
    ok &=
        CheckCuda(cudaMemset(buffers.remaining_parent_duties, 0, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    if (!ok) {
        DestroyDeviceArenaRuntimeBuffers(buffers);
        return false;
    }

    return true;
}

void DestroyDeviceArenaRuntimeBuffers(DeviceArenaRuntimeBuffers &buffers) noexcept {
    cudaFree(buffers.arena_storage);
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
    cudaFree(buffers.remaining_parent_duties);
    cudaFree(buffers.status);
    buffers = {};
}

bool TryUploadArenaToDevice(const HostGenotypeArena &host_arena, DeviceArenaRuntimeBuffers &buffers) {
    if (!IsValidHostGenotypeArena(host_arena) ||
        (host_arena.layout.action_count != buffers.arena_layout.action_count) ||
        (host_arena.layout.slot_count != buffers.arena_layout.slot_count) ||
        (host_arena.layout.slot_stride_bytes != buffers.arena_layout.slot_stride_bytes)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemcpy(buffers.arena_storage, host_arena.storage.get(), host_arena.layout.arena_bytes,
                               cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.slot_states, host_arena.slot_states.get(),
                               host_arena.layout.slot_count * sizeof(ArenaSlotState), cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.free_slot_stack, host_arena.free_slot_stack.get(),
                               host_arena.layout.slot_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    ok &= CheckCuda(
        cudaMemcpy(buffers.free_slot_count, &host_arena.free_slot_count, sizeof(std::size_t), cudaMemcpyHostToDevice));
    return ok;
}

bool TryDownloadArenaFromDevice(const DeviceArenaRuntimeBuffers &buffers, HostGenotypeArena &host_arena) {
    if (!IsValidGenotypeArenaLayout(buffers.arena_layout) || (buffers.arena_layout.slot_count == 0)) {
        return false;
    }

    if (!TryCreateHostGenotypeArena(host_arena, buffers.arena_layout.slot_count, buffers.arena_layout.action_count)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemcpy(host_arena.storage.get(), buffers.arena_storage, host_arena.layout.arena_bytes,
                               cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(host_arena.slot_states.get(), buffers.slot_states,
                               host_arena.layout.slot_count * sizeof(ArenaSlotState), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(host_arena.free_slot_stack.get(), buffers.free_slot_stack,
                               host_arena.layout.slot_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(
        cudaMemcpy(&host_arena.free_slot_count, buffers.free_slot_count, sizeof(std::size_t), cudaMemcpyDeviceToHost));
    return ok;
}

bool TryUploadCurrentGenerationToDevice(const ArenaGeneration &generation, DeviceArenaRuntimeBuffers &buffers) {
    if (!IsGenerationCompatibleWithArena(generation, buffers)) {
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

bool TryDownloadCurrentGenerationFromDevice(const DeviceArenaRuntimeBuffers &buffers, ArenaGeneration &generation) {
    if (buffers.current_generation_size == 0) {
        return false;
    }

    if (!TryCreateArenaGeneration(generation, buffers.current_generation_size, buffers.current_generation_index)) {
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

bool TryDownloadNextGenerationFromDevice(const DeviceArenaRuntimeBuffers &buffers, ArenaGeneration &generation) {
    if (buffers.next_generation_size == 0) {
        return false;
    }

    if (!TryCreateArenaGeneration(generation, buffers.next_generation_size, buffers.next_generation_index)) {
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

bool TryUploadAssemblyPlanToDevice(const ArenaAssemblyPlan &plan, DeviceArenaRuntimeBuffers &buffers) {
    if (!IsValidArenaAssemblyPlan(plan) || (plan.child_count > buffers.max_generation_size)) {
        return false;
    }

    buffers.planned_child_count = plan.child_count;
    return CheckCuda(cudaMemcpy(buffers.assembly_parent_pairs, plan.parent_pairs.get(),
                                plan.child_count * sizeof(ArenaParentPair), cudaMemcpyHostToDevice));
}

bool TryAssembleNextGenerationWithoutElitismOnDevice(DeviceArenaRuntimeBuffers &buffers,
                                                     const std::uint32_t generation_seed,
                                                     const ArenaDeviceAssemblyConfig &config) {
    if (!IsValidArenaDeviceAssemblyConfig(config) || !IsValidGenotypeArenaLayout(buffers.arena_layout) ||
        (buffers.current_generation_size == 0) || (buffers.planned_child_count == 0)) {
        (void)WriteDeviceStatus(buffers, DeviceArenaRuntimeStatusCode::kInvalidAssemblyConfig);
        return false;
    }

    buffers.next_generation_index = buffers.current_generation_index + 1;
    buffers.next_generation_size = buffers.planned_child_count;

    bool ok = true;
    ok &= ResetDeviceStatus(buffers);
    ok &= ClearGenerationBuffers(buffers.next_slot_indices, buffers.next_fitness, buffers.next_evaluation_counts,
                                 buffers.next_has_fitness, buffers.max_generation_size);
    ok &=
        CheckCuda(cudaMemset(buffers.remaining_parent_duties, 0, buffers.max_generation_size * sizeof(std::uint32_t)));
    if (!ok) {
        return false;
    }

    AssembleNextGenerationWithoutElitismKernel<<<1, kArenaRuntimeThreadBlockSize>>>(
        buffers.arena_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
        buffers.arena_layout, buffers.current_slot_indices, buffers.current_fitness, buffers.current_evaluation_counts,
        buffers.current_has_fitness, buffers.current_generation_index, buffers.current_generation_size,
        buffers.next_slot_indices, buffers.next_fitness, buffers.next_evaluation_counts, buffers.next_has_fitness,
        buffers.next_generation_index, buffers.next_generation_size, buffers.assembly_parent_pairs,
        buffers.remaining_parent_duties, generation_seed, config, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    int status_value = DeviceStatusValue(DeviceArenaRuntimeStatusCode::kCudaFailure);
    return ReadDeviceStatus(buffers, status_value) &&
           (status_value == DeviceStatusValue(DeviceArenaRuntimeStatusCode::kOk));
}

bool TryReadDeviceArenaRuntimeStatus(const DeviceArenaRuntimeBuffers &buffers,
                                     DeviceArenaRuntimeStatusCode &status_code) {
    int status_value = 0;
    if (!ReadDeviceStatus(buffers, status_value)) {
        return false;
    }

    status_code = static_cast<DeviceArenaRuntimeStatusCode>(status_value);
    return true;
}

const char *DeviceArenaRuntimeStatusCodeString(const DeviceArenaRuntimeStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceArenaRuntimeStatusCode::kOk:
        return "ok";
    case DeviceArenaRuntimeStatusCode::kCudaFailure:
        return "cuda failure";
    case DeviceArenaRuntimeStatusCode::kInvalidRuntimeConfig:
        return "invalid device arena runtime config";
    case DeviceArenaRuntimeStatusCode::kInvalidArena:
        return "invalid genotype arena state";
    case DeviceArenaRuntimeStatusCode::kInvalidGeneration:
        return "invalid arena generation state";
    case DeviceArenaRuntimeStatusCode::kInvalidAssemblyPlan:
        return "invalid arena assembly plan";
    case DeviceArenaRuntimeStatusCode::kInvalidAssemblyConfig:
        return "invalid arena assembly config";
    case DeviceArenaRuntimeStatusCode::kInvalidParentIndex:
        return "invalid parent index in arena assembly plan";
    case DeviceArenaRuntimeStatusCode::kArenaFull:
        return "arena is genuinely full";
    }

    return "unknown device arena runtime status";
}

} // namespace neuroevolution::genetic_algorithm::genotype_arena::device
