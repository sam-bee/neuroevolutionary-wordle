#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/genotype_slab/device_runtime.hpp"
#include "genetic_algorithm/genotype_slab/reference_counter.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::common::ToFloat16;
using neuroevolution::genetic_algorithm::genotype_slab::GenomePolicyModelParameters;
using neuroevolution::genetic_algorithm::genotype_slab::GenomeTailRows;
using neuroevolution::genetic_algorithm::genotype_slab::GenotypeSlabLayout;
using neuroevolution::genetic_algorithm::genotype_slab::GenotypeSlabView;
using neuroevolution::genetic_algorithm::genotype_slab::HostGenotypeSlab;
using neuroevolution::genetic_algorithm::genotype_slab::HostSlabSlotBytesAt;
using neuroevolution::genetic_algorithm::genotype_slab::kInvalidSlabSlotIndex;
using neuroevolution::genetic_algorithm::genotype_slab::SlabAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_slab::SlabGeneration;
using neuroevolution::genetic_algorithm::genotype_slab::SlabSlotState;
using neuroevolution::genetic_algorithm::genotype_slab::TryAllocateSlabSlot;
using neuroevolution::genetic_algorithm::genotype_slab::TryCreateHostGenotypeSlab;
using neuroevolution::genetic_algorithm::genotype_slab::TryCreateSlabAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_slab::TryCreateSlabGeneration;
using neuroevolution::genetic_algorithm::genotype_slab::TryDecrementParentReferenceCount;
using neuroevolution::genetic_algorithm::genotype_slab::TryIncrementParentReferenceCount;
using neuroevolution::genetic_algorithm::genotype_slab::TryReleaseSlabSlot;
using neuroevolution::genetic_algorithm::genotype_slab::TryRetainSlabSlot;
using neuroevolution::genetic_algorithm::genotype_slab::TrySetSlabGenerationSlot;
using neuroevolution::genetic_algorithm::genotype_slab::device::DestroyDeviceSlabRuntimeBuffers;
using neuroevolution::genetic_algorithm::genotype_slab::device::DeviceSlabRuntimeBuffers;
using neuroevolution::genetic_algorithm::genotype_slab::device::DeviceSlabRuntimeConfig;
using neuroevolution::genetic_algorithm::genotype_slab::device::DeviceSlabRuntimeStatusCode;
using neuroevolution::genetic_algorithm::genotype_slab::device::SlabDeviceAssemblyConfig;
using neuroevolution::genetic_algorithm::genotype_slab::device::TryAssembleNextGenerationOnDevice;
using neuroevolution::genetic_algorithm::genotype_slab::device::TryCreateDeviceSlabRuntimeBuffers;
using neuroevolution::genetic_algorithm::genotype_slab::device::TryDownloadCurrentGenerationFromDevice;
using neuroevolution::genetic_algorithm::genotype_slab::device::TryDownloadNextGenerationFromDevice;
using neuroevolution::genetic_algorithm::genotype_slab::device::TryDownloadSlabFromDevice;
using neuroevolution::genetic_algorithm::genotype_slab::device::TryPrepareSlabForExpandedActionCountOnDevice;
using neuroevolution::genetic_algorithm::genotype_slab::device::TryPrioritizeSlabAssemblyPlanForParentReleaseOnDevice;
using neuroevolution::genetic_algorithm::genotype_slab::device::TryReadDeviceSlabRuntimeStatus;
using neuroevolution::genetic_algorithm::genotype_slab::device::TryUploadAssemblyPlanToDevice;
using neuroevolution::genetic_algorithm::genotype_slab::device::TryUploadCurrentGenerationToDevice;
using neuroevolution::genetic_algorithm::genotype_slab::device::TryUploadSlabToDevice;

constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr float kTolerance = 1.0e-3f;

bool CheckCuda(const cudaError_t error, const std::string_view action) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA failure during " << action << ": " << cudaGetErrorString(error) << '\n';
        return false;
    }

    return true;
}

bool SelectVisibleCudaDevice() {
    int device_count = 0;
    if (!CheckCuda(cudaGetDeviceCount(&device_count), "querying visible CUDA device count")) {
        return false;
    }

    if (device_count <= kSelectedVisibleDeviceIndex) {
        std::cerr << "FAIL: selected logical device index " << kSelectedVisibleDeviceIndex
                  << " is not available in this process\n";
        return false;
    }

    return CheckCuda(cudaSetDevice(kSelectedVisibleDeviceIndex), "selecting visible CUDA device");
}

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

bool ExpectNear(const float actual, const float expected, const std::string_view label) {
    if (std::fabs(actual - expected) > kTolerance) {
        std::cerr << "FAIL: " << label << " expected " << expected << ", got " << actual << '\n';
        return false;
    }

    return true;
}

SlabDeviceAssemblyConfig MakeDeterministicAssemblyConfig() {
    SlabDeviceAssemblyConfig config{};
    config.breeding.first_parent_probability = 1.0f;
    config.mutation.mutation_probability = 0.0f;
    config.mutation.mutation_sigma = 0.0f;
    return config;
}

__global__ void ReferenceCounterWarpContentionKernel(std::uint32_t *parent_reference_counts,
                                                     std::uint32_t *final_release_counts, int *status) {
    constexpr unsigned kWarpMask = 0xFFFFFFFFU;
    if ((blockIdx.x != 0) || (threadIdx.x >= 32)) {
        return;
    }

    const std::uint32_t lane_index = threadIdx.x;
    if (!TryIncrementParentReferenceCount(parent_reference_counts, 0)) {
        atomicCAS(status, 0, 1);
    }

    if ((lane_index % 2U) == 0) {
        if (!TryIncrementParentReferenceCount(parent_reference_counts, 1)) {
            atomicCAS(status, 0, 2);
        }
    }

    __syncwarp(kWarpMask);
    if (lane_index == 0) {
        if ((parent_reference_counts[0] != 32U) || (parent_reference_counts[1] != 16U) ||
            (parent_reference_counts[2] != 0U)) {
            atomicCAS(status, 0, 3);
        }
    }

    __syncwarp(kWarpMask);
    std::uint32_t previous_count = 0;
    if (!TryDecrementParentReferenceCount(parent_reference_counts, 0, previous_count)) {
        atomicCAS(status, 0, 4);
    } else if (previous_count == 1U) {
        atomicAdd(&final_release_counts[0], 1U);
    }

    if ((lane_index % 2U) == 0) {
        previous_count = 0;
        if (!TryDecrementParentReferenceCount(parent_reference_counts, 1, previous_count)) {
            atomicCAS(status, 0, 5);
        } else if (previous_count == 1U) {
            atomicAdd(&final_release_counts[1], 1U);
        }
    }

    __syncwarp(kWarpMask);
    previous_count = 0;
    if (TryDecrementParentReferenceCount(parent_reference_counts, 2, previous_count)) {
        atomicCAS(status, 0, 6);
    }

    __syncwarp(kWarpMask);
    if (lane_index == 0) {
        if ((parent_reference_counts[0] != 0U) || (parent_reference_counts[1] != 0U) ||
            (parent_reference_counts[2] != 0U) || (final_release_counts[0] != 1U) || (final_release_counts[1] != 1U)) {
            atomicCAS(status, 0, 7);
        }
    }
}

bool TestDeviceReferenceCountersAreAtomicUnderWarpContention() {
    std::uint32_t *device_parent_reference_counts = nullptr;
    std::uint32_t *device_final_release_counts = nullptr;
    int *device_status = nullptr;

    bool ok = true;
    ok &= CheckCuda(cudaMalloc(&device_parent_reference_counts, 3 * sizeof(std::uint32_t)),
                    "allocating device reference counts");
    ok &= CheckCuda(cudaMalloc(&device_final_release_counts, 2 * sizeof(std::uint32_t)),
                    "allocating device final-release counts");
    ok &= CheckCuda(cudaMalloc(&device_status, sizeof(int)), "allocating device reference-counter status");
    ok &= CheckCuda(cudaMemset(device_parent_reference_counts, 0, 3 * sizeof(std::uint32_t)),
                    "clearing device reference counts");
    ok &= CheckCuda(cudaMemset(device_final_release_counts, 0, 2 * sizeof(std::uint32_t)),
                    "clearing device final-release counts");
    ok &= CheckCuda(cudaMemset(device_status, 0, sizeof(int)), "clearing device reference-counter status");
    if (!ok) {
        cudaFree(device_parent_reference_counts);
        cudaFree(device_final_release_counts);
        cudaFree(device_status);
        return false;
    }

    ReferenceCounterWarpContentionKernel<<<1, 32>>>(device_parent_reference_counts, device_final_release_counts,
                                                    device_status);
    ok &= CheckCuda(cudaGetLastError(), "launching reference-counter warp contention kernel");
    ok &= CheckCuda(cudaDeviceSynchronize(), "running reference-counter warp contention kernel");

    std::uint32_t host_parent_reference_counts[3]{};
    std::uint32_t host_final_release_counts[2]{};
    int host_status = 0;
    ok &= CheckCuda(cudaMemcpy(host_parent_reference_counts, device_parent_reference_counts,
                               sizeof(host_parent_reference_counts), cudaMemcpyDeviceToHost),
                    "copying reference counts after warp contention");
    ok &= CheckCuda(cudaMemcpy(host_final_release_counts, device_final_release_counts,
                               sizeof(host_final_release_counts), cudaMemcpyDeviceToHost),
                    "copying final-release counts after warp contention");
    ok &= CheckCuda(cudaMemcpy(&host_status, device_status, sizeof(int), cudaMemcpyDeviceToHost),
                    "copying reference-counter warp status");

    cudaFree(device_parent_reference_counts);
    cudaFree(device_final_release_counts);
    cudaFree(device_status);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(host_status == 0, "Expected atomic reference-counter warp contention status to stay ok");
    ok &= ExpectTrue((host_parent_reference_counts[0] == 0U) && (host_parent_reference_counts[1] == 0U) &&
                         (host_parent_reference_counts[2] == 0U),
                     "Expected warp-contended reference counters to return to zero");
    ok &= ExpectTrue((host_final_release_counts[0] == 1U) && (host_final_release_counts[1] == 1U),
                     "Expected exactly one final-release observation per contended counter");
    return ok;
}

__global__ void SlabFreeListWarpContentionKernel(std::uint8_t *slab_storage, SlabSlotState *slot_states,
                                                 std::uint32_t *free_slot_stack, std::uint32_t *free_slot_count,
                                                 std::uint32_t *free_slot_lock, const GenotypeSlabLayout slab_layout,
                                                 std::uint32_t *allocated_slots, std::uint32_t *slot_hit_counts,
                                                 int *status) {
    constexpr unsigned kWarpMask = 0xFFFFFFFFU;
    if ((blockIdx.x != 0) || (threadIdx.x >= 32)) {
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

    const std::uint32_t lane_index = threadIdx.x;
    std::uint32_t slot_index = kInvalidSlabSlotIndex;
    if (!TryAllocateSlabSlot(slab, slot_index)) {
        atomicCAS(status, 0, 1);
        allocated_slots[lane_index] = kInvalidSlabSlotIndex;
        return;
    }

    allocated_slots[lane_index] = slot_index;
    if (slot_index >= slab_layout.slot_count) {
        atomicCAS(status, 0, 2);
        return;
    }

    atomicAdd(&slot_hit_counts[slot_index], 1U);
    __syncwarp(kWarpMask);

    if (lane_index == 0) {
        if (*free_slot_count != 0U) {
            atomicCAS(status, 0, 3);
        }

        for (std::uint32_t checked_slot = 0; checked_slot < slab_layout.slot_count; ++checked_slot) {
            if (slot_hit_counts[checked_slot] != 1U) {
                atomicCAS(status, 0, 4);
            }

            if (!slot_states[checked_slot].occupied || (slot_states[checked_slot].reference_count != 1U)) {
                atomicCAS(status, 0, 5);
            }
        }
    }

    __syncwarp(kWarpMask);
    if (!TryReleaseSlabSlot(slab, allocated_slots[lane_index])) {
        atomicCAS(status, 0, 6);
    }

    __syncwarp(kWarpMask);
    if (lane_index == 0) {
        if (*free_slot_count != slab_layout.slot_count) {
            atomicCAS(status, 0, 7);
        }

        for (std::uint32_t checked_slot = 0; checked_slot < slab_layout.slot_count; ++checked_slot) {
            if (slot_states[checked_slot].occupied || (slot_states[checked_slot].reference_count != 0U)) {
                atomicCAS(status, 0, 8);
            }
        }
    }
}

bool TestDeviceSlabFreeListIsThreadSafeUnderWarpContention() {
    constexpr std::size_t kSlotCount = 32;
    constexpr std::size_t kActionCount = 4;

    HostGenotypeSlab host_buffer{};
    bool ok = TryCreateHostGenotypeSlab(host_buffer, kSlotCount, kActionCount);
    if (!ok) {
        std::cerr << "FAIL: could not allocate free-list contention host slab\n";
        return false;
    }

    DeviceSlabRuntimeConfig runtime_config{};
    runtime_config.slot_count = kSlotCount;
    runtime_config.action_count = kActionCount;
    runtime_config.max_generation_size = kSlotCount;

    DeviceSlabRuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadSlabToDevice(host_buffer, buffers);

    std::uint32_t *device_allocated_slots = nullptr;
    std::uint32_t *device_slot_hit_counts = nullptr;
    int *device_status = nullptr;
    ok &= CheckCuda(cudaMalloc(&device_allocated_slots, kSlotCount * sizeof(std::uint32_t)),
                    "allocating device allocated-slot records");
    ok &= CheckCuda(cudaMalloc(&device_slot_hit_counts, kSlotCount * sizeof(std::uint32_t)),
                    "allocating device slot-hit counts");
    ok &= CheckCuda(cudaMalloc(&device_status, sizeof(int)), "allocating device free-list status");
    ok &= CheckCuda(cudaMemset(device_allocated_slots, 0xFF, kSlotCount * sizeof(std::uint32_t)),
                    "clearing allocated-slot records");
    ok &= CheckCuda(cudaMemset(device_slot_hit_counts, 0, kSlotCount * sizeof(std::uint32_t)),
                    "clearing slot-hit counts");
    ok &= CheckCuda(cudaMemset(device_status, 0, sizeof(int)), "clearing free-list status");
    if (!ok) {
        cudaFree(device_allocated_slots);
        cudaFree(device_slot_hit_counts);
        cudaFree(device_status);
        DestroyDeviceSlabRuntimeBuffers(buffers);
        return false;
    }

    SlabFreeListWarpContentionKernel<<<1, 32>>>(buffers.slab_storage, buffers.slot_states, buffers.free_slot_stack,
                                                buffers.free_slot_count, buffers.free_slot_lock, buffers.slab_layout,
                                                device_allocated_slots, device_slot_hit_counts, device_status);
    ok &= CheckCuda(cudaGetLastError(), "launching free-list warp contention kernel");
    ok &= CheckCuda(cudaDeviceSynchronize(), "running free-list warp contention kernel");

    int host_status = 0;
    std::uint32_t host_slot_hit_counts[kSlotCount]{};
    ok &= CheckCuda(cudaMemcpy(&host_status, device_status, sizeof(int), cudaMemcpyDeviceToHost),
                    "copying free-list contention status");
    ok &= CheckCuda(
        cudaMemcpy(host_slot_hit_counts, device_slot_hit_counts, sizeof(host_slot_hit_counts), cudaMemcpyDeviceToHost),
        "copying free-list contention slot-hit counts");

    HostGenotypeSlab downloaded_buffer{};
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);

    cudaFree(device_allocated_slots);
    cudaFree(device_slot_hit_counts);
    cudaFree(device_status);
    DestroyDeviceSlabRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(host_status == 0, "Expected free-list warp contention status to stay ok");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == kSlotCount,
                     "Expected all contended slots to return to the genotype slab");
    for (std::size_t slot_index = 0; slot_index < kSlotCount; ++slot_index) {
        ok &= ExpectTrue(host_slot_hit_counts[slot_index] == 1U,
                         "Expected each genotype slab slot to be allocated exactly once under warp contention");
        ok &= ExpectTrue(!downloaded_buffer.slot_states[slot_index].occupied,
                         "Expected released contended slot to be unoccupied");
        ok &= ExpectTrue(downloaded_buffer.slot_states[slot_index].reference_count == 0U,
                         "Expected released contended slot to have no references");
    }

    return ok;
}

bool TestDeviceSlabRuntimeUploadsAndDownloadsBufferAndGenerationState() {
    HostGenotypeSlab host_buffer{};
    SlabGeneration current_generation{};
    bool ok = TryCreateHostGenotypeSlab(host_buffer, 3, 4);
    ok &= TryCreateSlabGeneration(current_generation, 2, 7);
    if (!ok) {
        std::cerr << "FAIL: could not allocate upload/download fixtures\n";
        return false;
    }

    std::uint32_t slot0 = 0;
    std::uint32_t slot1 = 0;
    ok &= TryAllocateSlabSlot(host_buffer, slot0);
    ok &= TryAllocateSlabSlot(host_buffer, slot1);
    ok &= TryRetainSlabSlot(host_buffer, slot0);
    ok &= TryReleaseSlabSlot(host_buffer, slot1);
    ok &= TrySetSlabGenerationSlot(current_generation, 0, slot0);
    current_generation.fitness[0] = 9.0f;
    current_generation.evaluation_counts[0] = 3;
    current_generation.has_fitness[0] = 1;
    GenomePolicyModelParameters(HostSlabSlotBytesAt(host_buffer, slot0)).dense_trunk.hidden1_to_output.biases[0] = 3.5f;
    GenomeTailRows(HostSlabSlotBytesAt(host_buffer, slot0))[1][0] = ToFloat16(-1.25f);

    DeviceSlabRuntimeConfig runtime_config{};
    runtime_config.slot_count = 3;
    runtime_config.action_count = 4;
    runtime_config.max_generation_size = 3;

    DeviceSlabRuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadSlabToDevice(host_buffer, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    if (!ok) {
        DestroyDeviceSlabRuntimeBuffers(buffers);
        std::cerr << "FAIL: could not upload slab runtime fixtures\n";
        return false;
    }

    HostGenotypeSlab downloaded_buffer{};
    SlabGeneration downloaded_generation{};
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_generation);
    DestroyDeviceSlabRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(downloaded_buffer.free_slot_count == 2, "Expected downloaded slab to preserve free-slot count");
    ok &= ExpectTrue(downloaded_buffer.slot_states[slot0].occupied,
                     "Expected downloaded slab to preserve live slot state");
    ok &= ExpectTrue(downloaded_buffer.slot_states[slot0].reference_count == 2,
                     "Expected downloaded slab to preserve slot reference counts");
    ok &= ExpectTrue(!downloaded_buffer.slot_states[slot1].occupied,
                     "Expected downloaded slab to preserve released slot state");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(downloaded_buffer, slot0))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     3.5f, "downloaded slot dense trunk bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(downloaded_buffer, slot0))[1][0]), -1.25f,
                     "downloaded slot trainable tail value");
    ok &= ExpectTrue(downloaded_generation.generation_index == 7, "Expected downloaded generation index to round-trip");
    ok &= ExpectTrue(downloaded_generation.slot_indices[0] == slot0,
                     "Expected downloaded generation slot handle to round-trip");
    ok &= ExpectTrue(downloaded_generation.slot_indices[1] ==
                         neuroevolution::genetic_algorithm::genotype_slab::kInvalidSlabSlotIndex,
                     "Expected downloaded invalid slot handle to round-trip");
    ok &= ExpectNear(downloaded_generation.fitness[0], 9.0f, "downloaded fitness");
    ok &= ExpectTrue(downloaded_generation.evaluation_counts[0] == 3, "Expected evaluation count to round-trip");
    ok &= ExpectTrue(downloaded_generation.has_fitness[0] == 1, "Expected has_fitness flag to round-trip");
    return ok;
}

bool TestDeviceSlabRuntimeReusesSweptAndReleasedSlotsDuringAssembly() {
    HostGenotypeSlab host_buffer{};
    SlabGeneration current_generation{};
    bool ok = TryCreateHostGenotypeSlab(host_buffer, 3, 4);
    ok &= TryCreateSlabGeneration(current_generation, 3, 5);
    if (!ok) {
        std::cerr << "FAIL: could not allocate assembly fixtures\n";
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = 0;
        ok &= TryAllocateSlabSlot(host_buffer, slot_index);
        ok &= TrySetSlabGenerationSlot(current_generation, individual_index, slot_index);
        GenomePolicyModelParameters(HostSlabSlotBytesAt(host_buffer, slot_index))
            .dense_trunk.hidden1_to_output.biases[0] = ToFloat16(11.0f * static_cast<float>(individual_index + 1));
        GenomeTailRows(HostSlabSlotBytesAt(host_buffer, slot_index))[0][0] =
            ToFloat16(2.0f * static_cast<float>(individual_index + 1));
    }
    if (!ok) {
        return false;
    }

    SlabAssemblyPlan plan{};
    ok &= TryCreateSlabAssemblyPlan(plan, 2);
    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};
    plan.parent_pairs[1] = {.first_parent_index = 0, .second_parent_index = 0};

    DeviceSlabRuntimeConfig runtime_config{};
    runtime_config.slot_count = 3;
    runtime_config.action_count = 4;
    runtime_config.max_generation_size = 3;

    DeviceSlabRuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadSlabToDevice(host_buffer, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    ok &= TryAssembleNextGenerationOnDevice(buffers, 17U, MakeDeterministicAssemblyConfig());
    if (!ok) {
        DeviceSlabRuntimeStatusCode status_code = DeviceSlabRuntimeStatusCode::kOk;
        (void)TryReadDeviceSlabRuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: device slab assembly failed with status " << static_cast<int>(status_code) << '\n';
        DestroyDeviceSlabRuntimeBuffers(buffers);
        return false;
    }

    HostGenotypeSlab downloaded_buffer{};
    SlabGeneration downloaded_current_generation{};
    SlabGeneration downloaded_next_generation{};
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    ok &= TryDownloadNextGenerationFromDevice(buffers, downloaded_next_generation);
    DestroyDeviceSlabRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(downloaded_next_generation.generation_index == 6,
                     "Expected device slab assembly to increment the generation index");
    ok &= ExpectTrue(downloaded_next_generation.slot_indices[0] == 2U,
                     "Expected the first child to reuse the garbage-collected zero-reference parent slot");
    ok &= ExpectTrue(downloaded_next_generation.slot_indices[1] == 1U,
                     "Expected the second child to reuse the parent slot freed after its final parent reference");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(
                                 HostSlabSlotBytesAt(downloaded_buffer, downloaded_next_generation.slot_indices[0]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     11.0f, "first child bias copied from first parent");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(
                                 HostSlabSlotBytesAt(downloaded_buffer, downloaded_next_generation.slot_indices[1]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     11.0f, "second child bias copied from first parent");
    ok &= ExpectNear(ToFloat(GenomeTailRows(
                         HostSlabSlotBytesAt(downloaded_buffer, downloaded_next_generation.slot_indices[0]))[0][0]),
                     2.0f, "first child trainable tail copied from first parent");
    ok &= ExpectNear(ToFloat(GenomeTailRows(
                         HostSlabSlotBytesAt(downloaded_buffer, downloaded_next_generation.slot_indices[1]))[0][0]),
                     2.0f, "second child trainable tail copied from first parent");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[0] ==
                         neuroevolution::genetic_algorithm::genotype_slab::kInvalidSlabSlotIndex,
                     "Expected the first parent slot handle to be cleared after its final parent reference");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[1] ==
                         neuroevolution::genetic_algorithm::genotype_slab::kInvalidSlabSlotIndex,
                     "Expected the second parent slot handle to be cleared after its final parent reference");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[2] ==
                         neuroevolution::genetic_algorithm::genotype_slab::kInvalidSlabSlotIndex,
                     "Expected zero-reference parents to be garbage-collected on-device before child assembly");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == 1,
                     "Expected one free slot to remain after two children occupy the reused slab slots");
    return ok;
}

bool TestDeviceAssemblyPlanPrioritisesChildrenThatFreeParents() {
    HostGenotypeSlab host_buffer{};
    SlabGeneration current_generation{};
    bool ok = TryCreateHostGenotypeSlab(host_buffer, 3, 4);
    ok &= TryCreateSlabGeneration(current_generation, 2, 13);
    if (!ok) {
        std::cerr << "FAIL: could not allocate assembly-plan prioritisation fixtures\n";
        return false;
    }

    std::uint32_t parent_slots[2]{};
    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        ok &= TryAllocateSlabSlot(host_buffer, parent_slots[parent_index]);
        ok &= TrySetSlabGenerationSlot(current_generation, parent_index, parent_slots[parent_index]);
    }
    GenomePolicyModelParameters(HostSlabSlotBytesAt(host_buffer, parent_slots[0]))
        .dense_trunk.hidden1_to_output.biases[0] = ToFloat16(11.0f);
    GenomePolicyModelParameters(HostSlabSlotBytesAt(host_buffer, parent_slots[1]))
        .dense_trunk.hidden1_to_output.biases[0] = ToFloat16(22.0f);
    if (!ok) {
        return false;
    }

    SlabAssemblyPlan plan{};
    ok &= TryCreateSlabAssemblyPlan(plan, 2);
    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 0};
    plan.parent_pairs[1] = {.first_parent_index = 0, .second_parent_index = 1};

    DeviceSlabRuntimeConfig runtime_config{};
    runtime_config.slot_count = 3;
    runtime_config.action_count = 4;
    runtime_config.max_generation_size = 2;

    DeviceSlabRuntimeBuffers failed_buffers{};
    ok &= TryCreateDeviceSlabRuntimeBuffers(failed_buffers, runtime_config);
    ok &= TryUploadSlabToDevice(host_buffer, failed_buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, failed_buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, failed_buffers);
    if (!ok) {
        DestroyDeviceSlabRuntimeBuffers(failed_buffers);
        return false;
    }

    ok &= ExpectTrue(!TryAssembleNextGenerationOnDevice(failed_buffers, 401U, MakeDeterministicAssemblyConfig()),
                     "Expected the unreordered assembly plan to fail when it frees parents too late");
    DeviceSlabRuntimeStatusCode failed_status = DeviceSlabRuntimeStatusCode::kOk;
    ok &= TryReadDeviceSlabRuntimeStatus(failed_buffers, failed_status);
    DestroyDeviceSlabRuntimeBuffers(failed_buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(failed_status == DeviceSlabRuntimeStatusCode::kSlabFull,
                     "Expected the unreordered late-release plan to report kSlabFull");

    DeviceSlabRuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadSlabToDevice(host_buffer, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    ok &= TryPrioritizeSlabAssemblyPlanForParentReleaseOnDevice(buffers);
    ok &= TryAssembleNextGenerationOnDevice(buffers, 401U, MakeDeterministicAssemblyConfig());
    if (!ok) {
        DeviceSlabRuntimeStatusCode status_code = DeviceSlabRuntimeStatusCode::kOk;
        (void)TryReadDeviceSlabRuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: prioritised assembly plan failed with status " << static_cast<int>(status_code) << '\n';
        DestroyDeviceSlabRuntimeBuffers(buffers);
        return false;
    }

    HostGenotypeSlab downloaded_buffer{};
    SlabGeneration downloaded_current_generation{};
    SlabGeneration downloaded_next_generation{};
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    ok &= TryDownloadNextGenerationFromDevice(buffers, downloaded_next_generation);
    DestroyDeviceSlabRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(downloaded_current_generation.slot_indices[0] == kInvalidSlabSlotIndex,
                     "Expected prioritised assembly to release the first parent");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[1] == kInvalidSlabSlotIndex,
                     "Expected prioritised assembly to release the second parent");
    ok &= ExpectTrue(downloaded_next_generation.generation_index == 14,
                     "Expected prioritised assembly to advance the generation");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == 1,
                     "Expected prioritised assembly to finish with one free slot");
    for (std::size_t child_index = 0; child_index < downloaded_next_generation.active_individual_count; ++child_index) {
        const std::uint32_t child_slot = downloaded_next_generation.slot_indices[child_index];
        ok &=
            ExpectTrue(child_slot != kInvalidSlabSlotIndex, "Expected prioritised assembly to materialise every child");
        ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(downloaded_buffer, child_slot))
                                     .dense_trunk.hidden1_to_output.biases[0]),
                         11.0f, "prioritised child bias copied from the first parent");
    }

    return ok;
}

bool TestDeviceSlabRuntimeRepacksAndAssemblesAfterActionCountGrowth() {
    constexpr std::size_t kInitialActionCount = 4;
    constexpr std::size_t kExpandedActionCount = 8;

    HostGenotypeSlab host_buffer{};
    SlabGeneration current_generation{};
    bool ok = TryCreateHostGenotypeSlab(host_buffer, 6, kInitialActionCount);
    ok &= TryCreateSlabGeneration(current_generation, 3, 20);
    if (!ok) {
        std::cerr << "FAIL: could not allocate growth-and-assembly fixtures\n";
        return false;
    }

    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        std::uint32_t slot_index = 0;
        ok &= TryAllocateSlabSlot(host_buffer, slot_index);
        ok &= TrySetSlabGenerationSlot(current_generation, parent_index, slot_index);
        GenomePolicyModelParameters(HostSlabSlotBytesAt(host_buffer, slot_index))
            .dense_trunk.hidden1_to_output.biases[0] = ToFloat16(11.0f * static_cast<float>(parent_index + 1));
        GenomeTailRows(HostSlabSlotBytesAt(host_buffer, slot_index))[0][0] =
            ToFloat16(2.0f * static_cast<float>(parent_index + 1));
    }
    if (!ok) {
        return false;
    }

    SlabAssemblyPlan plan{};
    ok &= TryCreateSlabAssemblyPlan(plan, 2);
    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 0};
    plan.parent_pairs[1] = {.first_parent_index = 2, .second_parent_index = 2};

    DeviceSlabRuntimeConfig runtime_config{};
    runtime_config.slot_count = 6;
    runtime_config.action_count = kInitialActionCount;
    runtime_config.max_generation_size = 3;

    DeviceSlabRuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadSlabToDevice(host_buffer, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    ok &= TryPrepareSlabForExpandedActionCountOnDevice(buffers, kExpandedActionCount);

    SlabDeviceAssemblyConfig assembly_config = MakeDeterministicAssemblyConfig();
    assembly_config.parent_action_count = kInitialActionCount;
    ok &= TryAssembleNextGenerationOnDevice(buffers, 211U, assembly_config);
    if (!ok) {
        DeviceSlabRuntimeStatusCode status_code = DeviceSlabRuntimeStatusCode::kOk;
        (void)TryReadDeviceSlabRuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: growth-and-assembly device slab test failed with status " << static_cast<int>(status_code)
                  << '\n';
        DestroyDeviceSlabRuntimeBuffers(buffers);
        return false;
    }

    HostGenotypeSlab downloaded_buffer{};
    SlabGeneration downloaded_current_generation{};
    SlabGeneration downloaded_next_generation{};
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    ok &= TryDownloadNextGenerationFromDevice(buffers, downloaded_next_generation);
    DestroyDeviceSlabRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(downloaded_buffer.layout.action_count == kExpandedActionCount,
                     "Expected device repacking to expand the slab action count");
    ok &= ExpectTrue(downloaded_next_generation.generation_index == 21,
                     "Expected post-growth assembly to increment the generation index");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == 3,
                     "Expected post-growth assembly to leave only child slots occupied");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[0] == kInvalidSlabSlotIndex,
                     "Expected post-growth assembly to release first referenced parent");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[1] == kInvalidSlabSlotIndex,
                     "Expected growth repacking to collect zero-reference parent");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[2] == kInvalidSlabSlotIndex,
                     "Expected post-growth assembly to release second referenced parent");

    const std::uint32_t first_child_slot = downloaded_next_generation.slot_indices[0];
    const std::uint32_t second_child_slot = downloaded_next_generation.slot_indices[1];
    ok &= ExpectTrue(first_child_slot != kInvalidSlabSlotIndex, "Expected first post-growth child to have a slab slot");
    ok &=
        ExpectTrue(second_child_slot != kInvalidSlabSlotIndex, "Expected second post-growth child to have a slab slot");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(downloaded_buffer, first_child_slot))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     11.0f, "first post-growth child bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(downloaded_buffer, first_child_slot))[0][0]), 2.0f,
                     "first post-growth child inherited tail");
    ok &= ExpectNear(
        ToFloat(GenomeTailRows(HostSlabSlotBytesAt(downloaded_buffer, first_child_slot))[kInitialActionCount][0]), 0.0f,
        "first post-growth child appended tail remains cleared without injection");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(downloaded_buffer, second_child_slot))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     33.0f, "second post-growth child bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(downloaded_buffer, second_child_slot))[0][0]), 6.0f,
                     "second post-growth child inherited tail");
    ok &= ExpectNear(
        ToFloat(GenomeTailRows(HostSlabSlotBytesAt(downloaded_buffer, second_child_slot))[kInitialActionCount][0]),
        0.0f, "second post-growth child appended tail remains cleared without injection");
    return ok;
}

bool TestDeviceSlabRuntimeRepackFailureDoesNotMutateBuffer() {
    constexpr std::size_t kInitialActionCount = 4;
    constexpr std::size_t kExpandedActionCount = 8;

    HostGenotypeSlab host_buffer{};
    SlabGeneration current_generation{};
    bool ok = TryCreateHostGenotypeSlab(host_buffer, 6, kInitialActionCount);
    ok &= TryCreateSlabGeneration(current_generation, 3, 24);
    if (!ok) {
        std::cerr << "FAIL: could not allocate failed-growth fixtures\n";
        return false;
    }

    std::uint32_t parent_slots[3]{};
    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        ok &= TryAllocateSlabSlot(host_buffer, parent_slots[parent_index]);
        ok &= TrySetSlabGenerationSlot(current_generation, parent_index, parent_slots[parent_index]);
        GenomePolicyModelParameters(HostSlabSlotBytesAt(host_buffer, parent_slots[parent_index]))
            .dense_trunk.hidden1_to_output.biases[0] = ToFloat16(5.0f + static_cast<float>(parent_index));
    }
    if (!ok) {
        return false;
    }

    SlabAssemblyPlan plan{};
    ok &= TryCreateSlabAssemblyPlan(plan, 4);
    for (std::size_t child_index = 0; child_index < plan.child_count; ++child_index) {
        plan.parent_pairs[child_index] = {.first_parent_index = 0, .second_parent_index = 2};
    }

    DeviceSlabRuntimeConfig runtime_config{};
    runtime_config.slot_count = 6;
    runtime_config.action_count = kInitialActionCount;
    runtime_config.max_generation_size = 4;

    DeviceSlabRuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadSlabToDevice(host_buffer, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    if (!ok) {
        DestroyDeviceSlabRuntimeBuffers(buffers);
        return false;
    }

    ok &= ExpectTrue(!TryPrepareSlabForExpandedActionCountOnDevice(buffers, kExpandedActionCount),
                     "Expected growth repacking to fail when planned children cannot fit");

    DeviceSlabRuntimeStatusCode status_code = DeviceSlabRuntimeStatusCode::kOk;
    ok &= TryReadDeviceSlabRuntimeStatus(buffers, status_code);

    HostGenotypeSlab downloaded_buffer{};
    SlabGeneration downloaded_current_generation{};
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    DestroyDeviceSlabRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(status_code == DeviceSlabRuntimeStatusCode::kSlabRepackFailed,
                     "Expected failed growth to report kSlabRepackFailed");
    ok &= ExpectTrue(downloaded_buffer.layout.action_count == kInitialActionCount,
                     "Expected failed growth to preserve the original action count");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == 3,
                     "Expected failed growth to preserve the original free-slot count");
    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        ok &= ExpectTrue(downloaded_current_generation.slot_indices[parent_index] == parent_slots[parent_index],
                         "Expected failed growth to preserve current-generation slot handles");
        ok &= ExpectTrue(downloaded_buffer.slot_states[parent_slots[parent_index]].occupied,
                         "Expected failed growth to preserve live parent slot state");
        ok &= ExpectTrue(downloaded_buffer.slot_states[parent_slots[parent_index]].reference_count == 1U,
                         "Expected failed growth to preserve live parent slot references");
        ok &= ExpectNear(
            ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(downloaded_buffer, parent_slots[parent_index]))
                        .dense_trunk.hidden1_to_output.biases[0]),
            5.0f + static_cast<float>(parent_index), "failed growth preserved parent bias");
    }

    return ok;
}

bool TestDeviceSlabRuntimeAssemblesChildBatchConcurrently() {
    constexpr std::size_t kParentCount = 32;
    constexpr std::size_t kChildCount = 32;
    constexpr std::size_t kSlotCount = 96;
    constexpr std::size_t kActionCount = 4;

    HostGenotypeSlab host_buffer{};
    SlabGeneration current_generation{};
    bool ok = TryCreateHostGenotypeSlab(host_buffer, kSlotCount, kActionCount);
    ok &= TryCreateSlabGeneration(current_generation, kParentCount, 11);
    if (!ok) {
        std::cerr << "FAIL: could not allocate concurrent assembly fixtures\n";
        return false;
    }

    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        std::uint32_t slot_index = 0;
        ok &= TryAllocateSlabSlot(host_buffer, slot_index);
        ok &= TrySetSlabGenerationSlot(current_generation, parent_index, slot_index);
        GenomePolicyModelParameters(HostSlabSlotBytesAt(host_buffer, slot_index))
            .dense_trunk.hidden1_to_output.biases[0] = ToFloat16(100.0f + static_cast<float>(parent_index));
        GenomeTailRows(HostSlabSlotBytesAt(host_buffer, slot_index))[0][0] =
            ToFloat16(20.0f + static_cast<float>(parent_index));
    }
    if (!ok) {
        return false;
    }

    SlabAssemblyPlan plan{};
    ok &= TryCreateSlabAssemblyPlan(plan, kChildCount);
    for (std::size_t child_index = 0; child_index < kChildCount; ++child_index) {
        plan.parent_pairs[child_index] = {
            .first_parent_index = static_cast<std::uint32_t>(child_index),
            .second_parent_index = static_cast<std::uint32_t>(child_index),
        };
    }

    DeviceSlabRuntimeConfig runtime_config{};
    runtime_config.slot_count = kSlotCount;
    runtime_config.action_count = kActionCount;
    runtime_config.max_generation_size = kSlotCount;

    DeviceSlabRuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadSlabToDevice(host_buffer, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    ok &= TryAssembleNextGenerationOnDevice(buffers, 101U, MakeDeterministicAssemblyConfig());
    if (!ok) {
        DeviceSlabRuntimeStatusCode status_code = DeviceSlabRuntimeStatusCode::kOk;
        (void)TryReadDeviceSlabRuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: concurrent device slab assembly failed with status " << static_cast<int>(status_code)
                  << '\n';
        DestroyDeviceSlabRuntimeBuffers(buffers);
        return false;
    }

    HostGenotypeSlab downloaded_buffer{};
    SlabGeneration downloaded_current_generation{};
    SlabGeneration downloaded_next_generation{};
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    ok &= TryDownloadNextGenerationFromDevice(buffers, downloaded_next_generation);
    DestroyDeviceSlabRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(downloaded_next_generation.generation_index == 12,
                     "Expected concurrent device assembly to increment the generation index");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == (kSlotCount - kChildCount),
                     "Expected concurrent child assembly to leave only child slots occupied");
    for (std::size_t child_index = 0; child_index < kChildCount; ++child_index) {
        const std::uint32_t child_slot = downloaded_next_generation.slot_indices[child_index];
        ok &= ExpectTrue(child_slot != kInvalidSlabSlotIndex, "Expected concurrent child slot to be set");
        ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(downloaded_buffer, child_slot))
                                     .dense_trunk.hidden1_to_output.biases[0]),
                         100.0f + static_cast<float>(child_index), "concurrent child bias copied from first parent");
        ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(downloaded_buffer, child_slot))[0][0]),
                         20.0f + static_cast<float>(child_index),
                         "concurrent child trainable tail copied from first parent");
        ok &= ExpectTrue(downloaded_current_generation.slot_indices[child_index] == kInvalidSlabSlotIndex,
                         "Expected concurrent assembly to release consumed parent slots");
    }

    return ok;
}

bool TestDeviceSlabRuntimeCleansUpPartialAssemblyWhenLaterBatchFails() {
    constexpr std::size_t kParentCount = 4;
    constexpr std::size_t kChildCount = 6;
    constexpr std::size_t kSlotCount = 5;
    constexpr std::size_t kActionCount = 4;

    HostGenotypeSlab host_buffer{};
    SlabGeneration current_generation{};
    bool ok = TryCreateHostGenotypeSlab(host_buffer, kSlotCount, kActionCount);
    ok &= TryCreateSlabGeneration(current_generation, kParentCount, 17);
    if (!ok) {
        std::cerr << "FAIL: could not allocate partial-failure assembly fixtures\n";
        return false;
    }

    std::uint32_t original_parent_slots[kParentCount]{};
    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        ok &= TryAllocateSlabSlot(host_buffer, original_parent_slots[parent_index]);
        ok &= TrySetSlabGenerationSlot(current_generation, parent_index, original_parent_slots[parent_index]);
    }
    if (!ok) {
        return false;
    }

    SlabAssemblyPlan plan{};
    ok &= TryCreateSlabAssemblyPlan(plan, kChildCount);
    for (std::size_t child_index = 0; child_index < kChildCount; ++child_index) {
        plan.parent_pairs[child_index] = {
            .first_parent_index = static_cast<std::uint32_t>(child_index % kParentCount),
            .second_parent_index = static_cast<std::uint32_t>(child_index % kParentCount),
        };
    }

    DeviceSlabRuntimeConfig runtime_config{};
    runtime_config.slot_count = kSlotCount;
    runtime_config.action_count = kActionCount;
    runtime_config.max_generation_size = kChildCount;

    DeviceSlabRuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadSlabToDevice(host_buffer, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    if (!ok) {
        DestroyDeviceSlabRuntimeBuffers(buffers);
        return false;
    }

    ok &= ExpectTrue(!TryAssembleNextGenerationOnDevice(buffers, 303U, MakeDeterministicAssemblyConfig()),
                     "Expected over-large device assembly to fail after a partial child batch");

    DeviceSlabRuntimeStatusCode status_code = DeviceSlabRuntimeStatusCode::kOk;
    ok &= TryReadDeviceSlabRuntimeStatus(buffers, status_code);

    HostGenotypeSlab downloaded_buffer{};
    SlabGeneration downloaded_current_generation{};
    SlabGeneration downloaded_next_generation{};
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    ok &= ExpectTrue(!TryDownloadNextGenerationFromDevice(buffers, downloaded_next_generation),
                     "Expected failed partial assembly to clear the next-generation handle");
    DestroyDeviceSlabRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(status_code == DeviceSlabRuntimeStatusCode::kSlabFull,
                     "Expected partial assembly failure to preserve kSlabFull status");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == 1,
                     "Expected partial assembly cleanup to release the allocated child slot");
    for (std::size_t parent_index = 0; parent_index < kParentCount; ++parent_index) {
        ok &=
            ExpectTrue(downloaded_current_generation.slot_indices[parent_index] == original_parent_slots[parent_index],
                       "Expected partial assembly cleanup to leave unreleased parent slots intact");
        ok &= ExpectTrue(downloaded_buffer.slot_states[original_parent_slots[parent_index]].occupied,
                         "Expected partial assembly cleanup to keep original parent slots occupied");
        ok &= ExpectTrue(downloaded_buffer.slot_states[original_parent_slots[parent_index]].reference_count == 1U,
                         "Expected partial assembly cleanup to preserve parent slot references");
    }

    return ok;
}

bool TestDeviceSlabRuntimeFailsCleanlyWhenBufferIsGenuinelyFull() {
    HostGenotypeSlab host_buffer{};
    SlabGeneration current_generation{};
    bool ok = TryCreateHostGenotypeSlab(host_buffer, 2, 4);
    ok &= TryCreateSlabGeneration(current_generation, 2, 4);
    if (!ok) {
        std::cerr << "FAIL: could not allocate full-slab fixtures\n";
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = 0;
        ok &= TryAllocateSlabSlot(host_buffer, slot_index);
        ok &= TrySetSlabGenerationSlot(current_generation, individual_index, slot_index);
    }
    if (!ok) {
        return false;
    }

    SlabAssemblyPlan plan{};
    ok &= TryCreateSlabAssemblyPlan(plan, 1);
    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};

    DeviceSlabRuntimeConfig runtime_config{};
    runtime_config.slot_count = 2;
    runtime_config.action_count = 4;
    runtime_config.max_generation_size = 2;

    DeviceSlabRuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadSlabToDevice(host_buffer, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    if (!ok) {
        DestroyDeviceSlabRuntimeBuffers(buffers);
        return false;
    }

    ok &= ExpectTrue(!TryAssembleNextGenerationOnDevice(buffers, 23U, MakeDeterministicAssemblyConfig()),
                     "Expected device slab assembly to fail when the slab is genuinely full");

    DeviceSlabRuntimeStatusCode status_code = DeviceSlabRuntimeStatusCode::kOk;
    ok &= TryReadDeviceSlabRuntimeStatus(buffers, status_code);

    HostGenotypeSlab downloaded_buffer{};
    SlabGeneration downloaded_current_generation{};
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    DestroyDeviceSlabRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(status_code == DeviceSlabRuntimeStatusCode::kSlabFull,
                     "Expected genuinely full device slab assembly to report kSlabFull");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == 0, "Expected failed assembly to leave the slab full");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[0] == 0U,
                     "Expected failed assembly to leave the first parent slot handle intact");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[1] == 1U,
                     "Expected failed assembly to leave the second parent slot handle intact");
    return ok;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    if (!TestDeviceReferenceCountersAreAtomicUnderWarpContention() ||
        !TestDeviceSlabFreeListIsThreadSafeUnderWarpContention() ||
        !TestDeviceSlabRuntimeUploadsAndDownloadsBufferAndGenerationState() ||
        !TestDeviceSlabRuntimeReusesSweptAndReleasedSlotsDuringAssembly() ||
        !TestDeviceAssemblyPlanPrioritisesChildrenThatFreeParents() ||
        !TestDeviceSlabRuntimeRepacksAndAssemblesAfterActionCountGrowth() ||
        !TestDeviceSlabRuntimeRepackFailureDoesNotMutateBuffer() ||
        !TestDeviceSlabRuntimeAssemblesChildBatchConcurrently() ||
        !TestDeviceSlabRuntimeCleansUpPartialAssemblyWhenLaterBatchFails() ||
        !TestDeviceSlabRuntimeFailsCleanlyWhenBufferIsGenuinelyFull()) {
        return 1;
    }

    std::cout << "PASS: genotype_slab_device_runtime_test\n";
    return 0;
}
