#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/genotype_buffer/device_runtime.hpp"
#include "genetic_algorithm/genotype_buffer/reference_counter.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::common::ToFloat16;
using neuroevolution::genetic_algorithm::genotype_buffer::BufferAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_buffer::BufferGeneration;
using neuroevolution::genetic_algorithm::genotype_buffer::BufferSlotState;
using neuroevolution::genetic_algorithm::genotype_buffer::GenomePolicyModelParameters;
using neuroevolution::genetic_algorithm::genotype_buffer::GenomeTailRows;
using neuroevolution::genetic_algorithm::genotype_buffer::GenotypeBufferLayout;
using neuroevolution::genetic_algorithm::genotype_buffer::GenotypeBufferView;
using neuroevolution::genetic_algorithm::genotype_buffer::HostBufferSlotBytesAt;
using neuroevolution::genetic_algorithm::genotype_buffer::HostGenotypeBuffer;
using neuroevolution::genetic_algorithm::genotype_buffer::kInvalidBufferSlotIndex;
using neuroevolution::genetic_algorithm::genotype_buffer::TryAllocateBufferSlot;
using neuroevolution::genetic_algorithm::genotype_buffer::TryCreateBufferAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_buffer::TryCreateBufferGeneration;
using neuroevolution::genetic_algorithm::genotype_buffer::TryCreateHostGenotypeBuffer;
using neuroevolution::genetic_algorithm::genotype_buffer::TryDecrementParentReferenceCount;
using neuroevolution::genetic_algorithm::genotype_buffer::TryIncrementParentReferenceCount;
using neuroevolution::genetic_algorithm::genotype_buffer::TryReleaseBufferSlot;
using neuroevolution::genetic_algorithm::genotype_buffer::TryRetainBufferSlot;
using neuroevolution::genetic_algorithm::genotype_buffer::TrySetBufferGenerationSlot;
using neuroevolution::genetic_algorithm::genotype_buffer::device::BufferDeviceAssemblyConfig;
using neuroevolution::genetic_algorithm::genotype_buffer::device::DestroyDeviceBufferRuntimeBuffers;
using neuroevolution::genetic_algorithm::genotype_buffer::device::DeviceBufferRuntimeBuffers;
using neuroevolution::genetic_algorithm::genotype_buffer::device::DeviceBufferRuntimeConfig;
using neuroevolution::genetic_algorithm::genotype_buffer::device::DeviceBufferRuntimeStatusCode;
using neuroevolution::genetic_algorithm::genotype_buffer::device::TryAssembleNextGenerationOnDevice;
using neuroevolution::genetic_algorithm::genotype_buffer::device::TryCreateDeviceBufferRuntimeBuffers;
using neuroevolution::genetic_algorithm::genotype_buffer::device::TryDownloadBufferFromDevice;
using neuroevolution::genetic_algorithm::genotype_buffer::device::TryDownloadCurrentGenerationFromDevice;
using neuroevolution::genetic_algorithm::genotype_buffer::device::TryDownloadNextGenerationFromDevice;
using neuroevolution::genetic_algorithm::genotype_buffer::device::TryReadDeviceBufferRuntimeStatus;
using neuroevolution::genetic_algorithm::genotype_buffer::device::TryUploadAssemblyPlanToDevice;
using neuroevolution::genetic_algorithm::genotype_buffer::device::TryUploadBufferToDevice;
using neuroevolution::genetic_algorithm::genotype_buffer::device::TryUploadCurrentGenerationToDevice;

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

BufferDeviceAssemblyConfig MakeDeterministicAssemblyConfig() {
    BufferDeviceAssemblyConfig config{};
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

__global__ void BufferFreeListWarpContentionKernel(std::uint8_t *buffer_storage, BufferSlotState *slot_states,
                                                   std::uint32_t *free_slot_stack, std::uint32_t *free_slot_count,
                                                   std::uint32_t *free_slot_lock,
                                                   const GenotypeBufferLayout buffer_layout,
                                                   std::uint32_t *allocated_slots, std::uint32_t *slot_hit_counts,
                                                   int *status) {
    constexpr unsigned kWarpMask = 0xFFFFFFFFU;
    if ((blockIdx.x != 0) || (threadIdx.x >= 32)) {
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

    const std::uint32_t lane_index = threadIdx.x;
    std::uint32_t slot_index = kInvalidBufferSlotIndex;
    if (!TryAllocateBufferSlot(buffer, slot_index)) {
        atomicCAS(status, 0, 1);
        allocated_slots[lane_index] = kInvalidBufferSlotIndex;
        return;
    }

    allocated_slots[lane_index] = slot_index;
    if (slot_index >= buffer_layout.slot_count) {
        atomicCAS(status, 0, 2);
        return;
    }

    atomicAdd(&slot_hit_counts[slot_index], 1U);
    __syncwarp(kWarpMask);

    if (lane_index == 0) {
        if (*free_slot_count != 0U) {
            atomicCAS(status, 0, 3);
        }

        for (std::uint32_t checked_slot = 0; checked_slot < buffer_layout.slot_count; ++checked_slot) {
            if (slot_hit_counts[checked_slot] != 1U) {
                atomicCAS(status, 0, 4);
            }

            if (!slot_states[checked_slot].occupied || (slot_states[checked_slot].reference_count != 1U)) {
                atomicCAS(status, 0, 5);
            }
        }
    }

    __syncwarp(kWarpMask);
    if (!TryReleaseBufferSlot(buffer, allocated_slots[lane_index])) {
        atomicCAS(status, 0, 6);
    }

    __syncwarp(kWarpMask);
    if (lane_index == 0) {
        if (*free_slot_count != buffer_layout.slot_count) {
            atomicCAS(status, 0, 7);
        }

        for (std::uint32_t checked_slot = 0; checked_slot < buffer_layout.slot_count; ++checked_slot) {
            if (slot_states[checked_slot].occupied || (slot_states[checked_slot].reference_count != 0U)) {
                atomicCAS(status, 0, 8);
            }
        }
    }
}

bool TestDeviceBufferFreeListIsThreadSafeUnderWarpContention() {
    constexpr std::size_t kSlotCount = 32;
    constexpr std::size_t kActionCount = 4;

    HostGenotypeBuffer host_buffer{};
    bool ok = TryCreateHostGenotypeBuffer(host_buffer, kSlotCount, kActionCount);
    if (!ok) {
        std::cerr << "FAIL: could not allocate free-list contention host buffer\n";
        return false;
    }

    DeviceBufferRuntimeConfig runtime_config{};
    runtime_config.slot_count = kSlotCount;
    runtime_config.action_count = kActionCount;
    runtime_config.max_generation_size = kSlotCount;

    DeviceBufferRuntimeBuffers buffers{};
    ok &= TryCreateDeviceBufferRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadBufferToDevice(host_buffer, buffers);

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
        DestroyDeviceBufferRuntimeBuffers(buffers);
        return false;
    }

    BufferFreeListWarpContentionKernel<<<1, 32>>>(
        buffers.buffer_storage, buffers.slot_states, buffers.free_slot_stack, buffers.free_slot_count,
        buffers.free_slot_lock, buffers.buffer_layout, device_allocated_slots, device_slot_hit_counts, device_status);
    ok &= CheckCuda(cudaGetLastError(), "launching free-list warp contention kernel");
    ok &= CheckCuda(cudaDeviceSynchronize(), "running free-list warp contention kernel");

    int host_status = 0;
    std::uint32_t host_slot_hit_counts[kSlotCount]{};
    ok &= CheckCuda(cudaMemcpy(&host_status, device_status, sizeof(int), cudaMemcpyDeviceToHost),
                    "copying free-list contention status");
    ok &= CheckCuda(
        cudaMemcpy(host_slot_hit_counts, device_slot_hit_counts, sizeof(host_slot_hit_counts), cudaMemcpyDeviceToHost),
        "copying free-list contention slot-hit counts");

    HostGenotypeBuffer downloaded_buffer{};
    ok &= TryDownloadBufferFromDevice(buffers, downloaded_buffer);

    cudaFree(device_allocated_slots);
    cudaFree(device_slot_hit_counts);
    cudaFree(device_status);
    DestroyDeviceBufferRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(host_status == 0, "Expected free-list warp contention status to stay ok");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == kSlotCount,
                     "Expected all contended slots to return to the genotype buffer");
    for (std::size_t slot_index = 0; slot_index < kSlotCount; ++slot_index) {
        ok &= ExpectTrue(host_slot_hit_counts[slot_index] == 1U,
                         "Expected each genotype buffer slot to be allocated exactly once under warp contention");
        ok &= ExpectTrue(!downloaded_buffer.slot_states[slot_index].occupied,
                         "Expected released contended slot to be unoccupied");
        ok &= ExpectTrue(downloaded_buffer.slot_states[slot_index].reference_count == 0U,
                         "Expected released contended slot to have no references");
    }

    return ok;
}

bool TestDeviceBufferRuntimeUploadsAndDownloadsBufferAndGenerationState() {
    HostGenotypeBuffer host_buffer{};
    BufferGeneration current_generation{};
    bool ok = TryCreateHostGenotypeBuffer(host_buffer, 3, 4);
    ok &= TryCreateBufferGeneration(current_generation, 2, 7);
    if (!ok) {
        std::cerr << "FAIL: could not allocate upload/download fixtures\n";
        return false;
    }

    std::uint32_t slot0 = 0;
    std::uint32_t slot1 = 0;
    ok &= TryAllocateBufferSlot(host_buffer, slot0);
    ok &= TryAllocateBufferSlot(host_buffer, slot1);
    ok &= TryRetainBufferSlot(host_buffer, slot0);
    ok &= TryReleaseBufferSlot(host_buffer, slot1);
    ok &= TrySetBufferGenerationSlot(current_generation, 0, slot0);
    current_generation.fitness[0] = 9.0f;
    current_generation.evaluation_counts[0] = 3;
    current_generation.has_fitness[0] = 1;
    GenomePolicyModelParameters(HostBufferSlotBytesAt(host_buffer, slot0)).dense_trunk.hidden1_to_output.biases[0] =
        3.5f;
    GenomeTailRows(HostBufferSlotBytesAt(host_buffer, slot0))[1][0] = ToFloat16(-1.25f);

    DeviceBufferRuntimeConfig runtime_config{};
    runtime_config.slot_count = 3;
    runtime_config.action_count = 4;
    runtime_config.max_generation_size = 3;

    DeviceBufferRuntimeBuffers buffers{};
    ok &= TryCreateDeviceBufferRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadBufferToDevice(host_buffer, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    if (!ok) {
        DestroyDeviceBufferRuntimeBuffers(buffers);
        std::cerr << "FAIL: could not upload buffer runtime fixtures\n";
        return false;
    }

    HostGenotypeBuffer downloaded_buffer{};
    BufferGeneration downloaded_generation{};
    ok &= TryDownloadBufferFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_generation);
    DestroyDeviceBufferRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(downloaded_buffer.free_slot_count == 2, "Expected downloaded buffer to preserve free-slot count");
    ok &= ExpectTrue(downloaded_buffer.slot_states[slot0].occupied,
                     "Expected downloaded buffer to preserve live slot state");
    ok &= ExpectTrue(downloaded_buffer.slot_states[slot0].reference_count == 2,
                     "Expected downloaded buffer to preserve slot reference counts");
    ok &= ExpectTrue(!downloaded_buffer.slot_states[slot1].occupied,
                     "Expected downloaded buffer to preserve released slot state");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostBufferSlotBytesAt(downloaded_buffer, slot0))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     3.5f, "downloaded slot dense trunk bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostBufferSlotBytesAt(downloaded_buffer, slot0))[1][0]), -1.25f,
                     "downloaded slot trainable tail value");
    ok &= ExpectTrue(downloaded_generation.generation_index == 7, "Expected downloaded generation index to round-trip");
    ok &= ExpectTrue(downloaded_generation.slot_indices[0] == slot0,
                     "Expected downloaded generation slot handle to round-trip");
    ok &= ExpectTrue(downloaded_generation.slot_indices[1] ==
                         neuroevolution::genetic_algorithm::genotype_buffer::kInvalidBufferSlotIndex,
                     "Expected downloaded invalid slot handle to round-trip");
    ok &= ExpectNear(downloaded_generation.fitness[0], 9.0f, "downloaded fitness");
    ok &= ExpectTrue(downloaded_generation.evaluation_counts[0] == 3, "Expected evaluation count to round-trip");
    ok &= ExpectTrue(downloaded_generation.has_fitness[0] == 1, "Expected has_fitness flag to round-trip");
    return ok;
}

bool TestDeviceBufferRuntimeReusesSweptAndReleasedSlotsDuringAssembly() {
    HostGenotypeBuffer host_buffer{};
    BufferGeneration current_generation{};
    bool ok = TryCreateHostGenotypeBuffer(host_buffer, 3, 4);
    ok &= TryCreateBufferGeneration(current_generation, 3, 5);
    if (!ok) {
        std::cerr << "FAIL: could not allocate assembly fixtures\n";
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = 0;
        ok &= TryAllocateBufferSlot(host_buffer, slot_index);
        ok &= TrySetBufferGenerationSlot(current_generation, individual_index, slot_index);
        GenomePolicyModelParameters(HostBufferSlotBytesAt(host_buffer, slot_index))
            .dense_trunk.hidden1_to_output.biases[0] = ToFloat16(11.0f * static_cast<float>(individual_index + 1));
        GenomeTailRows(HostBufferSlotBytesAt(host_buffer, slot_index))[0][0] =
            ToFloat16(2.0f * static_cast<float>(individual_index + 1));
    }
    if (!ok) {
        return false;
    }

    BufferAssemblyPlan plan{};
    ok &= TryCreateBufferAssemblyPlan(plan, 2);
    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};
    plan.parent_pairs[1] = {.first_parent_index = 0, .second_parent_index = 0};

    DeviceBufferRuntimeConfig runtime_config{};
    runtime_config.slot_count = 3;
    runtime_config.action_count = 4;
    runtime_config.max_generation_size = 3;

    DeviceBufferRuntimeBuffers buffers{};
    ok &= TryCreateDeviceBufferRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadBufferToDevice(host_buffer, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    ok &= TryAssembleNextGenerationOnDevice(buffers, 17U, MakeDeterministicAssemblyConfig());
    if (!ok) {
        DeviceBufferRuntimeStatusCode status_code = DeviceBufferRuntimeStatusCode::kOk;
        (void)TryReadDeviceBufferRuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: device buffer assembly failed with status " << static_cast<int>(status_code) << '\n';
        DestroyDeviceBufferRuntimeBuffers(buffers);
        return false;
    }

    HostGenotypeBuffer downloaded_buffer{};
    BufferGeneration downloaded_current_generation{};
    BufferGeneration downloaded_next_generation{};
    ok &= TryDownloadBufferFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    ok &= TryDownloadNextGenerationFromDevice(buffers, downloaded_next_generation);
    DestroyDeviceBufferRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(downloaded_next_generation.generation_index == 6,
                     "Expected device buffer assembly to increment the generation index");
    ok &= ExpectTrue(downloaded_next_generation.slot_indices[0] == 2U,
                     "Expected the first child to reuse the garbage-collected zero-reference parent slot");
    ok &= ExpectTrue(downloaded_next_generation.slot_indices[1] == 1U,
                     "Expected the second child to reuse the parent slot freed after its final parent reference");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(
                                 HostBufferSlotBytesAt(downloaded_buffer, downloaded_next_generation.slot_indices[0]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     11.0f, "first child bias copied from first parent");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(
                                 HostBufferSlotBytesAt(downloaded_buffer, downloaded_next_generation.slot_indices[1]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     11.0f, "second child bias copied from first parent");
    ok &= ExpectNear(ToFloat(GenomeTailRows(
                         HostBufferSlotBytesAt(downloaded_buffer, downloaded_next_generation.slot_indices[0]))[0][0]),
                     2.0f, "first child trainable tail copied from first parent");
    ok &= ExpectNear(ToFloat(GenomeTailRows(
                         HostBufferSlotBytesAt(downloaded_buffer, downloaded_next_generation.slot_indices[1]))[0][0]),
                     2.0f, "second child trainable tail copied from first parent");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[0] ==
                         neuroevolution::genetic_algorithm::genotype_buffer::kInvalidBufferSlotIndex,
                     "Expected the first parent slot handle to be cleared after its final parent reference");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[1] ==
                         neuroevolution::genetic_algorithm::genotype_buffer::kInvalidBufferSlotIndex,
                     "Expected the second parent slot handle to be cleared after its final parent reference");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[2] ==
                         neuroevolution::genetic_algorithm::genotype_buffer::kInvalidBufferSlotIndex,
                     "Expected zero-reference parents to be garbage-collected on-device before child assembly");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == 1,
                     "Expected one free slot to remain after two children occupy the reused buffer slots");
    return ok;
}

bool TestDeviceBufferRuntimeAssemblesChildBatchConcurrently() {
    constexpr std::size_t kParentCount = 32;
    constexpr std::size_t kChildCount = 32;
    constexpr std::size_t kSlotCount = 96;
    constexpr std::size_t kActionCount = 4;

    HostGenotypeBuffer host_buffer{};
    BufferGeneration current_generation{};
    bool ok = TryCreateHostGenotypeBuffer(host_buffer, kSlotCount, kActionCount);
    ok &= TryCreateBufferGeneration(current_generation, kParentCount, 11);
    if (!ok) {
        std::cerr << "FAIL: could not allocate concurrent assembly fixtures\n";
        return false;
    }

    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        std::uint32_t slot_index = 0;
        ok &= TryAllocateBufferSlot(host_buffer, slot_index);
        ok &= TrySetBufferGenerationSlot(current_generation, parent_index, slot_index);
        GenomePolicyModelParameters(HostBufferSlotBytesAt(host_buffer, slot_index))
            .dense_trunk.hidden1_to_output.biases[0] = ToFloat16(100.0f + static_cast<float>(parent_index));
        GenomeTailRows(HostBufferSlotBytesAt(host_buffer, slot_index))[0][0] =
            ToFloat16(20.0f + static_cast<float>(parent_index));
    }
    if (!ok) {
        return false;
    }

    BufferAssemblyPlan plan{};
    ok &= TryCreateBufferAssemblyPlan(plan, kChildCount);
    for (std::size_t child_index = 0; child_index < kChildCount; ++child_index) {
        plan.parent_pairs[child_index] = {
            .first_parent_index = static_cast<std::uint32_t>(child_index),
            .second_parent_index = static_cast<std::uint32_t>(child_index),
        };
    }

    DeviceBufferRuntimeConfig runtime_config{};
    runtime_config.slot_count = kSlotCount;
    runtime_config.action_count = kActionCount;
    runtime_config.max_generation_size = kSlotCount;

    DeviceBufferRuntimeBuffers buffers{};
    ok &= TryCreateDeviceBufferRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadBufferToDevice(host_buffer, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    ok &= TryAssembleNextGenerationOnDevice(buffers, 101U, MakeDeterministicAssemblyConfig());
    if (!ok) {
        DeviceBufferRuntimeStatusCode status_code = DeviceBufferRuntimeStatusCode::kOk;
        (void)TryReadDeviceBufferRuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: concurrent device buffer assembly failed with status " << static_cast<int>(status_code)
                  << '\n';
        DestroyDeviceBufferRuntimeBuffers(buffers);
        return false;
    }

    HostGenotypeBuffer downloaded_buffer{};
    BufferGeneration downloaded_current_generation{};
    BufferGeneration downloaded_next_generation{};
    ok &= TryDownloadBufferFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    ok &= TryDownloadNextGenerationFromDevice(buffers, downloaded_next_generation);
    DestroyDeviceBufferRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(downloaded_next_generation.generation_index == 12,
                     "Expected concurrent device assembly to increment the generation index");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == (kSlotCount - kChildCount),
                     "Expected concurrent child assembly to leave only child slots occupied");
    for (std::size_t child_index = 0; child_index < kChildCount; ++child_index) {
        const std::uint32_t child_slot = downloaded_next_generation.slot_indices[child_index];
        ok &= ExpectTrue(child_slot != kInvalidBufferSlotIndex, "Expected concurrent child slot to be set");
        ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostBufferSlotBytesAt(downloaded_buffer, child_slot))
                                     .dense_trunk.hidden1_to_output.biases[0]),
                         100.0f + static_cast<float>(child_index), "concurrent child bias copied from first parent");
        ok &= ExpectNear(ToFloat(GenomeTailRows(HostBufferSlotBytesAt(downloaded_buffer, child_slot))[0][0]),
                         20.0f + static_cast<float>(child_index),
                         "concurrent child trainable tail copied from first parent");
        ok &= ExpectTrue(downloaded_current_generation.slot_indices[child_index] == kInvalidBufferSlotIndex,
                         "Expected concurrent assembly to release consumed parent slots");
    }

    return ok;
}

bool TestDeviceBufferRuntimeCleansUpPartialAssemblyWhenLaterBatchFails() {
    constexpr std::size_t kParentCount = 4;
    constexpr std::size_t kChildCount = 6;
    constexpr std::size_t kSlotCount = 5;
    constexpr std::size_t kActionCount = 4;

    HostGenotypeBuffer host_buffer{};
    BufferGeneration current_generation{};
    bool ok = TryCreateHostGenotypeBuffer(host_buffer, kSlotCount, kActionCount);
    ok &= TryCreateBufferGeneration(current_generation, kParentCount, 17);
    if (!ok) {
        std::cerr << "FAIL: could not allocate partial-failure assembly fixtures\n";
        return false;
    }

    std::uint32_t original_parent_slots[kParentCount]{};
    for (std::size_t parent_index = 0; parent_index < current_generation.active_individual_count; ++parent_index) {
        ok &= TryAllocateBufferSlot(host_buffer, original_parent_slots[parent_index]);
        ok &= TrySetBufferGenerationSlot(current_generation, parent_index, original_parent_slots[parent_index]);
    }
    if (!ok) {
        return false;
    }

    BufferAssemblyPlan plan{};
    ok &= TryCreateBufferAssemblyPlan(plan, kChildCount);
    for (std::size_t child_index = 0; child_index < kChildCount; ++child_index) {
        plan.parent_pairs[child_index] = {
            .first_parent_index = static_cast<std::uint32_t>(child_index % kParentCount),
            .second_parent_index = static_cast<std::uint32_t>(child_index % kParentCount),
        };
    }

    DeviceBufferRuntimeConfig runtime_config{};
    runtime_config.slot_count = kSlotCount;
    runtime_config.action_count = kActionCount;
    runtime_config.max_generation_size = kChildCount;

    DeviceBufferRuntimeBuffers buffers{};
    ok &= TryCreateDeviceBufferRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadBufferToDevice(host_buffer, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    if (!ok) {
        DestroyDeviceBufferRuntimeBuffers(buffers);
        return false;
    }

    ok &= ExpectTrue(!TryAssembleNextGenerationOnDevice(buffers, 303U, MakeDeterministicAssemblyConfig()),
                     "Expected over-large device assembly to fail after a partial child batch");

    DeviceBufferRuntimeStatusCode status_code = DeviceBufferRuntimeStatusCode::kOk;
    ok &= TryReadDeviceBufferRuntimeStatus(buffers, status_code);

    HostGenotypeBuffer downloaded_buffer{};
    BufferGeneration downloaded_current_generation{};
    BufferGeneration downloaded_next_generation{};
    ok &= TryDownloadBufferFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    ok &= ExpectTrue(!TryDownloadNextGenerationFromDevice(buffers, downloaded_next_generation),
                     "Expected failed partial assembly to clear the next-generation handle");
    DestroyDeviceBufferRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(status_code == DeviceBufferRuntimeStatusCode::kBufferFull,
                     "Expected partial assembly failure to preserve kBufferFull status");
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

bool TestDeviceBufferRuntimeFailsCleanlyWhenBufferIsGenuinelyFull() {
    HostGenotypeBuffer host_buffer{};
    BufferGeneration current_generation{};
    bool ok = TryCreateHostGenotypeBuffer(host_buffer, 2, 4);
    ok &= TryCreateBufferGeneration(current_generation, 2, 4);
    if (!ok) {
        std::cerr << "FAIL: could not allocate full-buffer fixtures\n";
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = 0;
        ok &= TryAllocateBufferSlot(host_buffer, slot_index);
        ok &= TrySetBufferGenerationSlot(current_generation, individual_index, slot_index);
    }
    if (!ok) {
        return false;
    }

    BufferAssemblyPlan plan{};
    ok &= TryCreateBufferAssemblyPlan(plan, 1);
    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};

    DeviceBufferRuntimeConfig runtime_config{};
    runtime_config.slot_count = 2;
    runtime_config.action_count = 4;
    runtime_config.max_generation_size = 2;

    DeviceBufferRuntimeBuffers buffers{};
    ok &= TryCreateDeviceBufferRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadBufferToDevice(host_buffer, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    if (!ok) {
        DestroyDeviceBufferRuntimeBuffers(buffers);
        return false;
    }

    ok &= ExpectTrue(!TryAssembleNextGenerationOnDevice(buffers, 23U, MakeDeterministicAssemblyConfig()),
                     "Expected device buffer assembly to fail when the buffer is genuinely full");

    DeviceBufferRuntimeStatusCode status_code = DeviceBufferRuntimeStatusCode::kOk;
    ok &= TryReadDeviceBufferRuntimeStatus(buffers, status_code);

    HostGenotypeBuffer downloaded_buffer{};
    BufferGeneration downloaded_current_generation{};
    ok &= TryDownloadBufferFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    DestroyDeviceBufferRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(status_code == DeviceBufferRuntimeStatusCode::kBufferFull,
                     "Expected genuinely full device buffer assembly to report kBufferFull");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == 0, "Expected failed assembly to leave the buffer full");
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
        !TestDeviceBufferFreeListIsThreadSafeUnderWarpContention() ||
        !TestDeviceBufferRuntimeUploadsAndDownloadsBufferAndGenerationState() ||
        !TestDeviceBufferRuntimeReusesSweptAndReleasedSlotsDuringAssembly() ||
        !TestDeviceBufferRuntimeAssemblesChildBatchConcurrently() ||
        !TestDeviceBufferRuntimeCleansUpPartialAssemblyWhenLaterBatchFails() ||
        !TestDeviceBufferRuntimeFailsCleanlyWhenBufferIsGenuinelyFull()) {
        return 1;
    }

    std::cout << "PASS: genotype_buffer_device_runtime_test\n";
    return 0;
}
