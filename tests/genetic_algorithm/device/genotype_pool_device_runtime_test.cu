#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/genotype_pool/device_runtime.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::common::ToFloat16;
using neuroevolution::genetic_algorithm::genotype_pool::GenomePolicyModelParameters;
using neuroevolution::genetic_algorithm::genotype_pool::GenomeTailRows;
using neuroevolution::genetic_algorithm::genotype_pool::HostGenotypePool;
using neuroevolution::genetic_algorithm::genotype_pool::HostPoolSlotBytesAt;
using neuroevolution::genetic_algorithm::genotype_pool::PoolAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_pool::PoolGeneration;
using neuroevolution::genetic_algorithm::genotype_pool::TryAllocatePoolSlot;
using neuroevolution::genetic_algorithm::genotype_pool::TryCreateHostGenotypePool;
using neuroevolution::genetic_algorithm::genotype_pool::TryCreatePoolAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_pool::TryCreatePoolGeneration;
using neuroevolution::genetic_algorithm::genotype_pool::TryReleasePoolSlot;
using neuroevolution::genetic_algorithm::genotype_pool::TryRetainPoolSlot;
using neuroevolution::genetic_algorithm::genotype_pool::TrySetPoolGenerationSlot;
using neuroevolution::genetic_algorithm::genotype_pool::device::DestroyDevicePoolRuntimeBuffers;
using neuroevolution::genetic_algorithm::genotype_pool::device::DevicePoolRuntimeBuffers;
using neuroevolution::genetic_algorithm::genotype_pool::device::DevicePoolRuntimeConfig;
using neuroevolution::genetic_algorithm::genotype_pool::device::DevicePoolRuntimeStatusCode;
using neuroevolution::genetic_algorithm::genotype_pool::device::PoolDeviceAssemblyConfig;
using neuroevolution::genetic_algorithm::genotype_pool::device::TryAssembleNextGenerationOnDevice;
using neuroevolution::genetic_algorithm::genotype_pool::device::TryCreateDevicePoolRuntimeBuffers;
using neuroevolution::genetic_algorithm::genotype_pool::device::TryDownloadCurrentGenerationFromDevice;
using neuroevolution::genetic_algorithm::genotype_pool::device::TryDownloadNextGenerationFromDevice;
using neuroevolution::genetic_algorithm::genotype_pool::device::TryDownloadPoolFromDevice;
using neuroevolution::genetic_algorithm::genotype_pool::device::TryReadDevicePoolRuntimeStatus;
using neuroevolution::genetic_algorithm::genotype_pool::device::TryUploadAssemblyPlanToDevice;
using neuroevolution::genetic_algorithm::genotype_pool::device::TryUploadCurrentGenerationToDevice;
using neuroevolution::genetic_algorithm::genotype_pool::device::TryUploadPoolToDevice;

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

PoolDeviceAssemblyConfig MakeDeterministicAssemblyConfig() {
    PoolDeviceAssemblyConfig config{};
    config.breeding.first_parent_probability = 1.0f;
    config.mutation.mutation_probability = 0.0f;
    config.mutation.mutation_sigma = 0.0f;
    return config;
}

bool TestDevicePoolRuntimeUploadsAndDownloadsPoolAndGenerationState() {
    HostGenotypePool host_pool{};
    PoolGeneration current_generation{};
    bool ok = TryCreateHostGenotypePool(host_pool, 3, 4);
    ok &= TryCreatePoolGeneration(current_generation, 2, 7);
    if (!ok) {
        std::cerr << "FAIL: could not allocate upload/download fixtures\n";
        return false;
    }

    std::uint32_t slot0 = 0;
    std::uint32_t slot1 = 0;
    ok &= TryAllocatePoolSlot(host_pool, slot0);
    ok &= TryAllocatePoolSlot(host_pool, slot1);
    ok &= TryRetainPoolSlot(host_pool, slot0);
    ok &= TryReleasePoolSlot(host_pool, slot1);
    ok &= TrySetPoolGenerationSlot(current_generation, 0, slot0);
    current_generation.fitness[0] = 9.0f;
    current_generation.evaluation_counts[0] = 3;
    current_generation.has_fitness[0] = 1;
    GenomePolicyModelParameters(HostPoolSlotBytesAt(host_pool, slot0)).dense_trunk.hidden1_to_output.biases[0] = 3.5f;
    GenomeTailRows(HostPoolSlotBytesAt(host_pool, slot0))[1][0] = ToFloat16(-1.25f);

    DevicePoolRuntimeConfig runtime_config{};
    runtime_config.slot_count = 3;
    runtime_config.action_count = 4;
    runtime_config.max_generation_size = 3;

    DevicePoolRuntimeBuffers buffers{};
    ok &= TryCreateDevicePoolRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadPoolToDevice(host_pool, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    if (!ok) {
        DestroyDevicePoolRuntimeBuffers(buffers);
        std::cerr << "FAIL: could not upload pool runtime fixtures\n";
        return false;
    }

    HostGenotypePool downloaded_pool{};
    PoolGeneration downloaded_generation{};
    ok &= TryDownloadPoolFromDevice(buffers, downloaded_pool);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_generation);
    DestroyDevicePoolRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(downloaded_pool.free_slot_count == 2, "Expected downloaded pool to preserve free-slot count");
    ok &=
        ExpectTrue(downloaded_pool.slot_states[slot0].occupied, "Expected downloaded pool to preserve live slot state");
    ok &= ExpectTrue(downloaded_pool.slot_states[slot0].reference_count == 2,
                     "Expected downloaded pool to preserve slot reference counts");
    ok &= ExpectTrue(!downloaded_pool.slot_states[slot1].occupied,
                     "Expected downloaded pool to preserve released slot state");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostPoolSlotBytesAt(downloaded_pool, slot0))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     3.5f, "downloaded slot dense trunk bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostPoolSlotBytesAt(downloaded_pool, slot0))[1][0]), -1.25f,
                     "downloaded slot trainable tail value");
    ok &= ExpectTrue(downloaded_generation.generation_index == 7, "Expected downloaded generation index to round-trip");
    ok &= ExpectTrue(downloaded_generation.slot_indices[0] == slot0,
                     "Expected downloaded generation slot handle to round-trip");
    ok &= ExpectTrue(downloaded_generation.slot_indices[1] ==
                         neuroevolution::genetic_algorithm::genotype_pool::kInvalidPoolSlotIndex,
                     "Expected downloaded invalid slot handle to round-trip");
    ok &= ExpectNear(downloaded_generation.fitness[0], 9.0f, "downloaded fitness");
    ok &= ExpectTrue(downloaded_generation.evaluation_counts[0] == 3, "Expected evaluation count to round-trip");
    ok &= ExpectTrue(downloaded_generation.has_fitness[0] == 1, "Expected has_fitness flag to round-trip");
    return ok;
}

bool TestDevicePoolRuntimeReusesSweptAndReleasedSlotsDuringAssembly() {
    HostGenotypePool host_pool{};
    PoolGeneration current_generation{};
    bool ok = TryCreateHostGenotypePool(host_pool, 3, 4);
    ok &= TryCreatePoolGeneration(current_generation, 3, 5);
    if (!ok) {
        std::cerr << "FAIL: could not allocate assembly fixtures\n";
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = 0;
        ok &= TryAllocatePoolSlot(host_pool, slot_index);
        ok &= TrySetPoolGenerationSlot(current_generation, individual_index, slot_index);
        GenomePolicyModelParameters(HostPoolSlotBytesAt(host_pool, slot_index))
            .dense_trunk.hidden1_to_output.biases[0] = ToFloat16(11.0f * static_cast<float>(individual_index + 1));
        GenomeTailRows(HostPoolSlotBytesAt(host_pool, slot_index))[0][0] =
            ToFloat16(2.0f * static_cast<float>(individual_index + 1));
    }
    if (!ok) {
        return false;
    }

    PoolAssemblyPlan plan{};
    ok &= TryCreatePoolAssemblyPlan(plan, 2);
    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};
    plan.parent_pairs[1] = {.first_parent_index = 0, .second_parent_index = 0};

    DevicePoolRuntimeConfig runtime_config{};
    runtime_config.slot_count = 3;
    runtime_config.action_count = 4;
    runtime_config.max_generation_size = 3;

    DevicePoolRuntimeBuffers buffers{};
    ok &= TryCreateDevicePoolRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadPoolToDevice(host_pool, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    ok &= TryAssembleNextGenerationOnDevice(buffers, 17U, MakeDeterministicAssemblyConfig());
    if (!ok) {
        DevicePoolRuntimeStatusCode status_code = DevicePoolRuntimeStatusCode::kOk;
        (void)TryReadDevicePoolRuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: device pool assembly failed with status " << static_cast<int>(status_code) << '\n';
        DestroyDevicePoolRuntimeBuffers(buffers);
        return false;
    }

    HostGenotypePool downloaded_pool{};
    PoolGeneration downloaded_current_generation{};
    PoolGeneration downloaded_next_generation{};
    ok &= TryDownloadPoolFromDevice(buffers, downloaded_pool);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    ok &= TryDownloadNextGenerationFromDevice(buffers, downloaded_next_generation);
    DestroyDevicePoolRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(downloaded_next_generation.generation_index == 6,
                     "Expected device pool assembly to increment the generation index");
    ok &= ExpectTrue(downloaded_next_generation.slot_indices[0] == 2U,
                     "Expected the first child to reuse the garbage-collected zero-reference parent slot");
    ok &= ExpectTrue(downloaded_next_generation.slot_indices[1] == 1U,
                     "Expected the second child to reuse the parent slot freed after its final parent reference");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(
                                 HostPoolSlotBytesAt(downloaded_pool, downloaded_next_generation.slot_indices[0]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     11.0f, "first child bias copied from first parent");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(
                                 HostPoolSlotBytesAt(downloaded_pool, downloaded_next_generation.slot_indices[1]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     11.0f, "second child bias copied from first parent");
    ok &= ExpectNear(
        ToFloat(GenomeTailRows(HostPoolSlotBytesAt(downloaded_pool, downloaded_next_generation.slot_indices[0]))[0][0]),
        2.0f, "first child trainable tail copied from first parent");
    ok &= ExpectNear(
        ToFloat(GenomeTailRows(HostPoolSlotBytesAt(downloaded_pool, downloaded_next_generation.slot_indices[1]))[0][0]),
        2.0f, "second child trainable tail copied from first parent");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[0] ==
                         neuroevolution::genetic_algorithm::genotype_pool::kInvalidPoolSlotIndex,
                     "Expected the first parent slot handle to be cleared after its final parent reference");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[1] ==
                         neuroevolution::genetic_algorithm::genotype_pool::kInvalidPoolSlotIndex,
                     "Expected the second parent slot handle to be cleared after its final parent reference");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[2] ==
                         neuroevolution::genetic_algorithm::genotype_pool::kInvalidPoolSlotIndex,
                     "Expected zero-reference parents to be garbage-collected on-device before child assembly");
    ok &= ExpectTrue(downloaded_pool.free_slot_count == 1,
                     "Expected one free slot to remain after two children occupy the reused pool slots");
    return ok;
}

bool TestDevicePoolRuntimeFailsCleanlyWhenPoolIsGenuinelyFull() {
    HostGenotypePool host_pool{};
    PoolGeneration current_generation{};
    bool ok = TryCreateHostGenotypePool(host_pool, 2, 4);
    ok &= TryCreatePoolGeneration(current_generation, 2, 4);
    if (!ok) {
        std::cerr << "FAIL: could not allocate full-pool fixtures\n";
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = 0;
        ok &= TryAllocatePoolSlot(host_pool, slot_index);
        ok &= TrySetPoolGenerationSlot(current_generation, individual_index, slot_index);
    }
    if (!ok) {
        return false;
    }

    PoolAssemblyPlan plan{};
    ok &= TryCreatePoolAssemblyPlan(plan, 1);
    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};

    DevicePoolRuntimeConfig runtime_config{};
    runtime_config.slot_count = 2;
    runtime_config.action_count = 4;
    runtime_config.max_generation_size = 2;

    DevicePoolRuntimeBuffers buffers{};
    ok &= TryCreateDevicePoolRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadPoolToDevice(host_pool, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    if (!ok) {
        DestroyDevicePoolRuntimeBuffers(buffers);
        return false;
    }

    ok &= ExpectTrue(!TryAssembleNextGenerationOnDevice(buffers, 23U, MakeDeterministicAssemblyConfig()),
                     "Expected device pool assembly to fail when the pool is genuinely full");

    DevicePoolRuntimeStatusCode status_code = DevicePoolRuntimeStatusCode::kOk;
    ok &= TryReadDevicePoolRuntimeStatus(buffers, status_code);

    HostGenotypePool downloaded_pool{};
    PoolGeneration downloaded_current_generation{};
    ok &= TryDownloadPoolFromDevice(buffers, downloaded_pool);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    DestroyDevicePoolRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(status_code == DevicePoolRuntimeStatusCode::kPoolFull,
                     "Expected genuinely full device pool assembly to report kPoolFull");
    ok &= ExpectTrue(downloaded_pool.free_slot_count == 0, "Expected failed assembly to leave the pool full");
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

    if (!TestDevicePoolRuntimeUploadsAndDownloadsPoolAndGenerationState() ||
        !TestDevicePoolRuntimeReusesSweptAndReleasedSlotsDuringAssembly() ||
        !TestDevicePoolRuntimeFailsCleanlyWhenPoolIsGenuinelyFull()) {
        return 1;
    }

    std::cout << "PASS: genotype_pool_device_runtime_test\n";
    return 0;
}
