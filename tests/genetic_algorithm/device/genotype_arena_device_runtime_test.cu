#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/genotype_arena/device_runtime.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::common::ToFloat16;
using neuroevolution::genetic_algorithm::genotype_arena::ArenaAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_arena::ArenaGeneration;
using neuroevolution::genetic_algorithm::genotype_arena::GenomePolicyModelParameters;
using neuroevolution::genetic_algorithm::genotype_arena::GenomeTailRows;
using neuroevolution::genetic_algorithm::genotype_arena::HostArenaSlotBytesAt;
using neuroevolution::genetic_algorithm::genotype_arena::HostGenotypeArena;
using neuroevolution::genetic_algorithm::genotype_arena::TryAllocateArenaSlot;
using neuroevolution::genetic_algorithm::genotype_arena::TryCreateArenaAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_arena::TryCreateArenaGeneration;
using neuroevolution::genetic_algorithm::genotype_arena::TryCreateHostGenotypeArena;
using neuroevolution::genetic_algorithm::genotype_arena::TryReleaseArenaSlot;
using neuroevolution::genetic_algorithm::genotype_arena::TryRetainArenaSlot;
using neuroevolution::genetic_algorithm::genotype_arena::TrySetArenaGenerationSlot;
using neuroevolution::genetic_algorithm::genotype_arena::device::ArenaDeviceAssemblyConfig;
using neuroevolution::genetic_algorithm::genotype_arena::device::DestroyDeviceArenaRuntimeBuffers;
using neuroevolution::genetic_algorithm::genotype_arena::device::DeviceArenaRuntimeBuffers;
using neuroevolution::genetic_algorithm::genotype_arena::device::DeviceArenaRuntimeConfig;
using neuroevolution::genetic_algorithm::genotype_arena::device::DeviceArenaRuntimeStatusCode;
using neuroevolution::genetic_algorithm::genotype_arena::device::TryAssembleNextGenerationWithoutElitismOnDevice;
using neuroevolution::genetic_algorithm::genotype_arena::device::TryCreateDeviceArenaRuntimeBuffers;
using neuroevolution::genetic_algorithm::genotype_arena::device::TryDownloadArenaFromDevice;
using neuroevolution::genetic_algorithm::genotype_arena::device::TryDownloadCurrentGenerationFromDevice;
using neuroevolution::genetic_algorithm::genotype_arena::device::TryDownloadNextGenerationFromDevice;
using neuroevolution::genetic_algorithm::genotype_arena::device::TryReadDeviceArenaRuntimeStatus;
using neuroevolution::genetic_algorithm::genotype_arena::device::TryUploadArenaToDevice;
using neuroevolution::genetic_algorithm::genotype_arena::device::TryUploadAssemblyPlanToDevice;
using neuroevolution::genetic_algorithm::genotype_arena::device::TryUploadCurrentGenerationToDevice;

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

ArenaDeviceAssemblyConfig MakeDeterministicAssemblyConfig() {
    ArenaDeviceAssemblyConfig config{};
    config.breeding.first_parent_probability = 1.0f;
    config.mutation.mutation_probability = 0.0f;
    config.mutation.mutation_sigma = 0.0f;
    return config;
}

bool TestDeviceArenaRuntimeUploadsAndDownloadsArenaAndGenerationState() {
    HostGenotypeArena host_arena{};
    ArenaGeneration current_generation{};
    bool ok = TryCreateHostGenotypeArena(host_arena, 3, 4);
    ok &= TryCreateArenaGeneration(current_generation, 2, 7);
    if (!ok) {
        std::cerr << "FAIL: could not allocate upload/download fixtures\n";
        return false;
    }

    std::uint32_t slot0 = 0;
    std::uint32_t slot1 = 0;
    ok &= TryAllocateArenaSlot(host_arena, slot0);
    ok &= TryAllocateArenaSlot(host_arena, slot1);
    ok &= TryRetainArenaSlot(host_arena, slot0);
    ok &= TryReleaseArenaSlot(host_arena, slot1);
    ok &= TrySetArenaGenerationSlot(current_generation, 0, slot0);
    current_generation.fitness[0] = 9.0f;
    current_generation.evaluation_counts[0] = 3;
    current_generation.has_fitness[0] = 1;
    GenomePolicyModelParameters(HostArenaSlotBytesAt(host_arena, slot0)).dense_trunk.hidden1_to_output.biases[0] = 3.5f;
    GenomeTailRows(HostArenaSlotBytesAt(host_arena, slot0))[1][0] = ToFloat16(-1.25f);

    DeviceArenaRuntimeConfig runtime_config{};
    runtime_config.slot_count = 3;
    runtime_config.action_count = 4;
    runtime_config.max_generation_size = 3;

    DeviceArenaRuntimeBuffers buffers{};
    ok &= TryCreateDeviceArenaRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadArenaToDevice(host_arena, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    if (!ok) {
        DestroyDeviceArenaRuntimeBuffers(buffers);
        std::cerr << "FAIL: could not upload arena runtime fixtures\n";
        return false;
    }

    HostGenotypeArena downloaded_arena{};
    ArenaGeneration downloaded_generation{};
    ok &= TryDownloadArenaFromDevice(buffers, downloaded_arena);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_generation);
    DestroyDeviceArenaRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(downloaded_arena.free_slot_count == 2, "Expected downloaded arena to preserve free-slot count");
    ok &= ExpectTrue(downloaded_arena.slot_states[slot0].occupied,
                     "Expected downloaded arena to preserve live slot state");
    ok &= ExpectTrue(downloaded_arena.slot_states[slot0].reference_count == 2,
                     "Expected downloaded arena to preserve slot reference counts");
    ok &= ExpectTrue(!downloaded_arena.slot_states[slot1].occupied,
                     "Expected downloaded arena to preserve released slot state");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostArenaSlotBytesAt(downloaded_arena, slot0))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     3.5f, "downloaded slot dense trunk bias");
    ok &= ExpectNear(ToFloat(GenomeTailRows(HostArenaSlotBytesAt(downloaded_arena, slot0))[1][0]), -1.25f,
                     "downloaded slot trainable tail value");
    ok &= ExpectTrue(downloaded_generation.generation_index == 7, "Expected downloaded generation index to round-trip");
    ok &= ExpectTrue(downloaded_generation.slot_indices[0] == slot0,
                     "Expected downloaded generation slot handle to round-trip");
    ok &= ExpectTrue(downloaded_generation.slot_indices[1] ==
                         neuroevolution::genetic_algorithm::genotype_arena::kInvalidArenaSlotIndex,
                     "Expected downloaded invalid slot handle to round-trip");
    ok &= ExpectNear(downloaded_generation.fitness[0], 9.0f, "downloaded fitness");
    ok &= ExpectTrue(downloaded_generation.evaluation_counts[0] == 3, "Expected evaluation count to round-trip");
    ok &= ExpectTrue(downloaded_generation.has_fitness[0] == 1, "Expected has_fitness flag to round-trip");
    return ok;
}

bool TestDeviceArenaRuntimeReusesSweptAndReleasedSlotsDuringAssembly() {
    HostGenotypeArena host_arena{};
    ArenaGeneration current_generation{};
    bool ok = TryCreateHostGenotypeArena(host_arena, 3, 4);
    ok &= TryCreateArenaGeneration(current_generation, 3, 5);
    if (!ok) {
        std::cerr << "FAIL: could not allocate assembly fixtures\n";
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = 0;
        ok &= TryAllocateArenaSlot(host_arena, slot_index);
        ok &= TrySetArenaGenerationSlot(current_generation, individual_index, slot_index);
        GenomePolicyModelParameters(HostArenaSlotBytesAt(host_arena, slot_index))
            .dense_trunk.hidden1_to_output.biases[0] = ToFloat16(11.0f * static_cast<float>(individual_index + 1));
        GenomeTailRows(HostArenaSlotBytesAt(host_arena, slot_index))[0][0] =
            ToFloat16(2.0f * static_cast<float>(individual_index + 1));
    }
    if (!ok) {
        return false;
    }

    ArenaAssemblyPlan plan{};
    ok &= TryCreateArenaAssemblyPlan(plan, 2);
    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};
    plan.parent_pairs[1] = {.first_parent_index = 0, .second_parent_index = 0};

    DeviceArenaRuntimeConfig runtime_config{};
    runtime_config.slot_count = 3;
    runtime_config.action_count = 4;
    runtime_config.max_generation_size = 3;

    DeviceArenaRuntimeBuffers buffers{};
    ok &= TryCreateDeviceArenaRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadArenaToDevice(host_arena, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    ok &= TryAssembleNextGenerationWithoutElitismOnDevice(buffers, 17U, MakeDeterministicAssemblyConfig());
    if (!ok) {
        DeviceArenaRuntimeStatusCode status_code = DeviceArenaRuntimeStatusCode::kOk;
        (void)TryReadDeviceArenaRuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: device arena assembly failed with status " << static_cast<int>(status_code) << '\n';
        DestroyDeviceArenaRuntimeBuffers(buffers);
        return false;
    }

    HostGenotypeArena downloaded_arena{};
    ArenaGeneration downloaded_current_generation{};
    ArenaGeneration downloaded_next_generation{};
    ok &= TryDownloadArenaFromDevice(buffers, downloaded_arena);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    ok &= TryDownloadNextGenerationFromDevice(buffers, downloaded_next_generation);
    DestroyDeviceArenaRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(downloaded_next_generation.generation_index == 6,
                     "Expected device arena assembly to increment the generation index");
    ok &= ExpectTrue(downloaded_next_generation.slot_indices[0] == 2U,
                     "Expected the first child to reuse the pre-swept zero-duty parent slot");
    ok &= ExpectTrue(downloaded_next_generation.slot_indices[1] == 1U,
                     "Expected the second child to reuse the parent slot freed after its final duty");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(
                                 HostArenaSlotBytesAt(downloaded_arena, downloaded_next_generation.slot_indices[0]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     11.0f, "first child bias copied from first parent");
    ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(
                                 HostArenaSlotBytesAt(downloaded_arena, downloaded_next_generation.slot_indices[1]))
                                 .dense_trunk.hidden1_to_output.biases[0]),
                     11.0f, "second child bias copied from first parent");
    ok &= ExpectNear(ToFloat(GenomeTailRows(
                         HostArenaSlotBytesAt(downloaded_arena, downloaded_next_generation.slot_indices[0]))[0][0]),
                     2.0f, "first child trainable tail copied from first parent");
    ok &= ExpectNear(ToFloat(GenomeTailRows(
                         HostArenaSlotBytesAt(downloaded_arena, downloaded_next_generation.slot_indices[1]))[0][0]),
                     2.0f, "second child trainable tail copied from first parent");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[0] ==
                         neuroevolution::genetic_algorithm::genotype_arena::kInvalidArenaSlotIndex,
                     "Expected the first parent slot handle to be cleared after its final duty");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[1] ==
                         neuroevolution::genetic_algorithm::genotype_arena::kInvalidArenaSlotIndex,
                     "Expected the second parent slot handle to be cleared after its final duty");
    ok &= ExpectTrue(downloaded_current_generation.slot_indices[2] ==
                         neuroevolution::genetic_algorithm::genotype_arena::kInvalidArenaSlotIndex,
                     "Expected zero-duty parents to be swept on-device before child assembly");
    ok &= ExpectTrue(downloaded_arena.free_slot_count == 1,
                     "Expected one free slot to remain after two children occupy the reused arena slots");
    return ok;
}

bool TestDeviceArenaRuntimeFailsCleanlyWhenArenaIsGenuinelyFull() {
    HostGenotypeArena host_arena{};
    ArenaGeneration current_generation{};
    bool ok = TryCreateHostGenotypeArena(host_arena, 2, 4);
    ok &= TryCreateArenaGeneration(current_generation, 2, 4);
    if (!ok) {
        std::cerr << "FAIL: could not allocate full-arena fixtures\n";
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < current_generation.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = 0;
        ok &= TryAllocateArenaSlot(host_arena, slot_index);
        ok &= TrySetArenaGenerationSlot(current_generation, individual_index, slot_index);
    }
    if (!ok) {
        return false;
    }

    ArenaAssemblyPlan plan{};
    ok &= TryCreateArenaAssemblyPlan(plan, 1);
    plan.parent_pairs[0] = {.first_parent_index = 0, .second_parent_index = 1};

    DeviceArenaRuntimeConfig runtime_config{};
    runtime_config.slot_count = 2;
    runtime_config.action_count = 4;
    runtime_config.max_generation_size = 2;

    DeviceArenaRuntimeBuffers buffers{};
    ok &= TryCreateDeviceArenaRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadArenaToDevice(host_arena, buffers);
    ok &= TryUploadCurrentGenerationToDevice(current_generation, buffers);
    ok &= TryUploadAssemblyPlanToDevice(plan, buffers);
    if (!ok) {
        DestroyDeviceArenaRuntimeBuffers(buffers);
        return false;
    }

    ok &= ExpectTrue(!TryAssembleNextGenerationWithoutElitismOnDevice(buffers, 23U, MakeDeterministicAssemblyConfig()),
                     "Expected device arena assembly to fail when the arena is genuinely full");

    DeviceArenaRuntimeStatusCode status_code = DeviceArenaRuntimeStatusCode::kOk;
    ok &= TryReadDeviceArenaRuntimeStatus(buffers, status_code);

    HostGenotypeArena downloaded_arena{};
    ArenaGeneration downloaded_current_generation{};
    ok &= TryDownloadArenaFromDevice(buffers, downloaded_arena);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_current_generation);
    DestroyDeviceArenaRuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(status_code == DeviceArenaRuntimeStatusCode::kArenaFull,
                     "Expected genuinely full device arena assembly to report kArenaFull");
    ok &= ExpectTrue(downloaded_arena.free_slot_count == 0, "Expected failed assembly to leave the arena full");
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

    if (!TestDeviceArenaRuntimeUploadsAndDownloadsArenaAndGenerationState() ||
        !TestDeviceArenaRuntimeReusesSweptAndReleasedSlotsDuringAssembly() ||
        !TestDeviceArenaRuntimeFailsCleanlyWhenArenaIsGenuinelyFull()) {
        return 1;
    }

    std::cout << "PASS: genotype_arena_device_runtime_test\n";
    return 0;
}
