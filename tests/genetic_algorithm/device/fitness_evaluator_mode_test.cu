#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>

#include "genetic_algorithm/device/slab_runtime.hpp"
#include "genetic_algorithm/genetic_algorithm.hpp"
#include "genetic_algorithm/genotype_slab/layout.hpp"
#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::genetic_algorithm::genotype_slab::ComputeSlabSlotStrideBytes;
using neuroevolution::genetic_algorithm::genotype_slab::SlabGeneration;
using neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers;
using neuroevolution::genetic_algorithm::slab_device::DeviceSlabGARuntimeBuffers;
using neuroevolution::genetic_algorithm::slab_device::DeviceSlabGARuntimeConfig;
using neuroevolution::genetic_algorithm::slab_device::PopulationFitnessSummary;
using neuroevolution::genetic_algorithm::slab_device::RuntimeWordCounts;
using neuroevolution::genetic_algorithm::slab_device::TryBootstrapRandomCurrentGenerationOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryCreateDeviceSlabGARuntimeBuffers;
using neuroevolution::genetic_algorithm::slab_device::TryDownloadCurrentGenerationFromDevice;
using neuroevolution::genetic_algorithm::slab_device::TryEvaluateCurrentGenerationFitnessOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryReadPopulationFitnessSummaryFromDevice;
using neuroevolution::training_folder::LoadTrainingWordCatalogFromActionSpace;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::training_folder::UploadTrainingWordCatalogToDeviceConstantMemory;

constexpr int kSelectedVisibleDeviceIndex = 0;
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

bool ExpectInRange(const float value, const float minimum, const float maximum, const std::string_view label) {
    if ((value < minimum) || (value > maximum)) {
        std::cerr << "FAIL: " << label << " expected range [" << minimum << ", " << maximum << "], got " << value
                  << '\n';
        return false;
    }

    return true;
}

RuntimeWordCounts MakeRuntimeWordCounts(const std::size_t action_count) {
    RuntimeWordCounts runtime_word_counts{};
    runtime_word_counts.training_word_count = action_count;
    runtime_word_counts.action_space_word_count = action_count;
    runtime_word_counts.training_word_schedule.initial_word_count = action_count;
    runtime_word_counts.training_word_schedule.word_count_step = 0;
    runtime_word_counts.training_word_schedule.word_count_step_period_generations = 1;
    return runtime_word_counts;
}

DeviceSlabGARuntimeConfig MakeRuntimeConfig(const std::size_t slot_count, const std::size_t generation_size,
                                            const std::size_t action_count) {
    const std::size_t slot_stride_bytes = ComputeSlabSlotStrideBytes(action_count);

    DeviceSlabGARuntimeConfig runtime_config{};
    runtime_config.genotype_slab_byte_budget_bytes = slot_count * slot_stride_bytes;
    runtime_config.generation_byte_budget_bytes = generation_size * slot_stride_bytes;
    runtime_config.host_spillover_byte_budget_bytes = runtime_config.generation_byte_budget_bytes / 2;
    runtime_config.action_count = action_count;
    runtime_config.population_size_ceiling = generation_size;
    return runtime_config;
}

bool InitializeTrainingCatalog(TrainingWordCatalog &training_word_catalog) {
    training_word_catalog = LoadTrainingWordCatalogFromActionSpace();
    return UploadTrainingWordCatalogToDeviceConstantMemory(training_word_catalog);
}

bool EvaluateAndCheckSaneFitness(const std::size_t action_count, const std::uint32_t seed) {
    constexpr std::size_t kGenerationSize = 4;
    constexpr std::size_t kSlotCount = 16;

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(kSlotCount, kGenerationSize, action_count);
    const RuntimeWordCounts runtime_word_counts = MakeRuntimeWordCounts(action_count);

    DeviceSlabGARuntimeBuffers buffers{};
    bool ok = TryCreateDeviceSlabGARuntimeBuffers(buffers, runtime_config);
    ok &= TryBootstrapRandomCurrentGenerationOnDevice(buffers, kGenerationSize, seed, 0);
    ok &= TryEvaluateCurrentGenerationFitnessOnDevice(buffers, runtime_word_counts);

    PopulationFitnessSummary summary{};
    SlabGeneration generation{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, generation);
    DestroyDeviceSlabGARuntimeBuffers(buffers);
    if (!ok) {
        std::cerr << "FAIL: could not evaluate current generation at action_count=" << action_count << '\n';
        return false;
    }

    ok &= ExpectTrue(summary.population_size == kGenerationSize, "Expected summary to report the evaluated population");
    ok &= ExpectTrue(summary.action_count == action_count, "Expected summary to report the configured action count");
    ok &= ExpectTrue(summary.best_index < generation.active_individual_count,
                     "Expected best index to reference a live organism");
    ok &= ExpectInRange(summary.best_fitness, neuroevolution::spatial::kPositiveSelectionFitnessFloor, 1.0f,
                        "summary best fitness");
    ok &= ExpectInRange(summary.average_fitness, neuroevolution::spatial::kPositiveSelectionFitnessFloor, 1.0f,
                        "summary average fitness");

    for (std::size_t individual_index = 0; individual_index < generation.active_individual_count; ++individual_index) {
        ok &= ExpectTrue(generation.has_fitness[individual_index] == 1,
                         "Expected every organism to receive a fitness flag");
        ok &= ExpectTrue(generation.evaluation_counts[individual_index] == 1,
                         "Expected every organism to be evaluated exactly once");
        ok &= ExpectInRange(generation.fitness[individual_index], neuroevolution::spatial::kPositiveSelectionFitnessFloor,
                            1.0f, "current-generation fitness");
    }

    return ok;
}

bool TestFitnessEvaluatorCoversBothKernelModes() {
    bool ok = true;
    TrainingWordCatalog training_word_catalog{};
    ok &= InitializeTrainingCatalog(training_word_catalog);
    ok &= ExpectTrue(training_word_catalog.word_count >= 2000,
                     "Expected the action-space catalog to contain at least 2000 words");
    if (!ok) {
        return false;
    }

    ok &= EvaluateAndCheckSaneFitness(20, 7U);
    ok &= EvaluateAndCheckSaneFitness(2000, 11U);
    return ok;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    if (!TestFitnessEvaluatorCoversBothKernelModes()) {
        return 1;
    }

    std::cout << "PASS: fitness_evaluator_mode_test\n";
    return 0;
}
