#include <cuda_runtime.h>

#include <iostream>
#include <string_view>

#include "genetic_algorithm/device/device_runtime.hpp"
#include "genetic_algorithm/genetic_algorithm.hpp"
#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::genetic_algorithm::GenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::InitializePopulation;
using neuroevolution::genetic_algorithm::PopulationInitializationRandomEngine;
using neuroevolution::genetic_algorithm::device::DestroyDeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::device::DevicePopulation;
using neuroevolution::genetic_algorithm::device::DeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::device::PopulationFitnessSummary;
using neuroevolution::genetic_algorithm::device::SwapDevicePopulationBuffers;
using neuroevolution::genetic_algorithm::device::TryAssembleNextGenerationOnDevice;
using neuroevolution::genetic_algorithm::device::TryCreateDeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::device::TryEvaluatePopulationFitnessOnDevice;
using neuroevolution::genetic_algorithm::device::TryReadPopulationFitnessSummaryFromDevice;
using neuroevolution::genetic_algorithm::device::TryUploadCurrentPopulationToDevice;
using neuroevolution::training_folder::LoadInitialTrainingDataShardFromActionSpace;
using neuroevolution::training_folder::UploadTrainingDataShardToDeviceConstantMemory;

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

bool TestDeviceRuntimeEvaluatesAndAssemblesPopulationsOnDevice() {
    const auto training_shard = LoadInitialTrainingDataShardFromActionSpace();
    if (!UploadTrainingDataShardToDeviceConstantMemory(training_shard)) {
        std::cerr << "FAIL: could not upload training-data shard to device constant memory\n";
        return false;
    }

    PopulationInitializationRandomEngine initialization_random_engine(42);
    const DevicePopulation host_population =
        InitializePopulation<neuroevolution::genetic_algorithm::device::kDeviceActionCount,
                             neuroevolution::genetic_algorithm::device::kDevicePopulationSize>(
            initialization_random_engine);

    DeviceRuntimeBuffers buffers{};
    bool ok = TryCreateDeviceRuntimeBuffers(buffers);
    ok &= TryUploadCurrentPopulationToDevice(host_population, buffers);
    ok &= TryEvaluatePopulationFitnessOnDevice(buffers);

    PopulationFitnessSummary summary_generation_0{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary_generation_0);

    GenerationAssemblyConfig assembly_config{};
    assembly_config.genetic_algorithm.elite_count = 1;
    assembly_config.parent_selection.tournament_size = 3;
    assembly_config.parent_selection.allow_self_parenting = false;
    assembly_config.breeding.first_parent_probability = 0.5f;
    assembly_config.mutation.mutation_probability = 0.02f;
    assembly_config.mutation.mutation_sigma = 0.05f;

    ok &= TryAssembleNextGenerationOnDevice(buffers, 77U, assembly_config);
    SwapDevicePopulationBuffers(buffers);
    ok &= TryEvaluatePopulationFitnessOnDevice(buffers);

    PopulationFitnessSummary summary_generation_1{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary_generation_1);

    DestroyDeviceRuntimeBuffers(buffers);

    ok &= ExpectTrue(summary_generation_0.generation_index == 0, "Expected initial device population generation zero");
    ok &= ExpectTrue(summary_generation_1.generation_index == 1,
                     "Expected device-assembled next generation to increment generation index");
    ok &= ExpectTrue(summary_generation_0.best_index < neuroevolution::genetic_algorithm::device::kDevicePopulationSize,
                     "Expected valid best index for generation zero");
    ok &= ExpectTrue(summary_generation_1.best_index < neuroevolution::genetic_algorithm::device::kDevicePopulationSize,
                     "Expected valid best index for generation one");
    ok &= ExpectInRange(summary_generation_0.best_fitness, 0.0f, static_cast<float>(training_shard.entry_count),
                        "generation zero best fitness");
    ok &= ExpectInRange(summary_generation_0.average_fitness, 0.0f, static_cast<float>(training_shard.entry_count),
                        "generation zero average fitness");
    ok &= ExpectInRange(summary_generation_1.best_fitness, 0.0f, static_cast<float>(training_shard.entry_count),
                        "generation one best fitness");
    ok &= ExpectInRange(summary_generation_1.average_fitness, 0.0f, static_cast<float>(training_shard.entry_count),
                        "generation one average fitness");
    return ok;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    if (!TestDeviceRuntimeEvaluatesAndAssemblesPopulationsOnDevice()) {
        return 1;
    }

    std::cout << "PASS: device_runtime_test\n";
    return 0;
}
