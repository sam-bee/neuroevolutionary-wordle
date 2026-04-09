#pragma once

#include <cstddef>
#include <cstdint>

#include "genetic_algorithm/generation_assembly.hpp"
#include "training_folder/training_data.hpp"
#include "wordle/word.hpp"

namespace neuroevolution::genetic_algorithm::device {

constexpr std::size_t kDeviceActionCount = training_folder::kInitialTrainingDataShardEntryCount;
constexpr std::size_t kDevicePopulationSize = 6;

using DeviceGenome = ModelGenome<kDeviceActionCount>;
using DevicePopulation = Population<DeviceGenome, kDevicePopulationSize>;

enum class DeviceRuntimeStatusCode : int {
    kOk = 0,
    kCudaFailure = 1,
    kInvalidTrainingShard = 2,
    kOpeningGuessAppendFailed = 3,
    kPolicyForwardFailed = 4,
    kActionSelectionFailed = 5,
    kPopulationNotEvaluated = 6,
    kInvalidAssemblyConfig = 7,
    kParentSelectionFailed = 8,
};

struct PopulationFitnessSummary {
    float best_fitness = 0.0f;
    float average_fitness = 0.0f;
    std::size_t best_index = 0;
    std::size_t generation_index = 0;
};

struct DeviceRuntimeBuffers {
    DevicePopulation *current_population = nullptr;
    DevicePopulation *next_population = nullptr;
    PopulationFitnessSummary *summary = nullptr;
    int *status = nullptr;
};

bool TryCreateDeviceRuntimeBuffers(DeviceRuntimeBuffers &buffers);

void DestroyDeviceRuntimeBuffers(DeviceRuntimeBuffers &buffers) noexcept;

bool TryUploadCurrentPopulationToDevice(const DevicePopulation &host_population, DeviceRuntimeBuffers &buffers);

bool TryDownloadCurrentPopulationFromDevice(const DeviceRuntimeBuffers &buffers, DevicePopulation &host_population);

bool TryEvaluatePopulationFitnessOnDevice(DeviceRuntimeBuffers &buffers, const wordle::Word &opening_guess);

bool TryReadPopulationFitnessSummaryFromDevice(const DeviceRuntimeBuffers &buffers, PopulationFitnessSummary &summary);

bool TryReadDeviceRuntimeStatus(const DeviceRuntimeBuffers &buffers, DeviceRuntimeStatusCode &status_code);

bool TryAssembleNextGenerationOnDevice(DeviceRuntimeBuffers &buffers, std::uint32_t generation_seed,
                                       const GenerationAssemblyConfig &config = {});

void SwapDevicePopulationBuffers(DeviceRuntimeBuffers &buffers) noexcept;

const char *DeviceRuntimeStatusCodeString(DeviceRuntimeStatusCode status_code) noexcept;

} // namespace neuroevolution::genetic_algorithm::device
