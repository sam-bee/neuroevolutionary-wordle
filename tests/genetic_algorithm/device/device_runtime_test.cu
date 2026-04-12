#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/device/device_runtime.hpp"
#include "genetic_algorithm/genetic_algorithm.hpp"
#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::genetic_algorithm::GenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::PopulationInitializationRandomEngine;
using neuroevolution::genetic_algorithm::TrySeedOutputEmbeddingTailFromHintGrids;
using neuroevolution::genetic_algorithm::TryInitializePopulation;
using neuroevolution::genetic_algorithm::device::DeviceRuntimeStatusCode;
using neuroevolution::genetic_algorithm::device::DestroyDeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::device::DevicePopulation;
using neuroevolution::genetic_algorithm::device::DeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::device::PendingOutputEmbeddingInjection;
using neuroevolution::genetic_algorithm::device::PopulationFitnessSummary;
using neuroevolution::genetic_algorithm::device::RuntimeWordCounts;
using neuroevolution::genetic_algorithm::device::SwapDevicePopulationBuffers;
using neuroevolution::genetic_algorithm::device::TryAssembleNextGenerationOnDevice;
using neuroevolution::genetic_algorithm::device::TryCreateDeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::device::TryDownloadCurrentPopulationFromDevice;
using neuroevolution::genetic_algorithm::device::TryEvaluatePopulationFitnessOnDevice;
using neuroevolution::genetic_algorithm::device::TryReadDeviceRuntimeStatus;
using neuroevolution::genetic_algorithm::device::TryReadPopulationFitnessSummaryFromDevice;
using neuroevolution::genetic_algorithm::device::TryUploadCurrentPopulationToDevice;
using neuroevolution::training_folder::ActiveTrainingWordCountForGeneration;
using neuroevolution::training_folder::kPhasedCurriculumSecondShardGeneration;
using neuroevolution::training_folder::LoadTrainingWordCatalogFromActionSpace;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::training_folder::UploadTrainingWordCatalogToDeviceConstantMemory;
using neuroevolution::wordle::Word;

constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr std::size_t kTestPopulationSize = 6;
constexpr float kMaximumEpisodeWinScore = 15.0f;
constexpr float kEpisodesPerTrainingEntry = 3.0f;
constexpr float kTailTolerance = 1.0e-3f;

float MaximumPossibleFitnessForGeneration(const TrainingWordCatalog &training_word_catalog,
                                          const RuntimeWordCounts &runtime_word_counts,
                                          const std::size_t generation_index) {
    (void)training_word_catalog;
    return kEpisodesPerTrainingEntry * kMaximumEpisodeWinScore *
           static_cast<float>(ActiveTrainingWordCountForGeneration(runtime_word_counts.training_word_count,
                                                                   generation_index));
}

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

bool ExpectNear(const float actual, const float expected, const std::string_view label) {
    if (std::fabs(actual - expected) > kTailTolerance) {
        std::cerr << "FAIL: " << label << " expected " << expected << ", got " << actual << '\n';
        return false;
    }

    return true;
}

void SetPopulationActiveOutputEmbeddingCount(DevicePopulation &population, const std::size_t active_count) {
    for (std::size_t individual_index = 0; individual_index < population.active_individual_count; ++individual_index) {
        population.individuals[individual_index].genome.output_embedding.active_count = active_count;
    }
}

void CopyGenomeAcrossPopulation(DevicePopulation &population, const std::size_t source_index) {
    const auto genome = population.individuals[source_index].genome;

    for (std::size_t individual_index = 0; individual_index < population.active_individual_count; ++individual_index) {
        population.individuals[individual_index].genome = genome;
    }
}

bool ExpectPopulationTailMatchesExpected(const DevicePopulation &population, const std::size_t tail_index,
                                         const neuroevolution::model::output_embedding::TrainableActionEmbeddingTail &expected_tail,
                                         const std::string_view label_prefix) {
    bool ok = true;

    for (std::size_t individual_index = 0; individual_index < population.active_individual_count; ++individual_index) {
        for (std::size_t feature_index = 0;
             feature_index < neuroevolution::model::output_embedding::kTrainableFeatureDimension; ++feature_index) {
            ok &= ExpectNear(
                neuroevolution::common::ToFloat(
                    population.individuals[individual_index].genome.output_embedding.trainable_tails[tail_index][feature_index]),
                neuroevolution::common::ToFloat(expected_tail[feature_index]),
                std::string(label_prefix) + " individual " + std::to_string(individual_index) + " feature " +
                    std::to_string(feature_index));
        }
    }

    return ok;
}

bool TestDeviceRuntimeEvaluatesAndAssemblesPopulationsOnDevice() {
    const auto training_word_catalog = LoadTrainingWordCatalogFromActionSpace();
    if (!UploadTrainingWordCatalogToDeviceConstantMemory(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    RuntimeWordCounts runtime_word_counts{};

    PopulationInitializationRandomEngine initialization_random_engine(42);
    auto host_population = std::make_unique<DevicePopulation>();
    if (!TryInitializePopulation<neuroevolution::genetic_algorithm::device::kDeviceActionCount,
                                 neuroevolution::genetic_algorithm::device::kDevicePopulationCapacity>(
            *host_population, initialization_random_engine)) {
        std::cerr << "FAIL: could not initialize the host population\n";
        return false;
    }

    host_population->active_individual_count = kTestPopulationSize;

    DeviceRuntimeBuffers buffers{};
    bool ok = TryCreateDeviceRuntimeBuffers(buffers);
    ok &= TryUploadCurrentPopulationToDevice(*host_population, buffers);
    ok &= TryEvaluatePopulationFitnessOnDevice(buffers, runtime_word_counts);

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
    ok &= TryEvaluatePopulationFitnessOnDevice(buffers, runtime_word_counts);

    PopulationFitnessSummary summary_generation_1{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary_generation_1);

    DestroyDeviceRuntimeBuffers(buffers);

    ok &= ExpectTrue(summary_generation_0.generation_index == 0, "Expected initial device population generation zero");
    ok &= ExpectTrue(summary_generation_1.generation_index == 1,
                     "Expected device-assembled next generation to increment generation index");
    ok &= ExpectTrue(summary_generation_0.best_index < kTestPopulationSize,
                     "Expected valid best index for generation zero");
    ok &= ExpectTrue(summary_generation_1.best_index < kTestPopulationSize,
                     "Expected valid best index for generation one");
    ok &= ExpectInRange(summary_generation_0.best_fitness, 0.0f,
                        MaximumPossibleFitnessForGeneration(training_word_catalog, runtime_word_counts,
                                                           summary_generation_0.generation_index),
                        "generation zero best fitness");
    ok &= ExpectInRange(summary_generation_0.average_fitness, 0.0f,
                        MaximumPossibleFitnessForGeneration(training_word_catalog, runtime_word_counts,
                                                           summary_generation_0.generation_index),
                        "generation zero average fitness");
    ok &= ExpectInRange(summary_generation_1.best_fitness, 0.0f,
                        MaximumPossibleFitnessForGeneration(training_word_catalog, runtime_word_counts,
                                                           summary_generation_1.generation_index),
                        "generation one best fitness");
    ok &= ExpectInRange(summary_generation_1.average_fitness, 0.0f,
                        MaximumPossibleFitnessForGeneration(training_word_catalog, runtime_word_counts,
                                                           summary_generation_1.generation_index),
                        "generation one average fitness");
    return ok;
}

bool TestDeviceRuntimeActivatesSecondTrainingShardAtGenerationOneHundred() {
    const auto training_word_catalog = LoadTrainingWordCatalogFromActionSpace();
    if (!UploadTrainingWordCatalogToDeviceConstantMemory(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    RuntimeWordCounts runtime_word_counts{};

    PopulationInitializationRandomEngine initialization_random_engine(42);
    auto host_population = std::make_unique<DevicePopulation>();
    if (!TryInitializePopulation<neuroevolution::genetic_algorithm::device::kDeviceActionCount,
                                 neuroevolution::genetic_algorithm::device::kDevicePopulationCapacity>(
            *host_population, initialization_random_engine)) {
        std::cerr << "FAIL: could not initialize the host population\n";
        return false;
    }

    host_population->active_individual_count = kTestPopulationSize;
    host_population->generation_index = kPhasedCurriculumSecondShardGeneration - 1;

    DeviceRuntimeBuffers buffers{};
    bool ok = TryCreateDeviceRuntimeBuffers(buffers);
    ok &= TryUploadCurrentPopulationToDevice(*host_population, buffers);
    ok &= TryEvaluatePopulationFitnessOnDevice(buffers, runtime_word_counts);

    PopulationFitnessSummary summary_generation_99{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary_generation_99);

    GenerationAssemblyConfig assembly_config{};
    assembly_config.genetic_algorithm.elite_count = 1;
    assembly_config.parent_selection.tournament_size = 3;
    assembly_config.parent_selection.allow_self_parenting = false;
    assembly_config.breeding.first_parent_probability = 0.5f;
    assembly_config.mutation.mutation_probability = 0.02f;
    assembly_config.mutation.mutation_sigma = 0.05f;

    ok &= TryAssembleNextGenerationOnDevice(buffers, 77U, assembly_config);
    SwapDevicePopulationBuffers(buffers);
    ok &= TryEvaluatePopulationFitnessOnDevice(buffers, runtime_word_counts);

    PopulationFitnessSummary summary_generation_100{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary_generation_100);

    DestroyDeviceRuntimeBuffers(buffers);

    ok &= ExpectTrue(summary_generation_99.generation_index == (kPhasedCurriculumSecondShardGeneration - 1),
                     "Expected pre-curriculum-expansion generation index to remain at 99");
    ok &= ExpectTrue(summary_generation_100.generation_index == kPhasedCurriculumSecondShardGeneration,
                     "Expected next generation to cross the phased-curriculum activation threshold");
    ok &= ExpectInRange(summary_generation_99.best_fitness, 0.0f,
                        MaximumPossibleFitnessForGeneration(training_word_catalog, runtime_word_counts,
                                                           summary_generation_99.generation_index),
                        "generation 99 best fitness");
    ok &= ExpectInRange(summary_generation_99.average_fitness, 0.0f,
                        MaximumPossibleFitnessForGeneration(training_word_catalog, runtime_word_counts,
                                                           summary_generation_99.generation_index),
                        "generation 99 average fitness");
    ok &= ExpectInRange(summary_generation_100.best_fitness, 0.0f,
                        MaximumPossibleFitnessForGeneration(training_word_catalog, runtime_word_counts,
                                                           summary_generation_100.generation_index),
                        "generation 100 best fitness");
    ok &= ExpectInRange(summary_generation_100.average_fitness, 0.0f,
                        MaximumPossibleFitnessForGeneration(training_word_catalog, runtime_word_counts,
                                                           summary_generation_100.generation_index),
                        "generation 100 average fitness");
    ok &= ExpectTrue(MaximumPossibleFitnessForGeneration(training_word_catalog, runtime_word_counts,
                                                         summary_generation_100.generation_index) >
                         MaximumPossibleFitnessForGeneration(training_word_catalog, runtime_word_counts,
                                                            summary_generation_99.generation_index),
                     "Expected phased curriculum generation 100 to activate a larger training set");
    return ok;
}

bool TestDeviceRuntimeInjectsOutputEmbeddingDuringNextGenerationAssembly() {
    const auto training_word_catalog = LoadTrainingWordCatalogFromActionSpace();
    if (!UploadTrainingWordCatalogToDeviceConstantMemory(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    RuntimeWordCounts runtime_word_counts{};
    runtime_word_counts.training_word_count = neuroevolution::genetic_algorithm::device::kDeviceActionCount - 1;
    runtime_word_counts.action_space_word_count = neuroevolution::genetic_algorithm::device::kDeviceActionCount - 1;

    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
    pending_output_embedding_injection.enabled = true;
    pending_output_embedding_injection.catalog_word_index = runtime_word_counts.action_space_word_count;

    PopulationInitializationRandomEngine initialization_random_engine(42);
    auto host_population = std::make_unique<DevicePopulation>();
    if (!TryInitializePopulation<neuroevolution::genetic_algorithm::device::kDeviceActionCount,
                                 neuroevolution::genetic_algorithm::device::kDevicePopulationCapacity>(
            *host_population, initialization_random_engine)) {
        std::cerr << "FAIL: could not initialize the host population\n";
        return false;
    }

    host_population->active_individual_count = kTestPopulationSize;
    SetPopulationActiveOutputEmbeddingCount(*host_population, runtime_word_counts.action_space_word_count);
    CopyGenomeAcrossPopulation(*host_population, 0);

    neuroevolution::model::output_embedding::TrainableActionEmbeddingTail expected_tail{};
    const Word injected_word = training_word_catalog.words[pending_output_embedding_injection.catalog_word_index];
    bool ok = TrySeedOutputEmbeddingTailFromHintGrids(host_population->individuals[0].genome.policy_model, injected_word,
                                                      expected_tail);
    ok &= ExpectTrue(ok, "Expected host-side tail synthesis for pending injection to succeed");

    DeviceRuntimeBuffers buffers{};
    ok &= TryCreateDeviceRuntimeBuffers(buffers);
    ok &= TryUploadCurrentPopulationToDevice(*host_population, buffers);
    ok &= TryEvaluatePopulationFitnessOnDevice(buffers, runtime_word_counts);
    if (!ok) {
        DestroyDeviceRuntimeBuffers(buffers);
        return false;
    }

    GenerationAssemblyConfig assembly_config{};
    assembly_config.genetic_algorithm.elite_count = 1;
    assembly_config.parent_selection.tournament_size = 3;
    assembly_config.parent_selection.allow_self_parenting = false;
    assembly_config.breeding.first_parent_probability = 0.5f;
    assembly_config.mutation.mutation_probability = 0.0f;
    assembly_config.mutation.mutation_sigma = 0.0f;

    ok &= TryAssembleNextGenerationOnDevice(buffers, 77U, assembly_config, pending_output_embedding_injection);
    if (!ok) {
        DestroyDeviceRuntimeBuffers(buffers);
        return false;
    }

    SwapDevicePopulationBuffers(buffers);

    runtime_word_counts.training_word_count = pending_output_embedding_injection.catalog_word_index + 1;
    runtime_word_counts.action_space_word_count = pending_output_embedding_injection.catalog_word_index + 1;

    auto assembled_population = std::make_unique<DevicePopulation>();
    ok &= TryDownloadCurrentPopulationFromDevice(buffers, *assembled_population);
    ok &= TryEvaluatePopulationFitnessOnDevice(buffers, runtime_word_counts);

    PopulationFitnessSummary summary_generation_1{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary_generation_1);

    DestroyDeviceRuntimeBuffers(buffers);

    ok &= ExpectTrue(assembled_population->generation_index == 1,
                     "Expected injected next generation to advance the generation index");
    ok &= ExpectTrue(runtime_word_counts.training_word_count == runtime_word_counts.action_space_word_count,
                     "Expected injected runtime counts to stay equal for the current host policy");

    for (std::size_t individual_index = 0; individual_index < assembled_population->active_individual_count;
         ++individual_index) {
        ok &= ExpectTrue(
            assembled_population->individuals[individual_index].genome.output_embedding.active_count ==
                neuroevolution::genetic_algorithm::device::kDeviceActionCount,
            "Expected assembly injection to raise every genome active count to device action capacity");
    }

    ok &= ExpectPopulationTailMatchesExpected(
        *assembled_population, pending_output_embedding_injection.catalog_word_index, expected_tail,
        "expected injected population tail");
    ok &= ExpectInRange(summary_generation_1.best_fitness, 0.0f,
                        MaximumPossibleFitnessForGeneration(training_word_catalog, runtime_word_counts,
                                                           summary_generation_1.generation_index),
                        "injected generation best fitness");
    ok &= ExpectInRange(summary_generation_1.average_fitness, 0.0f,
                        MaximumPossibleFitnessForGeneration(training_word_catalog, runtime_word_counts,
                                                           summary_generation_1.generation_index),
                        "injected generation average fitness");
    return ok;
}

bool TestDeviceRuntimeReportsInjectionFailureWithoutSpareCapacity() {
    const auto training_word_catalog = LoadTrainingWordCatalogFromActionSpace();
    if (!UploadTrainingWordCatalogToDeviceConstantMemory(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    RuntimeWordCounts runtime_word_counts{};

    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
    pending_output_embedding_injection.enabled = true;
    pending_output_embedding_injection.catalog_word_index = runtime_word_counts.action_space_word_count;

    PopulationInitializationRandomEngine initialization_random_engine(42);
    auto host_population = std::make_unique<DevicePopulation>();
    if (!TryInitializePopulation<neuroevolution::genetic_algorithm::device::kDeviceActionCount,
                                 neuroevolution::genetic_algorithm::device::kDevicePopulationCapacity>(
            *host_population, initialization_random_engine)) {
        std::cerr << "FAIL: could not initialize the host population\n";
        return false;
    }

    host_population->active_individual_count = kTestPopulationSize;

    DeviceRuntimeBuffers buffers{};
    bool ok = TryCreateDeviceRuntimeBuffers(buffers);
    ok &= TryUploadCurrentPopulationToDevice(*host_population, buffers);
    ok &= TryEvaluatePopulationFitnessOnDevice(buffers, runtime_word_counts);
    if (!ok) {
        DestroyDeviceRuntimeBuffers(buffers);
        return false;
    }

    GenerationAssemblyConfig assembly_config{};
    assembly_config.genetic_algorithm.elite_count = 1;
    assembly_config.parent_selection.tournament_size = 3;
    assembly_config.parent_selection.allow_self_parenting = false;
    assembly_config.breeding.first_parent_probability = 0.5f;
    assembly_config.mutation.mutation_probability = 0.0f;
    assembly_config.mutation.mutation_sigma = 0.0f;

    ok &= ExpectTrue(!TryAssembleNextGenerationOnDevice(buffers, 77U, assembly_config, pending_output_embedding_injection),
                     "Expected injection request without spare capacity to fail");

    DeviceRuntimeStatusCode status_code = DeviceRuntimeStatusCode::kOk;
    ok &= TryReadDeviceRuntimeStatus(buffers, status_code);

    DestroyDeviceRuntimeBuffers(buffers);

    ok &= ExpectTrue(status_code == DeviceRuntimeStatusCode::kOutputEmbeddingInjectionFailed,
                     "Expected failed injection assembly to report output-embedding injection failure");
    return ok;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    if (!TestDeviceRuntimeEvaluatesAndAssemblesPopulationsOnDevice() ||
        !TestDeviceRuntimeActivatesSecondTrainingShardAtGenerationOneHundred() ||
        !TestDeviceRuntimeInjectsOutputEmbeddingDuringNextGenerationAssembly() ||
        !TestDeviceRuntimeReportsInjectionFailureWithoutSpareCapacity()) {
        return 1;
    }

    std::cout << "PASS: device_runtime_test\n";
    return 0;
}
