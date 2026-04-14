#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/device/dynamic_runtime.hpp"
#include "genetic_algorithm/genetic_algorithm.hpp"
#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::genetic_algorithm::GenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::TrySeedOutputEmbeddingTailFromHintGrids;
using neuroevolution::genetic_algorithm::dynamic_device::ComputeDynamicGenomeStrideBytes;
using neuroevolution::genetic_algorithm::dynamic_device::DestroyDeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::dynamic_device::DeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::dynamic_device::DeviceRuntimeConfig;
using neuroevolution::genetic_algorithm::dynamic_device::DeviceRuntimeStatusCode;
using neuroevolution::genetic_algorithm::dynamic_device::GenomePolicyModelParameters;
using neuroevolution::genetic_algorithm::dynamic_device::GenomeTailRows;
using neuroevolution::genetic_algorithm::dynamic_device::HostGenomeBytesAt;
using neuroevolution::genetic_algorithm::dynamic_device::HostPopulation;
using neuroevolution::genetic_algorithm::dynamic_device::PendingOutputEmbeddingInjection;
using neuroevolution::genetic_algorithm::dynamic_device::PopulationFitnessSummary;
using neuroevolution::genetic_algorithm::dynamic_device::PopulationSizeForGenotypeBudgetBytes;
using neuroevolution::genetic_algorithm::dynamic_device::RuntimeWordCounts;
using neuroevolution::genetic_algorithm::dynamic_device::SwapDevicePopulationBuffers;
using neuroevolution::genetic_algorithm::dynamic_device::TrainableActionEmbeddingTail;
using neuroevolution::genetic_algorithm::dynamic_device::TryAssembleNextGenerationOnDevice;
using neuroevolution::genetic_algorithm::dynamic_device::TryCreateDeviceRuntimeBuffers;
using neuroevolution::genetic_algorithm::dynamic_device::TryDownloadCurrentPopulationFromDevice;
using neuroevolution::genetic_algorithm::dynamic_device::TryEvaluatePopulationFitnessOnDevice;
using neuroevolution::genetic_algorithm::dynamic_device::TryInitializeRandomHostPopulation;
using neuroevolution::genetic_algorithm::dynamic_device::TryReadDeviceRuntimeStatus;
using neuroevolution::genetic_algorithm::dynamic_device::TryReadPopulationFitnessSummaryFromDevice;
using neuroevolution::genetic_algorithm::dynamic_device::TryUploadCurrentPopulationToDevice;
using neuroevolution::training_folder::LoadTrainingWordCatalogFromActionSpace;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::training_folder::UploadTrainingWordCatalogToDeviceConstantMemory;
using neuroevolution::wordle::Word;

constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr std::size_t kInitialPopulationSize = 6;
constexpr float kMaximumEpisodeWinScore = 15.0f;
constexpr float kEpisodesPerTrainingEntry = 3.0f;
constexpr float kTailTolerance = 1.0e-3f;

float MaximumPossibleFitnessForGeneration(const TrainingWordCatalog &training_word_catalog,
                                          const RuntimeWordCounts &runtime_word_counts,
                                          const std::size_t generation_index) {
    (void)training_word_catalog;
    (void)generation_index;
    return kEpisodesPerTrainingEntry * kMaximumEpisodeWinScore *
           static_cast<float>(runtime_word_counts.training_word_count);
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

GenerationAssemblyConfig MakeAssemblyConfig() {
    GenerationAssemblyConfig config{};
    config.parent_selection.tournament_size = 3;
    config.parent_selection.allow_self_parenting = false;
    config.breeding.first_parent_probability = 0.5f;
    config.mutation.mutation_probability = 0.0f;
    config.mutation.mutation_sigma = 0.0f;
    return config;
}

void CopyGenomeAcrossPopulation(HostPopulation &population, const std::size_t source_index) {
    const std::uint8_t *source_genome_bytes = HostGenomeBytesAt(population, source_index);

    for (std::size_t individual_index = 0; individual_index < population.layout.active_individual_count;
         ++individual_index) {
        if (individual_index == source_index) {
            continue;
        }

        std::memcpy(HostGenomeBytesAt(population, individual_index), source_genome_bytes,
                    population.layout.genome_stride_bytes);
    }
}

bool ExpectPopulationTailMatchesExpected(const HostPopulation &population, const std::size_t tail_index,
                                         const TrainableActionEmbeddingTail &expected_tail,
                                         const std::string_view label_prefix) {
    bool ok = true;

    for (std::size_t individual_index = 0; individual_index < population.layout.active_individual_count;
         ++individual_index) {
        const TrainableActionEmbeddingTail &tail =
            GenomeTailRows(HostGenomeBytesAt(population, individual_index))[tail_index];
        for (std::size_t feature_index = 0;
             feature_index < neuroevolution::model::output_embedding::kTrainableFeatureDimension; ++feature_index) {
            ok &= ExpectNear(ToFloat(tail[feature_index]), ToFloat(expected_tail[feature_index]),
                             std::string(label_prefix) + " individual " + std::to_string(individual_index) +
                                 " feature " + std::to_string(feature_index));
        }
    }

    return ok;
}

bool TestDynamicRuntimeInjectsAndResizesWithinFixedBudget() {
    const auto training_word_catalog = LoadTrainingWordCatalogFromActionSpace();
    if (!UploadTrainingWordCatalogToDeviceConstantMemory(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    RuntimeWordCounts runtime_word_counts{};
    const std::size_t initial_action_count = runtime_word_counts.action_space_word_count;
    constexpr std::size_t kInjectedWordCount = 3;
    const std::size_t genotype_budget_bytes =
        kInitialPopulationSize * ComputeDynamicGenomeStrideBytes(initial_action_count);
    const std::size_t expected_next_population_size = PopulationSizeForGenotypeBudgetBytes(
        genotype_budget_bytes, initial_action_count + kInjectedWordCount, kInitialPopulationSize);

    bool ok = true;
    ok &= ExpectTrue(expected_next_population_size > 0,
                     "Expected fixed genotype budget to fit at least one injected genome");
    ok &= ExpectTrue(expected_next_population_size < kInitialPopulationSize,
                     "Expected injected generation to shrink population under the same genotype byte budget");
    if (!ok) {
        return false;
    }

    HostPopulation host_population{};
    ok &= TryInitializeRandomHostPopulation(host_population, kInitialPopulationSize, initial_action_count, 42U);
    if (!ok) {
        std::cerr << "FAIL: could not initialize the dynamic host population\n";
        return false;
    }

    CopyGenomeAcrossPopulation(host_population, 0);

    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
    pending_output_embedding_injection.enabled = true;
    pending_output_embedding_injection.first_catalog_word_index = initial_action_count;
    pending_output_embedding_injection.injection_count = kInjectedWordCount;

    TrainableActionEmbeddingTail expected_tails[kInjectedWordCount]{};
    for (std::size_t injection_offset = 0; injection_offset < kInjectedWordCount; ++injection_offset) {
        ok &= TrySeedOutputEmbeddingTailFromHintGrids(
            GenomePolicyModelParameters(HostGenomeBytesAt(host_population, 0)),
            training_word_catalog.words[pending_output_embedding_injection.first_catalog_word_index + injection_offset],
            expected_tails[injection_offset]);
    }
    ok &= ExpectTrue(ok, "Expected host-side injected tail synthesis to succeed for every injected word");
    if (!ok) {
        return false;
    }

    DeviceRuntimeConfig runtime_config{};
    runtime_config.genotype_memory_budget_bytes = genotype_budget_bytes;
    runtime_config.population_size_ceiling = kInitialPopulationSize;
    runtime_config.initial_action_count = initial_action_count;

    DeviceRuntimeBuffers buffers{};
    ok &= TryCreateDeviceRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadCurrentPopulationToDevice(host_population, buffers);
    ok &= TryEvaluatePopulationFitnessOnDevice(buffers, runtime_word_counts);
    if (!ok) {
        DestroyDeviceRuntimeBuffers(buffers);
        return false;
    }

    PopulationFitnessSummary summary_generation_0{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary_generation_0);

    const GenerationAssemblyConfig assembly_config = MakeAssemblyConfig();
    ok &= TryAssembleNextGenerationOnDevice(buffers, 77U, assembly_config, pending_output_embedding_injection);
    if (!ok) {
        DestroyDeviceRuntimeBuffers(buffers);
        return false;
    }

    ok &= ExpectTrue(buffers.next_layout.active_individual_count == expected_next_population_size,
                     "Expected next generation to use the resized population count");
    ok &= ExpectTrue(buffers.next_layout.action_count == (initial_action_count + kInjectedWordCount),
                     "Expected injection batch to increase the next generation action count");
    ok &= ExpectTrue(buffers.next_layout.genome_stride_bytes > buffers.current_layout.genome_stride_bytes,
                     "Expected injected next generation genomes to have a larger byte stride");
    ok &= ExpectTrue(buffers.next_layout.genotype_bytes <= genotype_budget_bytes,
                     "Expected resized next generation genotypes to stay within the fixed byte budget");
    if (!ok) {
        DestroyDeviceRuntimeBuffers(buffers);
        return false;
    }

    SwapDevicePopulationBuffers(buffers);
    runtime_word_counts.training_word_count = buffers.current_layout.action_count;
    runtime_word_counts.action_space_word_count = buffers.current_layout.action_count;

    HostPopulation assembled_population{};
    ok &= TryDownloadCurrentPopulationFromDevice(buffers, assembled_population);
    ok &= TryEvaluatePopulationFitnessOnDevice(buffers, runtime_word_counts);

    PopulationFitnessSummary summary_generation_1{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary_generation_1);

    DestroyDeviceRuntimeBuffers(buffers);

    ok &= ExpectTrue(summary_generation_0.generation_index == 0, "Expected initial generation index to be zero");
    ok &= ExpectTrue(summary_generation_0.population_size == kInitialPopulationSize,
                     "Expected initial summary to report the starting population size");
    ok &= ExpectTrue(summary_generation_0.action_count == initial_action_count,
                     "Expected initial summary to report the starting action count");
    ok &= ExpectTrue(summary_generation_1.generation_index == 1,
                     "Expected resized injected generation to increment the generation index");
    ok &= ExpectTrue(summary_generation_1.population_size == expected_next_population_size,
                     "Expected injected generation summary to report the resized population");
    ok &= ExpectTrue(summary_generation_1.action_count == (initial_action_count + kInjectedWordCount),
                     "Expected injected generation summary to report the batched larger action count");
    ok &= ExpectTrue(assembled_population.layout.active_individual_count == expected_next_population_size,
                     "Expected downloaded population layout to reflect the resized generation");
    ok &= ExpectTrue(assembled_population.layout.action_count == (initial_action_count + kInjectedWordCount),
                     "Expected downloaded population layout to reflect the batched injected action count");
    for (std::size_t injection_offset = 0; injection_offset < kInjectedWordCount; ++injection_offset) {
        ok &= ExpectPopulationTailMatchesExpected(
            assembled_population, initial_action_count + injection_offset, expected_tails[injection_offset],
            std::string("expected injected dynamic population tail ") + std::to_string(injection_offset));
    }
    ok &= ExpectInRange(summary_generation_0.best_fitness, 0.0f,
                        MaximumPossibleFitnessForGeneration(training_word_catalog, RuntimeWordCounts{},
                                                            summary_generation_0.generation_index),
                        "generation zero best fitness");
    ok &= ExpectInRange(summary_generation_0.average_fitness, 0.0f,
                        MaximumPossibleFitnessForGeneration(training_word_catalog, RuntimeWordCounts{},
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

bool TestDynamicRuntimeRejectsInjectionWhenNoInjectedGenomeFitsBudget() {
    const auto training_word_catalog = LoadTrainingWordCatalogFromActionSpace();
    if (!UploadTrainingWordCatalogToDeviceConstantMemory(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    RuntimeWordCounts runtime_word_counts{};
    const std::size_t initial_action_count = runtime_word_counts.action_space_word_count;
    const std::size_t genotype_budget_bytes = ComputeDynamicGenomeStrideBytes(initial_action_count);

    HostPopulation host_population{};
    bool ok = TryInitializeRandomHostPopulation(host_population, 1, initial_action_count, 7U);
    if (!ok) {
        std::cerr << "FAIL: could not initialize the single-genome dynamic host population\n";
        return false;
    }

    DeviceRuntimeConfig runtime_config{};
    runtime_config.genotype_memory_budget_bytes = genotype_budget_bytes;
    runtime_config.population_size_ceiling = 1;
    runtime_config.initial_action_count = initial_action_count;

    DeviceRuntimeBuffers buffers{};
    ok &= TryCreateDeviceRuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadCurrentPopulationToDevice(host_population, buffers);
    ok &= TryEvaluatePopulationFitnessOnDevice(buffers, runtime_word_counts);
    if (!ok) {
        DestroyDeviceRuntimeBuffers(buffers);
        return false;
    }

    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
    pending_output_embedding_injection.enabled = true;
    pending_output_embedding_injection.first_catalog_word_index = initial_action_count;
    pending_output_embedding_injection.injection_count = 1;

    const GenerationAssemblyConfig assembly_config = MakeAssemblyConfig();
    ok &=
        ExpectTrue(!TryAssembleNextGenerationOnDevice(buffers, 9U, assembly_config, pending_output_embedding_injection),
                   "Expected injected assembly to fail when the budget cannot fit even one next-generation genome");

    DeviceRuntimeStatusCode status_code = DeviceRuntimeStatusCode::kOk;
    ok &= TryReadDeviceRuntimeStatus(buffers, status_code);

    DestroyDeviceRuntimeBuffers(buffers);

    ok &= ExpectTrue(status_code == DeviceRuntimeStatusCode::kInvalidAssemblyConfig,
                     "Expected insufficient injected budget to report invalid assembly config");
    return ok;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    if (!TestDynamicRuntimeInjectsAndResizesWithinFixedBudget() ||
        !TestDynamicRuntimeRejectsInjectionWhenNoInjectedGenomeFitsBudget()) {
        return 1;
    }

    std::cout << "PASS: dynamic_runtime_test\n";
    return 0;
}
