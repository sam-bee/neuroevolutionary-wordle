#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/device/dynamic_runtime.hpp"
#include "genetic_algorithm/genetic_algorithm.hpp"
#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::genetic_algorithm::GenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::TrySeedOutputEmbeddingTailFromHintGrids;
using neuroevolution::genetic_algorithm::dynamic_device::GenomePolicyModelParameters;
using neuroevolution::genetic_algorithm::dynamic_device::GenomeTailRows;
using neuroevolution::genetic_algorithm::dynamic_device::HostGenomeBytesAt;
using neuroevolution::genetic_algorithm::dynamic_device::HostPopulation;
using neuroevolution::genetic_algorithm::dynamic_device::TrainableActionEmbeddingTail;
using neuroevolution::genetic_algorithm::dynamic_device::TryInitializeRandomHostPopulation;
using neuroevolution::genetic_algorithm::genotype_pool::HostGenotypePool;
using neuroevolution::genetic_algorithm::genotype_pool::HostPoolSlotBytesAt;
using neuroevolution::genetic_algorithm::genotype_pool::PoolGeneration;
using neuroevolution::genetic_algorithm::genotype_pool::TryAllocatePoolSlot;
using neuroevolution::genetic_algorithm::genotype_pool::TryCopyGenomeBytesIntoPoolSlot;
using neuroevolution::genetic_algorithm::genotype_pool::TryCreateHostGenotypePool;
using neuroevolution::genetic_algorithm::genotype_pool::TryCreatePoolGeneration;
using neuroevolution::genetic_algorithm::genotype_pool::TrySetPoolGenerationSlot;
using neuroevolution::genetic_algorithm::pool_device::DevicePoolGARuntimeBuffers;
using neuroevolution::genetic_algorithm::pool_device::DevicePoolGARuntimeConfig;
using neuroevolution::genetic_algorithm::pool_device::DevicePoolGARuntimeStatusCode;
using neuroevolution::genetic_algorithm::pool_device::PendingOutputEmbeddingInjection;
using neuroevolution::genetic_algorithm::pool_device::PopulationFitnessSummary;
using neuroevolution::genetic_algorithm::pool_device::RuntimeWordCounts;
using neuroevolution::genetic_algorithm::pool_device::TryAdvanceGenerationOnDevice;
using neuroevolution::genetic_algorithm::pool_device::TryCreateDevicePoolGARuntimeBuffers;
using neuroevolution::genetic_algorithm::pool_device::TryDownloadCurrentGenerationFromDevice;
using neuroevolution::genetic_algorithm::pool_device::TryDownloadPoolFromDevice;
using neuroevolution::genetic_algorithm::pool_device::TryEvaluateCurrentGenerationFitnessOnDevice;
using neuroevolution::genetic_algorithm::pool_device::TryReadDevicePoolGARuntimeStatus;
using neuroevolution::genetic_algorithm::pool_device::TryReadPopulationFitnessSummaryFromDevice;
using neuroevolution::genetic_algorithm::pool_device::TryUploadCurrentPoolPopulationToDevice;
using neuroevolution::training_folder::LoadTrainingWordCatalogFromActionSpace;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::training_folder::UploadTrainingWordCatalogToDeviceConstantMemory;

constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr std::size_t kActionCount = neuroevolution::training_folder::kDefaultInitialActiveWordCount;
constexpr float kTolerance = 1.0e-3f;
constexpr float kMaximumEpisodeWinScore = 15.0f;
constexpr float kEpisodesPerTrainingEntry = 3.0f;

float MaximumPossibleFitness(const RuntimeWordCounts &runtime_word_counts) {
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

bool ExpectNear(const float actual, const float expected, const std::string_view label) {
    if (std::fabs(actual - expected) > kTolerance) {
        std::cerr << "FAIL: " << label << " expected " << expected << ", got " << actual << '\n';
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

bool ExpectTailEquals(const TrainableActionEmbeddingTail &actual, const TrainableActionEmbeddingTail &expected,
                      const std::string_view label_prefix) {
    bool ok = true;

    for (std::size_t feature_index = 0;
         feature_index < neuroevolution::model::output_embedding::kTrainableFeatureDimension; ++feature_index) {
        ok &= ExpectNear(ToFloat(actual[feature_index]), ToFloat(expected[feature_index]),
                         std::string(label_prefix) + " feature " + std::to_string(feature_index));
    }

    return ok;
}

RuntimeWordCounts MakeRuntimeWordCounts() {
    RuntimeWordCounts runtime_word_counts{};
    runtime_word_counts.training_word_count = kActionCount;
    runtime_word_counts.action_space_word_count = kActionCount;
    return runtime_word_counts;
}

GenerationAssemblyConfig MakeAssemblyConfig() {
    GenerationAssemblyConfig config{};
    config.parent_selection.tournament_size = 3;
    config.parent_selection.allow_self_parenting = false;
    config.breeding.first_parent_probability = 1.0f;
    config.mutation.mutation_probability = 0.0f;
    config.mutation.mutation_sigma = 0.0f;
    return config;
}

bool PopulatePoolGenerationFromHostPopulation(const HostPopulation &host_population, const std::size_t slot_count,
                                              const std::size_t generation_index, HostGenotypePool &host_pool,
                                              PoolGeneration &current_generation) {
    bool ok = TryCreateHostGenotypePool(host_pool, slot_count, host_population.layout.action_count);
    ok &= TryCreatePoolGeneration(current_generation, host_population.layout.active_individual_count, generation_index);
    if (!ok) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < host_population.layout.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = 0;
        ok &= TryAllocatePoolSlot(host_pool, slot_index);
        ok &= TrySetPoolGenerationSlot(current_generation, individual_index, slot_index);
        ok &=
            TryCopyGenomeBytesIntoPoolSlot(host_pool, slot_index, HostGenomeBytesAt(host_population, individual_index),
                                           host_population.layout.genome_stride_bytes);
    }

    return ok;
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

bool InitializeTrainingCatalog(TrainingWordCatalog &training_word_catalog) {
    training_word_catalog = LoadTrainingWordCatalogFromActionSpace();
    return UploadTrainingWordCatalogToDeviceConstantMemory(training_word_catalog);
}

bool TestPoolRuntimeEvaluatesAndSummarizesCurrentGeneration() {
    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    HostPopulation host_population{};
    bool ok = TryInitializeRandomHostPopulation(host_population, 3, kActionCount, 41U);
    HostGenotypePool host_pool{};
    PoolGeneration current_generation{};
    ok &= PopulatePoolGenerationFromHostPopulation(host_population, 6, 4, host_pool, current_generation);
    if (!ok) {
        std::cerr << "FAIL: could not build pool evaluation fixtures\n";
        return false;
    }

    DevicePoolGARuntimeConfig runtime_config{};
    runtime_config.slot_count = 6;
    runtime_config.action_count = kActionCount;
    runtime_config.max_generation_size = 3;

    DevicePoolGARuntimeBuffers buffers{};
    ok &= TryCreateDevicePoolGARuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadCurrentPoolPopulationToDevice(host_pool, current_generation, buffers);
    ok &= TryEvaluateCurrentGenerationFitnessOnDevice(buffers, MakeRuntimeWordCounts());
    if (!ok) {
        neuroevolution::genetic_algorithm::pool_device::DestroyDevicePoolGARuntimeBuffers(buffers);
        return false;
    }

    PopulationFitnessSummary summary{};
    PoolGeneration downloaded_generation{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_generation);
    neuroevolution::genetic_algorithm::pool_device::DestroyDevicePoolGARuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    const float maximum_possible_fitness = MaximumPossibleFitness(MakeRuntimeWordCounts());
    ok &= ExpectTrue(summary.population_size == 3, "Expected summary to report the current generation size");
    ok &= ExpectTrue(summary.action_count == kActionCount, "Expected summary to report the fixed action count");
    ok &= ExpectTrue(summary.generation_index == 4, "Expected summary to report the current generation index");
    ok &= ExpectInRange(summary.best_fitness, 0.0f, maximum_possible_fitness, "summary best fitness");
    ok &= ExpectInRange(summary.average_fitness, 0.0f, maximum_possible_fitness, "summary average fitness");

    for (std::size_t individual_index = 0; individual_index < downloaded_generation.active_individual_count;
         ++individual_index) {
        ok &= ExpectTrue(downloaded_generation.has_fitness[individual_index] == 1,
                         "Expected evaluation to mark every current-generation individual as fitted");
        ok &= ExpectTrue(downloaded_generation.evaluation_counts[individual_index] == 1,
                         "Expected evaluation to increment the count for every current-generation individual");
        ok &= ExpectInRange(downloaded_generation.fitness[individual_index], 0.0f, maximum_possible_fitness,
                            "downloaded current-generation fitness");
    }

    return ok;
}

bool TestPoolRuntimeAdvancesOneGeneration() {
    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    HostPopulation host_population{};
    bool ok = TryInitializeRandomHostPopulation(host_population, 3, kActionCount, 77U);
    CopyGenomeAcrossPopulation(host_population, 0);

    HostGenotypePool host_pool{};
    PoolGeneration current_generation{};
    ok &= PopulatePoolGenerationFromHostPopulation(host_population, 6, 2, host_pool, current_generation);
    if (!ok) {
        std::cerr << "FAIL: could not build pool generation-step fixtures\n";
        return false;
    }

    const float expected_bias = ToFloat(
        GenomePolicyModelParameters(HostGenomeBytesAt(host_population, 0)).dense_trunk.hidden1_to_output.biases[0]);
    const float expected_tail = ToFloat(GenomeTailRows(HostGenomeBytesAt(host_population, 0))[0][0]);

    DevicePoolGARuntimeConfig runtime_config{};
    runtime_config.slot_count = 6;
    runtime_config.action_count = kActionCount;
    runtime_config.max_generation_size = 3;

    DevicePoolGARuntimeBuffers buffers{};
    ok &= TryCreateDevicePoolGARuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadCurrentPoolPopulationToDevice(host_pool, current_generation, buffers);
    ok &= TryAdvanceGenerationOnDevice(buffers, 19U, MakeRuntimeWordCounts(), MakeAssemblyConfig());
    if (!ok) {
        DevicePoolGARuntimeStatusCode status_code = DevicePoolGARuntimeStatusCode::kOk;
        (void)TryReadDevicePoolGARuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: pool generation step failed with status " << static_cast<int>(status_code) << '\n';
        neuroevolution::genetic_algorithm::pool_device::DestroyDevicePoolGARuntimeBuffers(buffers);
        return false;
    }

    PopulationFitnessSummary summary{};
    HostGenotypePool downloaded_pool{};
    PoolGeneration downloaded_generation{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary);
    ok &= TryDownloadPoolFromDevice(buffers, downloaded_pool);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_generation);
    neuroevolution::genetic_algorithm::pool_device::DestroyDevicePoolGARuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(summary.generation_index == 2, "Expected summary to describe the evaluated parent generation");
    ok &= ExpectTrue(downloaded_generation.generation_index == 3,
                     "Expected a successful generation step to increment the current generation index");
    ok &= ExpectTrue(downloaded_pool.free_slot_count == 3,
                     "Expected three pool slots to be free after three parents are replaced by three children");

    for (std::size_t individual_index = 0; individual_index < downloaded_generation.active_individual_count;
         ++individual_index) {
        const std::uint32_t slot_index = downloaded_generation.slot_indices[individual_index];
        ok &= ExpectTrue(slot_index != neuroevolution::genetic_algorithm::genotype_pool::kInvalidPoolSlotIndex,
                         "Expected every child generation handle to reference a live pool slot");
        ok &= ExpectTrue(downloaded_generation.has_fitness[individual_index] == 0,
                         "Expected newly assembled children to start unevaluated");
        ok &= ExpectTrue(downloaded_generation.evaluation_counts[individual_index] == 0,
                         "Expected newly assembled children to start with zero evaluation count");
        ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostPoolSlotBytesAt(downloaded_pool, slot_index))
                                     .dense_trunk.hidden1_to_output.biases[0]),
                         expected_bias, "child dense-trunk bias");
        ok &= ExpectNear(ToFloat(GenomeTailRows(HostPoolSlotBytesAt(downloaded_pool, slot_index))[0][0]), expected_tail,
                         "child trainable tail value");
        ok &= ExpectTrue(downloaded_pool.slot_states[slot_index].reference_count == 1,
                         "Expected each child slot to hold exactly one generation reference");
    }

    return ok;
}

bool TestPoolRuntimeGrowsActionCountWithPoolRepacking() {
    constexpr std::size_t kInjectedWordCount = 3;

    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    HostPopulation host_population{};
    bool ok = TryInitializeRandomHostPopulation(host_population, 3, kActionCount, 91U);
    CopyGenomeAcrossPopulation(host_population, 0);

    HostGenotypePool host_pool{};
    PoolGeneration current_generation{};
    ok &= PopulatePoolGenerationFromHostPopulation(host_population, 6, 8, host_pool, current_generation);
    if (!ok) {
        std::cerr << "FAIL: could not build pool growth fixtures\n";
        return false;
    }

    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
    pending_output_embedding_injection.enabled = true;
    pending_output_embedding_injection.first_catalog_word_index = kActionCount;
    pending_output_embedding_injection.injection_count = kInjectedWordCount;

    TrainableActionEmbeddingTail expected_tails[kInjectedWordCount]{};
    for (std::size_t injection_offset = 0; injection_offset < kInjectedWordCount; ++injection_offset) {
        ok &= TrySeedOutputEmbeddingTailFromHintGrids(
            GenomePolicyModelParameters(HostGenomeBytesAt(host_population, 0)),
            training_word_catalog.words[pending_output_embedding_injection.first_catalog_word_index + injection_offset],
            expected_tails[injection_offset]);
    }
    if (!ok) {
        std::cerr << "FAIL: could not build expected injected tails on host\n";
        return false;
    }

    DevicePoolGARuntimeConfig runtime_config{};
    runtime_config.slot_count = 6;
    runtime_config.action_count = kActionCount;
    runtime_config.max_generation_size = 3;

    DevicePoolGARuntimeBuffers buffers{};
    ok &= TryCreateDevicePoolGARuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadCurrentPoolPopulationToDevice(host_pool, current_generation, buffers);
    ok &= TryAdvanceGenerationOnDevice(buffers, 31U, MakeRuntimeWordCounts(), MakeAssemblyConfig(),
                                       pending_output_embedding_injection);
    if (!ok) {
        DevicePoolGARuntimeStatusCode status_code = DevicePoolGARuntimeStatusCode::kOk;
        (void)TryReadDevicePoolGARuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: pool growth step failed with status " << static_cast<int>(status_code) << '\n';
        neuroevolution::genetic_algorithm::pool_device::DestroyDevicePoolGARuntimeBuffers(buffers);
        return false;
    }

    PopulationFitnessSummary summary{};
    HostGenotypePool downloaded_pool{};
    PoolGeneration downloaded_generation{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary);
    ok &= TryDownloadPoolFromDevice(buffers, downloaded_pool);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_generation);
    neuroevolution::genetic_algorithm::pool_device::DestroyDevicePoolGARuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &=
        ExpectTrue(summary.generation_index == 8, "Expected summary to still describe the evaluated parent generation");
    ok &= ExpectTrue(downloaded_generation.generation_index == 9,
                     "Expected a successful growth step to increment the generation index");
    ok &= ExpectTrue(downloaded_generation.active_individual_count == 2,
                     "Expected growth to shrink the next generation to the conservative repacked child capacity");
    ok &= ExpectTrue(downloaded_pool.layout.action_count == (kActionCount + kInjectedWordCount),
                     "Expected repacking to expand the pool action count before child assembly");

    for (std::size_t individual_index = 0; individual_index < downloaded_generation.active_individual_count;
         ++individual_index) {
        const std::uint32_t slot_index = downloaded_generation.slot_indices[individual_index];
        ok &= ExpectTrue(slot_index != neuroevolution::genetic_algorithm::genotype_pool::kInvalidPoolSlotIndex,
                         "Expected every grown child to occupy a live pool slot");
        for (std::size_t injection_offset = 0; injection_offset < kInjectedWordCount; ++injection_offset) {
            ok &= ExpectTailEquals(
                GenomeTailRows(HostPoolSlotBytesAt(downloaded_pool, slot_index))[kActionCount + injection_offset],
                expected_tails[injection_offset],
                std::string("grown child injected tail ") + std::to_string(individual_index) + ":" +
                    std::to_string(injection_offset));
        }
    }

    return ok;
}

bool TestPoolRuntimeFailsCleanlyWhenThePoolIsTooSmall() {
    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    HostPopulation host_population{};
    bool ok = TryInitializeRandomHostPopulation(host_population, 2, kActionCount, 11U);
    HostGenotypePool host_pool{};
    PoolGeneration current_generation{};
    ok &= PopulatePoolGenerationFromHostPopulation(host_population, 2, 0, host_pool, current_generation);
    if (!ok) {
        std::cerr << "FAIL: could not build full-pool runtime fixtures\n";
        return false;
    }

    DevicePoolGARuntimeConfig runtime_config{};
    runtime_config.slot_count = 2;
    runtime_config.action_count = kActionCount;
    runtime_config.max_generation_size = 2;

    DevicePoolGARuntimeBuffers buffers{};
    ok &= TryCreateDevicePoolGARuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadCurrentPoolPopulationToDevice(host_pool, current_generation, buffers);
    if (!ok) {
        neuroevolution::genetic_algorithm::pool_device::DestroyDevicePoolGARuntimeBuffers(buffers);
        return false;
    }

    ok &= ExpectTrue(!TryAdvanceGenerationOnDevice(buffers, 23U, MakeRuntimeWordCounts(), MakeAssemblyConfig()),
                     "Expected pool-backed generation stepping to fail when no child slot can be allocated");

    DevicePoolGARuntimeStatusCode status_code = DevicePoolGARuntimeStatusCode::kOk;
    ok &= TryReadDevicePoolGARuntimeStatus(buffers, status_code);
    neuroevolution::genetic_algorithm::pool_device::DestroyDevicePoolGARuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(status_code == DevicePoolGARuntimeStatusCode::kPoolFull,
                     "Expected an undersized pool to report kPoolFull during generation stepping");
    return ok;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    if (!TestPoolRuntimeEvaluatesAndSummarizesCurrentGeneration() || !TestPoolRuntimeAdvancesOneGeneration() ||
        !TestPoolRuntimeGrowsActionCountWithPoolRepacking() || !TestPoolRuntimeFailsCleanlyWhenThePoolIsTooSmall()) {
        return 1;
    }

    std::cout << "PASS: pool_runtime_test\n";
    return 0;
}
