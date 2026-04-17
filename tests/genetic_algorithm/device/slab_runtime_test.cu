#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/genetic_algorithm.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::genetic_algorithm::GenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::TrySeedOutputEmbeddingTailFromHintGrids;
using neuroevolution::genetic_algorithm::genome::GenomePolicyModelParameters;
using neuroevolution::genetic_algorithm::genome::GenomeTailRows;
using neuroevolution::genetic_algorithm::genome::HostGenomeBytesAt;
using neuroevolution::genetic_algorithm::genome::HostPopulation;
using neuroevolution::genetic_algorithm::genome::TrainableActionEmbeddingTail;
using neuroevolution::genetic_algorithm::genome::TryInitializeRandomHostPopulation;
using neuroevolution::genetic_algorithm::genotype_slab::ComputeSlabSlotStrideBytes;
using neuroevolution::genetic_algorithm::genotype_slab::HostGenotypeSlab;
using neuroevolution::genetic_algorithm::genotype_slab::HostSlabSlotBytesAt;
using neuroevolution::genetic_algorithm::genotype_slab::SlabGeneration;
using neuroevolution::genetic_algorithm::genotype_slab::SlabSlotCountForByteBudget;
using neuroevolution::genetic_algorithm::genotype_slab::TryAllocateSlabSlot;
using neuroevolution::genetic_algorithm::genotype_slab::TryCopyGenomeBytesIntoSlabSlot;
using neuroevolution::genetic_algorithm::genotype_slab::TryCreateHostGenotypeSlab;
using neuroevolution::genetic_algorithm::genotype_slab::TryCreateSlabGeneration;
using neuroevolution::genetic_algorithm::genotype_slab::TrySetSlabGenerationSlot;
using neuroevolution::genetic_algorithm::slab_device::DeviceSlabGARuntimeBuffers;
using neuroevolution::genetic_algorithm::slab_device::DeviceSlabGARuntimeConfig;
using neuroevolution::genetic_algorithm::slab_device::DeviceSlabGARuntimeStatusCode;
using neuroevolution::genetic_algorithm::slab_device::PendingOutputEmbeddingInjection;
using neuroevolution::genetic_algorithm::slab_device::PopulationFitnessSummary;
using neuroevolution::genetic_algorithm::slab_device::RuntimeWordCounts;
using neuroevolution::genetic_algorithm::slab_device::TryAdvanceGenerationOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryCreateDeviceSlabGARuntimeBuffers;
using neuroevolution::genetic_algorithm::slab_device::TryDownloadCurrentGenerationFromDevice;
using neuroevolution::genetic_algorithm::slab_device::TryDownloadSlabFromDevice;
using neuroevolution::genetic_algorithm::slab_device::TryEvaluateCurrentGenerationFitnessOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryReadDeviceSlabGARuntimeStatus;
using neuroevolution::genetic_algorithm::slab_device::TryReadPopulationFitnessSummaryFromDevice;
using neuroevolution::genetic_algorithm::slab_device::TryUploadCurrentSlabPopulationToDevice;
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

RuntimeWordCounts MakeRuntimeWordCounts(const std::size_t action_count) {
    RuntimeWordCounts runtime_word_counts{};
    runtime_word_counts.training_word_count = action_count;
    runtime_word_counts.action_space_word_count = action_count;
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

bool PopulateSlabGenerationFromHostPopulation(const HostPopulation &host_population, const std::size_t slot_count,
                                              const std::size_t generation_index, HostGenotypeSlab &host_buffer,
                                              SlabGeneration &current_generation) {
    bool ok = TryCreateHostGenotypeSlab(host_buffer, slot_count, host_population.layout.action_count);
    ok &= TryCreateSlabGeneration(current_generation, host_population.layout.active_individual_count, generation_index);
    if (!ok) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < host_population.layout.active_individual_count;
         ++individual_index) {
        std::uint32_t slot_index = 0;
        ok &= TryAllocateSlabSlot(host_buffer, slot_index);
        ok &= TrySetSlabGenerationSlot(current_generation, individual_index, slot_index);
        ok &= TryCopyGenomeBytesIntoSlabSlot(host_buffer, slot_index,
                                             HostGenomeBytesAt(host_population, individual_index),
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

std::size_t ComputeGenerationByteBudgetBytes(const DeviceSlabGARuntimeConfig &runtime_config) {
    return runtime_config.generation_byte_budget_bytes;
}

DeviceSlabGARuntimeConfig MakeRuntimeConfig(const std::size_t slot_count, const std::size_t generation_size,
                                            const std::size_t action_count) {
    const std::size_t slot_stride_bytes = ComputeSlabSlotStrideBytes(action_count);

    DeviceSlabGARuntimeConfig runtime_config{};
    runtime_config.genotype_slab_byte_budget_bytes = slot_count * slot_stride_bytes;
    runtime_config.generation_byte_budget_bytes = generation_size * slot_stride_bytes;
    runtime_config.action_count = action_count;
    runtime_config.population_size_ceiling = generation_size;
    return runtime_config;
}

bool TestSlabRuntimeEvaluatesAndSummarizesCurrentGeneration() {
    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    HostPopulation host_population{};
    bool ok = TryInitializeRandomHostPopulation(host_population, 3, kActionCount, 41U);
    HostGenotypeSlab host_buffer{};
    SlabGeneration current_generation{};
    ok &= PopulateSlabGenerationFromHostPopulation(host_population, 6, 4, host_buffer, current_generation);
    if (!ok) {
        std::cerr << "FAIL: could not build buffer evaluation fixtures\n";
        return false;
    }

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(6, 3, kActionCount);

    DeviceSlabGARuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabGARuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadCurrentSlabPopulationToDevice(host_buffer, current_generation, buffers);
    ok &= TryEvaluateCurrentGenerationFitnessOnDevice(buffers, MakeRuntimeWordCounts());
    if (!ok) {
        neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    PopulationFitnessSummary summary{};
    SlabGeneration downloaded_generation{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_generation);
    neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
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

bool TestSlabRuntimeAdvancesOneGeneration() {
    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    HostPopulation host_population{};
    bool ok = TryInitializeRandomHostPopulation(host_population, 3, kActionCount, 77U);
    CopyGenomeAcrossPopulation(host_population, 0);

    HostGenotypeSlab host_buffer{};
    SlabGeneration current_generation{};
    ok &= PopulateSlabGenerationFromHostPopulation(host_population, 6, 2, host_buffer, current_generation);
    if (!ok) {
        std::cerr << "FAIL: could not build slab generation-step fixtures\n";
        return false;
    }

    const float expected_bias = ToFloat(
        GenomePolicyModelParameters(HostGenomeBytesAt(host_population, 0)).dense_trunk.hidden1_to_output.biases[0]);
    const float expected_tail = ToFloat(GenomeTailRows(HostGenomeBytesAt(host_population, 0))[0][0]);

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(6, 3, kActionCount);

    DeviceSlabGARuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabGARuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadCurrentSlabPopulationToDevice(host_buffer, current_generation, buffers);
    ok &= TryAdvanceGenerationOnDevice(buffers, 19U, MakeRuntimeWordCounts(), MakeAssemblyConfig());
    if (!ok) {
        DeviceSlabGARuntimeStatusCode status_code = DeviceSlabGARuntimeStatusCode::kOk;
        (void)TryReadDeviceSlabGARuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: slab generation step failed with status " << static_cast<int>(status_code) << '\n';
        neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    PopulationFitnessSummary summary{};
    HostGenotypeSlab downloaded_buffer{};
    SlabGeneration downloaded_generation{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary);
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_generation);
    neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(summary.generation_index == 2, "Expected summary to describe the evaluated parent generation");
    ok &= ExpectTrue(downloaded_generation.generation_index == 3,
                     "Expected a successful generation step to increment the current generation index");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == 3,
                     "Expected three slab slots to be free after three parents are replaced by three children");

    for (std::size_t individual_index = 0; individual_index < downloaded_generation.active_individual_count;
         ++individual_index) {
        const std::uint32_t slot_index = downloaded_generation.slot_indices[individual_index];
        ok &= ExpectTrue(slot_index != neuroevolution::genetic_algorithm::genotype_slab::kInvalidSlabSlotIndex,
                         "Expected every child generation handle to reference a live slab slot");
        ok &= ExpectTrue(downloaded_generation.has_fitness[individual_index] == 0,
                         "Expected newly assembled children to start unevaluated");
        ok &= ExpectTrue(downloaded_generation.evaluation_counts[individual_index] == 0,
                         "Expected newly assembled children to start with zero evaluation count");
        ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(downloaded_buffer, slot_index))
                                     .dense_trunk.hidden1_to_output.biases[0]),
                         expected_bias, "child dense-trunk bias");
        ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(downloaded_buffer, slot_index))[0][0]),
                         expected_tail, "child trainable tail value");
        ok &= ExpectTrue(downloaded_buffer.slot_states[slot_index].reference_count == 1,
                         "Expected each child slot to hold exactly one generation reference");
    }

    return ok;
}

bool TestSlabRuntimeGrowsActionCountWithSlabRepacking() {
    constexpr std::size_t kInjectedWordCount = 3;

    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    HostPopulation host_population{};
    bool ok = TryInitializeRandomHostPopulation(host_population, 3, kActionCount, 91U);
    CopyGenomeAcrossPopulation(host_population, 0);

    HostGenotypeSlab host_buffer{};
    SlabGeneration current_generation{};
    ok &= PopulateSlabGenerationFromHostPopulation(host_population, 6, 8, host_buffer, current_generation);
    if (!ok) {
        std::cerr << "FAIL: could not build buffer growth fixtures\n";
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

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(6, 3, kActionCount);
    const std::size_t next_action_count = kActionCount + kInjectedWordCount;
    const std::size_t expected_next_generation_size = SlabSlotCountForByteBudget(
        ComputeGenerationByteBudgetBytes(runtime_config), next_action_count, runtime_config.population_size_ceiling);
    ok &= ExpectTrue(expected_next_generation_size == 2,
                     "Expected the fixed generation byte budget to shrink the grown generation to two children");
    if (!ok) {
        return false;
    }

    DeviceSlabGARuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabGARuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadCurrentSlabPopulationToDevice(host_buffer, current_generation, buffers);
    ok &= TryAdvanceGenerationOnDevice(buffers, 31U, MakeRuntimeWordCounts(), MakeAssemblyConfig(),
                                       pending_output_embedding_injection);
    if (!ok) {
        DeviceSlabGARuntimeStatusCode status_code = DeviceSlabGARuntimeStatusCode::kOk;
        (void)TryReadDeviceSlabGARuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: buffer growth step failed with status " << static_cast<int>(status_code) << '\n';
        neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    PopulationFitnessSummary summary{};
    PopulationFitnessSummary child_summary{};
    HostGenotypeSlab downloaded_buffer{};
    SlabGeneration downloaded_generation{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary);
    ok &= TryEvaluateCurrentGenerationFitnessOnDevice(buffers, MakeRuntimeWordCounts(next_action_count));
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, child_summary);
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_generation);
    neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &=
        ExpectTrue(summary.generation_index == 8, "Expected summary to still describe the evaluated parent generation");
    ok &= ExpectTrue(downloaded_generation.generation_index == 9,
                     "Expected a successful growth step to increment the generation index");
    ok &= ExpectTrue(downloaded_generation.active_individual_count == expected_next_generation_size,
                     "Expected growth to size the next generation from the fixed generation byte budget");
    ok &= ExpectTrue(downloaded_buffer.layout.action_count == next_action_count,
                     "Expected repacking to expand the slab action count before child assembly");
    ok &= ExpectTrue(child_summary.generation_index == 9,
                     "Expected child evaluation to report the grown current generation index");
    ok &= ExpectTrue(child_summary.population_size == expected_next_generation_size,
                     "Expected child evaluation to report the budget-sized grown population");
    ok &= ExpectTrue(child_summary.action_count == next_action_count,
                     "Expected child evaluation to report the grown action count");

    for (std::size_t individual_index = 0; individual_index < downloaded_generation.active_individual_count;
         ++individual_index) {
        const std::uint32_t slot_index = downloaded_generation.slot_indices[individual_index];
        ok &= ExpectTrue(slot_index != neuroevolution::genetic_algorithm::genotype_slab::kInvalidSlabSlotIndex,
                         "Expected every grown child to occupy a live slab slot");
        for (std::size_t injection_offset = 0; injection_offset < kInjectedWordCount; ++injection_offset) {
            ok &= ExpectTailEquals(
                GenomeTailRows(HostSlabSlotBytesAt(downloaded_buffer, slot_index))[kActionCount + injection_offset],
                expected_tails[injection_offset],
                std::string("grown child injected tail ") + std::to_string(individual_index) + ":" +
                    std::to_string(injection_offset));
        }
    }

    return ok;
}

bool TestSlabRuntimeRejectsGrowthWhenGenerationBudgetCannotFitOneChild() {
    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    HostPopulation host_population{};
    bool ok = TryInitializeRandomHostPopulation(host_population, 1, kActionCount, 7U);
    HostGenotypeSlab host_buffer{};
    SlabGeneration current_generation{};
    ok &= PopulateSlabGenerationFromHostPopulation(host_population, 2, 0, host_buffer, current_generation);
    if (!ok) {
        std::cerr << "FAIL: could not build budget-rejection fixtures\n";
        return false;
    }

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(2, 1, kActionCount);
    const std::size_t next_action_count = kActionCount + 1;
    const std::size_t expected_next_generation_size = SlabSlotCountForByteBudget(
        ComputeGenerationByteBudgetBytes(runtime_config), next_action_count, runtime_config.population_size_ceiling);
    ok &= ExpectTrue(expected_next_generation_size == 0,
                     "Expected the fixed generation byte budget to reject a larger genotype that cannot fit one child");
    if (!ok) {
        return false;
    }

    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
    pending_output_embedding_injection.enabled = true;
    pending_output_embedding_injection.first_catalog_word_index = kActionCount;
    pending_output_embedding_injection.injection_count = 1;

    DeviceSlabGARuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabGARuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadCurrentSlabPopulationToDevice(host_buffer, current_generation, buffers);
    if (!ok) {
        neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    ok &= ExpectTrue(!TryAdvanceGenerationOnDevice(buffers, 9U, MakeRuntimeWordCounts(), MakeAssemblyConfig(),
                                                   pending_output_embedding_injection),
                     "Expected growth assembly to fail when the fixed generation byte budget cannot fit one child");

    DeviceSlabGARuntimeStatusCode status_code = DeviceSlabGARuntimeStatusCode::kOk;
    ok &= TryReadDeviceSlabGARuntimeStatus(buffers, status_code);
    neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(status_code == DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig,
                     "Expected impossible growth under the fixed generation byte budget to report invalid assembly "
                     "config");
    return ok;
}

bool TestSlabRuntimeFailsCleanlyWhenTheSlabIsTooSmall() {
    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    HostPopulation host_population{};
    bool ok = TryInitializeRandomHostPopulation(host_population, 2, kActionCount, 11U);
    HostGenotypeSlab host_buffer{};
    SlabGeneration current_generation{};
    ok &= PopulateSlabGenerationFromHostPopulation(host_population, 2, 0, host_buffer, current_generation);
    if (!ok) {
        std::cerr << "FAIL: could not build full-slab runtime fixtures\n";
        return false;
    }

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(2, 2, kActionCount);

    DeviceSlabGARuntimeBuffers buffers{};
    ok &= TryCreateDeviceSlabGARuntimeBuffers(buffers, runtime_config);
    ok &= TryUploadCurrentSlabPopulationToDevice(host_buffer, current_generation, buffers);
    if (!ok) {
        neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    ok &= ExpectTrue(!TryAdvanceGenerationOnDevice(buffers, 23U, MakeRuntimeWordCounts(), MakeAssemblyConfig()),
                     "Expected slab-backed generation stepping to fail when no child slot can be allocated");

    DeviceSlabGARuntimeStatusCode status_code = DeviceSlabGARuntimeStatusCode::kOk;
    ok &= TryReadDeviceSlabGARuntimeStatus(buffers, status_code);
    neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(status_code == DeviceSlabGARuntimeStatusCode::kSlabFull,
                     "Expected an undersized buffer to report kSlabFull during generation stepping");
    return ok;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    if (!TestSlabRuntimeEvaluatesAndSummarizesCurrentGeneration() || !TestSlabRuntimeAdvancesOneGeneration() ||
        !TestSlabRuntimeGrowsActionCountWithSlabRepacking() ||
        !TestSlabRuntimeRejectsGrowthWhenGenerationBudgetCannotFitOneChild() ||
        !TestSlabRuntimeFailsCleanlyWhenTheSlabIsTooSmall()) {
        return 1;
    }

    std::cout << "PASS: slab_runtime_test\n";
    return 0;
}
