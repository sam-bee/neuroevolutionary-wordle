#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/device/evaluation_ops.cuh"
#include "genetic_algorithm/genetic_algorithm.hpp"
#include "genetic_algorithm/spatial/grid.hpp"
#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::genetic_algorithm::GenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::TrySeedOutputEmbeddingTailFromHintGrids;
using neuroevolution::genetic_algorithm::device_evaluation_ops::MaximumPossibleFitness;
using neuroevolution::genetic_algorithm::device_evaluation_ops::NormalizeFitnessForSelection;
using neuroevolution::genetic_algorithm::genome::GenomePolicyModelParameters;
using neuroevolution::genetic_algorithm::genome::GenomeTailRows;
using neuroevolution::genetic_algorithm::genome::TrainableActionEmbeddingTail;
using neuroevolution::genetic_algorithm::genotype_slab::ComputeSlabSlotStrideBytes;
using neuroevolution::genetic_algorithm::genotype_slab::HostGenotypeSlab;
using neuroevolution::genetic_algorithm::genotype_slab::HostSlabSlotBytesAt;
using neuroevolution::genetic_algorithm::genotype_slab::SlabAssemblyPlan;
using neuroevolution::genetic_algorithm::genotype_slab::SlabParentPair;
using neuroevolution::genetic_algorithm::genotype_slab::SlabGeneration;
using neuroevolution::genetic_algorithm::genotype_slab::SlabSlotCountForByteBudget;
using neuroevolution::genetic_algorithm::genotype_slab::TryCreateSlabAssemblyPlan;
using neuroevolution::genetic_algorithm::slab_device::DeviceSlabGARuntimeBuffers;
using neuroevolution::genetic_algorithm::slab_device::DeviceSlabGARuntimeConfig;
using neuroevolution::genetic_algorithm::slab_device::DeviceSlabGARuntimeStatusCode;
using neuroevolution::genetic_algorithm::slab_device::PendingOutputEmbeddingInjection;
using neuroevolution::genetic_algorithm::slab_device::PopulationFitnessSummary;
using neuroevolution::genetic_algorithm::slab_device::RuntimeWordCounts;
using neuroevolution::genetic_algorithm::slab_device::TryAdvanceGenerationOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryBootstrapRandomCurrentGenerationOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryCreateDeviceSlabGARuntimeBuffers;
using neuroevolution::genetic_algorithm::slab_device::TryDownloadCurrentGenerationFromDevice;
using neuroevolution::genetic_algorithm::slab_device::TryDownloadSlabFromDevice;
using neuroevolution::genetic_algorithm::slab_device::TryDownloadSlabSlotBytesFromDevice;
using neuroevolution::genetic_algorithm::slab_device::TryEvaluateCurrentGenerationFitnessOnDevice;
using neuroevolution::genetic_algorithm::slab_device::TryReadDeviceSlabGARuntimeStatus;
using neuroevolution::genetic_algorithm::slab_device::TryReadPopulationFitnessSummaryFromDevice;
using neuroevolution::genetic_algorithm::spatial::CellularGridShape;
using neuroevolution::genetic_algorithm::spatial::CellularNeighborList;
using neuroevolution::genetic_algorithm::spatial::ContainsNeighborIndex;
using neuroevolution::genetic_algorithm::spatial::FloorRowPreservingPopulationSize;
using neuroevolution::genetic_algorithm::spatial::FloorSquareRoot;
using neuroevolution::genetic_algorithm::spatial::TryCollectCellularSecondParentCandidates;
using neuroevolution::genetic_algorithm::spatial::TryMakeCellularGridShape;
using neuroevolution::training_folder::LoadTrainingWordCatalogFromActionSpace;
using neuroevolution::training_folder::DeterministicTrainingShardCenterCellIndex;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::training_folder::UploadTrainingWordCatalogToDeviceConstantMemory;

constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr std::size_t kActionCount = neuroevolution::training_folder::kDefaultInitialActiveWordCount;
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
    runtime_word_counts.training_word_schedule.initial_word_count = kActionCount;
    runtime_word_counts.training_word_schedule.word_count_step = 0;
    runtime_word_counts.training_word_schedule.word_count_step_period_generations = 1;
    return runtime_word_counts;
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

RuntimeWordCounts MakeSpatialShardRuntimeWordCounts(const std::size_t introduced_word_count,
                                                    const std::size_t action_count) {
    RuntimeWordCounts runtime_word_counts{};
    runtime_word_counts.training_word_count = introduced_word_count;
    runtime_word_counts.action_space_word_count = action_count;
    runtime_word_counts.training_word_schedule.initial_word_count = kActionCount;
    runtime_word_counts.training_word_schedule.word_count_step = introduced_word_count - kActionCount;
    runtime_word_counts.training_word_schedule.word_count_step_period_generations = 1;
    return runtime_word_counts;
}

GenerationAssemblyConfig MakeAssemblyConfig() {
    GenerationAssemblyConfig config{};
    config.parent_selection.tournament_size = 3;
    config.parent_selection.allow_self_parenting = false;
    config.breeding.first_parent_probability = 1.0f;
    config.breeding.output_tail_row_arithmetic_recombination_probability = 0.0f;
    config.mutation.mutation_probability = 0.0f;
    config.mutation.mutation_sigma = 0.0f;
    config.mutation.output_tail_row_scale_mutation_probability = 0.0f;
    return config;
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
    runtime_config.host_spillover_byte_budget_bytes = runtime_config.generation_byte_budget_bytes / 2;
    runtime_config.action_count = action_count;
    runtime_config.population_size_ceiling = generation_size;
    runtime_config.grid_column_count = FloorSquareRoot(generation_size);
    return runtime_config;
}

bool TryCreateAndBootstrapRuntime(DeviceSlabGARuntimeBuffers &buffers, const DeviceSlabGARuntimeConfig &runtime_config,
                                  const std::size_t generation_size, const std::uint32_t generation_seed,
                                  const std::size_t generation_index = 0) {
    return TryCreateDeviceSlabGARuntimeBuffers(buffers, runtime_config) &&
           TryBootstrapRandomCurrentGenerationOnDevice(buffers, generation_size, generation_seed, generation_index);
}

bool TryDownloadAssemblyPlanFromDevice(const DeviceSlabGARuntimeBuffers &buffers, const std::size_t child_count,
                                       SlabAssemblyPlan &plan) {
    if (!TryCreateSlabAssemblyPlan(plan, child_count)) {
        return false;
    }

    return CheckCuda(cudaMemcpy(plan.parent_pairs.get(), buffers.genotype_slab.assembly_parent_pairs,
                                child_count * sizeof(SlabParentPair), cudaMemcpyDeviceToHost),
                     "downloading slab assembly plan");
}

bool TestSlabRuntimeBootstrapsRandomCurrentGenerationOnDevice() {
    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(16, 9, kActionCount);

    DeviceSlabGARuntimeBuffers first_buffers{};
    DeviceSlabGARuntimeBuffers second_buffers{};
    bool ok = TryCreateDeviceSlabGARuntimeBuffers(first_buffers, runtime_config);
    ok &= TryCreateDeviceSlabGARuntimeBuffers(second_buffers, runtime_config);
    ok &= TryBootstrapRandomCurrentGenerationOnDevice(first_buffers, 9, 123U, 0);
    ok &= TryBootstrapRandomCurrentGenerationOnDevice(second_buffers, 9, 123U, 0);
    if (!ok) {
        neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(first_buffers);
        neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(second_buffers);
        std::cerr << "FAIL: could not bootstrap random device generations\n";
        return false;
    }

    HostGenotypeSlab first_buffer{};
    HostGenotypeSlab second_buffer{};
    SlabGeneration first_generation{};
    SlabGeneration second_generation{};
    ok &= TryDownloadSlabFromDevice(first_buffers, first_buffer);
    ok &= TryDownloadSlabFromDevice(second_buffers, second_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(first_buffers, first_generation);
    ok &= TryDownloadCurrentGenerationFromDevice(second_buffers, second_generation);
    neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(first_buffers);
    neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(second_buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(first_generation.generation_index == 0,
                     "Expected device bootstrap to start from generation index zero");
    ok &= ExpectTrue(first_generation.active_individual_count == 9,
                     "Expected device bootstrap to populate the requested generation size");
    ok &= ExpectTrue(first_buffer.free_slot_count == 7,
                     "Expected device bootstrap to leave the unused slab slots on the free list");
    ok &= ExpectTrue(second_generation.active_individual_count == 9,
                     "Expected repeated bootstrap with the same seed to use the same generation size");

    bool found_non_zero_parameter = false;
    for (std::size_t individual_index = 0; individual_index < first_generation.active_individual_count;
         ++individual_index) {
        const std::uint32_t first_slot_index = first_generation.slot_indices[individual_index];
        const std::uint32_t second_slot_index = second_generation.slot_indices[individual_index];
        ok &= ExpectTrue(first_slot_index != neuroevolution::genetic_algorithm::genotype_slab::kInvalidSlabSlotIndex,
                         "Expected every bootstrapped organism to occupy a live slab slot");
        ok &= ExpectTrue(second_slot_index != neuroevolution::genetic_algorithm::genotype_slab::kInvalidSlabSlotIndex,
                         "Expected repeat bootstrap to occupy a matching live slab slot");
        ok &= ExpectTrue(first_buffer.slot_states[first_slot_index].occupied,
                         "Expected bootstrapped slots to be marked occupied");
        ok &= ExpectTrue(first_buffer.slot_states[first_slot_index].liveness_count == 1U,
                         "Expected each bootstrapped slot to hold one generation liveness count");
        ok &= ExpectTrue(first_generation.has_fitness[individual_index] == 0,
                         "Expected bootstrapped organisms to start unevaluated");
        ok &= ExpectTrue(first_generation.evaluation_counts[individual_index] == 0,
                         "Expected bootstrapped organisms to start with zero evaluation count");
        ok &= ExpectTrue(std::memcmp(HostSlabSlotBytesAt(first_buffer, first_slot_index),
                                     HostSlabSlotBytesAt(second_buffer, second_slot_index),
                                     first_buffer.layout.slot_stride_bytes) == 0,
                         "Expected explicit seeding to make device bootstrap reproducible");

        const float sample_weight =
            ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(first_buffer, first_slot_index))
                        .dense_trunk.input_to_hidden0.weights[0]);
        found_non_zero_parameter |= (std::fabs(sample_weight) > 0.0f);
    }

    ok &= ExpectTrue(found_non_zero_parameter,
                     "Expected device bootstrap to produce non-zero random policy parameters");
    return ok;
}

bool TestSlabRuntimeDownloadsOneWinningSlabSlot() {
    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(8, 4, kActionCount);

    DeviceSlabGARuntimeBuffers buffers{};
    bool ok = TryCreateAndBootstrapRuntime(buffers, runtime_config, 4, 101U, 3);
    ok &= TryEvaluateCurrentGenerationFitnessOnDevice(buffers, MakeRuntimeWordCounts());
    if (!ok) {
        neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    PopulationFitnessSummary summary{};
    HostGenotypeSlab downloaded_buffer{};
    std::unique_ptr<std::uint8_t[]> downloaded_slot_bytes{};
    std::size_t downloaded_slot_byte_count = 0;
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary);
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadSlabSlotBytesFromDevice(buffers, summary.best_slot_index, downloaded_slot_bytes,
                                             downloaded_slot_byte_count);
    neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(downloaded_slot_byte_count == downloaded_buffer.layout.slot_stride_bytes,
                     "Expected single-slot download to copy one full genome stride");
    ok &= ExpectTrue(
        std::memcmp(downloaded_slot_bytes.get(), HostSlabSlotBytesAt(downloaded_buffer, summary.best_slot_index),
                    downloaded_slot_byte_count) == 0,
        "Expected single-slot download to match the corresponding downloaded slab slot");
    return ok;
}

bool TestSlabRuntimeRejectsInvalidSingleSlotDownloads() {
    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(8, 4, kActionCount);

    DeviceSlabGARuntimeBuffers buffers{};
    bool ok = TryCreateAndBootstrapRuntime(buffers, runtime_config, 4, 131U, 2);
    if (!ok) {
        neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    std::unique_ptr<std::uint8_t[]> downloaded_slot_bytes{};
    std::size_t downloaded_slot_byte_count = 0;
    ok &= ExpectTrue(!TryDownloadSlabSlotBytesFromDevice(buffers, 7U, downloaded_slot_bytes, downloaded_slot_byte_count),
                     "Expected single-slot download to reject an unoccupied slab slot");
    ok &= ExpectTrue(!TryDownloadSlabSlotBytesFromDevice(buffers, 8U, downloaded_slot_bytes, downloaded_slot_byte_count),
                     "Expected single-slot download to reject an out-of-range slab slot index");
    ok &= ExpectTrue(downloaded_slot_bytes == nullptr,
                     "Expected failed single-slot downloads to leave the output buffer empty");
    ok &= ExpectTrue(downloaded_slot_byte_count == 0,
                     "Expected failed single-slot downloads to leave the output byte count at zero");
    neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
    return ok;
}

bool TestSlabRuntimeEvaluatesAndSummarizesCurrentGeneration() {
    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(8, 4, kActionCount);

    DeviceSlabGARuntimeBuffers buffers{};
    bool ok = TryCreateAndBootstrapRuntime(buffers, runtime_config, 4, 41U, 4);
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

    const RuntimeWordCounts runtime_word_counts = MakeRuntimeWordCounts();
    const float minimum_possible_fitness = NormalizeFitnessForSelection(0.0f, runtime_word_counts);
    ok &= ExpectTrue(summary.population_size == 4, "Expected summary to report the current generation size");
    ok &= ExpectTrue(summary.action_count == kActionCount, "Expected summary to report the fixed action count");
    ok &= ExpectTrue(summary.generation_index == 4, "Expected summary to report the current generation index");
    ok &= ExpectTrue(summary.best_index < downloaded_generation.active_individual_count,
                     "Expected summary best index to point at a live current-generation organism");
    ok &= ExpectTrue(summary.best_slot_index == downloaded_generation.slot_indices[summary.best_index],
                     "Expected summary best slot index to match the winning current-generation slot");
    ok &= ExpectInRange(summary.best_fitness, minimum_possible_fitness, 1.0f, "summary best fitness");
    ok &= ExpectInRange(summary.average_fitness, minimum_possible_fitness, 1.0f, "summary average fitness");

    for (std::size_t individual_index = 0; individual_index < downloaded_generation.active_individual_count;
         ++individual_index) {
        ok &= ExpectTrue(downloaded_generation.has_fitness[individual_index] == 1,
                         "Expected evaluation to mark every current-generation individual as fitted");
        ok &= ExpectTrue(downloaded_generation.evaluation_counts[individual_index] == 1,
                         "Expected evaluation to increment the count for every current-generation individual");
        ok &= ExpectInRange(downloaded_generation.fitness[individual_index], minimum_possible_fitness, 1.0f,
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

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(8, 4, kActionCount);

    DeviceSlabGARuntimeBuffers buffers{};
    bool ok = TryCreateAndBootstrapRuntime(buffers, runtime_config, 4, 77U, 2);
    HostGenotypeSlab parent_buffer{};
    SlabGeneration parent_generation{};
    ok &= TryDownloadSlabFromDevice(buffers, parent_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, parent_generation);
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
    SlabAssemblyPlan downloaded_plan{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary);
    ok &= TryDownloadAssemblyPlanFromDevice(buffers, 4, downloaded_plan);
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_generation);
    neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(summary.generation_index == 2, "Expected summary to describe the evaluated parent generation");
    ok &= ExpectTrue(downloaded_generation.generation_index == 3,
                     "Expected a successful generation step to increment the current generation index");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == 4,
                     "Expected four slab slots to be free after four parents are replaced by four children");

    for (std::size_t individual_index = 0; individual_index < downloaded_generation.active_individual_count;
         ++individual_index) {
        const std::uint32_t slot_index = downloaded_generation.slot_indices[individual_index];
        const SlabParentPair &parent_pair = downloaded_plan.parent_pairs[individual_index];
        const std::uint32_t parent_slot_index = parent_generation.slot_indices[parent_pair.first_parent_index];
        ok &= ExpectTrue(slot_index != neuroevolution::genetic_algorithm::genotype_slab::kInvalidSlabSlotIndex,
                         "Expected every child generation handle to reference a live slab slot");
        ok &= ExpectTrue(downloaded_generation.has_fitness[individual_index] == 0,
                         "Expected newly assembled children to start unevaluated");
        ok &= ExpectTrue(downloaded_generation.evaluation_counts[individual_index] == 0,
                         "Expected newly assembled children to start with zero evaluation count");
        ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(downloaded_buffer, slot_index))
                                     .dense_trunk.hidden1_to_output.biases[0]),
                         ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(parent_buffer, parent_slot_index))
                                     .dense_trunk.hidden1_to_output.biases[0]),
                         "child dense-trunk bias");
        ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(downloaded_buffer, slot_index))[0][0]),
                         ToFloat(GenomeTailRows(HostSlabSlotBytesAt(parent_buffer, parent_slot_index))[0][0]),
                         "child trainable tail value");
        ok &= ExpectTrue(downloaded_buffer.slot_states[slot_index].liveness_count == 1,
                         "Expected each child slot to hold exactly one generation reference");
    }

    return ok;
}

bool TestSlabRuntimeBuildsCellularLocalAssemblyPlan() {
    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(50, 25, kActionCount);

    DeviceSlabGARuntimeBuffers buffers{};
    bool ok = TryCreateAndBootstrapRuntime(buffers, runtime_config, 25, 123U, 6);
    ok &= TryAdvanceGenerationOnDevice(buffers, 47U, MakeRuntimeWordCounts(), MakeAssemblyConfig());
    if (!ok) {
        DeviceSlabGARuntimeStatusCode status_code = DeviceSlabGARuntimeStatusCode::kOk;
        (void)TryReadDeviceSlabGARuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: cellular-plan generation step failed with status " << static_cast<int>(status_code)
                  << '\n';
        neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    SlabAssemblyPlan downloaded_plan{};
    ok &= TryDownloadAssemblyPlanFromDevice(buffers, 25, downloaded_plan);
    neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    CellularGridShape shape{};
    ok &= TryMakeCellularGridShape(25, shape);
    if (!ok) {
        return false;
    }

    for (std::size_t child_index = 0; child_index < downloaded_plan.child_count; ++child_index) {
        const SlabParentPair &parent_pair = downloaded_plan.parent_pairs[child_index];
        CellularNeighborList neighbors{};
        ok &= TryCollectCellularSecondParentCandidates(shape, child_index, neighbors);
        ok &= ExpectTrue(parent_pair.first_parent_index != parent_pair.second_parent_index,
                         "Expected each cellular child to have two different parents");
        ok &= ExpectTrue(ContainsNeighborIndex(neighbors, parent_pair.first_parent_index),
                         "Expected each cellular child to choose its first parent from the local radius-two "
                         "neighborhood");
        ok &= ExpectTrue(ContainsNeighborIndex(neighbors, parent_pair.second_parent_index),
                         "Expected each cellular child to choose its second parent from the local radius-two "
                         "neighborhood");
    }

    return ok;
}

bool TestSlabRuntimeEvaluatesSpatialTrainingShardExposurePerCell() {
    constexpr std::size_t kExpandedActionCount = kActionCount + 10;

    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(50, 25, kExpandedActionCount);

    DeviceSlabGARuntimeBuffers buffers{};
    bool ok = TryCreateAndBootstrapRuntime(buffers, runtime_config, 25, 151U, 1);
    ok &= TryEvaluateCurrentGenerationFitnessOnDevice(buffers,
                                                     MakeSpatialShardRuntimeWordCounts(kExpandedActionCount,
                                                                                       kExpandedActionCount));
    if (!ok) {
        DeviceSlabGARuntimeStatusCode status_code = DeviceSlabGARuntimeStatusCode::kOk;
        (void)TryReadDeviceSlabGARuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: spatial-shard evaluation failed with status " << static_cast<int>(status_code) << '\n';
        neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    std::uint32_t local_training_word_counts[25]{};
    ok &= CheckCuda(cudaMemcpy(local_training_word_counts, buffers.current_local_training_word_counts,
                               sizeof(local_training_word_counts), cudaMemcpyDeviceToHost),
                    "copying local training-word counts back to host");
    neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    const std::size_t shard_center_cell_index = DeterministicTrainingShardCenterCellIndex(0, 25);
    for (std::size_t cell_index = 0; cell_index < 25; ++cell_index) {
        const std::uint32_t expected_local_word_count =
            (cell_index == shard_center_cell_index) ? static_cast<std::uint32_t>(kExpandedActionCount)
                                                    : static_cast<std::uint32_t>(kActionCount);
        ok &= ExpectTrue(local_training_word_counts[cell_index] == expected_local_word_count,
                         "Expected only the shard-center cell to see the newly introduced local evaluation words");
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

    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
    pending_output_embedding_injection.enabled = true;
    pending_output_embedding_injection.first_catalog_word_index = kActionCount;
    pending_output_embedding_injection.injection_count = kInjectedWordCount;

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(18, 9, kActionCount);
    const std::size_t next_action_count = kActionCount + kInjectedWordCount;
    const std::size_t expected_next_generation_size =
        FloorRowPreservingPopulationSize(SlabSlotCountForByteBudget(ComputeGenerationByteBudgetBytes(runtime_config),
                                                                    next_action_count,
                                                                    runtime_config.population_size_ceiling),
                                         runtime_config.grid_column_count);
    bool ok = true;
    ok &= ExpectTrue(expected_next_generation_size == 6,
                     "Expected the fixed generation byte budget to shrink the grown generation by one row");
    if (!ok) {
        return false;
    }

    DeviceSlabGARuntimeBuffers buffers{};
    ok &= TryCreateAndBootstrapRuntime(buffers, runtime_config, 9, 91U, 8);
    HostGenotypeSlab parent_buffer{};
    SlabGeneration parent_generation{};
    ok &= TryDownloadSlabFromDevice(buffers, parent_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, parent_generation);
    ok &= TryAdvanceGenerationOnDevice(buffers, 31U, MakeRuntimeWordCounts(), MakeAssemblyConfig(),
                                       pending_output_embedding_injection, &training_word_catalog);
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
    SlabAssemblyPlan downloaded_plan{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary);
    ok &= TryDownloadAssemblyPlanFromDevice(buffers, expected_next_generation_size, downloaded_plan);
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
    ok &= ExpectTrue(summary.best_index < parent_generation.active_individual_count,
                     "Expected parent summary best index to point at a live parent organism");
    ok &= ExpectTrue(summary.best_slot_index == parent_generation.slot_indices[summary.best_index],
                     "Expected parent summary best slot index to match the winning parent slot");
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
    ok &= ExpectTrue(child_summary.best_index < downloaded_generation.active_individual_count,
                     "Expected child summary best index to point at a live child organism");
    ok &= ExpectTrue(child_summary.best_slot_index == downloaded_generation.slot_indices[child_summary.best_index],
                     "Expected child summary best slot index to match the winning child slot");

    for (std::size_t individual_index = 0; individual_index < downloaded_generation.active_individual_count;
         ++individual_index) {
        const std::uint32_t slot_index = downloaded_generation.slot_indices[individual_index];
        ok &= ExpectTrue(slot_index != neuroevolution::genetic_algorithm::genotype_slab::kInvalidSlabSlotIndex,
                         "Expected every grown child to occupy a live slab slot");
        const SlabParentPair &parent_pair = downloaded_plan.parent_pairs[individual_index];
        ok &= ExpectTrue(parent_pair.first_parent_index != parent_pair.second_parent_index,
                         "Expected grown children to have two different parents");
        for (std::size_t injection_offset = 0; injection_offset < kInjectedWordCount; ++injection_offset) {
            TrainableActionEmbeddingTail expected_tail{};
            ok &= TrySeedOutputEmbeddingTailFromHintGrids(
                GenomePolicyModelParameters(
                    HostSlabSlotBytesAt(parent_buffer, parent_generation.slot_indices[parent_pair.first_parent_index])),
                training_word_catalog.words[pending_output_embedding_injection.first_catalog_word_index +
                                            injection_offset],
                expected_tail);
            ok &= ExpectTailEquals(
                GenomeTailRows(HostSlabSlotBytesAt(downloaded_buffer, slot_index))[kActionCount + injection_offset],
                expected_tail,
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

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(2, 1, kActionCount);
    const std::size_t next_action_count = kActionCount + 1;
    const std::size_t expected_next_generation_size = SlabSlotCountForByteBudget(
        ComputeGenerationByteBudgetBytes(runtime_config), next_action_count, runtime_config.population_size_ceiling);
    bool ok = ExpectTrue(expected_next_generation_size == 0,
                     "Expected the fixed generation byte budget to reject a larger genotype that cannot fit one child");
    if (!ok) {
        return false;
    }

    PendingOutputEmbeddingInjection pending_output_embedding_injection{};
    pending_output_embedding_injection.enabled = true;
    pending_output_embedding_injection.first_catalog_word_index = kActionCount;
    pending_output_embedding_injection.injection_count = 1;

    DeviceSlabGARuntimeBuffers buffers{};
    ok &= TryCreateAndBootstrapRuntime(buffers, runtime_config, 1, 7U, 0);
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

bool TestSlabRuntimeAdvancesGenerationWithTightDeviceSlack() {
    TrainingWordCatalog training_word_catalog{};
    if (!InitializeTrainingCatalog(training_word_catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return false;
    }

    const DeviceSlabGARuntimeConfig runtime_config = MakeRuntimeConfig(6, 4, kActionCount);

    DeviceSlabGARuntimeBuffers buffers{};
    bool ok = TryCreateAndBootstrapRuntime(buffers, runtime_config, 4, 29U, 5);
    HostGenotypeSlab parent_buffer{};
    SlabGeneration parent_generation{};
    ok &= TryDownloadSlabFromDevice(buffers, parent_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, parent_generation);
    ok &= TryAdvanceGenerationOnDevice(buffers, 43U, MakeRuntimeWordCounts(), MakeAssemblyConfig());
    if (!ok) {
        DeviceSlabGARuntimeStatusCode status_code = DeviceSlabGARuntimeStatusCode::kOk;
        (void)TryReadDeviceSlabGARuntimeStatus(buffers, status_code);
        std::cerr << "FAIL: tight-slab assembly step failed with status " << static_cast<int>(status_code) << '\n';
        neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    PopulationFitnessSummary summary{};
    HostGenotypeSlab downloaded_buffer{};
    SlabGeneration downloaded_generation{};
    SlabAssemblyPlan downloaded_plan{};
    ok &= TryReadPopulationFitnessSummaryFromDevice(buffers, summary);
    ok &= TryDownloadAssemblyPlanFromDevice(buffers, 4, downloaded_plan);
    ok &= TryDownloadSlabFromDevice(buffers, downloaded_buffer);
    ok &= TryDownloadCurrentGenerationFromDevice(buffers, downloaded_generation);
    neuroevolution::genetic_algorithm::slab_device::DestroyDeviceSlabGARuntimeBuffers(buffers);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(summary.generation_index == 5,
                     "Expected tight-slab assembly to preserve the evaluated parent-generation summary");
    ok &= ExpectTrue(downloaded_generation.generation_index == 6,
                     "Expected tight-slab assembly to increment the current generation index");
    ok &= ExpectTrue(downloaded_buffer.free_slot_count == 2,
                     "Expected two slab slots to remain free after spill-assisted assembly finishes");

    for (std::size_t individual_index = 0; individual_index < downloaded_generation.active_individual_count;
         ++individual_index) {
        const std::uint32_t slot_index = downloaded_generation.slot_indices[individual_index];
        const SlabParentPair &parent_pair = downloaded_plan.parent_pairs[individual_index];
        const std::uint32_t parent_slot_index = parent_generation.slot_indices[parent_pair.first_parent_index];
        ok &= ExpectTrue(slot_index != neuroevolution::genetic_algorithm::genotype_slab::kInvalidSlabSlotIndex,
                         "Expected every spilled child handle to reference a live slab slot");
        ok &= ExpectTrue(downloaded_generation.has_fitness[individual_index] == 0,
                         "Expected spilled children to start unevaluated");
        ok &= ExpectTrue(downloaded_generation.evaluation_counts[individual_index] == 0,
                         "Expected spilled children to start with zero evaluation count");
        ok &= ExpectNear(ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(downloaded_buffer, slot_index))
                                     .dense_trunk.hidden1_to_output.biases[0]),
                         ToFloat(GenomePolicyModelParameters(HostSlabSlotBytesAt(parent_buffer, parent_slot_index))
                                     .dense_trunk.hidden1_to_output.biases[0]),
                         "tight-slab child dense-trunk bias");
        ok &= ExpectNear(ToFloat(GenomeTailRows(HostSlabSlotBytesAt(downloaded_buffer, slot_index))[0][0]),
                         ToFloat(GenomeTailRows(HostSlabSlotBytesAt(parent_buffer, parent_slot_index))[0][0]),
                         "tight-slab child trainable tail value");
        ok &= ExpectTrue(downloaded_buffer.slot_states[slot_index].liveness_count == 1,
                         "Expected each tight-slab child slot to hold exactly one generation reference");
    }

    return ok;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    if (!TestSlabRuntimeBootstrapsRandomCurrentGenerationOnDevice() ||
        !TestSlabRuntimeDownloadsOneWinningSlabSlot() || !TestSlabRuntimeRejectsInvalidSingleSlotDownloads() ||
        !TestSlabRuntimeEvaluatesAndSummarizesCurrentGeneration() || !TestSlabRuntimeAdvancesOneGeneration() ||
        !TestSlabRuntimeBuildsCellularLocalAssemblyPlan() ||
        !TestSlabRuntimeEvaluatesSpatialTrainingShardExposurePerCell() ||
        !TestSlabRuntimeGrowsActionCountWithSlabRepacking() ||
        !TestSlabRuntimeRejectsGrowthWhenGenerationBudgetCannotFitOneChild() ||
        !TestSlabRuntimeAdvancesGenerationWithTightDeviceSlack()) {
        return 1;
    }

    std::cout << "PASS: slab_runtime_test\n";
    return 0;
}
