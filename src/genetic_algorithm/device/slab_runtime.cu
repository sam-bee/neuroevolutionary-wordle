#include "genetic_algorithm/device/slab_runtime.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include "common/progress_log.hpp"
#include "genetic_algorithm/device/evaluation_ops.cuh"
#include "genetic_algorithm/device/genome_ops.cuh"
#include "genetic_algorithm/device/selection_ops.cuh"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "genetic_algorithm/genotype_slab/reference_counter.hpp"
#include "genetic_algorithm/output_embedding_injection.hpp"
#include "genetic_algorithm/spatial/grid.hpp"

namespace neuroevolution::genetic_algorithm::slab_device {

namespace {

using device_evaluation_ops::DeviceGenomeEvaluationStatusCode;
using device_evaluation_ops::GenomeEvaluationBlockScratch;
using device_evaluation_ops::GenomeEvaluationTensorActionTileScratch;
using device_evaluation_ops::GenomeEvaluationWarpScratch;
using device_evaluation_ops::NormalizeFitnessForSelection;
using device_evaluation_ops::ScoreCompletedEpisode;
using device_evaluation_ops::TryCountLocalTrainingWords;
using device_evaluation_ops::TryEvaluateGenomeEpisodeConcurrently;
using device_evaluation_ops::TryEvaluateGenomeFitnessConcurrently;
using device_evaluation_ops::TryInitializeEpisodeGrid;
using device_evaluation_ops::TryPlayWordleEpisodeTileWithTensorCoresConcurrently;
using device_genome_ops::DeviceRandomState;
using device_genome_ops::MakeDeviceRandomState;
using device_selection_ops::IsBetterFitness;
using device_selection_ops::TrySelectCellularParentPairDevice;
using neuroevolution::common::PrintTimestampedProgressDuration;
using neuroevolution::common::PrintTimestampedProgressLine;
using neuroevolution::common::ProgressClock;
using neuroevolution::inference::dynamic_policy::kDynamicPolicyWarpSize;
using spatial::CellularGridShape;
using spatial::FloorRowPreservingPopulationSize;
using spatial::TryMakeCellularGridShape;
using spatial::TryMakeCellularGridShapeForColumnCount;
using spatial::TryMakeRectangularCellularGridShape;
using training_folder::DeviceTrainingWordCatalog;
using training_folder::TrainingDataShardRuntimeSet;
using training_folder::TrainingWordCatalog;
using training_folder::TryBuildTrainingDataShardRuntimeSet;

constexpr std::size_t kSlabGARuntimeThreadBlockSize = 256;
constexpr std::size_t kEvaluationWarpsPerBlock = 4;
constexpr std::size_t kEvaluationThreadsPerBlock = kEvaluationWarpsPerBlock * kDynamicPolicyWarpSize;
constexpr std::size_t kMinimumEvaluationBlockCount = 1024;
constexpr std::size_t kMaxEvaluationBlocksPerIndividual = 16;
// Benchmarks showed the WMMA scorer only helping at very small action counts; larger action spaces
// are much faster with the simpler warp-tiled direct scorer.
constexpr std::size_t kTensorActionScoreActionCountThreshold = 128;
constexpr std::size_t kDynamicPolicyThreadsPerBlock = kEvaluationThreadsPerBlock;
NEUROEVOLUTION_HOST_DEVICE constexpr int DeviceStatusValue(const DeviceSlabGARuntimeStatusCode status_code) {
    return static_cast<int>(status_code);
}

constexpr std::size_t CeilDivide(const std::size_t numerator, const std::size_t denominator) noexcept {
    return (denominator == 0) ? 0 : ((numerator + denominator - 1) / denominator);
}

inline bool CheckCuda(const cudaError_t error) { return error == cudaSuccess; }

inline bool ReadDeviceStatus(const DeviceSlabGARuntimeBuffers &buffers, int &status_value) {
    return CheckCuda(cudaMemcpy(&status_value, buffers.status, sizeof(int), cudaMemcpyDeviceToHost));
}

inline bool WriteDeviceStatus(const DeviceSlabGARuntimeBuffers &buffers,
                              const DeviceSlabGARuntimeStatusCode status_code) {
    const int status_value = DeviceStatusValue(status_code);
    return CheckCuda(cudaMemcpy(buffers.status, &status_value, sizeof(int), cudaMemcpyHostToDevice));
}

inline bool ResetDeviceStatus(const DeviceSlabGARuntimeBuffers &buffers) {
    return WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kOk);
}

inline bool KernelCompletedSuccessfully(const DeviceSlabGARuntimeBuffers &buffers) {
    int status_value = DeviceStatusValue(DeviceSlabGARuntimeStatusCode::kCudaFailure);
    return ReadDeviceStatus(buffers, status_value) &&
           (status_value == DeviceStatusValue(DeviceSlabGARuntimeStatusCode::kOk));
}

inline bool IsCurrentGenerationCompatible(const DeviceSlabGARuntimeBuffers &buffers) noexcept {
    return genotype_slab::IsValidGenotypeSlabLayout(buffers.genotype_slab.slab_layout) &&
           (buffers.max_generation_size > 0) && (buffers.genotype_slab.current_generation_size > 0) &&
           (buffers.genotype_slab.current_generation_size <= buffers.genotype_slab.max_generation_size) &&
           (buffers.grid_column_count > 0) &&
           ((buffers.genotype_slab.current_generation_size % buffers.grid_column_count) == 0);
}

inline bool TryUploadActiveTrainingShardsForCurrentGeneration(DeviceSlabGARuntimeBuffers &buffers,
                                                              const RuntimeWordCounts &runtime_word_counts,
                                                              CellularGridShape &current_grid_shape_out) {
    current_grid_shape_out = {};
    buffers.active_training_shard_count = 0;
    if ((buffers.active_training_shards == nullptr) || (buffers.active_training_shard_capacity == 0) ||
        !TryMakeCellularGridShapeForColumnCount(buffers.genotype_slab.current_generation_size,
                                                buffers.grid_column_count, current_grid_shape_out)) {
        return false;
    }

    TrainingDataShardRuntimeSet runtime_set{};
    if (!TryBuildTrainingDataShardRuntimeSet(runtime_word_counts.training_word_schedule,
                                             runtime_word_counts.training_word_count,
                                             buffers.genotype_slab.current_generation_index, current_grid_shape_out,
                                             buffers.epicenter_grid_shape, runtime_word_counts.shard_initial_radius,
                                             runtime_word_counts.shard_radius_growth_period_generations, runtime_set) ||
        (runtime_set.shard_count == 0) || (runtime_set.shard_count > buffers.active_training_shard_capacity)) {
        return false;
    }

    if (!CheckCuda(cudaMemcpy(buffers.active_training_shards, runtime_set.shards.values,
                              runtime_set.shard_count * sizeof(training_folder::TrainingDataShardRuntime),
                              cudaMemcpyHostToDevice))) {
        return false;
    }

    buffers.active_training_shard_count = runtime_set.shard_count;
    return true;
}

NEUROEVOLUTION_HOST_DEVICE constexpr DeviceSlabGARuntimeStatusCode
MapEvaluationStatus(const DeviceGenomeEvaluationStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceGenomeEvaluationStatusCode::kOk:
        return DeviceSlabGARuntimeStatusCode::kOk;
    case DeviceGenomeEvaluationStatusCode::kInvalidTrainingShard:
        return DeviceSlabGARuntimeStatusCode::kInvalidTrainingShard;
    case DeviceGenomeEvaluationStatusCode::kGuessAppendFailed:
        return DeviceSlabGARuntimeStatusCode::kGuessAppendFailed;
    case DeviceGenomeEvaluationStatusCode::kPolicyForwardFailed:
        return DeviceSlabGARuntimeStatusCode::kPolicyForwardFailed;
    case DeviceGenomeEvaluationStatusCode::kActionSelectionFailed:
        return DeviceSlabGARuntimeStatusCode::kActionSelectionFailed;
    }

    return DeviceSlabGARuntimeStatusCode::kCudaFailure;
}

inline DeviceSlabGARuntimeStatusCode
MapSlabRuntimeStatus(const genotype_slab::device::DeviceSlabRuntimeStatusCode status_code) noexcept {
    using genotype_slab::device::DeviceSlabRuntimeStatusCode;

    switch (status_code) {
    case DeviceSlabRuntimeStatusCode::kOk:
        return DeviceSlabGARuntimeStatusCode::kOk;
    case DeviceSlabRuntimeStatusCode::kCudaFailure:
        return DeviceSlabGARuntimeStatusCode::kCudaFailure;
    case DeviceSlabRuntimeStatusCode::kInvalidRuntimeConfig:
        return DeviceSlabGARuntimeStatusCode::kInvalidRuntimeConfig;
    case DeviceSlabRuntimeStatusCode::kInvalidSlab:
        return DeviceSlabGARuntimeStatusCode::kInvalidSlab;
    case DeviceSlabRuntimeStatusCode::kInvalidGeneration:
        return DeviceSlabGARuntimeStatusCode::kInvalidGeneration;
    case DeviceSlabRuntimeStatusCode::kInvalidAssemblyPlan:
        return DeviceSlabGARuntimeStatusCode::kInvalidAssemblyPlan;
    case DeviceSlabRuntimeStatusCode::kInvalidAssemblyConfig:
        return DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig;
    case DeviceSlabRuntimeStatusCode::kInvalidParentIndex:
        return DeviceSlabGARuntimeStatusCode::kInvalidParentIndex;
    case DeviceSlabRuntimeStatusCode::kSlabFull:
        return DeviceSlabGARuntimeStatusCode::kSlabFull;
    case DeviceSlabRuntimeStatusCode::kOutputEmbeddingInjectionFailed:
        return DeviceSlabGARuntimeStatusCode::kOutputEmbeddingInjectionFailed;
    case DeviceSlabRuntimeStatusCode::kSlabRepackFailed:
        return DeviceSlabGARuntimeStatusCode::kSlabRepackFailed;
    }

    return DeviceSlabGARuntimeStatusCode::kCudaFailure;
}

__device__ void SetFailureStatus(int *status, const DeviceSlabGARuntimeStatusCode status_code) {
    atomicCAS(status, DeviceStatusValue(DeviceSlabGARuntimeStatusCode::kOk), DeviceStatusValue(status_code));
}

inline std::size_t SelectEvaluationBlocksPerIndividual(const std::size_t population_size,
                                                       const RuntimeWordCounts &runtime_word_counts) noexcept {
    const std::size_t minimum_blocks_per_individual =
        (population_size == 0) ? 1 : CeilDivide(kMinimumEvaluationBlockCount, population_size);
    const std::size_t maximum_useful_blocks_per_individual =
        CeilDivide(runtime_word_counts.training_word_count * device_evaluation_ops::kEpisodesPerTrainingWordCount,
                   kEvaluationWarpsPerBlock);
    const std::size_t bounded_minimum = (minimum_blocks_per_individual < kMaxEvaluationBlocksPerIndividual)
                                            ? minimum_blocks_per_individual
                                            : kMaxEvaluationBlocksPerIndividual;

    if (maximum_useful_blocks_per_individual == 0) {
        return 1;
    }

    const std::size_t candidate = (bounded_minimum > 0) ? bounded_minimum : 1;
    return (candidate < maximum_useful_blocks_per_individual) ? candidate : maximum_useful_blocks_per_individual;
}

__global__ void EvaluateSlabGenerationFitnessKernel(
    const std::uint8_t *slab_storage, const genotype_slab::SlabSlotState *slot_states,
    const genotype_slab::GenotypeSlabLayout slab_layout, const std::uint32_t *current_slot_indices,
    const std::size_t current_generation_size, float *current_fitness, std::uint32_t *current_evaluation_counts,
    std::uint8_t *current_has_fitness, std::uint32_t *current_local_training_word_counts,
    const RuntimeWordCounts runtime_word_counts,
    const training_folder::TrainingDataShardRuntime *active_training_shards,
    const std::size_t active_training_shard_count, const CellularGridShape current_grid_shape, int *status) {
    const std::size_t individual_index = blockIdx.x;
    __shared__ std::uint32_t shared_slot_index;
    __shared__ DeviceSlabGARuntimeStatusCode shared_status_code;
    __shared__ GenomeEvaluationBlockScratch<kDynamicPolicyThreadsPerBlock> evaluation_scratch;
    if (!genotype_slab::IsValidGenotypeSlabLayout(slab_layout) || (slab_storage == nullptr) ||
        (slot_states == nullptr) || (current_slot_indices == nullptr) || (current_fitness == nullptr) ||
        (current_evaluation_counts == nullptr) || (current_has_fitness == nullptr) ||
        (current_local_training_word_counts == nullptr) || (current_generation_size == 0) ||
        (active_training_shards == nullptr) || (active_training_shard_count == 0)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidSlab);
        }
        return;
    }

    if (individual_index >= current_generation_size) {
        return;
    }

    if (threadIdx.x == 0) {
        shared_slot_index = current_slot_indices[individual_index];
        shared_status_code = DeviceSlabGARuntimeStatusCode::kOk;
        if ((shared_slot_index == genotype_slab::kInvalidSlabSlotIndex) ||
            (shared_slot_index >= slab_layout.slot_count)) {
            shared_status_code = DeviceSlabGARuntimeStatusCode::kInvalidGeneration;
        } else if (!slot_states[shared_slot_index].occupied || (slot_states[shared_slot_index].liveness_count == 0)) {
            shared_status_code = DeviceSlabGARuntimeStatusCode::kInvalidSlab;
        }
    }
    __syncthreads();

    if (shared_status_code != DeviceSlabGARuntimeStatusCode::kOk) {
        if (threadIdx.x == 0) {
            SetFailureStatus(status, shared_status_code);
        }
        return;
    }

    float fitness = 0.0f;
    std::size_t local_training_word_count = 0;
    const DeviceGenomeEvaluationStatusCode evaluation_status =
        TryEvaluateGenomeFitnessConcurrently<kDynamicPolicyThreadsPerBlock>(
            genotype_slab::SlabSlotBytesAt(slab_storage, slab_layout, shared_slot_index), slab_layout.action_count,
            runtime_word_counts, active_training_shards, active_training_shard_count, current_grid_shape,
            individual_index, evaluation_scratch, fitness, local_training_word_count);
    if (evaluation_status != DeviceGenomeEvaluationStatusCode::kOk) {
        if (threadIdx.x == 0) {
            SetFailureStatus(status, MapEvaluationStatus(evaluation_status));
        }
        return;
    }

    if (threadIdx.x == 0) {
        current_fitness[individual_index] = fitness;
        current_local_training_word_counts[individual_index] = static_cast<std::uint32_t>(local_training_word_count);
        ++current_evaluation_counts[individual_index];
        current_has_fitness[individual_index] = 1;
    }
}

__global__ void EvaluateSlabGenerationFitnessByEpisodeKernel(
    const std::uint8_t *slab_storage, const genotype_slab::SlabSlotState *slot_states,
    const genotype_slab::GenotypeSlabLayout slab_layout, const std::uint32_t *current_slot_indices,
    const std::size_t current_generation_size, const RuntimeWordCounts runtime_word_counts,
    const training_folder::TrainingDataShardRuntime *active_training_shards, const std::size_t active_shard_count,
    const CellularGridShape current_grid_shape, const std::size_t blocks_per_individual, float *fitness_partial_sums,
    std::uint32_t *current_local_training_word_counts, int *status) {
    const std::size_t individual_index = blockIdx.x;
    const std::size_t block_tile_index = blockIdx.y;
    const std::size_t warp_index = static_cast<std::size_t>(threadIdx.x / kDynamicPolicyWarpSize);
    const std::size_t lane_index = static_cast<std::size_t>(threadIdx.x % kDynamicPolicyWarpSize);
    __shared__ std::uint32_t shared_slot_index;
    __shared__ DeviceSlabGARuntimeStatusCode shared_status_code;
    __shared__ std::size_t shared_local_training_word_count;
    __shared__ float shared_partial_sums[kEvaluationWarpsPerBlock];
    __shared__ GenomeEvaluationWarpScratch<kDynamicPolicyWarpSize> evaluation_scratch[kEvaluationWarpsPerBlock];
    __shared__ GenomeEvaluationTensorActionTileScratch<kEvaluationWarpsPerBlock> tensor_tile_scratch;
    if (!genotype_slab::IsValidGenotypeSlabLayout(slab_layout) || (slab_storage == nullptr) ||
        (slot_states == nullptr) || (current_slot_indices == nullptr) || (fitness_partial_sums == nullptr) ||
        (current_local_training_word_counts == nullptr) || (current_generation_size == 0) ||
        (active_training_shards == nullptr) || (active_shard_count == 0) || (blocks_per_individual == 0)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidSlab);
        }
        return;
    }

    if (individual_index >= current_generation_size) {
        return;
    }

    if (threadIdx.x == 0) {
        shared_slot_index = current_slot_indices[individual_index];
        shared_local_training_word_count = 0;
        shared_status_code = DeviceSlabGARuntimeStatusCode::kOk;
        if ((shared_slot_index == genotype_slab::kInvalidSlabSlotIndex) ||
            (shared_slot_index >= slab_layout.slot_count)) {
            shared_status_code = DeviceSlabGARuntimeStatusCode::kInvalidGeneration;
        } else if (!slot_states[shared_slot_index].occupied || (slot_states[shared_slot_index].liveness_count == 0)) {
            shared_status_code = DeviceSlabGARuntimeStatusCode::kInvalidSlab;
        } else {
            const DeviceGenomeEvaluationStatusCode local_word_status =
                TryCountLocalTrainingWords(active_training_shards, active_shard_count, current_grid_shape,
                                           individual_index, shared_local_training_word_count);
            if (local_word_status != DeviceGenomeEvaluationStatusCode::kOk) {
                shared_status_code = MapEvaluationStatus(local_word_status);
            }
        }
    }
    __syncthreads();

    if (shared_status_code != DeviceSlabGARuntimeStatusCode::kOk) {
        if (threadIdx.x == 0) {
            SetFailureStatus(status, shared_status_code);
        }
        return;
    }

    float warp_score_sum = 0.0f;
    const std::size_t episode_count =
        shared_local_training_word_count * device_evaluation_ops::kEpisodesPerTrainingWordCount;
    const std::uint8_t *genome_bytes = genotype_slab::SlabSlotBytesAt(slab_storage, slab_layout, shared_slot_index);
    const TrainingWordCatalog &training_word_catalog = DeviceTrainingWordCatalog();
    const std::size_t selectable_action_count = (slab_layout.action_count < runtime_word_counts.action_space_word_count)
                                                    ? slab_layout.action_count
                                                    : runtime_word_counts.action_space_word_count;
    const std::size_t episode_tile_stride = blocks_per_individual * kEvaluationWarpsPerBlock;

    for (std::size_t episode_tile_base = (block_tile_index * kEvaluationWarpsPerBlock);
         episode_tile_base < episode_count; episode_tile_base += episode_tile_stride) {
        const std::size_t episode_index = episode_tile_base + warp_index;
        if (lane_index == 0) {
            const bool warp_has_episode = episode_index < episode_count;
            tensor_tile_scratch.warp_has_episode[warp_index] = warp_has_episode ? 1U : 0U;
            tensor_tile_scratch.warp_active[warp_index] = warp_has_episode ? 1U : 0U;
            evaluation_scratch[warp_index].status = static_cast<int>(DeviceGenomeEvaluationStatusCode::kOk);
            if (warp_has_episode) {
                evaluation_scratch[warp_index].status = static_cast<int>(
                    TryInitializeEpisodeGrid(training_word_catalog, active_training_shards, active_shard_count,
                                             current_grid_shape, individual_index, episode_index,
                                             shared_local_training_word_count, evaluation_scratch[warp_index].grid));
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            shared_status_code = DeviceSlabGARuntimeStatusCode::kOk;
            for (std::size_t active_warp_index = 0; active_warp_index < kEvaluationWarpsPerBlock; ++active_warp_index) {
                if ((tensor_tile_scratch.warp_has_episode[active_warp_index] != 0) &&
                    (evaluation_scratch[active_warp_index].status !=
                     static_cast<int>(DeviceGenomeEvaluationStatusCode::kOk))) {
                    shared_status_code = MapEvaluationStatus(
                        static_cast<DeviceGenomeEvaluationStatusCode>(evaluation_scratch[active_warp_index].status));
                    break;
                }
            }
        }
        __syncthreads();

        if (shared_status_code != DeviceSlabGARuntimeStatusCode::kOk) {
            if (threadIdx.x == 0) {
                SetFailureStatus(status, shared_status_code);
            }
            return;
        }

        const DeviceGenomeEvaluationStatusCode evaluation_status =
            TryPlayWordleEpisodeTileWithTensorCoresConcurrently<kDynamicPolicyWarpSize, kEvaluationWarpsPerBlock>(
                genome_bytes, training_word_catalog, selectable_action_count, evaluation_scratch, tensor_tile_scratch);
        if (evaluation_status != DeviceGenomeEvaluationStatusCode::kOk) {
            if (threadIdx.x == 0) {
                SetFailureStatus(status, MapEvaluationStatus(evaluation_status));
            }
            return;
        }

        if ((lane_index == 0) && (tensor_tile_scratch.warp_has_episode[warp_index] != 0)) {
            warp_score_sum += ScoreCompletedEpisode(evaluation_scratch[warp_index].grid);
        }
        __syncthreads();
    }

    if (lane_index == 0) {
        shared_partial_sums[warp_index] = warp_score_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_score_sum = 0.0f;
        for (std::size_t partial_index = 0; partial_index < kEvaluationWarpsPerBlock; ++partial_index) {
            block_score_sum += shared_partial_sums[partial_index];
        }

        fitness_partial_sums[(individual_index * blocks_per_individual) + block_tile_index] = block_score_sum;
        if (block_tile_index == 0) {
            current_local_training_word_counts[individual_index] =
                static_cast<std::uint32_t>(shared_local_training_word_count);
        }
    }
}

__global__ void EvaluateSlabGenerationFitnessByEpisodeWarpKernel(
    const std::uint8_t *slab_storage, const genotype_slab::SlabSlotState *slot_states,
    const genotype_slab::GenotypeSlabLayout slab_layout, const std::uint32_t *current_slot_indices,
    const std::size_t current_generation_size, const RuntimeWordCounts runtime_word_counts,
    const training_folder::TrainingDataShardRuntime *active_training_shards, const std::size_t active_shard_count,
    const CellularGridShape current_grid_shape, const std::size_t blocks_per_individual, float *fitness_partial_sums,
    std::uint32_t *current_local_training_word_counts, int *status) {
    const std::size_t individual_index = blockIdx.x;
    const std::size_t block_tile_index = blockIdx.y;
    const std::size_t warp_index = static_cast<std::size_t>(threadIdx.x / kDynamicPolicyWarpSize);
    const std::size_t lane_index = static_cast<std::size_t>(threadIdx.x % kDynamicPolicyWarpSize);
    __shared__ std::uint32_t shared_slot_index;
    __shared__ DeviceSlabGARuntimeStatusCode shared_status_code;
    __shared__ std::size_t shared_local_training_word_count;
    __shared__ float shared_partial_sums[kEvaluationWarpsPerBlock];
    __shared__ GenomeEvaluationWarpScratch<kDynamicPolicyWarpSize> evaluation_scratch[kEvaluationWarpsPerBlock];
    if (!genotype_slab::IsValidGenotypeSlabLayout(slab_layout) || (slab_storage == nullptr) ||
        (slot_states == nullptr) || (current_slot_indices == nullptr) || (fitness_partial_sums == nullptr) ||
        (current_local_training_word_counts == nullptr) || (current_generation_size == 0) ||
        (active_training_shards == nullptr) || (active_shard_count == 0) || (blocks_per_individual == 0)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidSlab);
        }
        return;
    }

    if (individual_index >= current_generation_size) {
        return;
    }

    if (threadIdx.x == 0) {
        shared_slot_index = current_slot_indices[individual_index];
        shared_local_training_word_count = 0;
        shared_status_code = DeviceSlabGARuntimeStatusCode::kOk;
        if ((shared_slot_index == genotype_slab::kInvalidSlabSlotIndex) ||
            (shared_slot_index >= slab_layout.slot_count)) {
            shared_status_code = DeviceSlabGARuntimeStatusCode::kInvalidGeneration;
        } else if (!slot_states[shared_slot_index].occupied || (slot_states[shared_slot_index].liveness_count == 0)) {
            shared_status_code = DeviceSlabGARuntimeStatusCode::kInvalidSlab;
        } else {
            const DeviceGenomeEvaluationStatusCode local_word_status =
                TryCountLocalTrainingWords(active_training_shards, active_shard_count, current_grid_shape,
                                           individual_index, shared_local_training_word_count);
            if (local_word_status != DeviceGenomeEvaluationStatusCode::kOk) {
                shared_status_code = MapEvaluationStatus(local_word_status);
            }
        }
    }
    __syncthreads();

    if (shared_status_code != DeviceSlabGARuntimeStatusCode::kOk) {
        if (threadIdx.x == 0) {
            SetFailureStatus(status, shared_status_code);
        }
        return;
    }

    float warp_score_sum = 0.0f;
    const std::size_t episode_count =
        shared_local_training_word_count * device_evaluation_ops::kEpisodesPerTrainingWordCount;
    for (std::size_t episode_index = (block_tile_index * kEvaluationWarpsPerBlock) + warp_index;
         episode_index < episode_count; episode_index += (blocks_per_individual * kEvaluationWarpsPerBlock)) {
        float episode_score = 0.0f;
        const DeviceGenomeEvaluationStatusCode evaluation_status =
            TryEvaluateGenomeEpisodeConcurrently<kDynamicPolicyWarpSize>(
                genotype_slab::SlabSlotBytesAt(slab_storage, slab_layout, shared_slot_index), slab_layout.action_count,
                runtime_word_counts, active_training_shards, active_shard_count, current_grid_shape, individual_index,
                episode_index, shared_local_training_word_count, evaluation_scratch[warp_index], episode_score);
        if (evaluation_status != DeviceGenomeEvaluationStatusCode::kOk) {
            if (lane_index == 0) {
                SetFailureStatus(status, MapEvaluationStatus(evaluation_status));
            }
            return;
        }

        if (lane_index == 0) {
            warp_score_sum += episode_score;
        }
    }

    if (lane_index == 0) {
        shared_partial_sums[warp_index] = warp_score_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_score_sum = 0.0f;
        for (std::size_t partial_index = 0; partial_index < kEvaluationWarpsPerBlock; ++partial_index) {
            block_score_sum += shared_partial_sums[partial_index];
        }

        fitness_partial_sums[(individual_index * blocks_per_individual) + block_tile_index] = block_score_sum;
        if (block_tile_index == 0) {
            current_local_training_word_counts[individual_index] =
                static_cast<std::uint32_t>(shared_local_training_word_count);
        }
    }
}

__global__ void FinalizeSlabGenerationFitnessKernel(const float *fitness_partial_sums,
                                                    const std::size_t blocks_per_individual,
                                                    const std::size_t current_generation_size,
                                                    const std::uint32_t *current_local_training_word_counts,
                                                    float *current_fitness, std::uint32_t *current_evaluation_counts,
                                                    std::uint8_t *current_has_fitness, int *status) {
    const std::size_t individual_index =
        (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
    if ((fitness_partial_sums == nullptr) || (current_local_training_word_counts == nullptr) ||
        (current_fitness == nullptr) || (current_evaluation_counts == nullptr) || (current_has_fitness == nullptr) ||
        (current_generation_size == 0) || (blocks_per_individual == 0)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidGeneration);
        }
        return;
    }

    if (individual_index >= current_generation_size) {
        return;
    }

    const std::size_t local_training_word_count = current_local_training_word_counts[individual_index];
    if (local_training_word_count == 0) {
        SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidTrainingShard);
        return;
    }

    float score_sum = 0.0f;
    for (std::size_t block_tile_index = 0; block_tile_index < blocks_per_individual; ++block_tile_index) {
        score_sum += fitness_partial_sums[(individual_index * blocks_per_individual) + block_tile_index];
    }

    current_fitness[individual_index] = NormalizeFitnessForSelection(score_sum, local_training_word_count);
    ++current_evaluation_counts[individual_index];
    current_has_fitness[individual_index] = 1;
}

__global__ void SummarizeSlabGenerationKernel(const float *current_fitness, const std::uint8_t *current_has_fitness,
                                              const std::uint32_t *current_slot_indices,
                                              const std::size_t current_generation_index,
                                              const std::size_t current_generation_size, const std::size_t action_count,
                                              PopulationFitnessSummary *summary, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    if ((current_fitness == nullptr) || (current_has_fitness == nullptr) || (current_slot_indices == nullptr) ||
        (summary == nullptr) || (current_generation_size == 0) || (action_count == 0)) {
        SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidGeneration);
        return;
    }

    bool found_best = false;
    float best_fitness = 0.0f;
    float fitness_sum = 0.0f;
    std::size_t best_index = 0;

    for (std::size_t individual_index = 0; individual_index < current_generation_size; ++individual_index) {
        if (current_has_fitness[individual_index] == 0) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kPopulationNotEvaluated);
            return;
        }

        fitness_sum += current_fitness[individual_index];

        if (!found_best ||
            IsBetterFitness(current_fitness[individual_index], individual_index, best_fitness, best_index)) {
            found_best = true;
            best_fitness = current_fitness[individual_index];
            best_index = individual_index;
        }
    }

    summary->best_fitness = best_fitness;
    summary->average_fitness = fitness_sum / static_cast<float>(current_generation_size);
    summary->best_index = best_index;
    summary->best_slot_index = current_slot_indices[best_index];
    summary->generation_index = current_generation_index;
    summary->action_count = action_count;
    summary->population_size = current_generation_size;
}

__global__ void BuildSlabAssemblyPlanKernel(const float *current_fitness, const std::uint8_t *current_has_fitness,
                                            const std::size_t current_generation_size, const std::size_t child_count,
                                            const std::size_t grid_column_count,
                                            const std::size_t current_generation_index,
                                            const ParentSelectionConfig config, const std::uint32_t planning_seed,
                                            genotype_slab::SlabParentPair *parent_pairs, int *status) {
    const std::size_t child_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ((current_fitness == nullptr) || (current_has_fitness == nullptr) || (parent_pairs == nullptr) ||
        (current_generation_size == 0) || !IsValidParentSelectionConfig(config)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig);
        }
        return;
    }

    CellularGridShape current_grid_shape{};
    CellularGridShape next_grid_shape{};
    if (!TryMakeCellularGridShapeForColumnCount(current_generation_size, grid_column_count, current_grid_shape) ||
        !TryMakeCellularGridShapeForColumnCount(child_count, grid_column_count, next_grid_shape) ||
        (next_grid_shape.row_count > current_grid_shape.row_count)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig);
        }
        return;
    }

    if (child_index >= child_count) {
        return;
    }

    DeviceRandomState random_state = MakeDeviceRandomState(
        planning_seed, static_cast<std::uint32_t>(child_index + (current_generation_index * 8191U)));

    ParentPair parent_pair{};
    if (!TrySelectCellularParentPairDevice(current_fitness, current_has_fitness, current_grid_shape, next_grid_shape,
                                           child_index, random_state, config, parent_pair)) {
        SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kParentSelectionFailed);
        return;
    }

    parent_pairs[child_index].first_parent_index = static_cast<std::uint32_t>(parent_pair.first_parent_index);
    parent_pairs[child_index].second_parent_index = static_cast<std::uint32_t>(parent_pair.second_parent_index);
}

inline bool
IsValidPendingOutputEmbeddingInjection(const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
                                       const std::size_t current_action_count) noexcept {
    return !pending_output_embedding_injection.enabled ||
           ((pending_output_embedding_injection.injection_count > 0) &&
            (pending_output_embedding_injection.first_catalog_word_index == current_action_count));
}

inline bool TryComputeGenerationByteBudgetBytes(const std::size_t action_count, const std::size_t generation_size,
                                                std::size_t &generation_byte_budget_bytes_out) {
    generation_byte_budget_bytes_out = 0;

    const std::size_t slot_stride_bytes = genotype_slab::ComputeSlabSlotStrideBytes(action_count);
    if ((slot_stride_bytes == 0) || (generation_size > (std::numeric_limits<std::size_t>::max() / slot_stride_bytes))) {
        return false;
    }

    generation_byte_budget_bytes_out = generation_size * slot_stride_bytes;
    return generation_byte_budget_bytes_out > 0;
}

inline std::size_t RuntimeGenerationCapacity(const DeviceSlabGARuntimeConfig &config) noexcept {
    return FloorRowPreservingPopulationSize(
        genotype_slab::SlabSlotCountForByteBudget(config.generation_byte_budget_bytes, config.action_count,
                                                  config.population_size_ceiling),
        config.grid_column_count);
}

inline std::size_t RuntimeSlabSlotCount(const DeviceSlabGARuntimeConfig &config) noexcept {
    return genotype_slab::SlabSlotCountForByteBudget(config.genotype_slab_byte_budget_bytes, config.action_count);
}

inline bool TryCountAssembledChildPrefix(const DeviceSlabGARuntimeBuffers &buffers,
                                         std::size_t &assembled_child_count) {
    assembled_child_count = 0;

    genotype_slab::SlabGeneration next_generation{};
    if (!TryDownloadNextGenerationFromDevice(buffers, next_generation)) {
        return false;
    }

    while ((assembled_child_count < next_generation.active_individual_count) &&
           (next_generation.slot_indices[assembled_child_count] != genotype_slab::kInvalidSlabSlotIndex)) {
        ++assembled_child_count;
    }

    for (std::size_t child_index = assembled_child_count; child_index < next_generation.active_individual_count;
         ++child_index) {
        if (next_generation.slot_indices[child_index] != genotype_slab::kInvalidSlabSlotIndex) {
            return false;
        }
    }

    return true;
}

__global__ void ReleaseNextGenerationPrefixKernel(std::uint8_t *slab_storage, genotype_slab::SlabSlotState *slot_states,
                                                  std::uint32_t *free_slot_stack, std::uint32_t *free_slot_count,
                                                  std::uint32_t *free_slot_lock,
                                                  const genotype_slab::GenotypeSlabLayout slab_layout,
                                                  std::uint32_t *next_slot_indices, float *next_fitness,
                                                  std::uint32_t *next_evaluation_counts, std::uint8_t *next_has_fitness,
                                                  const std::size_t spilled_child_count, int *status) {
    genotype_slab::GenotypeSlabView slab{
        .layout = slab_layout,
        .storage = slab_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };

    const std::size_t worker_index =
        (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
    const std::size_t worker_count = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t child_index = worker_index; child_index < spilled_child_count; child_index += worker_count) {
        const std::uint32_t slot_index = next_slot_indices[child_index];
        if (slot_index == genotype_slab::kInvalidSlabSlotIndex) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidGeneration);
            return;
        }

        if (!genotype_slab::TryReleaseSlabSlot(slab, slot_index)) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidSlab);
            return;
        }

        next_slot_indices[child_index] = genotype_slab::kInvalidSlabSlotIndex;
        next_fitness[child_index] = 0.0f;
        next_evaluation_counts[child_index] = 0;
        next_has_fitness[child_index] = 0;
    }
}

__global__ void AllocateRestoredChildPrefixKernel(std::uint8_t *slab_storage, genotype_slab::SlabSlotState *slot_states,
                                                  std::uint32_t *free_slot_stack, std::uint32_t *free_slot_count,
                                                  std::uint32_t *free_slot_lock,
                                                  const genotype_slab::GenotypeSlabLayout slab_layout,
                                                  std::uint32_t *next_slot_indices, float *next_fitness,
                                                  std::uint32_t *next_evaluation_counts, std::uint8_t *next_has_fitness,
                                                  const std::size_t spilled_child_count, int *status) {
    genotype_slab::GenotypeSlabView slab{
        .layout = slab_layout,
        .storage = slab_storage,
        .slot_states = slot_states,
        .free_slot_stack = free_slot_stack,
        .free_slot_count = free_slot_count,
        .free_slot_lock = free_slot_lock,
    };

    const std::size_t worker_index =
        (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
    const std::size_t worker_count = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t child_index = worker_index; child_index < spilled_child_count; child_index += worker_count) {
        if (next_slot_indices[child_index] != genotype_slab::kInvalidSlabSlotIndex) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kInvalidGeneration);
            return;
        }

        std::uint32_t slot_index = genotype_slab::kInvalidSlabSlotIndex;
        if (!genotype_slab::TryAllocateSlabSlot(slab, slot_index)) {
            SetFailureStatus(status, DeviceSlabGARuntimeStatusCode::kSlabFull);
            return;
        }

        next_slot_indices[child_index] = slot_index;
        next_fitness[child_index] = 0.0f;
        next_evaluation_counts[child_index] = 0;
        next_has_fitness[child_index] = 0;
    }
}

inline bool TryCreateHostSpillStorage(const DeviceSlabGARuntimeBuffers &buffers, const std::size_t action_count,
                                      const std::size_t next_generation_size, const std::size_t next_generation_index,
                                      genotype_slab::HostGenotypeSlab &spill_slab,
                                      genotype_slab::SlabGeneration &spill_generation) {
    return genotype_slab::TryCreateHostGenotypeSlabForByteBudget(spill_slab, buffers.host_spillover_byte_budget_bytes,
                                                                 action_count) &&
           (spill_slab.layout.slot_count > 0) &&
           genotype_slab::TryCreateSlabGeneration(spill_generation, next_generation_size, next_generation_index);
}

inline bool TrySpillAssembledChildrenToHost(DeviceSlabGARuntimeBuffers &buffers, const std::size_t spilled_child_count,
                                            genotype_slab::HostGenotypeSlab &spill_slab,
                                            genotype_slab::SlabGeneration &spill_generation) {
    if ((spilled_child_count == 0) || !genotype_slab::IsValidHostGenotypeSlab(spill_slab) ||
        !genotype_slab::IsValidSlabGeneration(spill_generation) ||
        (spilled_child_count > spill_generation.active_individual_count) ||
        (spill_slab.layout.action_count != buffers.genotype_slab.slab_layout.action_count) ||
        (spill_slab.layout.slot_stride_bytes != buffers.genotype_slab.slab_layout.slot_stride_bytes)) {
        return false;
    }

    genotype_slab::SlabGeneration next_generation{};
    if (!TryDownloadNextGenerationFromDevice(buffers, next_generation)) {
        return false;
    }

    for (std::size_t child_index = 0; child_index < spilled_child_count; ++child_index) {
        const std::uint32_t device_slot_index = next_generation.slot_indices[child_index];
        if (device_slot_index == genotype_slab::kInvalidSlabSlotIndex) {
            return false;
        }

        std::uint32_t spill_slot_index = genotype_slab::kInvalidSlabSlotIndex;
        if (!genotype_slab::TryAllocateSlabSlot(spill_slab, spill_slot_index) ||
            !genotype_slab::TrySetSlabGenerationSlot(spill_generation, child_index, spill_slot_index) ||
            !CheckCuda(cudaMemcpy(genotype_slab::HostSlabSlotBytesAt(spill_slab, spill_slot_index),
                                  genotype_slab::SlabSlotBytesAt(buffers.genotype_slab.slab_storage,
                                                                 buffers.genotype_slab.slab_layout, device_slot_index),
                                  spill_slab.layout.slot_stride_bytes, cudaMemcpyDeviceToHost))) {
            return false;
        }
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    ReleaseNextGenerationPrefixKernel<<<(spilled_child_count + kSlabGARuntimeThreadBlockSize - 1) /
                                            kSlabGARuntimeThreadBlockSize,
                                        kSlabGARuntimeThreadBlockSize>>>(
        buffers.genotype_slab.slab_storage, buffers.genotype_slab.slot_states, buffers.genotype_slab.free_slot_stack,
        buffers.genotype_slab.free_slot_count, buffers.genotype_slab.free_slot_lock, buffers.genotype_slab.slab_layout,
        buffers.genotype_slab.next_slot_indices, buffers.genotype_slab.next_fitness,
        buffers.genotype_slab.next_evaluation_counts, buffers.genotype_slab.next_has_fitness, spilled_child_count,
        buffers.status);
    return CheckCuda(cudaGetLastError()) && CheckCuda(cudaDeviceSynchronize()) && KernelCompletedSuccessfully(buffers);
}

inline bool TryRestoreSpilledChildrenToDevice(DeviceSlabGARuntimeBuffers &buffers,
                                              const std::size_t spilled_child_count,
                                              const genotype_slab::HostGenotypeSlab &spill_slab,
                                              const genotype_slab::SlabGeneration &spill_generation) {
    if ((spilled_child_count == 0) || !genotype_slab::IsValidHostGenotypeSlab(spill_slab) ||
        !genotype_slab::IsValidSlabGeneration(spill_generation) ||
        (spilled_child_count > spill_generation.active_individual_count) ||
        (spill_slab.layout.action_count != buffers.genotype_slab.slab_layout.action_count) ||
        (spill_slab.layout.slot_stride_bytes != buffers.genotype_slab.slab_layout.slot_stride_bytes)) {
        return false;
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    AllocateRestoredChildPrefixKernel<<<(spilled_child_count + kSlabGARuntimeThreadBlockSize - 1) /
                                            kSlabGARuntimeThreadBlockSize,
                                        kSlabGARuntimeThreadBlockSize>>>(
        buffers.genotype_slab.slab_storage, buffers.genotype_slab.slot_states, buffers.genotype_slab.free_slot_stack,
        buffers.genotype_slab.free_slot_count, buffers.genotype_slab.free_slot_lock, buffers.genotype_slab.slab_layout,
        buffers.genotype_slab.next_slot_indices, buffers.genotype_slab.next_fitness,
        buffers.genotype_slab.next_evaluation_counts, buffers.genotype_slab.next_has_fitness, spilled_child_count,
        buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize()) ||
        !KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    genotype_slab::SlabGeneration next_generation{};
    if (!TryDownloadNextGenerationFromDevice(buffers, next_generation)) {
        return false;
    }

    for (std::size_t child_index = 0; child_index < spilled_child_count; ++child_index) {
        const std::uint32_t spill_slot_index = spill_generation.slot_indices[child_index];
        const std::uint32_t device_slot_index = next_generation.slot_indices[child_index];
        if ((spill_slot_index == genotype_slab::kInvalidSlabSlotIndex) ||
            (device_slot_index == genotype_slab::kInvalidSlabSlotIndex)) {
            return false;
        }

        if (!CheckCuda(cudaMemcpy(genotype_slab::SlabSlotBytesAt(buffers.genotype_slab.slab_storage,
                                                                 buffers.genotype_slab.slab_layout, device_slot_index),
                                  genotype_slab::HostSlabSlotBytesAt(spill_slab, spill_slot_index),
                                  spill_slab.layout.slot_stride_bytes, cudaMemcpyHostToDevice))) {
            return false;
        }
    }

    return true;
}

inline bool TryPlanNextGenerationShape(const DeviceSlabGARuntimeBuffers &buffers,
                                       const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
                                       std::size_t &next_action_count_out, std::size_t &next_generation_size_out) {
    next_action_count_out = buffers.genotype_slab.slab_layout.action_count;
    if (!IsValidPendingOutputEmbeddingInjection(pending_output_embedding_injection,
                                                buffers.genotype_slab.slab_layout.action_count)) {
        return false;
    }

    if (pending_output_embedding_injection.enabled) {
        next_action_count_out += pending_output_embedding_injection.injection_count;
    }

    next_generation_size_out = genotype_slab::SlabSlotCountForByteBudget(
        buffers.generation_byte_budget_bytes, next_action_count_out, buffers.max_generation_size);
    next_generation_size_out = FloorRowPreservingPopulationSize(next_generation_size_out, buffers.grid_column_count);
    if (next_generation_size_out > buffers.genotype_slab.current_generation_size) {
        next_generation_size_out = buffers.genotype_slab.current_generation_size;
    }
    return next_generation_size_out > 0;
}

} // namespace

bool TryCreateDeviceSlabGARuntimeBuffers(DeviceSlabGARuntimeBuffers &buffers, const DeviceSlabGARuntimeConfig &config) {
    buffers = {};
    if (!IsValidDeviceSlabGARuntimeConfig(config)) {
        return false;
    }

    const std::size_t slot_count = RuntimeSlabSlotCount(config);
    const std::size_t max_generation_size = RuntimeGenerationCapacity(config);
    if ((slot_count < max_generation_size) ||
        !TryComputeGenerationByteBudgetBytes(config.action_count, max_generation_size,
                                             buffers.generation_byte_budget_bytes)) {
        return false;
    }

    genotype_slab::device::DeviceSlabRuntimeConfig slab_config{};
    slab_config.slot_count = slot_count;
    slab_config.action_count = config.action_count;
    slab_config.max_generation_size = max_generation_size;
    if (!genotype_slab::device::TryCreateDeviceSlabRuntimeBuffers(buffers.genotype_slab, slab_config)) {
        return false;
    }

    buffers.max_generation_size = max_generation_size;
    buffers.host_spillover_byte_budget_bytes = config.host_spillover_byte_budget_bytes;
    buffers.grid_column_count = config.grid_column_count;
    if (!TryMakeRectangularCellularGridShape(max_generation_size / config.grid_column_count, config.grid_column_count,
                                             buffers.epicenter_grid_shape)) {
        DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }
    buffers.active_training_shard_capacity = training_folder::kTrainingWordCatalogCapacity;
    bool ok = true;
    ok &= CheckCuda(cudaMalloc(&buffers.summary, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMalloc(&buffers.status, sizeof(int)));
    ok &= CheckCuda(cudaMalloc(&buffers.active_training_shards, buffers.active_training_shard_capacity *
                                                                    sizeof(training_folder::TrainingDataShardRuntime)));
    ok &= CheckCuda(
        cudaMalloc(&buffers.current_local_training_word_counts, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.fitness_partial_sums,
                               buffers.max_generation_size * kMaxEvaluationBlocksPerIndividual * sizeof(float)));
    if (!ok) {
        DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    ok &= CheckCuda(cudaMemset(buffers.summary, 0, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    ok &= CheckCuda(
        cudaMemset(buffers.active_training_shards, 0,
                   buffers.active_training_shard_capacity * sizeof(training_folder::TrainingDataShardRuntime)));
    ok &= CheckCuda(
        cudaMemset(buffers.current_local_training_word_counts, 0, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.fitness_partial_sums, 0,
                               buffers.max_generation_size * kMaxEvaluationBlocksPerIndividual * sizeof(float)));
    if (!ok) {
        DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    return true;
}

void DestroyDeviceSlabGARuntimeBuffers(DeviceSlabGARuntimeBuffers &buffers) noexcept {
    genotype_slab::device::DestroyDeviceSlabRuntimeBuffers(buffers.genotype_slab);
    cudaFree(buffers.summary);
    cudaFree(buffers.status);
    cudaFree(buffers.active_training_shards);
    cudaFree(buffers.current_local_training_word_counts);
    cudaFree(buffers.fitness_partial_sums);
    buffers = {};
}

bool TryBootstrapRandomCurrentGenerationOnDevice(DeviceSlabGARuntimeBuffers &buffers, const std::size_t generation_size,
                                                 const std::uint32_t generation_seed,
                                                 const std::size_t generation_index,
                                                 const DeviceSlabBootstrapConfig &config) {
    if (!genotype_slab::device::TryBootstrapRandomCurrentGenerationOnDevice(
            buffers.genotype_slab, generation_size, generation_seed, generation_index, config)) {
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemset(buffers.summary, 0, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    ok &= CheckCuda(
        cudaMemset(buffers.current_local_training_word_counts, 0, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.fitness_partial_sums, 0,
                               buffers.max_generation_size * kMaxEvaluationBlocksPerIndividual * sizeof(float)));
    ok &= CheckCuda(
        cudaMemset(buffers.active_training_shards, 0,
                   buffers.active_training_shard_capacity * sizeof(training_folder::TrainingDataShardRuntime)));
    buffers.active_training_shard_count = 0;
    buffers.last_generation_used_host_spillover = false;
    return ok;
}

bool TryDownloadSlabFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                               genotype_slab::HostGenotypeSlab &host_buffer) {
    return genotype_slab::device::TryDownloadSlabFromDevice(buffers.genotype_slab, host_buffer);
}

bool TryDownloadSlabSlotBytesFromDevice(const DeviceSlabGARuntimeBuffers &buffers, const std::uint32_t slot_index,
                                        std::unique_ptr<std::uint8_t[]> &slot_bytes, std::size_t &slot_byte_count) {
    slot_bytes.reset();
    slot_byte_count = 0;

    if (!genotype_slab::IsValidGenotypeSlabLayout(buffers.genotype_slab.slab_layout) ||
        (buffers.genotype_slab.slab_storage == nullptr) || (buffers.genotype_slab.slot_states == nullptr) ||
        (slot_index >= buffers.genotype_slab.slab_layout.slot_count)) {
        return false;
    }

    genotype_slab::SlabSlotState slot_state{};
    if (!CheckCuda(cudaMemcpy(&slot_state, buffers.genotype_slab.slot_states + slot_index, sizeof(slot_state),
                              cudaMemcpyDeviceToHost)) ||
        !slot_state.occupied || (slot_state.liveness_count == 0)) {
        return false;
    }

    const std::size_t slot_stride_bytes = buffers.genotype_slab.slab_layout.slot_stride_bytes;
    slot_bytes.reset(new (std::nothrow) std::uint8_t[slot_stride_bytes]);
    if ((slot_bytes == nullptr) ||
        !CheckCuda(cudaMemcpy(slot_bytes.get(),
                              genotype_slab::SlabSlotBytesAt(buffers.genotype_slab.slab_storage,
                                                             buffers.genotype_slab.slab_layout, slot_index),
                              slot_stride_bytes, cudaMemcpyDeviceToHost))) {
        slot_bytes.reset();
        return false;
    }

    slot_byte_count = slot_stride_bytes;
    return true;
}

bool TryDownloadCurrentGenerationFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                                            genotype_slab::SlabGeneration &generation) {
    return genotype_slab::device::TryDownloadCurrentGenerationFromDevice(buffers.genotype_slab, generation);
}

bool TryDownloadNextGenerationFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                                         genotype_slab::SlabGeneration &generation) {
    return genotype_slab::device::TryDownloadNextGenerationFromDevice(buffers.genotype_slab, generation);
}

bool TryEvaluateCurrentGenerationFitnessOnDevice(DeviceSlabGARuntimeBuffers &buffers,
                                                 const RuntimeWordCounts &runtime_word_counts) {
    if (!IsCurrentGenerationCompatible(buffers) || !ResetDeviceStatus(buffers)) {
        return false;
    }

    CellularGridShape current_grid_shape{};
    if (!TryUploadActiveTrainingShardsForCurrentGeneration(buffers, runtime_word_counts, current_grid_shape)) {
        (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kInvalidTrainingShard);
        return false;
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemset(buffers.genotype_slab.current_fitness, 0,
                               buffers.genotype_slab.max_generation_size * sizeof(float)));
    ok &= CheckCuda(cudaMemset(buffers.genotype_slab.current_evaluation_counts, 0,
                               buffers.genotype_slab.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.genotype_slab.current_has_fitness, 0,
                               buffers.genotype_slab.max_generation_size * sizeof(std::uint8_t)));
    ok &= CheckCuda(
        cudaMemset(buffers.current_local_training_word_counts, 0, buffers.max_generation_size * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.fitness_partial_sums, 0,
                               buffers.max_generation_size * kMaxEvaluationBlocksPerIndividual * sizeof(float)));
    if (!ok) {
        return false;
    }

    const std::size_t blocks_per_individual =
        SelectEvaluationBlocksPerIndividual(buffers.genotype_slab.current_generation_size, runtime_word_counts);
    const std::size_t selectable_action_count =
        (buffers.genotype_slab.slab_layout.action_count < runtime_word_counts.action_space_word_count)
            ? buffers.genotype_slab.slab_layout.action_count
            : runtime_word_counts.action_space_word_count;
    dim3 evaluation_grid{};
    evaluation_grid.x = static_cast<unsigned int>(buffers.genotype_slab.current_generation_size);
    evaluation_grid.y = static_cast<unsigned int>(blocks_per_individual);
    evaluation_grid.z = 1;

    if (selectable_action_count <= kTensorActionScoreActionCountThreshold) {
        EvaluateSlabGenerationFitnessByEpisodeKernel<<<evaluation_grid, kEvaluationThreadsPerBlock>>>(
            buffers.genotype_slab.slab_storage, buffers.genotype_slab.slot_states, buffers.genotype_slab.slab_layout,
            buffers.genotype_slab.current_slot_indices, buffers.genotype_slab.current_generation_size,
            runtime_word_counts, buffers.active_training_shards, buffers.active_training_shard_count,
            current_grid_shape, blocks_per_individual, buffers.fitness_partial_sums,
            buffers.current_local_training_word_counts, buffers.status);
    } else {
        EvaluateSlabGenerationFitnessByEpisodeWarpKernel<<<evaluation_grid, kEvaluationThreadsPerBlock>>>(
            buffers.genotype_slab.slab_storage, buffers.genotype_slab.slot_states, buffers.genotype_slab.slab_layout,
            buffers.genotype_slab.current_slot_indices, buffers.genotype_slab.current_generation_size,
            runtime_word_counts, buffers.active_training_shards, buffers.active_training_shard_count,
            current_grid_shape, blocks_per_individual, buffers.fitness_partial_sums,
            buffers.current_local_training_word_counts, buffers.status);
    }
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize()) ||
        !KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    const std::size_t reduction_block_count =
        CeilDivide(buffers.genotype_slab.current_generation_size, kSlabGARuntimeThreadBlockSize);
    FinalizeSlabGenerationFitnessKernel<<<reduction_block_count, kSlabGARuntimeThreadBlockSize>>>(
        buffers.fitness_partial_sums, blocks_per_individual, buffers.genotype_slab.current_generation_size,
        buffers.current_local_training_word_counts, buffers.genotype_slab.current_fitness,
        buffers.genotype_slab.current_evaluation_counts, buffers.genotype_slab.current_has_fitness, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize()) ||
        !KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    SummarizeSlabGenerationKernel<<<1, 1>>>(
        buffers.genotype_slab.current_fitness, buffers.genotype_slab.current_has_fitness,
        buffers.genotype_slab.current_slot_indices, buffers.genotype_slab.current_generation_index,
        buffers.genotype_slab.current_generation_size, buffers.genotype_slab.slab_layout.action_count, buffers.summary,
        buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    return KernelCompletedSuccessfully(buffers);
}

bool TryReadPopulationFitnessSummaryFromDevice(const DeviceSlabGARuntimeBuffers &buffers,
                                               PopulationFitnessSummary &summary) {
    return CheckCuda(cudaMemcpy(&summary, buffers.summary, sizeof(PopulationFitnessSummary), cudaMemcpyDeviceToHost));
}

namespace {

constexpr std::uint64_t kCheckpointMagic = 0x4b4350474147524eULL; // NRGAGPCK, little-endian marker.

inline void MixCheckpointBytes(std::uint64_t &hash, const void *data, const std::size_t byte_count) noexcept {
    const auto *bytes = static_cast<const std::uint8_t *>(data);
    for (std::size_t byte_index = 0; byte_index < byte_count; ++byte_index) {
        hash ^= bytes[byte_index];
        hash *= 1099511628211ULL;
    }
}

template <typename Value> void MixCheckpointValue(std::uint64_t &hash, const Value &value) noexcept {
    MixCheckpointBytes(hash, &value, sizeof(Value));
}

std::uint64_t ComputeRuntimeCheckpointChecksum(const RuntimeCheckpoint &checkpoint) {
    std::uint64_t hash = 1469598103934665603ULL;
    MixCheckpointValue(hash, checkpoint.schema_version);
    MixCheckpointValue(hash, checkpoint.genome_layout_version);
    const auto resume_phase = static_cast<std::uint32_t>(checkpoint.resume_phase);
    MixCheckpointValue(hash, resume_phase);
    MixCheckpointValue(hash, checkpoint.training_data_identity_hash);
    MixCheckpointValue(hash, checkpoint.generation_seed);
    MixCheckpointBytes(hash, &checkpoint.runtime_word_counts, sizeof(checkpoint.runtime_word_counts));
    MixCheckpointBytes(hash, &checkpoint.assembly_config, sizeof(checkpoint.assembly_config));
    MixCheckpointBytes(hash, &checkpoint.pending_output_embedding_injection,
                       sizeof(checkpoint.pending_output_embedding_injection));
    MixCheckpointBytes(hash, &checkpoint.runtime_config, sizeof(checkpoint.runtime_config));
    MixCheckpointBytes(hash, &checkpoint.slab_layout, sizeof(checkpoint.slab_layout));
    MixCheckpointBytes(hash, &checkpoint.current_grid_shape, sizeof(checkpoint.current_grid_shape));
    MixCheckpointBytes(hash, &checkpoint.next_grid_shape, sizeof(checkpoint.next_grid_shape));
    MixCheckpointBytes(hash, &checkpoint.epicenter_grid_shape, sizeof(checkpoint.epicenter_grid_shape));
    MixCheckpointValue(hash, checkpoint.current_generation.generation_index);
    MixCheckpointValue(hash, checkpoint.current_generation.active_individual_count);
    for (std::size_t individual_index = 0; individual_index < checkpoint.current_generation.active_individual_count;
         ++individual_index) {
        MixCheckpointValue(hash, checkpoint.current_generation.slot_indices[individual_index]);
        MixCheckpointValue(hash, checkpoint.current_generation.fitness[individual_index]);
        MixCheckpointValue(hash, checkpoint.current_generation.evaluation_counts[individual_index]);
        MixCheckpointValue(hash, checkpoint.current_generation.has_fitness[individual_index]);
    }
    MixCheckpointValue(hash, checkpoint.assembly_plan.child_count);
    for (std::size_t child_index = 0; child_index < checkpoint.assembly_plan.child_count; ++child_index) {
        MixCheckpointValue(hash, checkpoint.assembly_plan.parent_pairs[child_index]);
    }
    const std::uint64_t live_count = checkpoint.live_genotypes.size();
    MixCheckpointValue(hash, live_count);
    for (const RuntimeCheckpointGenotypeRecord &record : checkpoint.live_genotypes) {
        MixCheckpointValue(hash, record.organism_index);
        const std::uint64_t genome_byte_count = record.genome_bytes.size();
        MixCheckpointValue(hash, genome_byte_count);
        if (!record.genome_bytes.empty()) {
            MixCheckpointBytes(hash, record.genome_bytes.data(), record.genome_bytes.size());
        }
    }
    return hash;
}

bool TryDownloadAssemblyPlanFromDevice(const DeviceSlabGARuntimeBuffers &buffers, const std::size_t child_count,
                                       genotype_slab::SlabAssemblyPlan &plan) {
    if (!genotype_slab::TryCreateSlabAssemblyPlan(plan, child_count)) {
        return false;
    }

    return CheckCuda(cudaMemcpy(plan.parent_pairs.get(), buffers.genotype_slab.assembly_parent_pairs,
                                child_count * sizeof(genotype_slab::SlabParentPair), cudaMemcpyDeviceToHost));
}

bool TryDownloadBoundaryCheckpointPayload(const DeviceSlabGARuntimeBuffers &buffers,
                                          const std::uint32_t generation_seed,
                                          const RuntimeWordCounts &runtime_word_counts,
                                          const GenerationAssemblyConfig &config,
                                          const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
                                          const DeviceSlabGARuntimeConfig &runtime_config,
                                          const training_folder::TrainingWordCatalog *host_training_word_catalog,
                                          RuntimeCheckpoint &checkpoint_out, const bool verbose) {
    checkpoint_out = {};
    const auto checkpoint_start_time = ProgressClock::now();
    const auto log_verbose_line = [&](const std::string &message) {
        if (verbose) {
            PrintTimestampedProgressLine(std::cout, message);
        }
    };
    const auto log_verbose_duration = [&](const std::string &message, const ProgressClock::time_point start_time) {
        if (verbose) {
            PrintTimestampedProgressDuration(std::cout, message, start_time);
        }
    };

    checkpoint_out.generation_seed = generation_seed;
    std::uint64_t training_identity_hash = 1469598103934665603ULL;
    MixCheckpointBytes(training_identity_hash, &runtime_word_counts, sizeof(runtime_word_counts));
    if (host_training_word_catalog != nullptr) {
        MixCheckpointValue(training_identity_hash, host_training_word_catalog->word_count);
        for (std::size_t word_index = 0; word_index < host_training_word_catalog->word_count; ++word_index) {
            MixCheckpointBytes(training_identity_hash, &host_training_word_catalog->words[word_index],
                               sizeof(host_training_word_catalog->words[word_index]));
        }
    }
    checkpoint_out.training_data_identity_hash = training_identity_hash;
    checkpoint_out.runtime_word_counts = runtime_word_counts;
    checkpoint_out.assembly_config = config;
    checkpoint_out.pending_output_embedding_injection = pending_output_embedding_injection;
    checkpoint_out.runtime_config = runtime_config;
    checkpoint_out.slab_layout = buffers.genotype_slab.slab_layout;
    checkpoint_out.epicenter_grid_shape = buffers.epicenter_grid_shape;

    const auto metadata_download_start_time = ProgressClock::now();
    log_verbose_line("Generation " + std::to_string(buffers.genotype_slab.current_generation_index) +
                     ": downloading checkpoint metadata and assembly plan");
    if (!TryMakeCellularGridShapeForColumnCount(buffers.genotype_slab.current_generation_size,
                                                buffers.grid_column_count, checkpoint_out.current_grid_shape) ||
        !TryMakeCellularGridShapeForColumnCount(buffers.genotype_slab.next_generation_size, buffers.grid_column_count,
                                                checkpoint_out.next_grid_shape) ||
        !genotype_slab::device::TryDownloadCurrentGenerationFromDevice(buffers.genotype_slab,
                                                                       checkpoint_out.current_generation) ||
        !TryDownloadAssemblyPlanFromDevice(buffers, buffers.genotype_slab.planned_child_count,
                                           checkpoint_out.assembly_plan)) {
        checkpoint_out = {};
        return false;
    }
    log_verbose_duration("Generation " + std::to_string(checkpoint_out.current_generation.generation_index) +
                             ": checkpoint metadata and assembly plan downloaded",
                         metadata_download_start_time);

    const auto genotype_copy_start_time = ProgressClock::now();
    if (verbose) {
        std::ostringstream stream;
        stream << "Generation " << checkpoint_out.current_generation.generation_index
               << ": copying live checkpoint genotypes to host"
               << " (population=" << checkpoint_out.current_generation.active_individual_count
               << ", slot_stride_bytes=" << checkpoint_out.slab_layout.slot_stride_bytes << ')';
        log_verbose_line(stream.str());
    }
    for (std::size_t individual_index = 0; individual_index < checkpoint_out.current_generation.active_individual_count;
         ++individual_index) {
        const std::uint32_t slot_index = checkpoint_out.current_generation.slot_indices[individual_index];
        if (slot_index == genotype_slab::kInvalidSlabSlotIndex) {
            continue;
        }

        RuntimeCheckpointGenotypeRecord record{};
        record.organism_index = static_cast<std::uint32_t>(individual_index);
        record.genome_bytes.resize(checkpoint_out.slab_layout.slot_stride_bytes);
        if (!CheckCuda(cudaMemcpy(record.genome_bytes.data(),
                                  genotype_slab::SlabSlotBytesAt(buffers.genotype_slab.slab_storage,
                                                                 checkpoint_out.slab_layout, slot_index),
                                  checkpoint_out.slab_layout.slot_stride_bytes, cudaMemcpyDeviceToHost))) {
            checkpoint_out = {};
            return false;
        }

        checkpoint_out.live_genotypes.push_back(std::move(record));
    }
    if (verbose) {
        std::ostringstream stream;
        stream << "Generation " << checkpoint_out.current_generation.generation_index
               << ": copied live checkpoint genotypes to host"
               << " (live_genotypes=" << checkpoint_out.live_genotypes.size() << ')';
        PrintTimestampedProgressDuration(std::cout, stream.str(), genotype_copy_start_time);
    }

    const auto checksum_start_time = ProgressClock::now();
    log_verbose_line("Generation " + std::to_string(checkpoint_out.current_generation.generation_index) +
                     ": checksumming runtime checkpoint");
    checkpoint_out.checksum = ComputeRuntimeCheckpointChecksum(checkpoint_out);
    log_verbose_duration("Generation " + std::to_string(checkpoint_out.current_generation.generation_index) +
                             ": runtime checkpoint checksummed",
                         checksum_start_time);
    log_verbose_duration("Generation " + std::to_string(checkpoint_out.current_generation.generation_index) +
                             ": checkpoint payload ready on host",
                         checkpoint_start_time);
    return true;
}

bool TryPreparePrebreedingBoundaryOnDevice(DeviceSlabGARuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                           const RuntimeWordCounts &runtime_word_counts,
                                           const GenerationAssemblyConfig &config,
                                           const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
                                           std::size_t &parent_action_count_out,
                                           DeviceSlabGARuntimeConfig &checkpoint_runtime_config_out,
                                           const bool verbose,
                                           const PostFitnessEvaluationCallback &post_fitness_evaluation_callback) {
    parent_action_count_out = 0;
    checkpoint_runtime_config_out = {};
    buffers.last_generation_used_host_spillover = false;
    const std::size_t next_generation_index = buffers.genotype_slab.current_generation_index + 1;
    const auto log_verbose_line = [&](const std::string &message) {
        if (verbose) {
            PrintTimestampedProgressLine(std::cout, message);
        }
    };
    const auto log_verbose_duration = [&](const std::string &message, const ProgressClock::time_point start_time) {
        if (verbose) {
            PrintTimestampedProgressDuration(std::cout, message, start_time);
        }
    };

    if (verbose) {
        std::ostringstream stream;
        stream << "Generation " << next_generation_index << ": starting advancement from generation "
               << buffers.genotype_slab.current_generation_index
               << " (population=" << buffers.genotype_slab.current_generation_size
               << ", action_count=" << buffers.genotype_slab.slab_layout.action_count
               << ", slot_stride_bytes=" << buffers.genotype_slab.slab_layout.slot_stride_bytes << ')';
        log_verbose_line(stream.str());
    }

    const auto evaluation_start_time = ProgressClock::now();
    log_verbose_line("Generation " + std::to_string(next_generation_index) + ": evaluating current generation fitness");
    if (!IsValidGenerationAssemblyConfig(config) || !IsCurrentGenerationCompatible(buffers) ||
        !TryEvaluateCurrentGenerationFitnessOnDevice(buffers, runtime_word_counts) || !ResetDeviceStatus(buffers)) {
        if (!IsValidGenerationAssemblyConfig(config)) {
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig);
        }
        return false;
    }
    log_verbose_duration("Generation " + std::to_string(next_generation_index) +
                             ": current generation fitness evaluation finished",
                         evaluation_start_time);
    if (post_fitness_evaluation_callback && !post_fitness_evaluation_callback(buffers)) {
        return false;
    }

    std::size_t next_action_count = buffers.genotype_slab.slab_layout.action_count;
    std::size_t next_generation_size = buffers.genotype_slab.current_generation_size;
    const auto planning_start_time = ProgressClock::now();
    log_verbose_line("Generation " + std::to_string(next_generation_index) + ": planning next generation shape");
    if (!TryPlanNextGenerationShape(buffers, pending_output_embedding_injection, next_action_count,
                                    next_generation_size)) {
        (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig);
        return false;
    }
    if (verbose) {
        std::ostringstream stream;
        stream << "Generation " << next_generation_index << ": next generation shape planned"
               << " (population=" << next_generation_size << ", action_count=" << next_action_count << ')';
        PrintTimestampedProgressDuration(std::cout, stream.str(), planning_start_time);
    }

    const std::size_t block_count =
        (next_generation_size + kSlabGARuntimeThreadBlockSize - 1) / kSlabGARuntimeThreadBlockSize;
    const std::uint32_t planning_seed = generation_seed ^ 0xA341316CU;
    const auto assembly_plan_start_time = ProgressClock::now();
    log_verbose_line("Generation " + std::to_string(next_generation_index) + ": building next-generation parent plan");
    BuildSlabAssemblyPlanKernel<<<block_count, kSlabGARuntimeThreadBlockSize>>>(
        buffers.genotype_slab.current_fitness, buffers.genotype_slab.current_has_fitness,
        buffers.genotype_slab.current_generation_size, next_generation_size, buffers.grid_column_count,
        buffers.genotype_slab.current_generation_index, config.parent_selection, planning_seed,
        buffers.genotype_slab.assembly_parent_pairs, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize()) ||
        !KernelCompletedSuccessfully(buffers)) {
        return false;
    }
    log_verbose_duration("Generation " + std::to_string(next_generation_index) +
                             ": next-generation parent plan finished",
                         assembly_plan_start_time);

    buffers.genotype_slab.planned_child_count = next_generation_size;

    const std::size_t parent_action_count = buffers.genotype_slab.slab_layout.action_count;
    if (pending_output_embedding_injection.enabled &&
        !genotype_slab::device::TryPrepareSlabForExpandedActionCountOnDevice(buffers.genotype_slab, next_action_count,
                                                                             verbose)) {
        genotype_slab::device::DeviceSlabRuntimeStatusCode slab_status =
            genotype_slab::device::DeviceSlabRuntimeStatusCode::kCudaFailure;
        if (genotype_slab::device::TryReadDeviceSlabRuntimeStatus(buffers.genotype_slab, slab_status)) {
            (void)WriteDeviceStatus(buffers, MapSlabRuntimeStatus(slab_status));
        } else {
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
        }
        return false;
    }

    genotype_slab::device::SlabDeviceAssemblyConfig slab_assembly_config{};
    slab_assembly_config.breeding = config.breeding;
    slab_assembly_config.mutation = config.mutation;
    slab_assembly_config.parent_action_count = parent_action_count;
    slab_assembly_config.pending_output_embedding_injection = pending_output_embedding_injection;

    const auto initialize_assembly_start_time = ProgressClock::now();
    log_verbose_line("Generation " + std::to_string(next_generation_index) +
                     ": initializing next-generation assembly state");
    if (!genotype_slab::device::TryInitializeNextGenerationAssemblyOnDevice(buffers.genotype_slab,
                                                                            slab_assembly_config)) {
        genotype_slab::device::DeviceSlabRuntimeStatusCode slab_status =
            genotype_slab::device::DeviceSlabRuntimeStatusCode::kCudaFailure;
        if (genotype_slab::device::TryReadDeviceSlabRuntimeStatus(buffers.genotype_slab, slab_status)) {
            (void)WriteDeviceStatus(buffers, MapSlabRuntimeStatus(slab_status));
        } else {
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
        }
        return false;
    }
    log_verbose_duration("Generation " + std::to_string(next_generation_index) +
                             ": next-generation assembly state initialized",
                         initialize_assembly_start_time);

    checkpoint_runtime_config_out.genotype_slab_byte_budget_bytes = buffers.genotype_slab.slab_layout.slab_bytes;
    checkpoint_runtime_config_out.generation_byte_budget_bytes = buffers.generation_byte_budget_bytes;
    checkpoint_runtime_config_out.host_spillover_byte_budget_bytes = buffers.host_spillover_byte_budget_bytes;
    checkpoint_runtime_config_out.action_count = buffers.genotype_slab.slab_layout.action_count;
    checkpoint_runtime_config_out.population_size_ceiling = buffers.max_generation_size;
    checkpoint_runtime_config_out.grid_column_count = buffers.grid_column_count;
    parent_action_count_out = parent_action_count;
    return true;
}

bool TryAssemblePreparedGenerationOnDevice(DeviceSlabGARuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                           const genotype_slab::device::SlabDeviceAssemblyConfig &slab_assembly_config,
                                           const bool verbose) {
    const std::size_t next_generation_index = buffers.genotype_slab.next_generation_index;
    const auto log_verbose_line = [&](const std::string &message) {
        if (verbose) {
            PrintTimestampedProgressLine(std::cout, message);
        }
    };
    const auto log_verbose_duration = [&](const std::string &message, const ProgressClock::time_point start_time) {
        if (verbose) {
            PrintTimestampedProgressDuration(std::cout, message, start_time);
        }
    };
    const auto overall_start_time = ProgressClock::now();

    bool used_host_spillover = false;
    std::size_t spilled_child_count = 0;
    genotype_slab::HostGenotypeSlab spill_slab{};
    genotype_slab::SlabGeneration spill_generation{};
    std::size_t child_offset = 0;
    std::size_t assembly_batch_index = 0;

    while (child_offset < buffers.genotype_slab.next_generation_size) {
        ++assembly_batch_index;
        const auto assembly_batch_start_time = ProgressClock::now();
        if (verbose) {
            std::ostringstream stream;
            stream << "Generation " << next_generation_index << ": assembly batch " << assembly_batch_index
                   << " starting at child_offset=" << child_offset;
            log_verbose_line(stream.str());
        }

        if (genotype_slab::device::TryContinueNextGenerationAssemblyOnDevice(buffers.genotype_slab, generation_seed,
                                                                             slab_assembly_config, child_offset)) {
            child_offset = buffers.genotype_slab.next_generation_size;
            log_verbose_duration("Generation " + std::to_string(next_generation_index) + ": assembly batch " +
                                     std::to_string(assembly_batch_index) + " finished",
                                 assembly_batch_start_time);
            break;
        }

        genotype_slab::device::DeviceSlabRuntimeStatusCode slab_status =
            genotype_slab::device::DeviceSlabRuntimeStatusCode::kCudaFailure;
        if (!genotype_slab::device::TryReadDeviceSlabRuntimeStatus(buffers.genotype_slab, slab_status)) {
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
            return false;
        }

        if (slab_status != genotype_slab::device::DeviceSlabRuntimeStatusCode::kSlabFull) {
            (void)WriteDeviceStatus(buffers, MapSlabRuntimeStatus(slab_status));
            return false;
        }

        if (verbose) {
            std::ostringstream stream;
            stream << "Generation " << next_generation_index << ": assembly batch " << assembly_batch_index
                   << " hit slab-full after child_offset=" << child_offset;
            PrintTimestampedProgressDuration(std::cout, stream.str(), assembly_batch_start_time);
        }

        if (used_host_spillover || (buffers.host_spillover_byte_budget_bytes == 0)) {
            (void)genotype_slab::device::TryCleanupFailedAssemblyOnDevice(buffers.genotype_slab);
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kSlabFull);
            return false;
        }

        std::size_t assembled_child_count = 0;
        const auto assembled_prefix_count_start_time = ProgressClock::now();
        log_verbose_line("Generation " + std::to_string(next_generation_index) +
                         ": counting assembled child prefix before host spillover");
        if (!TryCountAssembledChildPrefix(buffers, assembled_child_count) || (assembled_child_count == 0)) {
            (void)genotype_slab::device::TryCleanupFailedAssemblyOnDevice(buffers.genotype_slab);
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kSlabFull);
            return false;
        }
        if (verbose) {
            std::ostringstream stream;
            stream << "Generation " << next_generation_index
                   << ": assembled child prefix counted (assembled_child_count=" << assembled_child_count << ')';
            PrintTimestampedProgressDuration(std::cout, stream.str(), assembled_prefix_count_start_time);
        }

        const auto spill_storage_start_time = ProgressClock::now();
        log_verbose_line("Generation " + std::to_string(next_generation_index) +
                         ": creating host spillover storage for assembled children");
        if (!TryCreateHostSpillStorage(buffers, buffers.genotype_slab.slab_layout.action_count,
                                       buffers.genotype_slab.next_generation_size,
                                       buffers.genotype_slab.next_generation_index, spill_slab, spill_generation)) {
            (void)genotype_slab::device::TryCleanupFailedAssemblyOnDevice(buffers.genotype_slab);
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
            return false;
        }
        if (verbose) {
            std::ostringstream stream;
            stream << "Generation " << next_generation_index << ": host spillover storage ready"
                   << " (spill_slot_count=" << spill_slab.layout.slot_count << ')';
            PrintTimestampedProgressDuration(std::cout, stream.str(), spill_storage_start_time);
        }

        if (assembled_child_count > spill_slab.layout.slot_count) {
            (void)genotype_slab::device::TryCleanupFailedAssemblyOnDevice(buffers.genotype_slab);
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
            return false;
        }

        const auto spill_copy_start_time = ProgressClock::now();
        if (verbose) {
            std::ostringstream stream;
            stream << "Generation " << next_generation_index << ": spilling " << assembled_child_count
                   << " assembled children to host";
            log_verbose_line(stream.str());
        }
        if (!TrySpillAssembledChildrenToHost(buffers, assembled_child_count, spill_slab, spill_generation)) {
            (void)genotype_slab::device::TryCleanupFailedAssemblyOnDevice(buffers.genotype_slab);
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
            return false;
        }
        if (verbose) {
            std::ostringstream stream;
            stream << "Generation " << next_generation_index << ": spilled " << assembled_child_count
                   << " assembled children to host";
            PrintTimestampedProgressDuration(std::cout, stream.str(), spill_copy_start_time);
        }

        used_host_spillover = true;
        spilled_child_count = assembled_child_count;
        child_offset = assembled_child_count;
    }

    if (used_host_spillover) {
        const auto restore_spill_start_time = ProgressClock::now();
        if (verbose) {
            std::ostringstream stream;
            stream << "Generation " << next_generation_index << ": restoring " << spilled_child_count
                   << " spilled children back to device";
            log_verbose_line(stream.str());
        }
        if (!TryRestoreSpilledChildrenToDevice(buffers, spilled_child_count, spill_slab, spill_generation)) {
            (void)genotype_slab::device::TryCleanupFailedAssemblyOnDevice(buffers.genotype_slab);
            (void)WriteDeviceStatus(buffers, DeviceSlabGARuntimeStatusCode::kCudaFailure);
            return false;
        }
        if (verbose) {
            std::ostringstream stream;
            stream << "Generation " << next_generation_index << ": restored " << spilled_child_count
                   << " spilled children back to device";
            PrintTimestampedProgressDuration(std::cout, stream.str(), restore_spill_start_time);
        }
    }

    buffers.last_generation_used_host_spillover = used_host_spillover;
    if (used_host_spillover) {
        buffers.host_spillover_count += 1;
    }

    SwapDeviceSlabGenerationBuffers(buffers);
    log_verbose_duration("Generation " + std::to_string(next_generation_index) + ": advancement finished",
                         overall_start_time);
    return true;
}

} // namespace

bool TryAdvanceGenerationOnDevice(DeviceSlabGARuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                  const RuntimeWordCounts &runtime_word_counts, const GenerationAssemblyConfig &config,
                                  const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
                                  const training_folder::TrainingWordCatalog *host_training_word_catalog,
                                  const bool verbose,
                                  const PostFitnessEvaluationCallback &post_fitness_evaluation_callback) {
    (void)host_training_word_catalog;
    std::size_t parent_action_count = 0;
    DeviceSlabGARuntimeConfig checkpoint_runtime_config{};
    if (!TryPreparePrebreedingBoundaryOnDevice(buffers, generation_seed, runtime_word_counts, config,
                                               pending_output_embedding_injection, parent_action_count,
                                               checkpoint_runtime_config, verbose,
                                               post_fitness_evaluation_callback)) {
        return false;
    }

    genotype_slab::device::SlabDeviceAssemblyConfig slab_assembly_config{};
    slab_assembly_config.breeding = config.breeding;
    slab_assembly_config.mutation = config.mutation;
    slab_assembly_config.parent_action_count = parent_action_count;
    slab_assembly_config.pending_output_embedding_injection = pending_output_embedding_injection;
    return TryAssemblePreparedGenerationOnDevice(buffers, generation_seed, slab_assembly_config, verbose);
}

bool TryCreatePrebreedingCheckpointOnDevice(DeviceSlabGARuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                            const RuntimeWordCounts &runtime_word_counts,
                                            const GenerationAssemblyConfig &config,
                                            const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
                                            RuntimeCheckpoint &checkpoint_out,
                                            const training_folder::TrainingWordCatalog *host_training_word_catalog,
                                            const bool verbose,
                                            const PostFitnessEvaluationCallback &post_fitness_evaluation_callback) {
    std::size_t parent_action_count = 0;
    DeviceSlabGARuntimeConfig checkpoint_runtime_config{};
    if (!TryPreparePrebreedingBoundaryOnDevice(buffers, generation_seed, runtime_word_counts, config,
                                               pending_output_embedding_injection, parent_action_count,
                                               checkpoint_runtime_config, verbose,
                                               post_fitness_evaluation_callback)) {
        checkpoint_out = {};
        return false;
    }

    return TryDownloadBoundaryCheckpointPayload(buffers, generation_seed, runtime_word_counts, config,
                                                pending_output_embedding_injection, checkpoint_runtime_config,
                                                host_training_word_catalog, checkpoint_out, verbose);
}

bool TryRestorePrebreedingCheckpointToDevice(const RuntimeCheckpoint &checkpoint, DeviceSlabGARuntimeBuffers &buffers) {
    if ((checkpoint.schema_version != kRuntimeCheckpointSchemaVersion) ||
        (checkpoint.genome_layout_version != kRuntimeCheckpointGenomeLayoutVersion) ||
        (checkpoint.resume_phase != RuntimeCheckpointResumePhase::kPreRecombinationPreMutation) ||
        (checkpoint.checksum != ComputeRuntimeCheckpointChecksum(checkpoint)) ||
        !genotype_slab::IsValidGenotypeSlabLayout(checkpoint.slab_layout) ||
        !genotype_slab::IsValidSlabGeneration(checkpoint.current_generation) ||
        !genotype_slab::IsValidSlabAssemblyPlan(checkpoint.assembly_plan)) {
        return false;
    }

    const std::size_t restore_generation_capacity =
        std::max(checkpoint.current_generation.active_individual_count, checkpoint.assembly_plan.child_count);
    genotype_slab::device::DeviceSlabRuntimeConfig slab_config{};
    slab_config.slot_count = checkpoint.slab_layout.slot_count;
    slab_config.action_count = checkpoint.slab_layout.action_count;
    slab_config.max_generation_size = restore_generation_capacity;
    if (!genotype_slab::device::TryCreateDeviceSlabRuntimeBuffers(buffers.genotype_slab, slab_config)) {
        DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }
    buffers.max_generation_size = restore_generation_capacity;
    buffers.generation_byte_budget_bytes = checkpoint.runtime_config.generation_byte_budget_bytes;
    buffers.host_spillover_byte_budget_bytes = checkpoint.runtime_config.host_spillover_byte_budget_bytes;
    buffers.grid_column_count = checkpoint.runtime_config.grid_column_count;
    buffers.epicenter_grid_shape = checkpoint.epicenter_grid_shape;
    buffers.active_training_shard_capacity = training_folder::kTrainingWordCatalogCapacity;
    bool allocate_ok = true;
    allocate_ok &= CheckCuda(cudaMalloc(&buffers.summary, sizeof(PopulationFitnessSummary)));
    allocate_ok &= CheckCuda(cudaMalloc(&buffers.status, sizeof(int)));
    allocate_ok &=
        CheckCuda(cudaMalloc(&buffers.active_training_shards, buffers.active_training_shard_capacity *
                                                                  sizeof(training_folder::TrainingDataShardRuntime)));
    allocate_ok &= CheckCuda(
        cudaMalloc(&buffers.current_local_training_word_counts, buffers.max_generation_size * sizeof(std::uint32_t)));
    allocate_ok &=
        CheckCuda(cudaMalloc(&buffers.fitness_partial_sums,
                             buffers.max_generation_size * kMaxEvaluationBlocksPerIndividual * sizeof(float)));
    if (!allocate_ok || !CheckCuda(cudaMemset(buffers.summary, 0, sizeof(PopulationFitnessSummary))) ||
        !CheckCuda(cudaMemset(buffers.status, 0, sizeof(int))) ||
        !CheckCuda(
            cudaMemset(buffers.active_training_shards, 0,
                       buffers.active_training_shard_capacity * sizeof(training_folder::TrainingDataShardRuntime))) ||
        !CheckCuda(cudaMemset(buffers.current_local_training_word_counts, 0,
                              buffers.max_generation_size * sizeof(std::uint32_t))) ||
        !CheckCuda(cudaMemset(buffers.fitness_partial_sums, 0,
                              buffers.max_generation_size * kMaxEvaluationBlocksPerIndividual * sizeof(float)))) {
        DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    genotype_slab::GenotypeSlabLayout compact_restore_layout = checkpoint.slab_layout;
    compact_restore_layout.slab_bytes = compact_restore_layout.slot_count * compact_restore_layout.slot_stride_bytes;
    genotype_slab::HostGenotypeSlab host_slab{};
    if (!genotype_slab::TryCreateHostGenotypeSlab(host_slab, compact_restore_layout)) {
        DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    genotype_slab::SlabGeneration restored_generation{};
    if (!genotype_slab::TryCreateSlabGeneration(restored_generation,
                                                checkpoint.current_generation.active_individual_count,
                                                checkpoint.current_generation.generation_index)) {
        DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }
    for (std::size_t individual_index = 0; individual_index < restored_generation.active_individual_count;
         ++individual_index) {
        restored_generation.fitness[individual_index] = checkpoint.current_generation.fitness[individual_index];
        restored_generation.evaluation_counts[individual_index] =
            checkpoint.current_generation.evaluation_counts[individual_index];
        restored_generation.has_fitness[individual_index] = checkpoint.current_generation.has_fitness[individual_index];
    }

    std::vector<bool> restored_indices(restored_generation.active_individual_count, false);
    for (const RuntimeCheckpointGenotypeRecord &record : checkpoint.live_genotypes) {
        if ((record.organism_index >= restored_generation.active_individual_count) ||
            restored_indices[record.organism_index] ||
            (record.genome_bytes.size() != checkpoint.slab_layout.slot_stride_bytes) ||
            (checkpoint.current_generation.slot_indices[record.organism_index] ==
             genotype_slab::kInvalidSlabSlotIndex)) {
            DestroyDeviceSlabGARuntimeBuffers(buffers);
            return false;
        }

        std::uint32_t slot_index = genotype_slab::kInvalidSlabSlotIndex;
        if (!genotype_slab::TryAllocateSlabSlot(host_slab, slot_index) ||
            !genotype_slab::TryCopyGenomeBytesIntoSlabSlot(host_slab, slot_index, record.genome_bytes.data(),
                                                           record.genome_bytes.size()) ||
            !genotype_slab::TrySetSlabGenerationSlot(restored_generation, record.organism_index, slot_index)) {
            DestroyDeviceSlabGARuntimeBuffers(buffers);
            return false;
        }
        restored_indices[record.organism_index] = true;
    }

    for (std::size_t child_index = 0; child_index < checkpoint.assembly_plan.child_count; ++child_index) {
        const genotype_slab::SlabParentPair &parent_pair = checkpoint.assembly_plan.parent_pairs[child_index];
        if ((parent_pair.first_parent_index >= restored_generation.active_individual_count) ||
            (parent_pair.second_parent_index >= restored_generation.active_individual_count) ||
            !restored_indices[parent_pair.first_parent_index] || !restored_indices[parent_pair.second_parent_index]) {
            DestroyDeviceSlabGARuntimeBuffers(buffers);
            return false;
        }
    }

    if (!genotype_slab::device::TryUploadSlabToDevice(host_slab, buffers.genotype_slab)) {
        DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }
    if (!genotype_slab::device::TryUploadCurrentGenerationToDevice(restored_generation, buffers.genotype_slab)) {
        DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }
    if (!genotype_slab::device::TryUploadAssemblyPlanToDevice(checkpoint.assembly_plan, buffers.genotype_slab)) {
        DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    genotype_slab::device::SlabDeviceAssemblyConfig slab_assembly_config{};
    slab_assembly_config.breeding = checkpoint.assembly_config.breeding;
    slab_assembly_config.mutation = checkpoint.assembly_config.mutation;
    slab_assembly_config.parent_action_count =
        checkpoint.pending_output_embedding_injection.enabled
            ? checkpoint.pending_output_embedding_injection.first_catalog_word_index
            : checkpoint.slab_layout.action_count;
    slab_assembly_config.pending_output_embedding_injection = checkpoint.pending_output_embedding_injection;
    if (!genotype_slab::device::TryInitializeNextGenerationAssemblyOnDevice(buffers.genotype_slab,
                                                                            slab_assembly_config)) {
        DestroyDeviceSlabGARuntimeBuffers(buffers);
        return false;
    }

    return true;
}

bool TryResumeGenerationFromCheckpointOnDevice(DeviceSlabGARuntimeBuffers &buffers, const RuntimeCheckpoint &checkpoint,
                                               const bool verbose) {
    genotype_slab::device::SlabDeviceAssemblyConfig slab_assembly_config{};
    slab_assembly_config.breeding = checkpoint.assembly_config.breeding;
    slab_assembly_config.mutation = checkpoint.assembly_config.mutation;
    slab_assembly_config.parent_action_count =
        checkpoint.pending_output_embedding_injection.enabled
            ? checkpoint.pending_output_embedding_injection.first_catalog_word_index
            : checkpoint.slab_layout.action_count;
    slab_assembly_config.pending_output_embedding_injection = checkpoint.pending_output_embedding_injection;
    return TryAssemblePreparedGenerationOnDevice(buffers, checkpoint.generation_seed, slab_assembly_config, verbose);
}

namespace {

template <typename Value> bool WriteBinaryValue(std::ofstream &stream, const Value &value) {
    stream.write(reinterpret_cast<const char *>(&value), sizeof(Value));
    return static_cast<bool>(stream);
}

bool WriteBinaryBytes(std::ofstream &stream, const void *data, const std::size_t byte_count) {
    if (byte_count == 0) {
        return true;
    }
    stream.write(static_cast<const char *>(data), static_cast<std::streamsize>(byte_count));
    return static_cast<bool>(stream);
}

template <typename Value> bool ReadBinaryValue(std::ifstream &stream, Value &value) {
    stream.read(reinterpret_cast<char *>(&value), sizeof(Value));
    return static_cast<bool>(stream);
}

bool ReadBinaryBytes(std::ifstream &stream, void *data, const std::size_t byte_count) {
    if (byte_count == 0) {
        return true;
    }
    stream.read(static_cast<char *>(data), static_cast<std::streamsize>(byte_count));
    return static_cast<bool>(stream);
}

} // namespace

bool TryWriteRuntimeCheckpointAtomically(const RuntimeCheckpoint &checkpoint,
                                         const std::filesystem::path &checkpoint_path) {
    if (checkpoint.checksum != ComputeRuntimeCheckpointChecksum(checkpoint)) {
        return false;
    }

    const std::filesystem::path temporary_path = checkpoint_path.string() + ".tmp";
    std::error_code directory_error;
    if (!checkpoint_path.parent_path().empty()) {
        std::filesystem::create_directories(checkpoint_path.parent_path(), directory_error);
        if (directory_error) {
            return false;
        }
    }
    std::ofstream stream(temporary_path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return false;
    }

    bool ok = true;
    ok &= WriteBinaryValue(stream, kCheckpointMagic);
    ok &= WriteBinaryValue(stream, checkpoint.schema_version);
    ok &= WriteBinaryValue(stream, checkpoint.genome_layout_version);
    ok &= WriteBinaryValue(stream, checkpoint.checksum);
    ok &= WriteBinaryValue(stream, checkpoint.training_data_identity_hash);
    const auto resume_phase = static_cast<std::uint32_t>(checkpoint.resume_phase);
    ok &= WriteBinaryValue(stream, resume_phase);
    ok &= WriteBinaryValue(stream, checkpoint.generation_seed);
    ok &= WriteBinaryValue(stream, checkpoint.runtime_word_counts);
    ok &= WriteBinaryValue(stream, checkpoint.assembly_config);
    ok &= WriteBinaryValue(stream, checkpoint.pending_output_embedding_injection);
    ok &= WriteBinaryValue(stream, checkpoint.runtime_config);
    ok &= WriteBinaryValue(stream, checkpoint.slab_layout);
    ok &= WriteBinaryValue(stream, checkpoint.current_grid_shape);
    ok &= WriteBinaryValue(stream, checkpoint.next_grid_shape);
    ok &= WriteBinaryValue(stream, checkpoint.epicenter_grid_shape);
    ok &= WriteBinaryValue(stream, checkpoint.current_generation.generation_index);
    ok &= WriteBinaryValue(stream, checkpoint.current_generation.active_individual_count);
    for (std::size_t individual_index = 0;
         ok && (individual_index < checkpoint.current_generation.active_individual_count); ++individual_index) {
        ok &= WriteBinaryValue(stream, checkpoint.current_generation.slot_indices[individual_index]);
        ok &= WriteBinaryValue(stream, checkpoint.current_generation.fitness[individual_index]);
        ok &= WriteBinaryValue(stream, checkpoint.current_generation.evaluation_counts[individual_index]);
        ok &= WriteBinaryValue(stream, checkpoint.current_generation.has_fitness[individual_index]);
    }
    ok &= WriteBinaryValue(stream, checkpoint.assembly_plan.child_count);
    for (std::size_t child_index = 0; ok && (child_index < checkpoint.assembly_plan.child_count); ++child_index) {
        ok &= WriteBinaryValue(stream, checkpoint.assembly_plan.parent_pairs[child_index]);
    }
    const std::uint64_t live_count = checkpoint.live_genotypes.size();
    ok &= WriteBinaryValue(stream, live_count);
    for (const RuntimeCheckpointGenotypeRecord &record : checkpoint.live_genotypes) {
        const std::uint64_t byte_count = record.genome_bytes.size();
        ok &= WriteBinaryValue(stream, record.organism_index);
        ok &= WriteBinaryValue(stream, byte_count);
        ok &= WriteBinaryBytes(stream, record.genome_bytes.data(), record.genome_bytes.size());
    }
    stream.close();
    if (!ok || !stream) {
        std::filesystem::remove(temporary_path);
        return false;
    }

    std::error_code error_code;
    std::filesystem::rename(temporary_path, checkpoint_path, error_code);
    if (error_code) {
        std::filesystem::remove(checkpoint_path, error_code);
        error_code.clear();
        std::filesystem::rename(temporary_path, checkpoint_path, error_code);
    }
    if (error_code) {
        std::filesystem::remove(temporary_path);
        return false;
    }
    return true;
}

bool TryReadRuntimeCheckpoint(const std::filesystem::path &checkpoint_path, RuntimeCheckpoint &checkpoint_out) {
    checkpoint_out = {};
    std::ifstream stream(checkpoint_path, std::ios::binary);
    if (!stream) {
        return false;
    }

    std::uint64_t magic = 0;
    std::uint32_t resume_phase = 0;
    bool ok = true;
    ok &= ReadBinaryValue(stream, magic);
    ok &= (magic == kCheckpointMagic);
    ok &= ReadBinaryValue(stream, checkpoint_out.schema_version);
    ok &= ReadBinaryValue(stream, checkpoint_out.genome_layout_version);
    ok &= ReadBinaryValue(stream, checkpoint_out.checksum);
    ok &= ReadBinaryValue(stream, checkpoint_out.training_data_identity_hash);
    ok &= ReadBinaryValue(stream, resume_phase);
    checkpoint_out.resume_phase = static_cast<RuntimeCheckpointResumePhase>(resume_phase);
    ok &= ReadBinaryValue(stream, checkpoint_out.generation_seed);
    ok &= ReadBinaryValue(stream, checkpoint_out.runtime_word_counts);
    ok &= ReadBinaryValue(stream, checkpoint_out.assembly_config);
    ok &= ReadBinaryValue(stream, checkpoint_out.pending_output_embedding_injection);
    ok &= ReadBinaryValue(stream, checkpoint_out.runtime_config);
    ok &= ReadBinaryValue(stream, checkpoint_out.slab_layout);
    ok &= ReadBinaryValue(stream, checkpoint_out.current_grid_shape);
    ok &= ReadBinaryValue(stream, checkpoint_out.next_grid_shape);
    ok &= ReadBinaryValue(stream, checkpoint_out.epicenter_grid_shape);
    std::size_t generation_index = 0;
    std::size_t active_individual_count = 0;
    ok &= ReadBinaryValue(stream, generation_index);
    ok &= ReadBinaryValue(stream, active_individual_count);
    ok &= genotype_slab::TryCreateSlabGeneration(checkpoint_out.current_generation, active_individual_count,
                                                 generation_index);
    for (std::size_t individual_index = 0; ok && (individual_index < active_individual_count); ++individual_index) {
        ok &= ReadBinaryValue(stream, checkpoint_out.current_generation.slot_indices[individual_index]);
        ok &= ReadBinaryValue(stream, checkpoint_out.current_generation.fitness[individual_index]);
        ok &= ReadBinaryValue(stream, checkpoint_out.current_generation.evaluation_counts[individual_index]);
        ok &= ReadBinaryValue(stream, checkpoint_out.current_generation.has_fitness[individual_index]);
    }
    std::size_t child_count = 0;
    ok &= ReadBinaryValue(stream, child_count);
    ok &= genotype_slab::TryCreateSlabAssemblyPlan(checkpoint_out.assembly_plan, child_count);
    for (std::size_t child_index = 0; ok && (child_index < child_count); ++child_index) {
        ok &= ReadBinaryValue(stream, checkpoint_out.assembly_plan.parent_pairs[child_index]);
    }
    std::uint64_t live_count = 0;
    ok &= ReadBinaryValue(stream, live_count);
    if (live_count > active_individual_count) {
        ok = false;
    }
    checkpoint_out.live_genotypes.resize(static_cast<std::size_t>(live_count));
    for (RuntimeCheckpointGenotypeRecord &record : checkpoint_out.live_genotypes) {
        std::uint64_t byte_count = 0;
        ok &= ReadBinaryValue(stream, record.organism_index);
        ok &= ReadBinaryValue(stream, byte_count);
        ok &= (byte_count == checkpoint_out.slab_layout.slot_stride_bytes);
        record.genome_bytes.resize(static_cast<std::size_t>(byte_count));
        ok &= ReadBinaryBytes(stream, record.genome_bytes.data(), record.genome_bytes.size());
    }

    if (!ok || (checkpoint_out.schema_version != kRuntimeCheckpointSchemaVersion) ||
        (checkpoint_out.genome_layout_version != kRuntimeCheckpointGenomeLayoutVersion) ||
        (checkpoint_out.checksum != ComputeRuntimeCheckpointChecksum(checkpoint_out))) {
        checkpoint_out = {};
        return false;
    }

    return true;
}

bool RuntimeCheckpointAsyncWriter::IsWriteInProgress() {
    return pending_write_.valid() && (pending_write_.wait_for(std::chrono::seconds(0)) != std::future_status::ready);
}

bool RuntimeCheckpointAsyncWriter::TryCollectFinishedWrite(bool &write_finished_out, bool &write_succeeded_out) {
    write_finished_out = false;
    write_succeeded_out = true;
    if (!pending_write_.valid()) {
        return true;
    }

    if (IsWriteInProgress()) {
        return true;
    }

    write_finished_out = true;
    write_succeeded_out = pending_write_.get();
    return write_succeeded_out;
}

bool RuntimeCheckpointAsyncWriter::TryStartWrite(RuntimeCheckpoint checkpoint, std::filesystem::path checkpoint_path) {
    if (IsWriteInProgress()) {
        return false;
    }
    if (pending_write_.valid()) {
        if (!pending_write_.get()) {
            return false;
        }
    }

    pending_write_ = std::async(std::launch::async, [checkpoint = std::move(checkpoint),
                                                     checkpoint_path = std::move(checkpoint_path)]() mutable {
        return TryWriteRuntimeCheckpointAtomically(checkpoint, checkpoint_path);
    });
    return true;
}

bool RuntimeCheckpointAsyncWriter::TryWaitForWrite() {
    if (!pending_write_.valid()) {
        return true;
    }

    return pending_write_.get();
}

void SwapDeviceSlabGenerationBuffers(DeviceSlabGARuntimeBuffers &buffers) noexcept {
    std::swap(buffers.genotype_slab.current_slot_indices, buffers.genotype_slab.next_slot_indices);
    std::swap(buffers.genotype_slab.current_fitness, buffers.genotype_slab.next_fitness);
    std::swap(buffers.genotype_slab.current_evaluation_counts, buffers.genotype_slab.next_evaluation_counts);
    std::swap(buffers.genotype_slab.current_has_fitness, buffers.genotype_slab.next_has_fitness);
    std::swap(buffers.genotype_slab.current_generation_index, buffers.genotype_slab.next_generation_index);
    std::swap(buffers.genotype_slab.current_generation_size, buffers.genotype_slab.next_generation_size);
}

bool TryReadDeviceSlabGARuntimeStatus(const DeviceSlabGARuntimeBuffers &buffers,
                                      DeviceSlabGARuntimeStatusCode &status_code) {
    int status_value = 0;
    if (!ReadDeviceStatus(buffers, status_value)) {
        return false;
    }

    status_code = static_cast<DeviceSlabGARuntimeStatusCode>(status_value);
    return true;
}

const char *DeviceSlabGARuntimeStatusCodeString(const DeviceSlabGARuntimeStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceSlabGARuntimeStatusCode::kOk:
        return "ok";
    case DeviceSlabGARuntimeStatusCode::kCudaFailure:
        return "cuda failure";
    case DeviceSlabGARuntimeStatusCode::kInvalidRuntimeConfig:
        return "invalid slab ga runtime config";
    case DeviceSlabGARuntimeStatusCode::kInvalidSlab:
        return "invalid slab state";
    case DeviceSlabGARuntimeStatusCode::kInvalidGeneration:
        return "invalid generation state";
    case DeviceSlabGARuntimeStatusCode::kInvalidTrainingShard:
        return "invalid training-data shard";
    case DeviceSlabGARuntimeStatusCode::kGuessAppendFailed:
        return "could not append a model-selected guess to the Wordle grid";
    case DeviceSlabGARuntimeStatusCode::kPolicyForwardFailed:
        return "policy model forward pass failed";
    case DeviceSlabGARuntimeStatusCode::kActionSelectionFailed:
        return "output-embedding action selection failed";
    case DeviceSlabGARuntimeStatusCode::kPopulationNotEvaluated:
        return "population fitness summary requested before the population was fully evaluated";
    case DeviceSlabGARuntimeStatusCode::kInvalidAssemblyConfig:
        return "invalid next-generation assembly config";
    case DeviceSlabGARuntimeStatusCode::kParentSelectionFailed:
        return "device parent selection failed";
    case DeviceSlabGARuntimeStatusCode::kInvalidAssemblyPlan:
        return "invalid slab assembly plan";
    case DeviceSlabGARuntimeStatusCode::kInvalidParentIndex:
        return "invalid parent index in slab assembly plan";
    case DeviceSlabGARuntimeStatusCode::kSlabFull:
        return "slab is genuinely full";
    case DeviceSlabGARuntimeStatusCode::kOutputEmbeddingInjectionFailed:
        return "device output-embedding injection failed";
    case DeviceSlabGARuntimeStatusCode::kSlabRepackFailed:
        return "slab compaction/repacking failed";
    }

    return "unknown slab ga runtime status";
}

} // namespace neuroevolution::genetic_algorithm::slab_device
