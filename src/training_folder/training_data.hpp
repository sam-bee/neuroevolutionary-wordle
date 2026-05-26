#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "spatial/grid.hpp"
#include "wordle/word.hpp"

namespace neuroevolution::training_folder {

constexpr std::size_t kDefaultInitialActiveWordCount = 20;
constexpr std::size_t kDefaultTrainingShardInitialRadius = 0;
constexpr std::size_t kDefaultShardRadiusGrowthPeriodGenerations = 2;
constexpr std::size_t kDefaultTrainingShardReleaseMinimumGapGenerations = 3;
constexpr std::size_t kDefaultFirstNewTrainingShardReleaseGeneration = 10;
constexpr float kDefaultTrainingShardReleaseCentroidDistanceThreshold = 6.0f;
constexpr float kDefaultTrainingShardReleaseFitnessP99Threshold = 0.20f;
constexpr std::size_t kEffectivelyInfiniteTrainingShardRadius = static_cast<std::size_t>(-1);
constexpr std::size_t kTrainingWordCatalogCapacity = 4739;

struct WordCountSchedule {
    std::size_t initial_word_count = kDefaultInitialActiveWordCount;
    std::size_t word_count_step = 0;
    std::size_t word_count_step_period_generations = 1;
};

struct TrainingWordCatalog {
    common::FixedBuffer<wordle::Word, kTrainingWordCatalogCapacity> words{};
    std::size_t word_count = 0;
};

struct TrainingDataShardRuntime {
    std::size_t first_catalog_word_index = 0;
    std::size_t word_count = 0;
    std::size_t center_row = 0;
    std::size_t center_column = 0;
    std::size_t radius = 0;
    std::uint8_t global_from_outset = 0;
};

struct TrainingDataShardRuntimeSet {
    common::FixedBuffer<TrainingDataShardRuntime, kTrainingWordCatalogCapacity> shards{};
    std::size_t shard_count = 0;
};

struct TrainingDataShardReleaseHistory {
    common::FixedBuffer<std::size_t, kTrainingWordCatalogCapacity> release_generations{};
    common::FixedBuffer<float, kTrainingWordCatalogCapacity> release_fitness_p99{};
    std::size_t release_count = 0;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidTrainingWordCatalog(const TrainingWordCatalog &catalog) noexcept {
    if (catalog.word_count > kTrainingWordCatalogCapacity) {
        return false;
    }

    for (std::size_t word_index = 0; word_index < catalog.word_count; ++word_index) {
        if (!wordle::IsValidWord(catalog.words[word_index])) {
            return false;
        }
    }

    return true;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidWordCountSchedule(const WordCountSchedule &schedule) noexcept {
    return (schedule.initial_word_count > 0) && (schedule.word_count_step_period_generations > 0);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
TrainingDataShardCountForIntroducedWordCount(const WordCountSchedule &schedule,
                                             const std::size_t introduced_word_count) noexcept {
    if (!IsValidWordCountSchedule(schedule) || (introduced_word_count == 0)) {
        return 0;
    }

    const std::size_t foundation_word_count =
        (schedule.initial_word_count < introduced_word_count) ? schedule.initial_word_count : introduced_word_count;
    if (foundation_word_count == 0) {
        return 0;
    }

    if (introduced_word_count == foundation_word_count) {
        return 1;
    }

    if (schedule.word_count_step == 0) {
        return 0;
    }

    const std::size_t remaining_word_count = introduced_word_count - foundation_word_count;
    return 1 + ((remaining_word_count + schedule.word_count_step - 1) / schedule.word_count_step);
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidTrainingDataShardReleaseHistory(const TrainingDataShardReleaseHistory &history) noexcept {
    if ((history.release_count == 0) || (history.release_count > kTrainingWordCatalogCapacity)) {
        return false;
    }

    for (std::size_t release_index = 1; release_index < history.release_count; ++release_index) {
        if (history.release_generations[release_index] <= history.release_generations[release_index - 1]) {
            return false;
        }
    }

    return true;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidTrainingDataShardReleaseHistoryForIntroducedWordCount(const WordCountSchedule &schedule,
                                                             const std::size_t introduced_word_count,
                                                             const TrainingDataShardReleaseHistory &history) noexcept {
    const std::size_t required_shard_count =
        TrainingDataShardCountForIntroducedWordCount(schedule, introduced_word_count);
    return (required_shard_count > 0) && IsValidTrainingDataShardReleaseHistory(history) &&
           (history.release_count >= required_shard_count);
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool TryRecordTrainingDataShardRelease(TrainingDataShardReleaseHistory &history,
                                                                            const std::size_t release_generation,
                                                                            const float release_fitness_p99) noexcept {
    if (history.release_count >= kTrainingWordCatalogCapacity) {
        return false;
    }
    if ((history.release_count > 0) && (release_generation <= history.release_generations[history.release_count - 1])) {
        return false;
    }

    history.release_generations[history.release_count] = release_generation;
    history.release_fitness_p99[history.release_count] = release_fitness_p99;
    ++history.release_count;
    return true;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool
IsValidTrainingDataShardRuntime(const TrainingDataShardRuntime &shard, const std::size_t catalog_word_count,
                                const spatial::CellularGridShape &grid_shape) noexcept {
    return (shard.word_count > 0) && (shard.first_catalog_word_index < catalog_word_count) &&
           (shard.word_count <= (catalog_word_count - shard.first_catalog_word_index)) &&
           ((shard.global_from_outset != 0) ||
            (spatial::IsValidCellularGridShape(grid_shape) && (shard.center_row < grid_shape.row_count) &&
             (shard.center_column < grid_shape.column_count)));
}

struct TrainingShardCenterCoordinate {
    std::size_t row = 0;
    std::size_t column = 0;
};

constexpr NEUROEVOLUTION_HOST_DEVICE TrainingShardCenterCoordinate DeterministicTrainingShardCenterCoordinate(
    const std::size_t shard_ordinal, const spatial::CellularGridShape &grid_shape) noexcept {
    TrainingShardCenterCoordinate coordinate{};
    if (!spatial::IsValidCellularGridShape(grid_shape)) {
        return coordinate;
    }

    constexpr std::uint64_t kGoldenRatioStride = 11400714819323198485ull;
    const std::size_t cell_index = static_cast<std::size_t>(
        (static_cast<std::uint64_t>(shard_ordinal + 1) * kGoldenRatioStride) % grid_shape.cell_count);
    coordinate.row = spatial::GridRowFromIndex(grid_shape, cell_index);
    coordinate.column = spatial::GridColumnFromIndex(grid_shape, cell_index);
    return coordinate;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
DeterministicTrainingShardCenterCellIndex(const std::size_t shard_ordinal,
                                          const spatial::CellularGridShape &grid_shape) noexcept {
    const TrainingShardCenterCoordinate coordinate =
        DeterministicTrainingShardCenterCoordinate(shard_ordinal, grid_shape);
    return spatial::GridIndexFromRowColumn(grid_shape, coordinate.row, coordinate.column);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
DeterministicTrainingShardCenterCellIndex(const std::size_t shard_ordinal,
                                          const std::size_t square_cell_count) noexcept {
    spatial::CellularGridShape grid_shape{};
    if (!spatial::TryMakeCellularGridShape(square_cell_count, grid_shape)) {
        return 0;
    }

    return DeterministicTrainingShardCenterCellIndex(shard_ordinal, grid_shape);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
TrainingShardRadiusAtGeneration(const std::size_t introduction_generation, const std::size_t current_generation_index,
                                const std::size_t initial_radius,
                                const std::size_t radius_growth_period_generations) noexcept {
    if ((radius_growth_period_generations == 0) || (current_generation_index < introduction_generation)) {
        return 0;
    }

    const std::size_t growth_steps =
        (current_generation_index - introduction_generation) / radius_growth_period_generations;
    if (growth_steps > (kEffectivelyInfiniteTrainingShardRadius - initial_radius)) {
        return kEffectivelyInfiniteTrainingShardRadius;
    }

    return initial_radius + growth_steps;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
TrainingShardRadiusAtGeneration(const std::size_t introduction_generation, const std::size_t current_generation_index,
                                const std::size_t radius_growth_period_generations) noexcept {
    return TrainingShardRadiusAtGeneration(introduction_generation, current_generation_index,
                                           kDefaultTrainingShardInitialRadius, radius_growth_period_generations);
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool DoesTrainingDataShardCoverCell(const TrainingDataShardRuntime &shard,
                                                                         const spatial::CellularGridShape &grid_shape,
                                                                         const std::size_t cell_index) noexcept {
    if (!IsValidTrainingDataShardRuntime(shard, kTrainingWordCatalogCapacity, grid_shape) ||
        (cell_index >= grid_shape.cell_count)) {
        return false;
    }

    if (shard.global_from_outset != 0) {
        return true;
    }

    const std::size_t center_cell_index =
        spatial::GridIndexFromRowColumn(grid_shape, shard.center_row, shard.center_column);
    return spatial::ToroidalChebyshevDistance(grid_shape, center_cell_index, cell_index) <= shard.radius;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t
ScheduledWordCountForGeneration(const WordCountSchedule &schedule, const std::size_t catalog_word_count,
                                const std::size_t generation_index) noexcept {
    if (!IsValidWordCountSchedule(schedule) || (catalog_word_count == 0)) {
        return 0;
    }

    const std::size_t initial_word_count =
        (schedule.initial_word_count < catalog_word_count) ? schedule.initial_word_count : catalog_word_count;
    if (schedule.word_count_step == 0) {
        return initial_word_count;
    }

    const std::size_t completed_schedule_steps = generation_index / schedule.word_count_step_period_generations;
    if (completed_schedule_steps == 0) {
        return initial_word_count;
    }

    if (schedule.word_count_step > ((catalog_word_count - initial_word_count) / completed_schedule_steps)) {
        return catalog_word_count;
    }

    const std::size_t scheduled_word_count = initial_word_count + (completed_schedule_steps * schedule.word_count_step);
    return (scheduled_word_count < catalog_word_count) ? scheduled_word_count : catalog_word_count;
}

constexpr bool TryBuildTrainingDataShardRuntimeSet(
    const WordCountSchedule &schedule, const std::size_t introduced_word_count, const std::size_t generation_index,
    const spatial::CellularGridShape &grid_shape, const spatial::CellularGridShape &epicenter_grid_shape,
    const std::size_t shard_initial_radius, const std::size_t radius_growth_period_generations,
    const TrainingDataShardReleaseHistory *release_history, TrainingDataShardRuntimeSet &runtime_set_out) noexcept {
    runtime_set_out = {};
    if (!IsValidWordCountSchedule(schedule) || (introduced_word_count == 0) ||
        (introduced_word_count > kTrainingWordCatalogCapacity) || !spatial::IsValidCellularGridShape(grid_shape) ||
        !spatial::IsValidCellularGridShape(epicenter_grid_shape) ||
        (epicenter_grid_shape.column_count != grid_shape.column_count) || (radius_growth_period_generations == 0)) {
        return false;
    }
    if ((release_history != nullptr) && !IsValidTrainingDataShardReleaseHistoryForIntroducedWordCount(
                                            schedule, introduced_word_count, *release_history)) {
        return false;
    }

    const std::size_t foundation_word_count =
        (schedule.initial_word_count < introduced_word_count) ? schedule.initial_word_count : introduced_word_count;
    if (foundation_word_count == 0) {
        return false;
    }

    runtime_set_out.shards[0] = {
        .first_catalog_word_index = 0,
        .word_count = foundation_word_count,
        .center_row = 0,
        .center_column = 0,
        .radius = 0,
        .global_from_outset = 1,
    };
    runtime_set_out.shard_count = 1;

    if ((schedule.word_count_step == 0) || (foundation_word_count == introduced_word_count)) {
        return true;
    }

    std::size_t next_catalog_word_index = foundation_word_count;
    std::size_t local_shard_index = 0;
    while (next_catalog_word_index < introduced_word_count) {
        const std::size_t shard_word_count =
            ((introduced_word_count - next_catalog_word_index) < schedule.word_count_step)
                ? (introduced_word_count - next_catalog_word_index)
                : schedule.word_count_step;
        const std::size_t release_history_index = runtime_set_out.shard_count;
        const std::size_t introduction_generation =
            (release_history == nullptr) ? ((local_shard_index + 1) * schedule.word_count_step_period_generations)
                                         : release_history->release_generations[release_history_index];

        const TrainingShardCenterCoordinate unclamped_center =
            DeterministicTrainingShardCenterCoordinate(local_shard_index, epicenter_grid_shape);
        const std::size_t center_row =
            (unclamped_center.row < grid_shape.row_count) ? unclamped_center.row : (grid_shape.row_count - 1);

        runtime_set_out.shards[runtime_set_out.shard_count] = {
            .first_catalog_word_index = next_catalog_word_index,
            .word_count = shard_word_count,
            .center_row = center_row,
            .center_column = unclamped_center.column,
            .radius = TrainingShardRadiusAtGeneration(introduction_generation, generation_index, shard_initial_radius,
                                                      radius_growth_period_generations),
            .global_from_outset = 0,
        };
        ++runtime_set_out.shard_count;
        next_catalog_word_index += shard_word_count;
        ++local_shard_index;
    }

    return true;
}

constexpr bool TryBuildTrainingDataShardRuntimeSet(
    const WordCountSchedule &schedule, const std::size_t introduced_word_count, const std::size_t generation_index,
    const spatial::CellularGridShape &grid_shape, const std::size_t shard_initial_radius,
    const std::size_t radius_growth_period_generations, const TrainingDataShardReleaseHistory *release_history,
    TrainingDataShardRuntimeSet &runtime_set_out) noexcept {
    return TryBuildTrainingDataShardRuntimeSet(schedule, introduced_word_count, generation_index, grid_shape,
                                               grid_shape, shard_initial_radius, radius_growth_period_generations,
                                               release_history, runtime_set_out);
}

constexpr bool TryBuildTrainingDataShardRuntimeSet(
    const WordCountSchedule &schedule, const std::size_t introduced_word_count, const std::size_t generation_index,
    const spatial::CellularGridShape &grid_shape, const TrainingDataShardReleaseHistory *release_history,
    const std::size_t radius_growth_period_generations, TrainingDataShardRuntimeSet &runtime_set_out) noexcept {
    return TryBuildTrainingDataShardRuntimeSet(schedule, introduced_word_count, generation_index, grid_shape,
                                               kDefaultTrainingShardInitialRadius, radius_growth_period_generations,
                                               release_history, runtime_set_out);
}

constexpr bool TryBuildTrainingDataShardRuntimeSet(
    const WordCountSchedule &schedule, const std::size_t introduced_word_count, const std::size_t generation_index,
    const spatial::CellularGridShape &grid_shape, const spatial::CellularGridShape &epicenter_grid_shape,
    const std::size_t shard_initial_radius, const std::size_t radius_growth_period_generations,
    TrainingDataShardRuntimeSet &runtime_set_out) noexcept {
    return TryBuildTrainingDataShardRuntimeSet(schedule, introduced_word_count, generation_index, grid_shape,
                                               epicenter_grid_shape, shard_initial_radius,
                                               radius_growth_period_generations, nullptr, runtime_set_out);
}

constexpr bool TryBuildTrainingDataShardRuntimeSet(
    const WordCountSchedule &schedule, const std::size_t introduced_word_count, const std::size_t generation_index,
    const spatial::CellularGridShape &grid_shape, const std::size_t shard_initial_radius,
    const std::size_t radius_growth_period_generations, TrainingDataShardRuntimeSet &runtime_set_out) noexcept {
    return TryBuildTrainingDataShardRuntimeSet(schedule, introduced_word_count, generation_index, grid_shape,
                                               shard_initial_radius, radius_growth_period_generations, nullptr,
                                               runtime_set_out);
}

constexpr bool TryBuildTrainingDataShardRuntimeSet(const WordCountSchedule &schedule,
                                                   const std::size_t introduced_word_count,
                                                   const std::size_t generation_index,
                                                   const spatial::CellularGridShape &grid_shape,
                                                   const std::size_t radius_growth_period_generations,
                                                   TrainingDataShardRuntimeSet &runtime_set_out) noexcept {
    return TryBuildTrainingDataShardRuntimeSet(schedule, introduced_word_count, generation_index, grid_shape, nullptr,
                                               radius_growth_period_generations, runtime_set_out);
}

std::filesystem::path DefaultActionSpacePath();

bool TryLoadTrainingWordCatalogFromActionSpace(const std::filesystem::path &action_space_path,
                                               TrainingWordCatalog &catalog);

TrainingWordCatalog LoadTrainingWordCatalogFromActionSpace(const std::filesystem::path &action_space_path);

TrainingWordCatalog LoadTrainingWordCatalogFromActionSpace();

bool UploadTrainingWordCatalogToDeviceConstantMemory(const TrainingWordCatalog &catalog);

void UploadTrainingWordCatalogToDeviceConstantMemoryOrThrow(const TrainingWordCatalog &catalog);

#if defined(__CUDACC__)
__device__ const TrainingWordCatalog &DeviceTrainingWordCatalog() noexcept;
#endif

} // namespace neuroevolution::training_folder
