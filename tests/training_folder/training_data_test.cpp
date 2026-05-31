#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::spatial::CellularGridShape;
using neuroevolution::spatial::GridIndexFromRowColumn;
using neuroevolution::spatial::TryMakeCellularGridShape;
using neuroevolution::spatial::TryMakeRectangularCellularGridShape;
using neuroevolution::training_folder::DefaultActionSpacePath;
using neuroevolution::training_folder::DecideTrainingDataShardReleaseTrigger;
using neuroevolution::training_folder::DeterministicTrainingShardCenterCellIndex;
using neuroevolution::training_folder::DeterministicTrainingShardCenterCoordinate;
using neuroevolution::training_folder::DoesTrainingDataShardCoverCell;
using neuroevolution::training_folder::IsValidTrainingWordCatalog;
using neuroevolution::training_folder::kDefaultInitialActiveWordCount;
using neuroevolution::training_folder::kDefaultShardRadiusGrowthPeriodGenerations;
using neuroevolution::training_folder::kDefaultTrainingShardInitialRadius;
using neuroevolution::training_folder::kEffectivelyInfiniteTrainingShardRadius;
using neuroevolution::training_folder::kTrainingWordCatalogCapacity;
using neuroevolution::training_folder::LoadTrainingWordCatalogFromActionSpace;
using neuroevolution::training_folder::ScheduledWordCountForGeneration;
using neuroevolution::training_folder::TrainingDataShardCountForIntroducedWordCount;
using neuroevolution::training_folder::TrainingDataShardReleaseHistory;
using neuroevolution::training_folder::TrainingDataShardRuntime;
using neuroevolution::training_folder::TrainingDataShardRuntimeSet;
using neuroevolution::training_folder::TrainingShardCenterCoordinate;
using neuroevolution::training_folder::TrainingShardRadiusAtGeneration;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::training_folder::TryBuildTrainingDataShardRuntimeSet;
using neuroevolution::training_folder::TryRecordTrainingDataShardRelease;
using neuroevolution::training_folder::WordCountSchedule;
using neuroevolution::wordle::Word;

constexpr std::array<const char *, kDefaultInitialActiveWordCount> kExpectedTrainingWords = {
    "MINOS", "VODKA", "RAZOR", "GRADS", "CURLS", "BILGE", "GREET", "PYLON", "ENTER", "READY",
    "VERDE", "AUGER", "FOOTS", "BRACE", "PURTY", "SPORT", "TIRES", "FRISK", "AFFIX", "CHUMS",
};

Word MakeWord(const std::string_view letters) {
    if (letters.size() != neuroevolution::wordle::kWordLength) {
        throw std::invalid_argument("Training-data test word view must contain exactly five characters.");
    }

    Word word{};
    for (std::size_t position = 0; position < neuroevolution::wordle::kWordLength; ++position) {
        const char value = letters[position];
        if (!neuroevolution::wordle::IsAsciiUppercaseLetter(value)) {
            throw std::invalid_argument("Training-data test word view must contain only uppercase ASCII letters.");
        }

        word.letter_indices[position] = neuroevolution::wordle::LetterIndexFromAscii(value);
    }

    return word;
}

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

bool ExpectWordEquals(const Word &actual, const Word &expected, const std::string_view label) {
    bool ok = true;

    for (std::size_t position = 0; position < neuroevolution::wordle::kWordLength; ++position) {
        if (actual.letter_indices[position] != expected.letter_indices[position]) {
            std::cerr << "FAIL: " << label << " mismatch at position " << position << '\n';
            ok = false;
        }
    }

    return ok;
}

bool ExpectTrainingWordCatalogMatchesExpectedPrefix(const TrainingWordCatalog &catalog,
                                                    const std::string_view label_prefix) {
    bool ok = true;

    ok &= ExpectTrue(catalog.word_count == kTrainingWordCatalogCapacity,
                     std::string(label_prefix) + " should contain the full training-word catalog");

    for (std::size_t entry_index = 0; entry_index < kExpectedTrainingWords.size(); ++entry_index) {
        ok &= ExpectWordEquals(catalog.words[entry_index], MakeWord(kExpectedTrainingWords[entry_index]),
                               std::string(label_prefix) + " word " + std::to_string(entry_index));
    }

    return ok;
}

bool TestLoadTrainingWordCatalogReadsRandomisedActionSpaceWords() {
    const std::filesystem::path action_space_path = DefaultActionSpacePath();
    const TrainingWordCatalog catalog = LoadTrainingWordCatalogFromActionSpace(action_space_path);

    bool ok = true;
    ok &= ExpectTrue(std::filesystem::exists(action_space_path), "Expected randomized action-space file to exist");
    ok &= ExpectTrue(IsValidTrainingWordCatalog(catalog), "Expected loaded training-word catalog to be valid");
    ok &= ExpectTrainingWordCatalogMatchesExpectedPrefix(catalog, "training-word catalog");
    return ok;
}

bool TestWordCountScheduleAdvancesByConfiguredStepAndClampsToCatalog() {
    constexpr WordCountSchedule kSchedule{
        .initial_word_count = 50,
        .word_count_step = 50,
        .word_count_step_period_generations = 20,
    };

    bool ok = true;
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 0) == 50,
                     "Expected schedule generation zero to use the initial word count");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 19) == 50,
                     "Expected schedule to hold steady before the first configured period boundary");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 20) == 100,
                     "Expected schedule to add one configured step at generation 20");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 39) == 100,
                     "Expected schedule to hold the stepped count until the next boundary");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 40) == 150,
                     "Expected schedule to add the second configured step at generation 40");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, 120, 1000) == 120,
                     "Expected schedule to clamp to the catalog size when repeated steps would exceed it");
    return ok;
}

bool TestWordCountScheduleCanStayFixedWidthWithZeroStep() {
    constexpr WordCountSchedule kSchedule{
        .initial_word_count = 37,
        .word_count_step = 0,
        .word_count_step_period_generations = 5,
    };

    bool ok = true;
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 0) == 37,
                     "Expected zero-step schedule generation zero to use the initial word count");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 4) == 37,
                     "Expected zero-step schedule to stay fixed before a period boundary");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, kTrainingWordCatalogCapacity, 500) == 37,
                     "Expected zero-step schedule to remain fixed for later generations");
    ok &= ExpectTrue(ScheduledWordCountForGeneration(kSchedule, 20, 500) == 20,
                     "Expected zero-step schedule to still clamp to the catalog size");
    return ok;
}

bool TestTrainingShardRadiusGrowthUsesConfiguredCadence() {
    bool ok = true;
    ok &= ExpectTrue(TrainingShardRadiusAtGeneration(4, 4, kDefaultShardRadiusGrowthPeriodGenerations) == 0,
                     "Expected a new shard to start at radius zero on its introduction generation");
    ok &= ExpectTrue(TrainingShardRadiusAtGeneration(4, 5, kDefaultShardRadiusGrowthPeriodGenerations) == 0,
                     "Expected radius to hold steady before the first configured growth boundary");
    ok &= ExpectTrue(TrainingShardRadiusAtGeneration(4, 6, kDefaultShardRadiusGrowthPeriodGenerations) == 1,
                     "Expected radius to grow by one after two generations");
    ok &= ExpectTrue(TrainingShardRadiusAtGeneration(4, 8, kDefaultShardRadiusGrowthPeriodGenerations) == 2,
                     "Expected radius to continue growing at the configured cadence");
    ok &= ExpectTrue(TrainingShardRadiusAtGeneration(4, 4, 7, kDefaultShardRadiusGrowthPeriodGenerations) == 7,
                     "Expected a configured initial radius to apply on the introduction generation");
    ok &= ExpectTrue(TrainingShardRadiusAtGeneration(4, 8, 7, kDefaultShardRadiusGrowthPeriodGenerations) == 9,
                     "Expected radius growth to be added to the configured initial radius");
    return ok;
}

bool TestTrainingShardCoverageUsesToroidalChebyshevRadius() {
    CellularGridShape grid_shape{};
    bool ok = TryMakeCellularGridShape(25, grid_shape);
    ok &= ExpectTrue(ok, "Expected a 25-cell population to form a valid 5x5 cellular grid");
    if (!ok) {
        return false;
    }

    const TrainingDataShardRuntime foundation_shard{
        .first_catalog_word_index = 0,
        .word_count = 20,
        .center_row = 0,
        .center_column = 0,
        .radius = 0,
        .global_from_outset = 1,
    };
    const TrainingDataShardRuntime local_shard{
        .first_catalog_word_index = 20,
        .word_count = 10,
        .center_row = 0,
        .center_column = 0,
        .radius = 1,
        .global_from_outset = 0,
    };

    ok &= ExpectTrue(DoesTrainingDataShardCoverCell(foundation_shard, grid_shape, 12),
                     "Expected the foundation shard to cover every cell");
    ok &= ExpectTrue(DoesTrainingDataShardCoverCell(local_shard, grid_shape, 24),
                     "Expected toroidal wrap to make the opposite corner fall within radius one");
    ok &= ExpectTrue(!DoesTrainingDataShardCoverCell(local_shard, grid_shape, 12),
                     "Expected a cell at Chebyshev distance two to fall outside a radius-one shard");
    return ok;
}

bool TestTrainingShardRuntimeSetBuildsFoundationAndLocalPhaseShards() {
    constexpr WordCountSchedule kSchedule{
        .initial_word_count = 20,
        .word_count_step = 10,
        .word_count_step_period_generations = 3,
    };

    CellularGridShape grid_shape{};
    bool ok = TryMakeCellularGridShape(25, grid_shape);
    ok &= ExpectTrue(ok, "Expected a 25-cell population to form a valid 5x5 cellular grid");
    if (!ok) {
        return false;
    }

    TrainingDataShardRuntimeSet runtime_set{};
    ok &= ExpectTrue(TryBuildTrainingDataShardRuntimeSet(kSchedule, 40, 6, grid_shape,
                                                         kDefaultShardRadiusGrowthPeriodGenerations, runtime_set),
                     "Expected runtime shard scheduling to succeed for a two-phase local curriculum");
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(runtime_set.shard_count == 3, "Expected one global foundation shard plus two later local shards");

    const TrainingDataShardRuntime &foundation_shard = runtime_set.shards[0];
    ok &= ExpectTrue(foundation_shard.global_from_outset != 0,
                     "Expected the first shard to stay globally active from generation zero");
    ok &= ExpectTrue(foundation_shard.first_catalog_word_index == 0,
                     "Expected the foundation shard to begin at the top of the catalog");
    ok &= ExpectTrue(foundation_shard.word_count == 20,
                     "Expected the foundation shard to contain the configured initial word count");

    const TrainingDataShardRuntime &first_local_shard = runtime_set.shards[1];
    ok &= ExpectTrue(first_local_shard.global_from_outset == 0, "Expected later phases to become local spatial shards");
    ok &= ExpectTrue(first_local_shard.first_catalog_word_index == 20,
                     "Expected the first local shard to start immediately after the foundation shard");
    ok &= ExpectTrue(first_local_shard.word_count == 10,
                     "Expected the first local shard to contain one configured word-count step");
    ok &=
        ExpectTrue(GridIndexFromRowColumn(grid_shape, first_local_shard.center_row, first_local_shard.center_column) ==
                       DeterministicTrainingShardCenterCellIndex(0, 25),
                   "Expected the first local shard center to be derived deterministically from shard ordinal");
    ok &= ExpectTrue(first_local_shard.radius == 1,
                     "Expected the first local shard radius to have grown once after two generations");

    const TrainingDataShardRuntime &second_local_shard = runtime_set.shards[2];
    ok &= ExpectTrue(second_local_shard.first_catalog_word_index == 30,
                     "Expected later local shards to keep consuming contiguous catalog ranges");
    ok &= ExpectTrue(second_local_shard.word_count == 10,
                     "Expected the second local shard to contain one configured word-count step");
    ok &= ExpectTrue(
        GridIndexFromRowColumn(grid_shape, second_local_shard.center_row, second_local_shard.center_column) ==
            DeterministicTrainingShardCenterCellIndex(1, 25),
        "Expected the second local shard center to be derived deterministically from shard ordinal");
    ok &= ExpectTrue(second_local_shard.radius == 0,
                     "Expected a shard introduced on the current generation to start at radius zero");
    return ok;
}

bool TestTrainingShardRuntimeSetUsesRecordedReleaseGenerationsForRadius() {
    constexpr WordCountSchedule kSchedule{
        .initial_word_count = 20,
        .word_count_step = 10,
        .word_count_step_period_generations = 3,
    };

    CellularGridShape grid_shape{};
    bool ok = TryMakeCellularGridShape(25, grid_shape);
    ok &= ExpectTrue(ok, "Expected a 25-cell population to form a valid 5x5 cellular grid");
    if (!ok) {
        return false;
    }

    TrainingDataShardReleaseHistory release_history{};
    ok &= ExpectTrue(TryRecordTrainingDataShardRelease(release_history, 0, 0.40f),
                     "Expected foundation release history recording to succeed");
    ok &= ExpectTrue(TryRecordTrainingDataShardRelease(release_history, 10, 0.55f),
                     "Expected adaptive local release history recording to succeed");
    ok &= ExpectTrue(TrainingDataShardCountForIntroducedWordCount(kSchedule, 30) == 2,
                     "Expected introduced word count to require one foundation and one local shard");

    TrainingDataShardRuntimeSet runtime_set{};
    ok &= ExpectTrue(
        TryBuildTrainingDataShardRuntimeSet(kSchedule, 30, 10, grid_shape, kDefaultTrainingShardInitialRadius,
                                            kDefaultShardRadiusGrowthPeriodGenerations, &release_history, runtime_set),
        "Expected runtime shard scheduling to accept adaptive release history");
    if (!ok) {
        return false;
    }

    ok &=
        ExpectTrue(runtime_set.shard_count == 2, "Expected one global foundation shard plus one adaptive local shard");
    ok &= ExpectTrue(runtime_set.shards[1].radius == 0,
                     "Expected adaptively released shard to start at radius zero on its actual release generation");

    TrainingDataShardRuntimeSet later_runtime_set{};
    ok &= ExpectTrue(TryBuildTrainingDataShardRuntimeSet(
                         kSchedule, 30, 12, grid_shape, kDefaultTrainingShardInitialRadius,
                         kDefaultShardRadiusGrowthPeriodGenerations, &release_history, later_runtime_set),
                     "Expected runtime shard scheduling to rebuild from adaptive release history later");
    ok &= ExpectTrue(later_runtime_set.shards[1].radius == 1,
                     "Expected adaptively released shard radius to grow from its actual release generation");
    return ok;
}

bool TestTrainingShardCentersClampWhenRowsAreRemoved() {
    constexpr WordCountSchedule kSchedule{
        .initial_word_count = 20,
        .word_count_step = 10,
        .word_count_step_period_generations = 3,
    };

    CellularGridShape original_grid_shape{};
    CellularGridShape shrunken_grid_shape{};
    bool ok = TryMakeCellularGridShape(25, original_grid_shape);
    ok &= TryMakeRectangularCellularGridShape(4, 5, shrunken_grid_shape);
    if (!ok) {
        return false;
    }

    TrainingDataShardRuntimeSet runtime_set{};
    ok &= ExpectTrue(TryBuildTrainingDataShardRuntimeSet(kSchedule, 30, 3, shrunken_grid_shape, original_grid_shape,
                                                         kDefaultTrainingShardInitialRadius,
                                                         kDefaultShardRadiusGrowthPeriodGenerations, runtime_set),
                     "Expected runtime shard scheduling to accept an original epicenter grid");
    if (!ok) {
        return false;
    }

    const TrainingShardCenterCoordinate original_center =
        DeterministicTrainingShardCenterCoordinate(0, original_grid_shape);
    const TrainingDataShardRuntime &local_shard = runtime_set.shards[1];
    ok &= ExpectTrue(local_shard.center_column == original_center.column,
                     "Expected row deletion to preserve the shard center column");
    ok &= ExpectTrue(local_shard.center_row < shrunken_grid_shape.row_count,
                     "Expected row deletion to keep the shard center on the shrunken grid");
    if (original_center.row >= shrunken_grid_shape.row_count) {
        ok &= ExpectTrue(local_shard.center_row == shrunken_grid_shape.row_count - 1,
                         "Expected deleted-row shard centers to clamp to the last surviving row");
    } else {
        ok &= ExpectTrue(local_shard.center_row == original_center.row,
                         "Expected surviving shard center rows to stay unchanged");
    }
    return ok;
}

bool TestTrainingShardRuntimeSetCanStartLocalShardsWithInfiniteRadius() {
    constexpr WordCountSchedule kSchedule{
        .initial_word_count = 20,
        .word_count_step = 10,
        .word_count_step_period_generations = 3,
    };

    CellularGridShape grid_shape{};
    bool ok = TryMakeCellularGridShape(25, grid_shape);
    ok &= ExpectTrue(ok, "Expected a 25-cell population to form a valid 5x5 cellular grid");
    if (!ok) {
        return false;
    }

    TrainingDataShardRuntimeSet runtime_set{};
    ok &= ExpectTrue(TryBuildTrainingDataShardRuntimeSet(kSchedule, 30, 3, grid_shape,
                                                         kEffectivelyInfiniteTrainingShardRadius,
                                                         kDefaultShardRadiusGrowthPeriodGenerations, runtime_set),
                     "Expected runtime shard scheduling to accept an effectively infinite initial radius");
    if (!ok) {
        return false;
    }

    ok &=
        ExpectTrue(runtime_set.shard_count == 2, "Expected one foundation shard plus one newly introduced local shard");
    const TrainingDataShardRuntime &local_shard = runtime_set.shards[1];
    ok &= ExpectTrue(local_shard.radius == kEffectivelyInfiniteTrainingShardRadius,
                     "Expected the local shard to start with the configured infinite radius");
    ok &= ExpectTrue(DoesTrainingDataShardCoverCell(local_shard, grid_shape, 12),
                     "Expected an infinite-radius local shard to cover every cell immediately");

    TrainingDataShardRuntimeSet default_runtime_set{};
    ok &=
        ExpectTrue(TryBuildTrainingDataShardRuntimeSet(kSchedule, 30, 3, grid_shape, kDefaultTrainingShardInitialRadius,
                                                       kDefaultShardRadiusGrowthPeriodGenerations, default_runtime_set),
                   "Expected default initial radius scheduling to remain valid");
    ok &= ExpectTrue(default_runtime_set.shards[1].radius == 0,
                     "Expected default local shard introductions to remain radius zero");
    return ok;
}

bool TestTrainingShardReleaseTriggerUsesRelativeFitnessOrConvergence() {
    bool ok = true;

    const auto below_fitness_gain =
        DecideTrainingDataShardReleaseTrigger(0.549f, 0.500f, 0.050f, 0.700f, 8.0f, 4.0f);
    ok &= ExpectTrue(!below_fitness_gain.fitness_p99_triggered,
                     "Expected p99 gain below the configured threshold not to trigger release");
    ok &= ExpectTrue(!below_fitness_gain.convergence_triggered,
                     "Expected centroid distance above the configured threshold not to trigger release");

    const auto above_fitness_gain =
        DecideTrainingDataShardReleaseTrigger(0.560f, 0.500f, 0.050f, 0.700f, 8.0f, 4.0f);
    ok &= ExpectTrue(above_fitness_gain.fitness_p99_triggered,
                     "Expected p99 gain above the configured threshold to trigger release");
    ok &= ExpectTrue(!above_fitness_gain.convergence_triggered,
                     "Expected fitness-triggered release not to require centroid convergence");
    ok &= ExpectTrue((above_fitness_gain.fitness_p99_target > 0.549f) &&
                         (above_fitness_gain.fitness_p99_target < 0.551f),
                     "Expected fitness p99 target to use previous release baseline plus gain while below ceiling");

    const auto below_uncapped_gain_above_target_ceiling =
        DecideTrainingDataShardReleaseTrigger(0.700f, 0.680f, 0.050f, 0.700f, 8.0f, 4.0f);
    ok &= ExpectTrue(below_uncapped_gain_above_target_ceiling.fitness_p99_triggered,
                     "Expected p99 at the target ceiling to trigger even without a full p99 gain step");
    ok &= ExpectTrue((below_uncapped_gain_above_target_ceiling.fitness_p99_target > 0.699f) &&
                         (below_uncapped_gain_above_target_ceiling.fitness_p99_target < 0.701f),
                     "Expected fitness p99 target to cap at the configured ceiling");

    const auto at_centroid_threshold =
        DecideTrainingDataShardReleaseTrigger(0.540f, 0.500f, 0.050f, 0.700f, 4.0f, 4.0f);
    ok &= ExpectTrue(!at_centroid_threshold.convergence_triggered,
                     "Expected centroid convergence to require dipping below the threshold");

    const auto below_centroid_threshold =
        DecideTrainingDataShardReleaseTrigger(0.540f, 0.500f, 0.050f, 0.700f, 3.999f, 4.0f);
    ok &= ExpectTrue(!below_centroid_threshold.fitness_p99_triggered,
                     "Expected convergence-triggered release not to require the p99 gain threshold");
    ok &= ExpectTrue(below_centroid_threshold.convergence_triggered,
                     "Expected centroid distance below the configured threshold to trigger release");
    return ok;
}

} // namespace

int main() {
    if (!TestLoadTrainingWordCatalogReadsRandomisedActionSpaceWords() ||
        !TestWordCountScheduleAdvancesByConfiguredStepAndClampsToCatalog() ||
        !TestWordCountScheduleCanStayFixedWidthWithZeroStep() ||
        !TestTrainingShardRadiusGrowthUsesConfiguredCadence() ||
        !TestTrainingShardCoverageUsesToroidalChebyshevRadius() ||
        !TestTrainingShardRuntimeSetBuildsFoundationAndLocalPhaseShards() ||
        !TestTrainingShardRuntimeSetUsesRecordedReleaseGenerationsForRadius() ||
        !TestTrainingShardCentersClampWhenRowsAreRemoved() ||
        !TestTrainingShardRuntimeSetCanStartLocalShardsWithInfiniteRadius() ||
        !TestTrainingShardReleaseTriggerUsesRelativeFitnessOrConvergence()) {
        return 1;
    }

    std::cout << "PASS: training_data_test\n";
    return 0;
}
