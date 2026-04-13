#include "genetic_algorithm/device/dynamic_runtime.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "common/float16.hpp"
#include "genetic_algorithm/output_embedding_injection.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "model/policy_model/policy_model.hpp"
#include "training_folder/training_data.hpp"
#include "wordle/wordle_grid.hpp"

namespace neuroevolution::genetic_algorithm::dynamic_device {

namespace {

using neuroevolution::genetic_algorithm::TrySeedOutputEmbeddingTailFromHintGrids;
using neuroevolution::model::output_embedding::ActionEmbedding;
using neuroevolution::model::output_embedding::ScoreActionEmbedding;
using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::model::policy_model::PolicyVector;
using neuroevolution::model::policy_model::TryForwardPolicyModel;
using neuroevolution::training_folder::DeviceTrainingWordCatalog;
using neuroevolution::training_folder::IsValidTrainingWordCatalog;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::wordle::MakeWordleGrid;
using neuroevolution::wordle::TryAppendGuess;
using neuroevolution::wordle::Word;
using neuroevolution::wordle::WordleGrid;

constexpr float kWinScoreBase = 10.0f;
constexpr std::uint32_t kPlanningRandomSeedSalt = 0x9E37'79B9U;
constexpr std::uint32_t kBreedingRandomSeedSalt = 0x7F4A'7C15U;

NEUROEVOLUTION_HOST_DEVICE constexpr int DeviceStatusValue(const DeviceRuntimeStatusCode status_code) {
    return static_cast<int>(status_code);
}

NEUROEVOLUTION_HOST_DEVICE constexpr bool IsValidRuntimeWordCounts(const TrainingWordCatalog &training_word_catalog,
                                                                   const RuntimeWordCounts &runtime_word_counts) {
    return (runtime_word_counts.training_word_count <= training_word_catalog.word_count) &&
           (runtime_word_counts.action_space_word_count <= training_word_catalog.word_count);
}

NEUROEVOLUTION_HOST_DEVICE constexpr bool
IsValidPendingOutputEmbeddingInjection(const TrainingWordCatalog &training_word_catalog,
                                       const PendingOutputEmbeddingInjection &pending_output_embedding_injection) {
    return !pending_output_embedding_injection.enabled ||
           ((pending_output_embedding_injection.injection_count > 0) &&
            (pending_output_embedding_injection.first_catalog_word_index < training_word_catalog.word_count) &&
            (pending_output_embedding_injection.injection_count <=
             (training_word_catalog.word_count - pending_output_embedding_injection.first_catalog_word_index)));
}

constexpr bool IsValidDynamicGenerationAssemblyConfig(const GenerationAssemblyConfig &config) {
    return (config.genetic_algorithm.elite_count > 0) && IsValidParentSelectionConfig(config.parent_selection) &&
           IsValidBreedingConfig(config.breeding) && IsValidMutationConfig(config.mutation);
}

NEUROEVOLUTION_HOST_DEVICE constexpr std::size_t TailRowSlotCountForPopulationLayout(
    const DynamicPopulationLayout &layout) noexcept {
    return layout.active_individual_count * layout.action_count;
}

struct DeviceRandomState {
    std::uint64_t state = 0;
};

__device__ std::uint64_t NextUInt64(DeviceRandomState &state) {
    std::uint64_t x = state.state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.state = x;
    return x * 2685821657736338717ULL;
}

__device__ DeviceRandomState MakeDeviceRandomState(const std::uint32_t seed, const std::uint32_t stream) {
    DeviceRandomState state{};
    state.state =
        (static_cast<std::uint64_t>(seed) << 32) ^ (static_cast<std::uint64_t>(stream) + 0x9E3779B97F4A7C15ULL);
    if (state.state == 0) {
        state.state = 0xA5A5A5A5ULL;
    }

    (void)NextUInt64(state);
    return state;
}

__device__ DeviceRandomState MakePlanningRandomState(const std::uint32_t generation_seed,
                                                     const std::size_t generation_index,
                                                     const std::size_t slot_index) {
    return MakeDeviceRandomState(generation_seed ^ kPlanningRandomSeedSalt,
                                 static_cast<std::uint32_t>(slot_index + (generation_index * 4099U)));
}

__device__ DeviceRandomState MakeBreedingRandomState(const std::uint32_t generation_seed,
                                                     const std::size_t generation_index,
                                                     const std::size_t slot_index) {
    return MakeDeviceRandomState(generation_seed ^ kBreedingRandomSeedSalt,
                                 static_cast<std::uint32_t>(slot_index + (generation_index * 4099U)));
}

__device__ float NextUniform01(DeviceRandomState &state) {
    const std::uint32_t bits = static_cast<std::uint32_t>(NextUInt64(state) >> 32);
    return (static_cast<float>(bits) + 1.0f) / 4294967297.0f;
}

__device__ bool SampleBernoulli(DeviceRandomState &state, const float probability) {
    return NextUniform01(state) < probability;
}

__device__ std::size_t SampleIndex(DeviceRandomState &state, const std::size_t upper_bound_exclusive) {
    if (upper_bound_exclusive <= 1) {
        return 0;
    }

    return static_cast<std::size_t>(NextUInt64(state) % upper_bound_exclusive);
}

__device__ float SampleStandardNormal(DeviceRandomState &state) {
    constexpr float kTwoPi = 6.28318530717958647692f;
    const float u1 = NextUniform01(state);
    const float u2 = NextUniform01(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(kTwoPi * u2);
}

__device__ void SetFailureStatus(int *status, const DeviceRuntimeStatusCode status_code) {
    atomicCAS(status, DeviceStatusValue(DeviceRuntimeStatusCode::kOk), DeviceStatusValue(status_code));
}

template <std::size_t Size>
__device__ void BreedAndMutateFixedBuffer(const common::FixedBuffer<common::Float16, Size> &first_parent,
                                          const common::FixedBuffer<common::Float16, Size> &second_parent,
                                          common::FixedBuffer<common::Float16, Size> &child,
                                          DeviceRandomState &random_state, const BreedingConfig &breeding_config,
                                          const MutationConfig &mutation_config) {
    for (std::size_t index = 0; index < Size; ++index) {
        float value = SampleBernoulli(random_state, breeding_config.first_parent_probability)
                          ? common::ToFloat(first_parent[index])
                          : common::ToFloat(second_parent[index]);

        if ((mutation_config.mutation_probability > 0.0f) &&
            SampleBernoulli(random_state, mutation_config.mutation_probability) &&
            (mutation_config.mutation_sigma > 0.0f)) {
            value += mutation_config.mutation_sigma * SampleStandardNormal(random_state);
        }

        child[index] = common::ToFloat16(value);
    }
}

template <std::size_t InputSize, std::size_t OutputSize>
__device__ void
BreedAndMutateDenseLayer(const model::input_encoder::DenseLayerParameters<InputSize, OutputSize> &first_parent,
                         const model::input_encoder::DenseLayerParameters<InputSize, OutputSize> &second_parent,
                         model::input_encoder::DenseLayerParameters<InputSize, OutputSize> &child,
                         DeviceRandomState &random_state, const BreedingConfig &breeding_config,
                         const MutationConfig &mutation_config) {
    BreedAndMutateFixedBuffer(first_parent.weights, second_parent.weights, child.weights, random_state, breeding_config,
                              mutation_config);
    BreedAndMutateFixedBuffer(first_parent.biases, second_parent.biases, child.biases, random_state, breeding_config,
                              mutation_config);
}

template <std::size_t InputSize, std::size_t OutputSize>
__device__ void
BreedAndMutateDenseLayer(const model::dense_trunk::DenseLayerParameters<InputSize, OutputSize> &first_parent,
                         const model::dense_trunk::DenseLayerParameters<InputSize, OutputSize> &second_parent,
                         model::dense_trunk::DenseLayerParameters<InputSize, OutputSize> &child,
                         DeviceRandomState &random_state, const BreedingConfig &breeding_config,
                         const MutationConfig &mutation_config) {
    BreedAndMutateFixedBuffer(first_parent.weights, second_parent.weights, child.weights, random_state, breeding_config,
                              mutation_config);
    BreedAndMutateFixedBuffer(first_parent.biases, second_parent.biases, child.biases, random_state, breeding_config,
                              mutation_config);
}

__device__ bool IsBetterFitness(const float candidate_fitness, const std::size_t candidate_index,
                                const float reference_fitness, const std::size_t reference_index) {
    return (candidate_fitness > reference_fitness) ||
           ((candidate_fitness == reference_fitness) && (candidate_index < reference_index));
}

__device__ bool TryFindEliteIndexByRank(const float *fitness_values, const std::uint8_t *has_fitness_flags,
                                        const std::size_t active_population_size, const std::size_t elite_rank,
                                        std::size_t &elite_index) {
    for (std::size_t candidate_index = 0; candidate_index < active_population_size; ++candidate_index) {
        if (has_fitness_flags[candidate_index] == 0) {
            continue;
        }

        std::size_t better_count = 0;
        for (std::size_t other_index = 0; other_index < active_population_size; ++other_index) {
            if ((other_index == candidate_index) || (has_fitness_flags[other_index] == 0)) {
                continue;
            }

            if (IsBetterFitness(fitness_values[other_index], other_index, fitness_values[candidate_index],
                                candidate_index)) {
                ++better_count;
            }
        }

        if (better_count == elite_rank) {
            elite_index = candidate_index;
            return true;
        }
    }

    return false;
}

__device__ bool TrySampleSelectableIndex(const std::uint8_t *has_fitness_flags,
                                         const std::size_t active_population_size, DeviceRandomState &random_state,
                                         const std::size_t excluded_index, std::size_t &selected_index) {
    if (active_population_size == 0) {
        return false;
    }

    const std::size_t start_index = SampleIndex(random_state, active_population_size);
    for (std::size_t offset = 0; offset < active_population_size; ++offset) {
        const std::size_t candidate_index = (start_index + offset) % active_population_size;
        if ((candidate_index == excluded_index) || (has_fitness_flags[candidate_index] == 0)) {
            continue;
        }

        selected_index = candidate_index;
        return true;
    }

    return false;
}

__device__ bool TrySelectParentIndexDevice(const float *fitness_values, const std::uint8_t *has_fitness_flags,
                                           const std::size_t active_population_size, DeviceRandomState &random_state,
                                           const ParentSelectionConfig &config, std::size_t &selected_parent_index,
                                           const std::size_t excluded_index) {
    std::size_t selectable_count = 0;
    for (std::size_t individual_index = 0; individual_index < active_population_size; ++individual_index) {
        if ((individual_index != excluded_index) && (has_fitness_flags[individual_index] != 0)) {
            ++selectable_count;
        }
    }

    if (selectable_count == 0) {
        return false;
    }

    const std::size_t tournament_size =
        (config.tournament_size < selectable_count) ? config.tournament_size : selectable_count;
    if (tournament_size == 0) {
        return false;
    }

    bool found_parent = false;
    float best_fitness = 0.0f;

    for (std::size_t sample_index = 0; sample_index < tournament_size; ++sample_index) {
        std::size_t candidate_index = 0;
        if (!TrySampleSelectableIndex(has_fitness_flags, active_population_size, random_state, excluded_index,
                                      candidate_index)) {
            return false;
        }

        const float candidate_fitness = fitness_values[candidate_index];
        if (!found_parent || IsBetterFitness(candidate_fitness, candidate_index, best_fitness, selected_parent_index)) {
            found_parent = true;
            selected_parent_index = candidate_index;
            best_fitness = candidate_fitness;
        }
    }

    return found_parent;
}

__device__ bool TrySelectParentPairDevice(const float *fitness_values, const std::uint8_t *has_fitness_flags,
                                          const std::size_t active_population_size, DeviceRandomState &random_state,
                                          const ParentSelectionConfig &config, ParentPair &parent_pair) {
    constexpr std::size_t kNoIndividualIndex = static_cast<std::size_t>(-1);

    if (!TrySelectParentIndexDevice(fitness_values, has_fitness_flags, active_population_size, random_state, config,
                                    parent_pair.first_parent_index, kNoIndividualIndex)) {
        return false;
    }

    const std::size_t excluded_index =
        config.allow_self_parenting ? kNoIndividualIndex : parent_pair.first_parent_index;
    return TrySelectParentIndexDevice(fitness_values, has_fitness_flags, active_population_size, random_state, config,
                                      parent_pair.second_parent_index, excluded_index);
}

__device__ float ScoreDynamicActionEmbedding(const PolicyVector &policy_vector, const Word &action_word,
                                             const TrainableActionEmbeddingTail &trainable_tail) {
    ActionEmbedding action_embedding{};
    action_embedding.word = action_word;
    action_embedding.trainable_tail = trainable_tail;
    return ScoreActionEmbedding(policy_vector, action_embedding);
}

__device__ bool TrySelectBestDynamicAction(const PolicyVector &policy_vector,
                                           const TrainingWordCatalog &training_word_catalog,
                                           const ConstDynamicGenomeView genome_view, const std::size_t action_count,
                                           SelectedAction &selected_action) {
    if ((action_count == 0) || (action_count > training_word_catalog.word_count)) {
        return false;
    }

    selected_action.action_index = 0;
    selected_action.word = training_word_catalog.words[0];
    selected_action.score = ScoreDynamicActionEmbedding(policy_vector, selected_action.word, GenomeTailRow(genome_view, 0));

    for (std::size_t action_index = 1; action_index < action_count; ++action_index) {
        const float score = ScoreDynamicActionEmbedding(policy_vector, training_word_catalog.words[action_index],
                                                        GenomeTailRow(genome_view, action_index));
        if (score > selected_action.score) {
            selected_action.action_index = action_index;
            selected_action.word = training_word_catalog.words[action_index];
            selected_action.score = score;
        }
    }

    return true;
}

__device__ std::size_t WrapTrainingWordIndex(const std::size_t index, const std::size_t word_count) {
    return (word_count == 0) ? 0 : (index % word_count);
}

__device__ DeviceRuntimeStatusCode TryInitializePrefilledGrid(const TrainingWordCatalog &training_word_catalog,
                                                              const Word &solution, const std::size_t first_guess_index,
                                                              const std::size_t second_guess_index,
                                                              const std::size_t active_training_word_count,
                                                              WordleGrid &grid_out) {
    grid_out = MakeWordleGrid(solution);

    const Word first_guess = training_word_catalog.words[WrapTrainingWordIndex(first_guess_index, active_training_word_count)];
    const Word second_guess =
        training_word_catalog.words[WrapTrainingWordIndex(second_guess_index, active_training_word_count)];

    if (!TryAppendGuess(grid_out, first_guess) || !TryAppendGuess(grid_out, second_guess)) {
        return DeviceRuntimeStatusCode::kGuessAppendFailed;
    }

    return DeviceRuntimeStatusCode::kOk;
}

__device__ DeviceRuntimeStatusCode TryPlayWordleToCompletion(const ConstDynamicGenomeView genome_view,
                                                             const TrainingWordCatalog &training_word_catalog,
                                                             const std::size_t action_count, WordleGrid &grid,
                                                             float &episode_score_out) {
    while (!grid.IsFinished()) {
        PolicyVector policy_vector{};
        if (!TryForwardPolicyModel(GenomeBodyParameters(genome_view), grid, policy_vector)) {
            return DeviceRuntimeStatusCode::kPolicyForwardFailed;
        }

        SelectedAction selected_action{};
        if (!TrySelectBestDynamicAction(policy_vector, training_word_catalog, genome_view, action_count,
                                        selected_action)) {
            return DeviceRuntimeStatusCode::kActionSelectionFailed;
        }

        if (!TryAppendGuess(grid, selected_action.word)) {
            return DeviceRuntimeStatusCode::kGuessAppendFailed;
        }
    }

    if (!grid.IsWon()) {
        episode_score_out = 0.0f;
        return DeviceRuntimeStatusCode::kOk;
    }

    episode_score_out = kWinScoreBase + static_cast<float>(neuroevolution::wordle::kMaxTurnCount - grid.turn_count);
    return DeviceRuntimeStatusCode::kOk;
}

__device__ DeviceRuntimeStatusCode TryEvaluateIndividualFitness(const ConstDynamicGenomeView genome_view,
                                                                const DynamicPopulationLayout &population_layout,
                                                                const std::size_t generation_index,
                                                                const RuntimeWordCounts runtime_word_counts,
                                                                float &fitness_out) {
    const TrainingWordCatalog &training_word_catalog = DeviceTrainingWordCatalog();
    (void)generation_index;
    const std::size_t active_training_word_count = runtime_word_counts.training_word_count;
    const std::size_t selectable_action_count =
        (population_layout.action_count < runtime_word_counts.action_space_word_count) ? population_layout.action_count
                                                                                       : runtime_word_counts.action_space_word_count;

    if (!IsValidTrainingWordCatalog(training_word_catalog) ||
        !IsValidRuntimeWordCounts(training_word_catalog, runtime_word_counts) ||
        !IsValidDynamicPopulationLayout(population_layout) || (active_training_word_count == 0) ||
        (selectable_action_count == 0) || (selectable_action_count > population_layout.action_count)) {
        return DeviceRuntimeStatusCode::kInvalidTrainingShard;
    }

    float score_sum = 0.0f;

    for (std::size_t entry_index = 0; entry_index < active_training_word_count; ++entry_index) {
        const Word solution = training_word_catalog.words[entry_index];

        {
            WordleGrid fresh_grid = MakeWordleGrid(solution);
            float episode_score = 0.0f;
            const DeviceRuntimeStatusCode episode_status =
                TryPlayWordleToCompletion(genome_view, training_word_catalog, selectable_action_count, fresh_grid,
                                          episode_score);
            if (episode_status != DeviceRuntimeStatusCode::kOk) {
                return episode_status;
            }

            score_sum += episode_score;
        }

        {
            WordleGrid prefilled_grid{};
            const DeviceRuntimeStatusCode initialize_status =
                TryInitializePrefilledGrid(training_word_catalog, solution, entry_index + 1, entry_index + 2,
                                           active_training_word_count, prefilled_grid);
            if (initialize_status != DeviceRuntimeStatusCode::kOk) {
                return initialize_status;
            }

            float episode_score = 0.0f;
            const DeviceRuntimeStatusCode episode_status =
                TryPlayWordleToCompletion(genome_view, training_word_catalog, selectable_action_count, prefilled_grid,
                                          episode_score);
            if (episode_status != DeviceRuntimeStatusCode::kOk) {
                return episode_status;
            }

            score_sum += episode_score;
        }

        {
            WordleGrid prefilled_grid{};
            const DeviceRuntimeStatusCode initialize_status =
                TryInitializePrefilledGrid(training_word_catalog, solution, entry_index + 3, entry_index + 4,
                                           active_training_word_count, prefilled_grid);
            if (initialize_status != DeviceRuntimeStatusCode::kOk) {
                return initialize_status;
            }

            float episode_score = 0.0f;
            const DeviceRuntimeStatusCode episode_status =
                TryPlayWordleToCompletion(genome_view, training_word_catalog, selectable_action_count, prefilled_grid,
                                          episode_score);
            if (episode_status != DeviceRuntimeStatusCode::kOk) {
                return episode_status;
            }

            score_sum += episode_score;
        }
    }

    fitness_out = score_sum;
    return DeviceRuntimeStatusCode::kOk;
}

__device__ void CopyGenome(const ConstDynamicGenomeView source_genome_view, const std::size_t source_action_count,
                           const DynamicGenomeView target_genome_view, const std::size_t target_action_count) {
    GenomeBodyParameters(target_genome_view) = GenomeBodyParameters(source_genome_view);

    const std::size_t copied_action_count =
        (source_action_count < target_action_count) ? source_action_count : target_action_count;
    for (std::size_t action_index = 0; action_index < copied_action_count; ++action_index) {
        GenomeTailRow(target_genome_view, action_index) = GenomeTailRow(source_genome_view, action_index);
    }
}

__device__ void BreedAndMutateGenome(const ConstDynamicGenomeView first_parent_genome_view,
                                     const ConstDynamicGenomeView second_parent_genome_view,
                                     const std::size_t action_count, const DynamicGenomeView child_genome_view,
                                     DeviceRandomState &random_state,
                                     const BreedingConfig &breeding_config,
                                     const MutationConfig &mutation_config) {
    BreedAndMutateDenseLayer(GenomeBodyParameters(first_parent_genome_view).input_encoder.input_to_hidden,
                             GenomeBodyParameters(second_parent_genome_view).input_encoder.input_to_hidden,
                             GenomeBodyParameters(child_genome_view).input_encoder.input_to_hidden, random_state,
                             breeding_config, mutation_config);
    BreedAndMutateDenseLayer(GenomeBodyParameters(first_parent_genome_view).input_encoder.hidden_to_output,
                             GenomeBodyParameters(second_parent_genome_view).input_encoder.hidden_to_output,
                             GenomeBodyParameters(child_genome_view).input_encoder.hidden_to_output, random_state,
                             breeding_config, mutation_config);
    BreedAndMutateDenseLayer(GenomeBodyParameters(first_parent_genome_view).dense_trunk.input_to_hidden0,
                             GenomeBodyParameters(second_parent_genome_view).dense_trunk.input_to_hidden0,
                             GenomeBodyParameters(child_genome_view).dense_trunk.input_to_hidden0, random_state,
                             breeding_config, mutation_config);
    BreedAndMutateDenseLayer(GenomeBodyParameters(first_parent_genome_view).dense_trunk.hidden0_to_hidden1,
                             GenomeBodyParameters(second_parent_genome_view).dense_trunk.hidden0_to_hidden1,
                             GenomeBodyParameters(child_genome_view).dense_trunk.hidden0_to_hidden1, random_state,
                             breeding_config, mutation_config);
    BreedAndMutateDenseLayer(GenomeBodyParameters(first_parent_genome_view).dense_trunk.hidden1_to_output,
                             GenomeBodyParameters(second_parent_genome_view).dense_trunk.hidden1_to_output,
                             GenomeBodyParameters(child_genome_view).dense_trunk.hidden1_to_output, random_state,
                             breeding_config, mutation_config);

    for (std::size_t action_index = 0; action_index < action_count; ++action_index) {
        BreedAndMutateFixedBuffer(GenomeTailRow(first_parent_genome_view, action_index),
                                  GenomeTailRow(second_parent_genome_view, action_index),
                                  GenomeTailRow(child_genome_view, action_index), random_state, breeding_config,
                                  mutation_config);
    }
}

__device__ void MarkIndividualUnevaluated(float *fitness_values, std::uint32_t *evaluation_counts,
                                          std::uint8_t *has_fitness_flags, const std::size_t individual_index) {
    fitness_values[individual_index] = 0.0f;
    evaluation_counts[individual_index] = 0;
    has_fitness_flags[individual_index] = 0;
}

__device__ DeviceRuntimeStatusCode TryApplyPendingOutputEmbeddingInjection(
    const DynamicGenomeView genome_view, const std::size_t current_action_count,
    const PendingOutputEmbeddingInjection pending_output_embedding_injection) {
    if (!pending_output_embedding_injection.enabled) {
        return DeviceRuntimeStatusCode::kOk;
    }

    const TrainingWordCatalog &training_word_catalog = DeviceTrainingWordCatalog();
    if (!IsValidTrainingWordCatalog(training_word_catalog) ||
        !IsValidPendingOutputEmbeddingInjection(training_word_catalog, pending_output_embedding_injection) ||
        (pending_output_embedding_injection.first_catalog_word_index != current_action_count)) {
        return DeviceRuntimeStatusCode::kOutputEmbeddingInjectionFailed;
    }

    for (std::size_t injection_offset = 0; injection_offset < pending_output_embedding_injection.injection_count;
         ++injection_offset) {
        if (!TrySeedOutputEmbeddingTailFromHintGrids(
                GenomeBodyParameters(genome_view),
                training_word_catalog.words[pending_output_embedding_injection.first_catalog_word_index +
                                            injection_offset],
                GenomeTailRow(genome_view, current_action_count + injection_offset))) {
            return DeviceRuntimeStatusCode::kOutputEmbeddingInjectionFailed;
        }
    }

    return DeviceRuntimeStatusCode::kOk;
}

NEUROEVOLUTION_HOST_DEVICE inline ConstDynamicGenomeView CurrentArenaGenomeView(
    const PolicyModelParameters *body_slots, const TrainableActionEmbeddingTail *tail_row_slots,
    const DynamicArenaSlotId *body_slot_ids, const DynamicArenaSlotId *tail_row_slot_ids,
    const std::size_t tail_row_slot_id_stride, const DynamicPopulationLayout &layout,
    const std::size_t individual_index) noexcept {
    return ArenaGenomeView(body_slots, tail_row_slots, body_slot_ids, tail_row_slot_ids, tail_row_slot_id_stride, layout,
                           individual_index);
}

__device__ bool TryBindArenaGenomeView(PolicyModelParameters *body_slots, TrainableActionEmbeddingTail *tail_row_slots,
                                       DynamicArenaSlotId *body_slot_ids, DynamicArenaSlotId *tail_row_slot_ids,
                                       const std::size_t tail_row_slot_id_stride,
                                       const std::size_t body_slot_capacity,
                                       const std::size_t tail_row_slot_capacity,
                                       const DynamicPopulationLayout &layout, const std::size_t individual_index,
                                       DynamicGenomeView &genome_view_out) {
    const DynamicArenaSlotId body_slot_id = body_slot_ids[individual_index];
    if (!IsValidArenaSlotId(body_slot_id, body_slot_capacity)) {
        return false;
    }

    const DynamicArenaSlotId *individual_tail_row_slot_ids =
        TailRowSlotIdsForIndividual(tail_row_slot_ids, tail_row_slot_id_stride, individual_index);
    for (std::size_t action_index = 0; action_index < layout.action_count; ++action_index) {
        if (!IsValidArenaSlotId(individual_tail_row_slot_ids[action_index], tail_row_slot_capacity)) {
            return false;
        }
    }

    genome_view_out =
        ArenaGenomeView(body_slots, tail_row_slots, body_slot_ids, tail_row_slot_ids, tail_row_slot_id_stride, layout,
                        individual_index);
    return true;
}

__device__ bool TryAllocateNextArenaGenomeView(
    PolicyModelParameters *body_slots, TrainableActionEmbeddingTail *tail_row_slots, DynamicArenaSlotId *body_slot_ids,
    DynamicArenaSlotId *tail_row_slot_ids, const std::size_t tail_row_slot_id_stride,
    const std::size_t body_slot_capacity, const std::size_t tail_row_slot_capacity,
    const DynamicPopulationLayout &layout, const std::size_t individual_index, DynamicArenaSlotId *body_free_slot_ids,
    std::uint32_t &body_free_slot_count, DynamicArenaSlotId *tail_row_free_slot_ids,
    std::uint32_t &tail_row_free_slot_count, DynamicGenomeView &genome_view_out) {
    if (!TryAssignArenaGenomeSlotIds(body_slot_ids, tail_row_slot_ids, tail_row_slot_id_stride, layout, individual_index,
                                     body_free_slot_ids, body_free_slot_count, body_slot_capacity, tail_row_free_slot_ids,
                                     tail_row_free_slot_count, tail_row_slot_capacity)) {
        return false;
    }

    return TryBindArenaGenomeView(body_slots, tail_row_slots, body_slot_ids, tail_row_slot_ids, tail_row_slot_id_stride,
                                  body_slot_capacity, tail_row_slot_capacity, layout, individual_index, genome_view_out);
}

__global__ void EvaluatePopulationFitnessKernel(const PolicyModelParameters *body_slots,
                                                const TrainableActionEmbeddingTail *tail_row_slots,
                                                const DynamicArenaSlotId *current_body_slot_ids,
                                                const DynamicArenaSlotId *current_tail_row_slot_ids,
                                                const std::size_t tail_row_slot_id_stride,
                                                const DynamicPopulationLayout current_layout, float *current_fitness,
                                                std::uint32_t *current_evaluation_counts,
                                                std::uint8_t *current_has_fitness,
                                                const RuntimeWordCounts runtime_word_counts, int *status) {
    const std::size_t individual_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (!IsValidDynamicPopulationLayout(current_layout)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceRuntimeStatusCode::kInvalidPopulationLayout);
        }
        return;
    }

    if (individual_index >= current_layout.active_individual_count) {
        return;
    }

    const ConstDynamicGenomeView genome_view = CurrentArenaGenomeView(
        body_slots, tail_row_slots, current_body_slot_ids, current_tail_row_slot_ids, tail_row_slot_id_stride,
        current_layout, individual_index);
    float fitness = 0.0f;
    const DeviceRuntimeStatusCode evaluation_status =
        TryEvaluateIndividualFitness(genome_view, current_layout, current_layout.generation_index, runtime_word_counts,
                                     fitness);
    if (evaluation_status != DeviceRuntimeStatusCode::kOk) {
        SetFailureStatus(status, evaluation_status);
        return;
    }

    current_fitness[individual_index] = fitness;
    ++current_evaluation_counts[individual_index];
    current_has_fitness[individual_index] = 1;
}

__global__ void SummarizePopulationKernel(const float *current_fitness, const std::uint8_t *current_has_fitness,
                                          const DynamicPopulationLayout current_layout,
                                          PopulationFitnessSummary *summary, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    if (!IsValidDynamicPopulationLayout(current_layout)) {
        SetFailureStatus(status, DeviceRuntimeStatusCode::kInvalidPopulationLayout);
        return;
    }

    bool found_best = false;
    float best_fitness = 0.0f;
    float fitness_sum = 0.0f;
    std::size_t best_index = 0;

    for (std::size_t individual_index = 0; individual_index < current_layout.active_individual_count; ++individual_index) {
        if (current_has_fitness[individual_index] == 0) {
            SetFailureStatus(status, DeviceRuntimeStatusCode::kPopulationNotEvaluated);
            return;
        }

        fitness_sum += current_fitness[individual_index];

        if (!found_best || IsBetterFitness(current_fitness[individual_index], individual_index, best_fitness, best_index)) {
            found_best = true;
            best_fitness = current_fitness[individual_index];
            best_index = individual_index;
        }
    }

    summary->best_fitness = best_fitness;
    summary->average_fitness = fitness_sum / static_cast<float>(current_layout.active_individual_count);
    summary->best_index = best_index;
    summary->generation_index = current_layout.generation_index;
    summary->action_count = current_layout.action_count;
    summary->population_size = current_layout.active_individual_count;
}

__global__ void PlanNextGenerationKernel(const float *current_fitness, const std::uint8_t *current_has_fitness,
                                         const DynamicPopulationLayout current_layout,
                                         const DynamicPopulationLayout next_layout, const std::uint32_t generation_seed,
                                         const GenerationAssemblyConfig config,
                                         PlannedGenerationMember *next_generation_plan, int *status) {
    const std::size_t slot_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (!IsValidDynamicPopulationLayout(current_layout) || !IsValidDynamicPopulationLayout(next_layout)) {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
            SetFailureStatus(status, DeviceRuntimeStatusCode::kInvalidPopulationLayout);
        }
        return;
    }

    if (slot_index >= next_layout.active_individual_count) {
        return;
    }

    if (slot_index < config.genetic_algorithm.elite_count) {
        std::size_t elite_index = 0;
        if (!TryFindEliteIndexByRank(current_fitness, current_has_fitness, current_layout.active_individual_count,
                                     slot_index, elite_index)) {
            SetFailureStatus(status, DeviceRuntimeStatusCode::kPopulationNotEvaluated);
            return;
        }

        next_generation_plan[slot_index] = MakeEliteCopyPlannedGenerationMember(elite_index);
        return;
    }

    DeviceRandomState random_state =
        MakePlanningRandomState(generation_seed, current_layout.generation_index, slot_index);

    ParentPair parent_pair{};
    if (!TrySelectParentPairDevice(current_fitness, current_has_fitness, current_layout.active_individual_count,
                                   random_state, config.parent_selection, parent_pair)) {
        SetFailureStatus(status, DeviceRuntimeStatusCode::kParentSelectionFailed);
        return;
    }

    next_generation_plan[slot_index] = MakeBreedingPlannedGenerationMember(parent_pair);
}

__global__ void CountCurrentParentRemainingUsesKernel(const PlannedGenerationMember *next_generation_plan,
                                                      const std::size_t next_generation_member_count,
                                                      const std::size_t current_population_size,
                                                      std::uint32_t *current_parent_remaining_use_counts, int *status) {
    const std::size_t slot_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (slot_index >= next_generation_member_count) {
        return;
    }

    const PlannedGenerationMember planned_member = next_generation_plan[slot_index];
    if (!IsValidPlannedGenerationMember(planned_member, current_population_size)) {
        SetFailureStatus(status, DeviceRuntimeStatusCode::kParentSelectionFailed);
        return;
    }

    atomicAdd(&current_parent_remaining_use_counts[planned_member.parent_pair.first_parent_index], 1U);
    if (planned_member.operation == PlannedGenerationMemberOperation::kBreedChild) {
        atomicAdd(&current_parent_remaining_use_counts[planned_member.parent_pair.second_parent_index], 1U);
    }
}

__global__ void InitializeArenaFreeSlotStacksKernel(
    const DynamicArenaSlotId *current_body_slot_ids, const DynamicArenaSlotId *current_tail_row_slot_ids,
    const std::uint32_t *current_parent_remaining_use_counts, const std::size_t tail_row_slot_id_stride,
    const DynamicPopulationLayout current_layout, const std::size_t body_slot_capacity,
    const std::size_t tail_row_slot_capacity, std::uint8_t *body_slot_live_flags, std::uint8_t *tail_row_slot_live_flags,
    DynamicArenaSlotId *body_free_slot_ids, DynamicArenaSlotId *tail_row_free_slot_ids, std::uint32_t *body_free_slot_count,
    std::uint32_t *tail_row_free_slot_count, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    if (!IsValidDynamicPopulationLayout(current_layout)) {
        SetFailureStatus(status, DeviceRuntimeStatusCode::kInvalidPopulationLayout);
        return;
    }

    std::uint32_t body_free_count_local = 0;
    std::uint32_t tail_row_free_count_local = 0;

    for (std::size_t parent_index = 0; parent_index < current_layout.active_individual_count; ++parent_index) {
        if (current_parent_remaining_use_counts[parent_index] == 0) {
            continue;
        }

        const DynamicArenaSlotId body_slot_id = current_body_slot_ids[parent_index];
        if (!IsValidArenaSlotId(body_slot_id, body_slot_capacity)) {
            SetFailureStatus(status, DeviceRuntimeStatusCode::kInvalidPopulationLayout);
            return;
        }

        body_slot_live_flags[body_slot_id] = 1;

        const DynamicArenaSlotId *individual_tail_row_slot_ids =
            TailRowSlotIdsForIndividual(current_tail_row_slot_ids, tail_row_slot_id_stride, parent_index);
        for (std::size_t action_index = 0; action_index < current_layout.action_count; ++action_index) {
            const DynamicArenaSlotId tail_row_slot_id = individual_tail_row_slot_ids[action_index];
            if (!IsValidArenaSlotId(tail_row_slot_id, tail_row_slot_capacity)) {
                SetFailureStatus(status, DeviceRuntimeStatusCode::kInvalidPopulationLayout);
                return;
            }

            tail_row_slot_live_flags[tail_row_slot_id] = 1;
        }
    }

    for (std::size_t slot_index = 0; slot_index < body_slot_capacity; ++slot_index) {
        if (body_slot_live_flags[slot_index] != 0) {
            continue;
        }

        body_free_slot_ids[body_free_count_local] = static_cast<DynamicArenaSlotId>(slot_index);
        ++body_free_count_local;
    }

    for (std::size_t slot_index = 0; slot_index < tail_row_slot_capacity; ++slot_index) {
        if (tail_row_slot_live_flags[slot_index] != 0) {
            continue;
        }

        tail_row_free_slot_ids[tail_row_free_count_local] = static_cast<DynamicArenaSlotId>(slot_index);
        ++tail_row_free_count_local;
    }

    *body_free_slot_count = body_free_count_local;
    *tail_row_free_slot_count = tail_row_free_count_local;
}

__global__ void AssembleNextGenerationKernel(
    PolicyModelParameters *body_slots, TrainableActionEmbeddingTail *tail_row_slots,
    DynamicArenaSlotId *current_body_slot_ids, DynamicArenaSlotId *current_tail_row_slot_ids,
    DynamicArenaSlotId *next_body_slot_ids, DynamicArenaSlotId *next_tail_row_slot_ids,
    DynamicArenaSlotId *body_free_slot_ids, DynamicArenaSlotId *tail_row_free_slot_ids,
    std::uint32_t *body_free_slot_count_device, std::uint32_t *tail_row_free_slot_count_device,
    std::uint32_t *current_parent_remaining_use_counts, const std::size_t tail_row_slot_id_stride,
    const std::size_t body_slot_capacity, const std::size_t tail_row_slot_capacity,
    const DynamicPopulationLayout current_layout, float *next_fitness, std::uint32_t *next_evaluation_counts,
    std::uint8_t *next_has_fitness, const DynamicPopulationLayout next_layout,
    const PlannedGenerationMember *next_generation_plan, const std::uint32_t generation_seed,
    const GenerationAssemblyConfig config,
    const PendingOutputEmbeddingInjection pending_output_embedding_injection, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    if (!IsValidDynamicPopulationLayout(current_layout) || !IsValidDynamicPopulationLayout(next_layout)) {
        SetFailureStatus(status, DeviceRuntimeStatusCode::kInvalidPopulationLayout);
        return;
    }

    std::uint32_t body_free_slot_count = *body_free_slot_count_device;
    std::uint32_t tail_row_free_slot_count = *tail_row_free_slot_count_device;

    for (std::size_t slot_index = 0; slot_index < next_layout.active_individual_count; ++slot_index) {
        const PlannedGenerationMember planned_member = next_generation_plan[slot_index];
        if (!IsValidPlannedGenerationMember(planned_member, current_layout.active_individual_count)) {
            SetFailureStatus(status, DeviceRuntimeStatusCode::kParentSelectionFailed);
            return;
        }

        DynamicGenomeView child_genome_view{};
        if (!TryAllocateNextArenaGenomeView(body_slots, tail_row_slots, next_body_slot_ids, next_tail_row_slot_ids,
                                            tail_row_slot_id_stride, body_slot_capacity, tail_row_slot_capacity,
                                            next_layout, slot_index, body_free_slot_ids, body_free_slot_count,
                                            tail_row_free_slot_ids, tail_row_free_slot_count, child_genome_view)) {
            SetFailureStatus(status, DeviceRuntimeStatusCode::kArenaExhausted);
            return;
        }

        if (planned_member.operation == PlannedGenerationMemberOperation::kEliteCopy) {
            const ConstDynamicGenomeView elite_genome_view = CurrentArenaGenomeView(
                body_slots, tail_row_slots, current_body_slot_ids, current_tail_row_slot_ids, tail_row_slot_id_stride,
                current_layout, planned_member.parent_pair.first_parent_index);
            CopyGenome(elite_genome_view, current_layout.action_count, child_genome_view, next_layout.action_count);
            const DeviceRuntimeStatusCode injection_status = TryApplyPendingOutputEmbeddingInjection(
                child_genome_view, current_layout.action_count, pending_output_embedding_injection);
            if (injection_status != DeviceRuntimeStatusCode::kOk) {
                SetFailureStatus(status, injection_status);
                return;
            }

            if (!TryConsumeParentUseAndMaybeRecycleArenaGenomeSlotIds(
                    planned_member.parent_pair.first_parent_index, current_parent_remaining_use_counts,
                    current_body_slot_ids, current_tail_row_slot_ids, tail_row_slot_id_stride, current_layout,
                    body_free_slot_ids, body_free_slot_count, body_slot_capacity, tail_row_free_slot_ids,
                    tail_row_free_slot_count, tail_row_slot_capacity)) {
                SetFailureStatus(status, DeviceRuntimeStatusCode::kInvalidPopulationLayout);
                return;
            }

            MarkIndividualUnevaluated(next_fitness, next_evaluation_counts, next_has_fitness, slot_index);
            continue;
        }

        DeviceRandomState random_state =
            MakeBreedingRandomState(generation_seed, current_layout.generation_index, slot_index);

        const ConstDynamicGenomeView first_parent_genome_view = CurrentArenaGenomeView(
            body_slots, tail_row_slots, current_body_slot_ids, current_tail_row_slot_ids, tail_row_slot_id_stride,
            current_layout, planned_member.parent_pair.first_parent_index);
        const ConstDynamicGenomeView second_parent_genome_view = CurrentArenaGenomeView(
            body_slots, tail_row_slots, current_body_slot_ids, current_tail_row_slot_ids, tail_row_slot_id_stride,
            current_layout, planned_member.parent_pair.second_parent_index);
        BreedAndMutateGenome(first_parent_genome_view, second_parent_genome_view, current_layout.action_count,
                             child_genome_view, random_state, config.breeding, config.mutation);

        const DeviceRuntimeStatusCode injection_status = TryApplyPendingOutputEmbeddingInjection(
            child_genome_view, current_layout.action_count, pending_output_embedding_injection);
        if (injection_status != DeviceRuntimeStatusCode::kOk) {
            SetFailureStatus(status, injection_status);
            return;
        }

        if (!TryConsumeParentUseAndMaybeRecycleArenaGenomeSlotIds(
                planned_member.parent_pair.first_parent_index, current_parent_remaining_use_counts,
                current_body_slot_ids, current_tail_row_slot_ids, tail_row_slot_id_stride, current_layout,
                body_free_slot_ids, body_free_slot_count, body_slot_capacity, tail_row_free_slot_ids,
                tail_row_free_slot_count, tail_row_slot_capacity) ||
            !TryConsumeParentUseAndMaybeRecycleArenaGenomeSlotIds(
                planned_member.parent_pair.second_parent_index, current_parent_remaining_use_counts,
                current_body_slot_ids, current_tail_row_slot_ids, tail_row_slot_id_stride, current_layout,
                body_free_slot_ids, body_free_slot_count, body_slot_capacity, tail_row_free_slot_ids,
                tail_row_free_slot_count, tail_row_slot_capacity)) {
            SetFailureStatus(status, DeviceRuntimeStatusCode::kInvalidPopulationLayout);
            return;
        }

        MarkIndividualUnevaluated(next_fitness, next_evaluation_counts, next_has_fitness, slot_index);
    }

    *body_free_slot_count_device = body_free_slot_count;
    *tail_row_free_slot_count_device = tail_row_free_slot_count;
}

bool CheckCuda(const cudaError_t error) { return error == cudaSuccess; }

bool ReadDeviceStatus(const DeviceRuntimeBuffers &buffers, int &status_value) {
    return CheckCuda(cudaMemcpy(&status_value, buffers.status, sizeof(int), cudaMemcpyDeviceToHost));
}

bool WriteDeviceStatus(const DeviceRuntimeBuffers &buffers, const DeviceRuntimeStatusCode status_code) {
    const int status_value = DeviceStatusValue(status_code);
    return CheckCuda(cudaMemcpy(buffers.status, &status_value, sizeof(int), cudaMemcpyHostToDevice));
}

bool ResetDeviceStatus(const DeviceRuntimeBuffers &buffers) {
    return WriteDeviceStatus(buffers, DeviceRuntimeStatusCode::kOk);
}

bool KernelCompletedSuccessfully(const DeviceRuntimeBuffers &buffers) {
    int status_value = DeviceStatusValue(DeviceRuntimeStatusCode::kCudaFailure);
    return ReadDeviceStatus(buffers, status_value) && (status_value == DeviceStatusValue(DeviceRuntimeStatusCode::kOk));
}

bool TryPlanNextPopulationLayout(const DeviceRuntimeBuffers &buffers,
                                 const PendingOutputEmbeddingInjection &pending_output_embedding_injection,
                                 DynamicPopulationLayout &next_layout) {
    next_layout = {};

    const std::size_t next_action_count =
        buffers.current_layout.action_count +
        (pending_output_embedding_injection.enabled ? pending_output_embedding_injection.injection_count : 0);
    if ((next_action_count == 0) || (next_action_count > buffers.max_action_count) ||
        (buffers.current_layout.active_individual_count == 0)) {
        return false;
    }

    const std::size_t next_schema_epoch =
        buffers.current_layout.schema_epoch + (pending_output_embedding_injection.enabled ? 1U : 0U);
    next_layout = MakeDynamicPopulationLayout(buffers.current_layout.active_individual_count,
                                              buffers.current_layout.generation_index + 1, next_action_count,
                                              buffers.tail_chunk_action_capacity, next_schema_epoch);
    return IsValidDynamicPopulationLayout(next_layout);
}

std::size_t ComputeMaxTailRowSlotCapacity(const std::size_t max_population_count, const DeviceRuntimeConfig &config) {
    if ((max_population_count == 0) || (config.max_action_count == 0) || (config.max_action_count < config.initial_action_count)) {
        return 0;
    }

    return max_population_count * config.max_action_count;
}

bool ResetNextGenerationStorage(const DeviceRuntimeBuffers &buffers) {
    bool ok = true;
    ok &= CheckCuda(cudaMemset(buffers.next_body_slot_ids, 0xFF, buffers.max_population_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(
        cudaMemset(buffers.next_tail_row_slot_ids, 0xFF,
                   buffers.max_population_count * buffers.max_action_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(cudaMemset(buffers.next_fitness, 0, buffers.max_population_count * sizeof(float)));
    ok &= CheckCuda(cudaMemset(buffers.next_evaluation_counts, 0, buffers.max_population_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.next_has_fitness, 0, buffers.max_population_count * sizeof(std::uint8_t)));
    return ok;
}

bool ResetArenaReuseState(const DeviceRuntimeBuffers &buffers) {
    bool ok = true;
    ok &= CheckCuda(cudaMemset(buffers.body_free_slot_ids, 0xFF, buffers.body_slot_capacity * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(
        cudaMemset(buffers.tail_row_free_slot_ids, 0xFF, buffers.tail_row_slot_capacity * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(cudaMemset(buffers.body_slot_live_flags, 0, buffers.body_slot_capacity * sizeof(std::uint8_t)));
    ok &= CheckCuda(cudaMemset(buffers.tail_row_slot_live_flags, 0, buffers.tail_row_slot_capacity * sizeof(std::uint8_t)));
    ok &= CheckCuda(cudaMemset(buffers.body_free_slot_count, 0, sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.tail_row_free_slot_count, 0, sizeof(std::uint32_t)));
    return ok;
}

bool ResetMatingPlanStorage(const DeviceRuntimeBuffers &buffers) {
    bool ok = true;
    ok &= CheckCuda(
        cudaMemset(buffers.next_generation_plan, 0xFF, buffers.max_population_count * sizeof(PlannedGenerationMember)));
    ok &= CheckCuda(cudaMemset(buffers.current_parent_remaining_use_counts, 0,
                               buffers.max_population_count * sizeof(std::uint32_t)));
    return ok;
}

bool TryUploadHostPopulationIntoCurrentArena(const HostPopulation &host_population, DeviceRuntimeBuffers &buffers) {
    const std::size_t population_size = host_population.layout.active_individual_count;
    const std::size_t action_count = host_population.layout.action_count;

    std::vector<PolicyModelParameters> host_body_slots(buffers.body_slot_capacity);
    std::vector<TrainableActionEmbeddingTail> host_tail_row_slots(buffers.tail_row_slot_capacity);
    std::vector<DynamicArenaSlotId> host_body_slot_ids(buffers.max_population_count, kInvalidDynamicArenaSlotId);
    std::vector<DynamicArenaSlotId> host_tail_row_slot_ids(buffers.max_population_count * buffers.max_action_count,
                                                           kInvalidDynamicArenaSlotId);

    for (std::size_t individual_index = 0; individual_index < population_size; ++individual_index) {
        const ConstDynamicGenomeView genome_view = HostGenomeViewAt(host_population, individual_index);
        host_body_slots[individual_index] = GenomeBodyParameters(genome_view);
        host_body_slot_ids[individual_index] = static_cast<DynamicArenaSlotId>(individual_index);

        DynamicArenaSlotId *individual_tail_row_slot_ids =
            TailRowSlotIdsForIndividual(host_tail_row_slot_ids.data(), buffers.max_action_count, individual_index);
        for (std::size_t action_index = 0; action_index < action_count; ++action_index) {
            host_tail_row_slots[(individual_index * action_count) + action_index] = GenomeTailRow(genome_view, action_index);
            individual_tail_row_slot_ids[action_index] =
                static_cast<DynamicArenaSlotId>((individual_index * action_count) + action_index);
        }
    }

    bool ok = true;
    ok &= CheckCuda(cudaMemcpy(buffers.body_slots, host_body_slots.data(),
                               host_body_slots.size() * sizeof(PolicyModelParameters), cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.tail_row_slots, host_tail_row_slots.data(),
                               host_tail_row_slots.size() * sizeof(TrainableActionEmbeddingTail), cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.current_body_slot_ids, host_body_slot_ids.data(),
                               host_body_slot_ids.size() * sizeof(DynamicArenaSlotId), cudaMemcpyHostToDevice));
    ok &= CheckCuda(cudaMemcpy(buffers.current_tail_row_slot_ids, host_tail_row_slot_ids.data(),
                               host_tail_row_slot_ids.size() * sizeof(DynamicArenaSlotId), cudaMemcpyHostToDevice));
    return ok;
}

bool TryPlanNextGenerationMatingPlanOnDevice(DeviceRuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                             const GenerationAssemblyConfig &config) {
    if (!ResetMatingPlanStorage(buffers)) {
        return false;
    }

    const std::size_t block_count =
        (buffers.next_layout.active_individual_count + kDynamicThreadBlockSize - 1) / kDynamicThreadBlockSize;

    PlanNextGenerationKernel<<<block_count, kDynamicThreadBlockSize>>>(
        buffers.current_fitness, buffers.current_has_fitness, buffers.current_layout, buffers.next_layout, generation_seed,
        config, buffers.next_generation_plan, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    if (!KernelCompletedSuccessfully(buffers) || !ResetDeviceStatus(buffers)) {
        return false;
    }

    CountCurrentParentRemainingUsesKernel<<<block_count, kDynamicThreadBlockSize>>>(
        buffers.next_generation_plan, buffers.next_layout.active_individual_count,
        buffers.current_layout.active_individual_count, buffers.current_parent_remaining_use_counts, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    return KernelCompletedSuccessfully(buffers);
}

bool TryInitializeArenaFreeSlotStacksOnDevice(DeviceRuntimeBuffers &buffers) {
    if (!ResetArenaReuseState(buffers) || !ResetDeviceStatus(buffers)) {
        return false;
    }

    InitializeArenaFreeSlotStacksKernel<<<1, 1>>>(
        buffers.current_body_slot_ids, buffers.current_tail_row_slot_ids, buffers.current_parent_remaining_use_counts,
        buffers.max_action_count, buffers.current_layout, buffers.body_slot_capacity, buffers.tail_row_slot_capacity,
        buffers.body_slot_live_flags, buffers.tail_row_slot_live_flags, buffers.body_free_slot_ids,
        buffers.tail_row_free_slot_ids, buffers.body_free_slot_count, buffers.tail_row_free_slot_count, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    return KernelCompletedSuccessfully(buffers);
}

} // namespace

bool TryCreateDeviceRuntimeBuffers(DeviceRuntimeBuffers &buffers, const DeviceRuntimeConfig &config) {
    buffers = {};

    const std::size_t max_population_count = PopulationSizeForGenotypeBudgetBytes(
        config.genotype_memory_budget_bytes, config.initial_action_count, config.population_size_ceiling);
    const std::size_t max_tail_row_slot_capacity_per_generation =
        ComputeMaxTailRowSlotCapacity(max_population_count, config);
    if ((config.genotype_memory_budget_bytes == 0) || (config.initial_action_count == 0) ||
        (config.max_action_count < config.initial_action_count) || (config.tail_chunk_action_capacity == 0) ||
        (max_population_count == 0) || (max_tail_row_slot_capacity_per_generation == 0)) {
        return false;
    }

    buffers.genotype_memory_budget_bytes = config.genotype_memory_budget_bytes;
    buffers.population_size_ceiling = config.population_size_ceiling;
    buffers.max_population_count = max_population_count;
    buffers.max_action_count = config.max_action_count;
    buffers.tail_chunk_action_capacity = config.tail_chunk_action_capacity;
    buffers.body_slot_capacity = 2 * max_population_count;
    buffers.tail_row_slot_capacity = 2 * max_tail_row_slot_capacity_per_generation;

    bool ok = true;
    ok &= CheckCuda(cudaMalloc(&buffers.body_slots, buffers.body_slot_capacity * sizeof(PolicyModelParameters)));
    ok &= CheckCuda(cudaMalloc(&buffers.tail_row_slots,
                               buffers.tail_row_slot_capacity * sizeof(TrainableActionEmbeddingTail)));
    ok &= CheckCuda(cudaMalloc(&buffers.current_body_slot_ids, buffers.max_population_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(cudaMalloc(&buffers.next_body_slot_ids, buffers.max_population_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(cudaMalloc(&buffers.current_tail_row_slot_ids,
                               buffers.max_population_count * buffers.max_action_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(cudaMalloc(&buffers.next_tail_row_slot_ids,
                               buffers.max_population_count * buffers.max_action_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(cudaMalloc(&buffers.body_free_slot_ids, buffers.body_slot_capacity * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(cudaMalloc(&buffers.tail_row_free_slot_ids,
                               buffers.tail_row_slot_capacity * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(cudaMalloc(&buffers.body_slot_live_flags, buffers.body_slot_capacity * sizeof(std::uint8_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.tail_row_slot_live_flags, buffers.tail_row_slot_capacity * sizeof(std::uint8_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.body_free_slot_count, sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.tail_row_free_slot_count, sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.next_generation_plan,
                               buffers.max_population_count * sizeof(PlannedGenerationMember)));
    ok &= CheckCuda(cudaMalloc(&buffers.current_parent_remaining_use_counts,
                               buffers.max_population_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.current_fitness, buffers.max_population_count * sizeof(float)));
    ok &= CheckCuda(cudaMalloc(&buffers.next_fitness, buffers.max_population_count * sizeof(float)));
    ok &= CheckCuda(cudaMalloc(&buffers.current_evaluation_counts, buffers.max_population_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.next_evaluation_counts, buffers.max_population_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.current_has_fitness, buffers.max_population_count * sizeof(std::uint8_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.next_has_fitness, buffers.max_population_count * sizeof(std::uint8_t)));
    ok &= CheckCuda(cudaMalloc(&buffers.summary, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMalloc(&buffers.status, sizeof(int)));

    if (!ok) {
        DestroyDeviceRuntimeBuffers(buffers);
        return false;
    }

    ok &= CheckCuda(cudaMemset(buffers.body_slots, 0, buffers.body_slot_capacity * sizeof(PolicyModelParameters)));
    ok &= CheckCuda(
        cudaMemset(buffers.tail_row_slots, 0, buffers.tail_row_slot_capacity * sizeof(TrainableActionEmbeddingTail)));
    ok &= CheckCuda(cudaMemset(buffers.current_body_slot_ids, 0xFF, buffers.max_population_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(cudaMemset(buffers.next_body_slot_ids, 0xFF, buffers.max_population_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(
        cudaMemset(buffers.current_tail_row_slot_ids, 0xFF,
                   buffers.max_population_count * buffers.max_action_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(
        cudaMemset(buffers.next_tail_row_slot_ids, 0xFF,
                   buffers.max_population_count * buffers.max_action_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(cudaMemset(buffers.body_free_slot_ids, 0xFF, buffers.body_slot_capacity * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(
        cudaMemset(buffers.tail_row_free_slot_ids, 0xFF, buffers.tail_row_slot_capacity * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(cudaMemset(buffers.body_slot_live_flags, 0, buffers.body_slot_capacity * sizeof(std::uint8_t)));
    ok &= CheckCuda(cudaMemset(buffers.tail_row_slot_live_flags, 0, buffers.tail_row_slot_capacity * sizeof(std::uint8_t)));
    ok &= CheckCuda(cudaMemset(buffers.body_free_slot_count, 0, sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.tail_row_free_slot_count, 0, sizeof(std::uint32_t)));
    ok &= CheckCuda(
        cudaMemset(buffers.next_generation_plan, 0xFF, buffers.max_population_count * sizeof(PlannedGenerationMember)));
    ok &= CheckCuda(cudaMemset(buffers.current_parent_remaining_use_counts, 0,
                               buffers.max_population_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.current_fitness, 0, buffers.max_population_count * sizeof(float)));
    ok &= CheckCuda(cudaMemset(buffers.next_fitness, 0, buffers.max_population_count * sizeof(float)));
    ok &= CheckCuda(
        cudaMemset(buffers.current_evaluation_counts, 0, buffers.max_population_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.next_evaluation_counts, 0, buffers.max_population_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.current_has_fitness, 0, buffers.max_population_count * sizeof(std::uint8_t)));
    ok &= CheckCuda(cudaMemset(buffers.next_has_fitness, 0, buffers.max_population_count * sizeof(std::uint8_t)));
    ok &= CheckCuda(cudaMemset(buffers.summary, 0, sizeof(PopulationFitnessSummary)));
    ok &= CheckCuda(cudaMemset(buffers.status, 0, sizeof(int)));
    return ok;
}

void DestroyDeviceRuntimeBuffers(DeviceRuntimeBuffers &buffers) noexcept {
    cudaFree(buffers.body_slots);
    cudaFree(buffers.tail_row_slots);
    cudaFree(buffers.current_body_slot_ids);
    cudaFree(buffers.next_body_slot_ids);
    cudaFree(buffers.current_tail_row_slot_ids);
    cudaFree(buffers.next_tail_row_slot_ids);
    cudaFree(buffers.body_free_slot_ids);
    cudaFree(buffers.tail_row_free_slot_ids);
    cudaFree(buffers.body_slot_live_flags);
    cudaFree(buffers.tail_row_slot_live_flags);
    cudaFree(buffers.body_free_slot_count);
    cudaFree(buffers.tail_row_free_slot_count);
    cudaFree(buffers.next_generation_plan);
    cudaFree(buffers.current_parent_remaining_use_counts);
    cudaFree(buffers.current_fitness);
    cudaFree(buffers.next_fitness);
    cudaFree(buffers.current_evaluation_counts);
    cudaFree(buffers.next_evaluation_counts);
    cudaFree(buffers.current_has_fitness);
    cudaFree(buffers.next_has_fitness);
    cudaFree(buffers.summary);
    cudaFree(buffers.status);
    buffers = {};
}

bool TryUploadCurrentPopulationToDevice(const HostPopulation &host_population, DeviceRuntimeBuffers &buffers) {
    const DynamicPopulationLayout current_layout = MakeDynamicPopulationLayout(
        host_population.layout.active_individual_count, host_population.layout.generation_index,
        host_population.layout.action_count, buffers.tail_chunk_action_capacity, host_population.layout.schema_epoch);
    if (!IsValidDynamicPopulationLayout(host_population.layout) || !IsValidDynamicPopulationLayout(current_layout) ||
        (host_population.layout.genotype_bytes > buffers.genotype_memory_budget_bytes) ||
        (host_population.layout.active_individual_count > buffers.max_population_count) ||
        (host_population.layout.action_count > buffers.max_action_count) || (host_population.genomes == nullptr) ||
        (TailRowSlotCountForPopulationLayout(current_layout) > buffers.tail_row_slot_capacity)) {
        return false;
    }

    buffers.current_layout = current_layout;
    buffers.next_layout = {};

    bool ok = true;
    ok &= CheckCuda(cudaMemset(buffers.body_slots, 0, buffers.body_slot_capacity * sizeof(PolicyModelParameters)));
    ok &= CheckCuda(
        cudaMemset(buffers.tail_row_slots, 0, buffers.tail_row_slot_capacity * sizeof(TrainableActionEmbeddingTail)));
    ok &= CheckCuda(
        cudaMemset(buffers.current_body_slot_ids, 0xFF, buffers.max_population_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(
        cudaMemset(buffers.current_tail_row_slot_ids, 0xFF,
                   buffers.max_population_count * buffers.max_action_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(
        cudaMemset(buffers.next_body_slot_ids, 0xFF, buffers.max_population_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(
        cudaMemset(buffers.next_tail_row_slot_ids, 0xFF,
                   buffers.max_population_count * buffers.max_action_count * sizeof(DynamicArenaSlotId)));
    ok &= CheckCuda(cudaMemset(buffers.current_fitness, 0, buffers.max_population_count * sizeof(float)));
    ok &= CheckCuda(cudaMemset(buffers.next_fitness, 0, buffers.max_population_count * sizeof(float)));
    ok &= CheckCuda(
        cudaMemset(buffers.current_evaluation_counts, 0, buffers.max_population_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.next_evaluation_counts, 0, buffers.max_population_count * sizeof(std::uint32_t)));
    ok &= CheckCuda(cudaMemset(buffers.current_has_fitness, 0, buffers.max_population_count * sizeof(std::uint8_t)));
    ok &= CheckCuda(cudaMemset(buffers.next_has_fitness, 0, buffers.max_population_count * sizeof(std::uint8_t)));
    ok &= ResetMatingPlanStorage(buffers);
    ok &= ResetArenaReuseState(buffers);
    if (!ok) {
        return false;
    }

    return TryUploadHostPopulationIntoCurrentArena(host_population, buffers);
}

bool TryDownloadCurrentPopulationFromDevice(const DeviceRuntimeBuffers &buffers, HostPopulation &host_population) {
    if (!IsValidDynamicPopulationLayout(buffers.current_layout)) {
        return false;
    }

    host_population = {};
    host_population.layout = MakeDynamicPopulationLayout(buffers.current_layout.active_individual_count,
                                                         buffers.current_layout.generation_index,
                                                         buffers.current_layout.action_count,
                                                         buffers.current_layout.tail_chunk_action_capacity,
                                                         buffers.current_layout.schema_epoch);
    if (!TryAllocateHostGenomeStorage(host_population)) {
        return false;
    }

    std::vector<PolicyModelParameters> host_body_slots(buffers.body_slot_capacity);
    std::vector<TrainableActionEmbeddingTail> host_tail_row_slots(buffers.tail_row_slot_capacity);
    std::vector<DynamicArenaSlotId> host_body_slot_ids(buffers.max_population_count, kInvalidDynamicArenaSlotId);
    std::vector<DynamicArenaSlotId> host_tail_row_slot_ids(buffers.max_population_count * buffers.max_action_count,
                                                           kInvalidDynamicArenaSlotId);

    bool ok = true;
    ok &= CheckCuda(cudaMemcpy(host_body_slots.data(), buffers.body_slots,
                               host_body_slots.size() * sizeof(PolicyModelParameters), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(host_tail_row_slots.data(), buffers.tail_row_slots,
                               host_tail_row_slots.size() * sizeof(TrainableActionEmbeddingTail),
                               cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(host_body_slot_ids.data(), buffers.current_body_slot_ids,
                               host_body_slot_ids.size() * sizeof(DynamicArenaSlotId), cudaMemcpyDeviceToHost));
    ok &= CheckCuda(cudaMemcpy(host_tail_row_slot_ids.data(), buffers.current_tail_row_slot_ids,
                               host_tail_row_slot_ids.size() * sizeof(DynamicArenaSlotId), cudaMemcpyDeviceToHost));
    if (!ok) {
        return false;
    }

    for (std::size_t individual_index = 0; individual_index < host_population.layout.active_individual_count;
         ++individual_index) {
        const DynamicGenomeView genome_view = HostGenomeViewAt(host_population, individual_index);
        const DynamicArenaSlotId body_slot_id = host_body_slot_ids[individual_index];
        if (!IsValidArenaSlotId(body_slot_id, buffers.body_slot_capacity)) {
            return false;
        }

        GenomeBodyParameters(genome_view) = host_body_slots[body_slot_id];
        const DynamicArenaSlotId *individual_tail_row_slot_ids =
            TailRowSlotIdsForIndividual(host_tail_row_slot_ids.data(), buffers.max_action_count, individual_index);
        for (std::size_t action_index = 0; action_index < host_population.layout.action_count; ++action_index) {
            const DynamicArenaSlotId tail_row_slot_id = individual_tail_row_slot_ids[action_index];
            if (!IsValidArenaSlotId(tail_row_slot_id, buffers.tail_row_slot_capacity)) {
                return false;
            }

            GenomeTailRow(genome_view, action_index) = host_tail_row_slots[tail_row_slot_id];
        }
    }

    return true;
}

bool TryEvaluatePopulationFitnessOnDevice(DeviceRuntimeBuffers &buffers, const RuntimeWordCounts &runtime_word_counts) {
    if (!IsValidDynamicPopulationLayout(buffers.current_layout) || !ResetDeviceStatus(buffers)) {
        return false;
    }

    const std::size_t block_count =
        (buffers.current_layout.active_individual_count + kDynamicThreadBlockSize - 1) / kDynamicThreadBlockSize;

    EvaluatePopulationFitnessKernel<<<block_count, kDynamicThreadBlockSize>>>(
        buffers.body_slots, buffers.tail_row_slots, buffers.current_body_slot_ids, buffers.current_tail_row_slot_ids,
        buffers.max_action_count, buffers.current_layout, buffers.current_fitness, buffers.current_evaluation_counts,
        buffers.current_has_fitness, runtime_word_counts, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    if (!KernelCompletedSuccessfully(buffers)) {
        return false;
    }

    if (!ResetDeviceStatus(buffers)) {
        return false;
    }

    SummarizePopulationKernel<<<1, 1>>>(buffers.current_fitness, buffers.current_has_fitness, buffers.current_layout,
                                        buffers.summary, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    return KernelCompletedSuccessfully(buffers);
}

bool TryReadPopulationFitnessSummaryFromDevice(const DeviceRuntimeBuffers &buffers, PopulationFitnessSummary &summary) {
    return CheckCuda(cudaMemcpy(&summary, buffers.summary, sizeof(PopulationFitnessSummary), cudaMemcpyDeviceToHost));
}

bool TryReadDeviceRuntimeStatus(const DeviceRuntimeBuffers &buffers, DeviceRuntimeStatusCode &status_code) {
    int status_value = 0;
    if (!ReadDeviceStatus(buffers, status_value)) {
        return false;
    }

    status_code = static_cast<DeviceRuntimeStatusCode>(status_value);
    return true;
}

bool TryAssembleNextGenerationOnDevice(DeviceRuntimeBuffers &buffers, const std::uint32_t generation_seed,
                                       const GenerationAssemblyConfig &config,
                                       const PendingOutputEmbeddingInjection &pending_output_embedding_injection) {
    if (!IsValidDynamicGenerationAssemblyConfig(config) || !IsValidDynamicPopulationLayout(buffers.current_layout)) {
        (void)WriteDeviceStatus(buffers, DeviceRuntimeStatusCode::kInvalidAssemblyConfig);
        return false;
    }

    DynamicPopulationLayout next_layout{};
    if (!TryPlanNextPopulationLayout(buffers, pending_output_embedding_injection, next_layout) ||
        (next_layout.active_individual_count < config.genetic_algorithm.elite_count)) {
        (void)WriteDeviceStatus(buffers, DeviceRuntimeStatusCode::kInvalidAssemblyConfig);
        return false;
    }

    buffers.next_layout = next_layout;

    bool ok = true;
    ok &= ResetDeviceStatus(buffers);
    ok &= ResetNextGenerationStorage(buffers);
    if (!ok) {
        return false;
    }

    if (!TryPlanNextGenerationMatingPlanOnDevice(buffers, generation_seed, config) || !ResetDeviceStatus(buffers)) {
        return false;
    }

    if (!TryInitializeArenaFreeSlotStacksOnDevice(buffers) || !ResetDeviceStatus(buffers)) {
        return false;
    }

    AssembleNextGenerationKernel<<<1, 1>>>(
        buffers.body_slots, buffers.tail_row_slots, buffers.current_body_slot_ids, buffers.current_tail_row_slot_ids,
        buffers.next_body_slot_ids, buffers.next_tail_row_slot_ids, buffers.body_free_slot_ids,
        buffers.tail_row_free_slot_ids, buffers.body_free_slot_count, buffers.tail_row_free_slot_count,
        buffers.current_parent_remaining_use_counts, buffers.max_action_count, buffers.body_slot_capacity,
        buffers.tail_row_slot_capacity, buffers.current_layout, buffers.next_fitness,
        buffers.next_evaluation_counts, buffers.next_has_fitness, buffers.next_layout, buffers.next_generation_plan,
        generation_seed, config, pending_output_embedding_injection, buffers.status);
    if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
        return false;
    }

    return KernelCompletedSuccessfully(buffers);
}

void SwapDevicePopulationBuffers(DeviceRuntimeBuffers &buffers) noexcept {
    std::swap(buffers.current_body_slot_ids, buffers.next_body_slot_ids);
    std::swap(buffers.current_tail_row_slot_ids, buffers.next_tail_row_slot_ids);
    std::swap(buffers.current_fitness, buffers.next_fitness);
    std::swap(buffers.current_evaluation_counts, buffers.next_evaluation_counts);
    std::swap(buffers.current_has_fitness, buffers.next_has_fitness);
    std::swap(buffers.current_layout, buffers.next_layout);
}

const char *DeviceRuntimeStatusCodeString(const DeviceRuntimeStatusCode status_code) noexcept {
    switch (status_code) {
    case DeviceRuntimeStatusCode::kOk:
        return "ok";
    case DeviceRuntimeStatusCode::kCudaFailure:
        return "cuda failure";
    case DeviceRuntimeStatusCode::kInvalidTrainingShard:
        return "invalid training-data shard";
    case DeviceRuntimeStatusCode::kGuessAppendFailed:
        return "could not append a model-selected guess to the Wordle grid";
    case DeviceRuntimeStatusCode::kPolicyForwardFailed:
        return "policy model forward pass failed";
    case DeviceRuntimeStatusCode::kActionSelectionFailed:
        return "output-embedding action selection failed";
    case DeviceRuntimeStatusCode::kPopulationNotEvaluated:
        return "population fitness summary requested before the population was fully evaluated";
    case DeviceRuntimeStatusCode::kInvalidAssemblyConfig:
        return "invalid next-generation assembly config";
    case DeviceRuntimeStatusCode::kParentSelectionFailed:
        return "device parent selection failed";
    case DeviceRuntimeStatusCode::kOutputEmbeddingInjectionFailed:
        return "device output-embedding injection failed";
    case DeviceRuntimeStatusCode::kInvalidPopulationLayout:
        return "invalid dynamic population layout";
    case DeviceRuntimeStatusCode::kArenaExhausted:
        return "dynamic genome arena exhausted during next-generation assembly";
    }

    return "unknown device-runtime status";
}

} // namespace neuroevolution::genetic_algorithm::dynamic_device
