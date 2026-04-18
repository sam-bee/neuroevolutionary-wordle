#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "common/fixed_buffer.hpp"
#include "genetic_algorithm/genetic_algorithm.hpp"

namespace {

using neuroevolution::common::FixedBuffer;
using neuroevolution::common::ToFloat;
using neuroevolution::genetic_algorithm::BeginNextGeneration;
using neuroevolution::genetic_algorithm::BreedChildGenome;
using neuroevolution::genetic_algorithm::BreedingConfig;
using neuroevolution::genetic_algorithm::BreedingRandomEngine;
using neuroevolution::genetic_algorithm::ClearPopulationFitness;
using neuroevolution::genetic_algorithm::EvaluatePopulationFitness;
using neuroevolution::genetic_algorithm::FitnessEvaluationConfig;
using neuroevolution::genetic_algorithm::FitnessRandomEngine;
using neuroevolution::genetic_algorithm::GenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::GenerationAssemblyRandomEngine;
using neuroevolution::genetic_algorithm::Individual;
using neuroevolution::genetic_algorithm::IsValidBreedingConfig;
using neuroevolution::genetic_algorithm::IsValidFitnessEvaluationConfig;
using neuroevolution::genetic_algorithm::IsValidGenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::IsValidMutationConfig;
using neuroevolution::genetic_algorithm::IsValidParentSelectionConfig;
using neuroevolution::genetic_algorithm::IsValidPopulationInitializationConfig;
using neuroevolution::genetic_algorithm::ModelGenome;
using neuroevolution::genetic_algorithm::MutateGenome;
using neuroevolution::genetic_algorithm::MutationConfig;
using neuroevolution::genetic_algorithm::MutationRandomEngine;
using neuroevolution::genetic_algorithm::ParentPair;
using neuroevolution::genetic_algorithm::ParentSelectionConfig;
using neuroevolution::genetic_algorithm::Population;
using neuroevolution::genetic_algorithm::PopulationInitializationConfig;
using neuroevolution::genetic_algorithm::PopulationInitializationRandomEngine;
using neuroevolution::genetic_algorithm::SelectionRandomEngine;
using neuroevolution::genetic_algorithm::TryAssembleNextGeneration;
using neuroevolution::genetic_algorithm::TryBreedChildGenome;
using neuroevolution::genetic_algorithm::TryBreedChildGenomeFromPopulation;
using neuroevolution::genetic_algorithm::TryFindBestIndividualIndex;
using neuroevolution::genetic_algorithm::TryInitializePopulation;
using neuroevolution::genetic_algorithm::TryMaterializeActionEmbeddings;
using neuroevolution::genetic_algorithm::TryMutateGenome;
using neuroevolution::genetic_algorithm::TrySelectParentIndex;
using neuroevolution::genetic_algorithm::TrySelectParentPair;
using neuroevolution::model::output_embedding::ActionEmbedding;
using neuroevolution::model::output_embedding::kTrainableFeatureDimension;
using neuroevolution::wordle::Word;

constexpr float kTolerance = 1.0e-6f;

Word MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    Word word{};

    if (!neuroevolution::wordle::TryMakeWordFromAscii(letters, word)) {
        throw std::invalid_argument("Genetic-algorithm test word literal must contain five uppercase ASCII letters.");
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

bool ExpectNear(const float actual, const float expected, const std::string_view label) {
    if (std::fabs(actual - expected) > kTolerance) {
        std::cerr << "FAIL: " << label << " expected " << expected << ", got " << actual << '\n';
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

bool TestPopulationBookkeeping() {
    Population<ModelGenome<2>, 3> population{};
    population.individuals[0].fitness = 2.0f;
    population.individuals[0].evaluation_count = 8;
    population.individuals[0].has_fitness = true;
    population.individuals[1].fitness = 3.5f;
    population.individuals[1].evaluation_count = 8;
    population.individuals[1].has_fitness = true;

    std::size_t best_index = 0;
    const bool found_best_before_reset = TryFindBestIndividualIndex(population, best_index);

    ClearPopulationFitness(population);

    std::size_t best_index_after_reset = 0;
    const bool found_best_after_reset = TryFindBestIndividualIndex(population, best_index_after_reset);

    population.individuals[2].fitness = 9.0f;
    population.individuals[2].evaluation_count = 12;
    population.individuals[2].has_fitness = true;

    BeginNextGeneration(population);

    bool ok = true;
    ok &= ExpectTrue(found_best_before_reset, "Expected a best individual before clearing fitness");
    ok &= ExpectTrue(best_index == 1, "Expected highest-fitness individual to be selected");
    ok &= ExpectTrue(!found_best_after_reset, "Expected cleared population to have no best individual");
    ok &= ExpectTrue(population.generation_index == 1, "Expected next-generation bookkeeping to increment generation");
    ok &= ExpectTrue(!population.individuals[2].has_fitness,
                     "Expected next-generation bookkeeping to clear stale fitness flags");
    ok &= ExpectTrue(population.individuals[2].evaluation_count == 0,
                     "Expected next-generation bookkeeping to clear evaluation counts");
    return ok;
}

bool TestGenomeMaterializesOutputEmbeddingTrainableParametersAgainstFixedWords() {
    constexpr std::size_t kActionCount = 2;

    ModelGenome<kActionCount> genome{};
    genome.output_embedding.trainable_tails[0][0] = 0.25f;
    genome.output_embedding.trainable_tails[0][1] = -0.5f;
    genome.output_embedding.trainable_tails[1][0] = 1.25f;
    genome.output_embedding.trainable_tails[1][1] = 2.0f;

    FixedBuffer<Word, kActionCount> action_words{};
    action_words[0] = MakeWord("CRANE");
    action_words[1] = MakeWord("SLATE");

    FixedBuffer<ActionEmbedding, kActionCount> action_embeddings{};
    const bool materialize_ok =
        TryMaterializeActionEmbeddings(genome.output_embedding, action_words, action_embeddings);

    bool ok = true;
    ok &= ExpectTrue(materialize_ok, "Expected valid fixed action words to materialize cleanly");
    ok &= ExpectWordEquals(action_embeddings[0].word, MakeWord("CRANE"), "first action word");
    ok &= ExpectWordEquals(action_embeddings[1].word, MakeWord("SLATE"), "second action word");
    ok &= ExpectNear(ToFloat(action_embeddings[0].trainable_tail[0]), 0.25f, "first action first tail feature");
    ok &= ExpectNear(ToFloat(action_embeddings[0].trainable_tail[1]), -0.5f, "first action second tail feature");
    ok &= ExpectNear(ToFloat(action_embeddings[1].trainable_tail[0]), 1.25f, "second action first tail feature");
    ok &= ExpectNear(ToFloat(action_embeddings[1].trainable_tail[1]), 2.0f, "second action second tail feature");

    for (std::size_t feature_index = 2; feature_index < kTrainableFeatureDimension; ++feature_index) {
        ok &= ExpectNear(ToFloat(action_embeddings[0].trainable_tail[feature_index]), 0.0f,
                         "remaining first-action tail features");
        ok &= ExpectNear(ToFloat(action_embeddings[1].trainable_tail[feature_index]), 0.0f,
                         "remaining second-action tail features");
    }

    return ok;
}

bool TestPlaceholderFitnessEvaluationAssignsSeededScoresAndBookkeeping() {
    constexpr std::size_t kPopulationSize = 3;

    FitnessEvaluationConfig valid_config{};
    valid_config.minimum_fitness = -2.0f;
    valid_config.maximum_fitness = 3.0f;

    FitnessEvaluationConfig invalid_config = valid_config;
    invalid_config.minimum_fitness = 5.0f;
    invalid_config.maximum_fitness = 4.0f;

    Population<ModelGenome<2>, kPopulationSize> population_a{};
    Population<ModelGenome<2>, kPopulationSize> population_b{};
    Population<ModelGenome<2>, kPopulationSize> population_c{};

    FitnessRandomEngine random_engine_a(123);
    FitnessRandomEngine random_engine_b(123);
    FitnessRandomEngine random_engine_c(456);

    EvaluatePopulationFitness(population_a, random_engine_a, valid_config);
    EvaluatePopulationFitness(population_b, random_engine_b, valid_config);
    EvaluatePopulationFitness(population_c, random_engine_c, valid_config);

    bool same_seed_matches = true;
    bool different_seed_differs = false;
    bool scores_in_range = true;
    bool bookkeeping_ok = true;

    for (std::size_t individual_index = 0; individual_index < kPopulationSize; ++individual_index) {
        const auto &individual_a = population_a.individuals[individual_index];
        const auto &individual_b = population_b.individuals[individual_index];
        const auto &individual_c = population_c.individuals[individual_index];

        same_seed_matches &= (individual_a.fitness == individual_b.fitness);
        different_seed_differs |= (individual_a.fitness != individual_c.fitness);
        scores_in_range &= (individual_a.fitness >= valid_config.minimum_fitness) &&
                           (individual_a.fitness <= valid_config.maximum_fitness);
        bookkeeping_ok &= individual_a.has_fitness && (individual_a.evaluation_count == 1);
    }

    bool ok = true;
    ok &= ExpectTrue(IsValidFitnessEvaluationConfig(valid_config), "Expected valid fitness config");
    ok &= ExpectTrue(!IsValidFitnessEvaluationConfig(invalid_config),
                     "Expected fitness config with inverted range to be rejected");
    ok &= ExpectTrue(same_seed_matches, "Expected placeholder fitness scores to be reproducible with the same seed");
    ok &=
        ExpectTrue(different_seed_differs, "Expected placeholder fitness scores to differ when using a different seed");
    ok &= ExpectTrue(scores_in_range, "Expected placeholder fitness scores to stay within configured bounds");
    ok &= ExpectTrue(bookkeeping_ok, "Expected placeholder fitness evaluation to set flags and increment counts");
    return ok;
}

bool TestParentSelectionUsesTournamentLogicAndSupportsParentPairs() {
    constexpr std::size_t kPopulationSize = 4;

    Population<ModelGenome<2>, kPopulationSize> population{};
    population.individuals[0].fitness = 1.0f;
    population.individuals[1].fitness = 6.0f;
    population.individuals[2].fitness = 8.0f;
    population.individuals[3].fitness = 10.0f;

    for (std::size_t individual_index = 0; individual_index < kPopulationSize; ++individual_index) {
        population.individuals[individual_index].has_fitness = true;
    }

    ParentSelectionConfig valid_config{};
    valid_config.tournament_size = kPopulationSize;
    valid_config.allow_self_parenting = false;

    ParentSelectionConfig invalid_config = valid_config;
    invalid_config.tournament_size = 0;

    SelectionRandomEngine random_engine_a(7);
    SelectionRandomEngine random_engine_b(7);

    std::size_t selected_parent_index = 0;
    ParentPair parent_pair{};

    const bool select_ok = TrySelectParentIndex(population, random_engine_a, selected_parent_index, valid_config);
    const bool pair_ok = TrySelectParentPair(population, random_engine_b, parent_pair, valid_config);

    ParentSelectionConfig self_parenting_config = valid_config;
    self_parenting_config.allow_self_parenting = true;

    SelectionRandomEngine random_engine_c(7);
    ParentPair self_parenting_pair{};
    const bool self_pair_ok =
        TrySelectParentPair(population, random_engine_c, self_parenting_pair, self_parenting_config);

    Population<ModelGenome<2>, 1> one_individual_population{};
    one_individual_population.individuals[0].fitness = 5.0f;
    one_individual_population.individuals[0].has_fitness = true;
    SelectionRandomEngine random_engine_d(11);
    ParentPair impossible_pair{};
    const bool impossible_pair_ok = TrySelectParentPair(one_individual_population, random_engine_d, impossible_pair);

    bool ok = true;
    ok &= ExpectTrue(IsValidParentSelectionConfig(valid_config), "Expected valid parent-selection config");
    ok &= ExpectTrue(!IsValidParentSelectionConfig(invalid_config), "Expected zero tournament size to be rejected");
    ok &= ExpectTrue(select_ok, "Expected parent selection to succeed for a fitted population");
    ok &= ExpectTrue(selected_parent_index == 3,
                     "Expected full-population tournament selection to choose the fittest parent");
    ok &= ExpectTrue(pair_ok, "Expected distinct-parent selection to succeed when enough fitted individuals exist");
    ok &=
        ExpectTrue(parent_pair.first_parent_index == 3, "Expected first selected parent to be the fittest individual");
    ok &= ExpectTrue(parent_pair.second_parent_index == 2,
                     "Expected second selected parent to be the best remaining individual");
    ok &= ExpectTrue(self_pair_ok, "Expected self-parenting selection to succeed");
    ok &= ExpectTrue(self_parenting_pair.first_parent_index == 3,
                     "Expected self-parenting mode to select the fittest first parent");
    ok &= ExpectTrue(self_parenting_pair.second_parent_index == 3,
                     "Expected self-parenting mode to allow the same best individual twice");
    ok &= ExpectTrue(!impossible_pair_ok,
                     "Expected distinct-parent selection to fail for a population with only one fitted individual");
    return ok;
}

ModelGenome<2> MakeBreedingParentGenome(const float base_value) {
    ModelGenome<2> genome{};

    genome.policy_model.input_encoder.input_to_hidden.weights[0] = base_value + 1.0f;
    genome.policy_model.input_encoder.input_to_hidden.biases[0] = base_value + 2.0f;
    genome.policy_model.input_encoder.hidden_to_output.weights[0] = base_value + 3.0f;
    genome.policy_model.dense_trunk.input_to_hidden0.weights[0] = base_value + 4.0f;
    genome.policy_model.dense_trunk.hidden0_to_hidden1.biases[0] = base_value + 5.0f;
    genome.policy_model.dense_trunk.hidden1_to_output.weights[0] = base_value + 6.0f;
    genome.output_embedding.trainable_tails[0][0] = base_value + 7.0f;
    genome.output_embedding.trainable_tails[1][1] = base_value + 8.0f;

    return genome;
}

bool ExpectSelectedGenomeValuesMatch(const ModelGenome<2> &actual, const ModelGenome<2> &expected,
                                     const std::string_view label_prefix) {
    bool ok = true;
    ok &= ExpectNear(ToFloat(actual.policy_model.input_encoder.input_to_hidden.weights[0]),
                     ToFloat(expected.policy_model.input_encoder.input_to_hidden.weights[0]),
                     std::string(label_prefix) + " input->hidden weight");
    ok &= ExpectNear(ToFloat(actual.policy_model.input_encoder.input_to_hidden.biases[0]),
                     ToFloat(expected.policy_model.input_encoder.input_to_hidden.biases[0]),
                     std::string(label_prefix) + " input->hidden bias");
    ok &= ExpectNear(ToFloat(actual.policy_model.input_encoder.hidden_to_output.weights[0]),
                     ToFloat(expected.policy_model.input_encoder.hidden_to_output.weights[0]),
                     std::string(label_prefix) + " hidden->output weight");
    ok &= ExpectNear(ToFloat(actual.policy_model.dense_trunk.input_to_hidden0.weights[0]),
                     ToFloat(expected.policy_model.dense_trunk.input_to_hidden0.weights[0]),
                     std::string(label_prefix) + " trunk input->hidden0 weight");
    ok &= ExpectNear(ToFloat(actual.policy_model.dense_trunk.hidden0_to_hidden1.biases[0]),
                     ToFloat(expected.policy_model.dense_trunk.hidden0_to_hidden1.biases[0]),
                     std::string(label_prefix) + " trunk hidden0->hidden1 bias");
    ok &= ExpectNear(ToFloat(actual.policy_model.dense_trunk.hidden1_to_output.weights[0]),
                     ToFloat(expected.policy_model.dense_trunk.hidden1_to_output.weights[0]),
                     std::string(label_prefix) + " trunk hidden1->output weight");
    ok &= ExpectNear(ToFloat(actual.output_embedding.trainable_tails[0][0]),
                     ToFloat(expected.output_embedding.trainable_tails[0][0]),
                     std::string(label_prefix) + " output tail[0][0]");
    ok &= ExpectNear(ToFloat(actual.output_embedding.trainable_tails[1][1]),
                     ToFloat(expected.output_embedding.trainable_tails[1][1]),
                     std::string(label_prefix) + " output tail[1][1]");
    return ok;
}

bool TestBreedingCanProduceChildGenomeFromEitherParentOrPopulationPair() {
    const ModelGenome<2> first_parent = MakeBreedingParentGenome(10.0f);
    const ModelGenome<2> second_parent = MakeBreedingParentGenome(100.0f);

    BreedingConfig first_parent_only_config{};
    first_parent_only_config.first_parent_probability = 1.0f;
    first_parent_only_config.output_tail_row_arithmetic_recombination_probability = 0.0f;

    BreedingConfig second_parent_only_config{};
    second_parent_only_config.first_parent_probability = 0.0f;
    second_parent_only_config.output_tail_row_arithmetic_recombination_probability = 0.0f;

    BreedingConfig invalid_config{};
    invalid_config.first_parent_probability = 1.5f;

    BreedingRandomEngine random_engine_a(3);
    BreedingRandomEngine random_engine_b(3);

    ModelGenome<2> first_child{};
    ModelGenome<2> second_child{};

    const bool first_breed_ok =
        TryBreedChildGenome(first_parent, second_parent, first_child, random_engine_a, first_parent_only_config);
    const bool second_breed_ok =
        TryBreedChildGenome(first_parent, second_parent, second_child, random_engine_b, second_parent_only_config);

    Population<ModelGenome<2>, 3> population{};
    population.individuals[1].genome = first_parent;
    population.individuals[2].genome = second_parent;

    ParentPair parent_pair{};
    parent_pair.first_parent_index = 1;
    parent_pair.second_parent_index = 2;

    BreedingRandomEngine random_engine_c(5);
    ModelGenome<2> population_child{};
    const bool population_breed_ok = TryBreedChildGenomeFromPopulation(population, parent_pair, population_child,
                                                                       random_engine_c, second_parent_only_config);

    ParentPair invalid_parent_pair{};
    invalid_parent_pair.first_parent_index = 0;
    invalid_parent_pair.second_parent_index = 4;

    BreedingRandomEngine random_engine_d(7);
    ModelGenome<2> invalid_child{};
    const bool invalid_population_breed_ok = TryBreedChildGenomeFromPopulation(
        population, invalid_parent_pair, invalid_child, random_engine_d, first_parent_only_config);

    BreedingRandomEngine random_engine_e(9);
    const ModelGenome<2> convenience_child =
        BreedChildGenome(first_parent, second_parent, random_engine_e, first_parent_only_config);

    bool ok = true;
    ok &= ExpectTrue(IsValidBreedingConfig(first_parent_only_config), "Expected first-parent-only breeding config");
    ok &= ExpectTrue(IsValidBreedingConfig(second_parent_only_config), "Expected second-parent-only breeding config");
    ok &= ExpectTrue(!IsValidBreedingConfig(invalid_config), "Expected breeding probability above one to be rejected");
    ok &= ExpectTrue(first_breed_ok, "Expected breeding to succeed with valid first-parent-only config");
    ok &= ExpectTrue(second_breed_ok, "Expected breeding to succeed with valid second-parent-only config");
    ok &= ExpectTrue(population_breed_ok, "Expected population-based breeding to succeed with valid parent indices");
    ok &= ExpectTrue(!invalid_population_breed_ok,
                     "Expected population-based breeding to reject out-of-range parent indices");
    ok &= ExpectSelectedGenomeValuesMatch(first_child, first_parent, "first-only child");
    ok &= ExpectSelectedGenomeValuesMatch(second_child, second_parent, "second-only child");
    ok &= ExpectSelectedGenomeValuesMatch(population_child, second_parent, "population child");
    ok &= ExpectSelectedGenomeValuesMatch(convenience_child, first_parent, "convenience child");
    return ok;
}

bool TrackedGenomeValuesEqual(const ModelGenome<2> &lhs, const ModelGenome<2> &rhs) {
    return (ToFloat(lhs.policy_model.input_encoder.input_to_hidden.weights[0]) ==
            ToFloat(rhs.policy_model.input_encoder.input_to_hidden.weights[0])) &&
           (ToFloat(lhs.policy_model.input_encoder.input_to_hidden.biases[0]) ==
            ToFloat(rhs.policy_model.input_encoder.input_to_hidden.biases[0])) &&
           (ToFloat(lhs.policy_model.input_encoder.hidden_to_output.weights[0]) ==
            ToFloat(rhs.policy_model.input_encoder.hidden_to_output.weights[0])) &&
           (ToFloat(lhs.policy_model.dense_trunk.input_to_hidden0.weights[0]) ==
            ToFloat(rhs.policy_model.dense_trunk.input_to_hidden0.weights[0])) &&
           (ToFloat(lhs.policy_model.dense_trunk.hidden0_to_hidden1.biases[0]) ==
            ToFloat(rhs.policy_model.dense_trunk.hidden0_to_hidden1.biases[0])) &&
           (ToFloat(lhs.policy_model.dense_trunk.hidden1_to_output.weights[0]) ==
            ToFloat(rhs.policy_model.dense_trunk.hidden1_to_output.weights[0])) &&
           (ToFloat(lhs.output_embedding.trainable_tails[0][0]) ==
            ToFloat(rhs.output_embedding.trainable_tails[0][0])) &&
           (ToFloat(lhs.output_embedding.trainable_tails[1][1]) == ToFloat(rhs.output_embedding.trainable_tails[1][1]));
}

bool TestMutationCanBeDisabledOrApplySeededNoise() {
    const ModelGenome<2> baseline = MakeBreedingParentGenome(10.0f);

    MutationConfig zero_probability_config{};
    zero_probability_config.mutation_probability = 0.0f;
    zero_probability_config.mutation_sigma = 1.0f;
    zero_probability_config.output_tail_row_scale_mutation_probability = 0.0f;

    MutationConfig zero_sigma_config{};
    zero_sigma_config.mutation_probability = 1.0f;
    zero_sigma_config.mutation_sigma = 0.0f;
    zero_sigma_config.output_tail_row_scale_mutation_probability = 0.0f;

    MutationConfig active_config{};
    active_config.mutation_probability = 1.0f;
    active_config.mutation_sigma = 0.5f;
    active_config.output_tail_row_scale_mutation_probability = 0.0f;

    MutationConfig invalid_config{};
    invalid_config.mutation_probability = 1.5f;

    ModelGenome<2> zero_probability_genome = baseline;
    ModelGenome<2> zero_sigma_genome = baseline;
    ModelGenome<2> active_genome_a = baseline;
    ModelGenome<2> active_genome_b = baseline;
    ModelGenome<2> active_genome_c = baseline;

    MutationRandomEngine random_engine_a(1);
    MutationRandomEngine random_engine_b(2);
    MutationRandomEngine random_engine_c(123);
    MutationRandomEngine random_engine_d(123);
    MutationRandomEngine random_engine_e(456);

    const bool zero_probability_ok = TryMutateGenome(zero_probability_genome, random_engine_a, zero_probability_config);
    const bool zero_sigma_ok = TryMutateGenome(zero_sigma_genome, random_engine_b, zero_sigma_config);
    const bool active_ok_a = TryMutateGenome(active_genome_a, random_engine_c, active_config);
    const bool active_ok_b = TryMutateGenome(active_genome_b, random_engine_d, active_config);
    const bool active_ok_c = TryMutateGenome(active_genome_c, random_engine_e, active_config);

    MutationRandomEngine random_engine_f(789);
    ModelGenome<2> convenience_genome = baseline;
    MutateGenome(convenience_genome, random_engine_f, zero_probability_config);

    bool ok = true;
    ok &= ExpectTrue(IsValidMutationConfig(zero_probability_config), "Expected zero-probability mutation config");
    ok &= ExpectTrue(IsValidMutationConfig(zero_sigma_config), "Expected zero-sigma mutation config");
    ok &= ExpectTrue(IsValidMutationConfig(active_config), "Expected active mutation config");
    ok &= ExpectTrue(!IsValidMutationConfig(invalid_config), "Expected mutation probability above one to be rejected");
    ok &= ExpectTrue(zero_probability_ok, "Expected zero-probability mutation call to succeed");
    ok &= ExpectTrue(zero_sigma_ok, "Expected zero-sigma mutation call to succeed");
    ok &= ExpectTrue(active_ok_a && active_ok_b && active_ok_c, "Expected active mutation calls to succeed");
    ok &= ExpectTrue(TrackedGenomeValuesEqual(zero_probability_genome, baseline),
                     "Expected zero mutation probability to leave the genome unchanged");
    ok &= ExpectTrue(TrackedGenomeValuesEqual(zero_sigma_genome, baseline),
                     "Expected zero mutation sigma to leave the genome unchanged");
    ok &= ExpectTrue(TrackedGenomeValuesEqual(active_genome_a, active_genome_b),
                     "Expected same mutation seed to reproduce the same mutated genome");
    ok &= ExpectTrue(!TrackedGenomeValuesEqual(active_genome_a, baseline),
                     "Expected active mutation to change at least one tracked genome value");
    ok &= ExpectTrue(!TrackedGenomeValuesEqual(active_genome_a, active_genome_c),
                     "Expected different mutation seeds to produce different mutated genomes");
    ok &= ExpectTrue(TrackedGenomeValuesEqual(convenience_genome, baseline),
                     "Expected the convenience mutation wrapper to respect zero-probability config");
    return ok;
}

bool TestPopulationInitializationSeedsRandomGenomesAndClearsMetadata() {
    constexpr std::size_t kPopulationSize = 3;

    PopulationInitializationConfig valid_config{};

    PopulationInitializationConfig invalid_config = valid_config;
    invalid_config.parameter_initialization.dense_weight_gain = 0.0f;

    Population<ModelGenome<2>, kPopulationSize> population_a{};
    Population<ModelGenome<2>, kPopulationSize> population_b{};
    Population<ModelGenome<2>, kPopulationSize> population_c{};

    PopulationInitializationRandomEngine random_engine_a(100);
    PopulationInitializationRandomEngine random_engine_b(100);
    PopulationInitializationRandomEngine random_engine_c(200);

    const bool init_ok_a = TryInitializePopulation(population_a, random_engine_a, valid_config);
    const bool init_ok_b = TryInitializePopulation(population_b, random_engine_b, valid_config);
    const bool init_ok_c = TryInitializePopulation(population_c, random_engine_c, valid_config);

    bool same_seed_matches = true;
    bool different_seed_differs = false;
    bool metadata_cleared = true;

    for (std::size_t individual_index = 0; individual_index < kPopulationSize; ++individual_index) {
        same_seed_matches &= TrackedGenomeValuesEqual(population_a.individuals[individual_index].genome,
                                                      population_b.individuals[individual_index].genome);
        different_seed_differs |= !TrackedGenomeValuesEqual(population_a.individuals[individual_index].genome,
                                                            population_c.individuals[individual_index].genome);

        const auto &individual = population_a.individuals[individual_index];
        metadata_cleared &=
            !individual.has_fitness && (individual.evaluation_count == 0) && (individual.fitness == 0.0f);
    }

    bool ok = true;
    ok &= ExpectTrue(IsValidPopulationInitializationConfig(valid_config),
                     "Expected valid population-initialization config");
    ok &= ExpectTrue(!IsValidPopulationInitializationConfig(invalid_config),
                     "Expected invalid parameter-init config to invalidate population initialization");
    ok &= ExpectTrue(init_ok_a && init_ok_b && init_ok_c, "Expected population initialization to succeed");
    ok &= ExpectTrue(same_seed_matches, "Expected same initialization seed to reproduce the same population");
    ok &=
        ExpectTrue(different_seed_differs, "Expected different initialization seeds to produce a different population");
    ok &= ExpectTrue(metadata_cleared, "Expected initialized population metadata to be reset");
    ok &= ExpectTrue(population_a.generation_index == 0, "Expected initialized population generation index to be zero");
    return ok;
}

bool TestNextGenerationAssemblyBreedsEveryChild() {
    constexpr std::size_t kPopulationSize = 4;

    Population<ModelGenome<2>, kPopulationSize> current_population{};
    current_population.generation_index = 5;
    current_population.individuals[0].genome = MakeBreedingParentGenome(10.0f);
    current_population.individuals[1].genome = MakeBreedingParentGenome(20.0f);
    current_population.individuals[2].genome = MakeBreedingParentGenome(30.0f);
    current_population.individuals[3].genome = MakeBreedingParentGenome(40.0f);

    current_population.individuals[0].fitness = 1.0f;
    current_population.individuals[1].fitness = 4.0f;
    current_population.individuals[2].fitness = 7.0f;
    current_population.individuals[3].fitness = 9.0f;

    for (std::size_t individual_index = 0; individual_index < kPopulationSize; ++individual_index) {
        current_population.individuals[individual_index].has_fitness = true;
        current_population.individuals[individual_index].evaluation_count = 3;
    }

    GenerationAssemblyConfig valid_config{};
    valid_config.parent_selection.tournament_size = kPopulationSize;
    valid_config.parent_selection.allow_self_parenting = false;
    valid_config.breeding.first_parent_probability = 0.0f;
    valid_config.mutation.mutation_probability = 0.0f;
    valid_config.mutation.mutation_sigma = 1.0f;

    GenerationAssemblyRandomEngine random_engine(13);
    Population<ModelGenome<2>, kPopulationSize> next_population{};
    const bool assemble_ok =
        TryAssembleNextGeneration(current_population, next_population, random_engine, valid_config);

    bool metadata_ok = true;
    for (std::size_t individual_index = 0; individual_index < kPopulationSize; ++individual_index) {
        const auto &individual = next_population.individuals[individual_index];
        metadata_ok &= !individual.has_fitness && (individual.evaluation_count == 0) && (individual.fitness == 0.0f);
    }

    bool ok = true;
    ok &= ExpectTrue(IsValidGenerationAssemblyConfig(valid_config), "Expected valid generation-assembly config");
    ok &= ExpectTrue(assemble_ok, "Expected next-generation assembly to succeed for a fitted population");
    ok &= ExpectTrue(next_population.generation_index == 6, "Expected next population generation index to increment");
    ok &= ExpectSelectedGenomeValuesMatch(next_population.individuals[0].genome,
                                          current_population.individuals[2].genome, "bred slot 0");
    ok &= ExpectSelectedGenomeValuesMatch(next_population.individuals[1].genome,
                                          current_population.individuals[2].genome, "bred slot 1");
    ok &= ExpectSelectedGenomeValuesMatch(next_population.individuals[2].genome,
                                          current_population.individuals[2].genome, "bred slot 2");
    ok &= ExpectSelectedGenomeValuesMatch(next_population.individuals[3].genome,
                                          current_population.individuals[2].genome, "bred slot 3");
    ok &= ExpectTrue(metadata_ok, "Expected all next-generation individuals to be marked unevaluated");
    ok &= ExpectTrue(current_population.generation_index == 5,
                     "Expected current population generation metadata to stay unchanged");
    ok &= ExpectTrue(current_population.individuals[3].has_fitness,
                     "Expected current population fitness metadata to stay unchanged");
    return ok;
}

bool TestNextGenerationAssemblyRejectsInvalidConfigAndImpossibleBreedingCases() {
    GenerationAssemblyConfig invalid_config{};
    invalid_config.parent_selection.tournament_size = 0;

    Population<ModelGenome<2>, 2> one_parent_population{};
    one_parent_population.individuals[0].genome = MakeBreedingParentGenome(10.0f);
    one_parent_population.individuals[0].fitness = 5.0f;
    one_parent_population.individuals[0].has_fitness = true;

    GenerationAssemblyConfig valid_but_impossible_config{};
    valid_but_impossible_config.parent_selection.allow_self_parenting = false;

    GenerationAssemblyRandomEngine random_engine_a(1);
    Population<ModelGenome<2>, 2> next_population_a{};
    const bool invalid_config_ok =
        TryAssembleNextGeneration(one_parent_population, next_population_a, random_engine_a, invalid_config);

    GenerationAssemblyRandomEngine random_engine_b(2);
    Population<ModelGenome<2>, 2> next_population_b{};
    const bool impossible_breeding_ok = TryAssembleNextGeneration(one_parent_population, next_population_b,
                                                                  random_engine_b, valid_but_impossible_config);

    bool ok = true;
    ok &= ExpectTrue(!IsValidGenerationAssemblyConfig(invalid_config),
                     "Expected zero tournament size to invalidate generation assembly");
    ok &= ExpectTrue(!invalid_config_ok, "Expected next-generation assembly to reject invalid config");
    ok &= ExpectTrue(!impossible_breeding_ok,
                     "Expected next-generation assembly to fail when no distinct parent pair can be selected");
    return ok;
}

bool TestNextGenerationAssemblyAppliesMutationToEveryChild() {
    constexpr std::size_t kPopulationSize = 3;

    Population<ModelGenome<2>, kPopulationSize> current_population{};
    current_population.generation_index = 2;
    current_population.individuals[0].genome = MakeBreedingParentGenome(10.0f);
    current_population.individuals[1].genome = MakeBreedingParentGenome(20.0f);
    current_population.individuals[2].genome = MakeBreedingParentGenome(30.0f);

    current_population.individuals[0].fitness = 1.0f;
    current_population.individuals[1].fitness = 5.0f;
    current_population.individuals[2].fitness = 9.0f;

    for (std::size_t individual_index = 0; individual_index < kPopulationSize; ++individual_index) {
        current_population.individuals[individual_index].has_fitness = true;
    }

    GenerationAssemblyConfig config{};
    config.parent_selection.tournament_size = kPopulationSize;
    config.parent_selection.allow_self_parenting = false;
    config.breeding.first_parent_probability = 0.0f;
    config.mutation.mutation_probability = 1.0f;
    config.mutation.mutation_sigma = 0.5f;

    GenerationAssemblyRandomEngine random_engine(17);
    Population<ModelGenome<2>, kPopulationSize> next_population{};
    const bool assemble_ok = TryAssembleNextGeneration(current_population, next_population, random_engine, config);

    bool ok = true;
    ok &= ExpectTrue(assemble_ok, "Expected next-generation assembly with mutation to succeed");
    ok &= ExpectTrue(
        !TrackedGenomeValuesEqual(next_population.individuals[0].genome, current_population.individuals[1].genome),
        "Expected first child to differ from its pre-mutation source parent");
    ok &= ExpectTrue(
        !TrackedGenomeValuesEqual(next_population.individuals[1].genome, current_population.individuals[1].genome),
        "Expected second child to differ from its pre-mutation source parent");
    ok &= ExpectTrue(
        !TrackedGenomeValuesEqual(next_population.individuals[2].genome, current_population.individuals[1].genome),
        "Expected third child to differ from its pre-mutation source parent");
    return ok;
}

} // namespace

int main() {
    if (!TestPopulationBookkeeping()) {
        return 1;
    }

    if (!TestGenomeMaterializesOutputEmbeddingTrainableParametersAgainstFixedWords()) {
        return 1;
    }

    if (!TestPlaceholderFitnessEvaluationAssignsSeededScoresAndBookkeeping()) {
        return 1;
    }

    if (!TestParentSelectionUsesTournamentLogicAndSupportsParentPairs()) {
        return 1;
    }

    if (!TestBreedingCanProduceChildGenomeFromEitherParentOrPopulationPair()) {
        return 1;
    }

    if (!TestMutationCanBeDisabledOrApplySeededNoise()) {
        return 1;
    }

    if (!TestPopulationInitializationSeedsRandomGenomesAndClearsMetadata()) {
        return 1;
    }

    if (!TestNextGenerationAssemblyBreedsEveryChild()) {
        return 1;
    }

    if (!TestNextGenerationAssemblyRejectsInvalidConfigAndImpossibleBreedingCases()) {
        return 1;
    }

    if (!TestNextGenerationAssemblyAppliesMutationToEveryChild()) {
        return 1;
    }

    std::cout << "PASS: genetic_algorithm_test\n";
    return 0;
}
