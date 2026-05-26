#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "common/fixed_buffer.hpp"
#include "genetic_algorithm/genetic_algorithm.hpp"

namespace {

using neuroevolution::common::FixedBuffer;
using neuroevolution::common::ToFloat;
using neuroevolution::genetic_algorithm::BeginNextGeneration;
using neuroevolution::genetic_algorithm::BreedingConfig;
using neuroevolution::genetic_algorithm::ClearPopulationFitness;
using neuroevolution::genetic_algorithm::EvaluatePopulationFitness;
using neuroevolution::genetic_algorithm::FitnessEvaluationConfig;
using neuroevolution::genetic_algorithm::FitnessRandomEngine;
using neuroevolution::genetic_algorithm::GenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::IsValidBreedingConfig;
using neuroevolution::genetic_algorithm::IsValidFitnessEvaluationConfig;
using neuroevolution::genetic_algorithm::IsValidGenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::IsValidMutationConfig;
using neuroevolution::genetic_algorithm::IsValidParentSelectionConfig;
using neuroevolution::genetic_algorithm::IsValidPopulationInitializationConfig;
using neuroevolution::genetic_algorithm::ModelGenome;
using neuroevolution::genetic_algorithm::MutationConfig;
using neuroevolution::genetic_algorithm::ParentPair;
using neuroevolution::genetic_algorithm::ParentSelectionConfig;
using neuroevolution::genetic_algorithm::Population;
using neuroevolution::genetic_algorithm::PopulationInitializationConfig;
using neuroevolution::genetic_algorithm::PopulationInitializationRandomEngine;
using neuroevolution::genetic_algorithm::SelectionRandomEngine;
using neuroevolution::genetic_algorithm::TryFindBestIndividualIndex;
using neuroevolution::genetic_algorithm::TryInitializePopulation;
using neuroevolution::genetic_algorithm::TryMaterializeActionEmbeddings;
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
    ParentSelectionConfig invalid_rank_config = valid_config;
    invalid_rank_config.rank_exponent = -0.1f;

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
    ok &= ExpectTrue(!IsValidParentSelectionConfig(invalid_rank_config),
                     "Expected negative parent-selection rank exponent to be rejected");
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

bool TestGenerationAssemblyConfigValidationKeepsOnlySharedCudaConfig() {
    BreedingConfig valid_breeding_config{};
    MutationConfig valid_mutation_config{};
    GenerationAssemblyConfig valid_assembly_config{};

    BreedingConfig invalid_breeding_config{};
    invalid_breeding_config.crossover_temperature_level1 = 1.5f;

    MutationConfig invalid_mutation_config{};
    invalid_mutation_config.mutation_probability = 1.5f;

    GenerationAssemblyConfig invalid_assembly_config{};
    invalid_assembly_config.parent_selection.tournament_size = 0;

    bool ok = true;
    ok &= ExpectTrue(IsValidBreedingConfig(valid_breeding_config), "Expected default breeding config to be valid");
    ok &= ExpectTrue(!IsValidBreedingConfig(invalid_breeding_config),
                     "Expected crossover temperature above one to be rejected");
    ok &= ExpectTrue(IsValidMutationConfig(valid_mutation_config), "Expected default mutation config to be valid");
    ok &= ExpectTrue(!IsValidMutationConfig(invalid_mutation_config),
                     "Expected mutation probability above one to be rejected");
    ok &= ExpectTrue(IsValidGenerationAssemblyConfig(valid_assembly_config),
                     "Expected default generation-assembly config to be valid");
    ok &= ExpectTrue(!IsValidGenerationAssemblyConfig(invalid_assembly_config),
                     "Expected invalid parent-selection config to invalidate generation assembly");
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

    if (!TestGenerationAssemblyConfigValidationKeepsOnlySharedCudaConfig()) {
        return 1;
    }

    if (!TestPopulationInitializationSeedsRandomGenomesAndClearsMetadata()) {
        return 1;
    }

    std::cout << "PASS: genetic_algorithm_test\n";
    return 0;
}
