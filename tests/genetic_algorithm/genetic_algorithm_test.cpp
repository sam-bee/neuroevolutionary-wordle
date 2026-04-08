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
using neuroevolution::genetic_algorithm::GeneticAlgorithmConfig;
using neuroevolution::genetic_algorithm::Individual;
using neuroevolution::genetic_algorithm::IsValidBreedingConfig;
using neuroevolution::genetic_algorithm::IsValidFitnessEvaluationConfig;
using neuroevolution::genetic_algorithm::IsValidGeneticAlgorithmConfig;
using neuroevolution::genetic_algorithm::IsValidParentSelectionConfig;
using neuroevolution::genetic_algorithm::ModelGenome;
using neuroevolution::genetic_algorithm::ParentPair;
using neuroevolution::genetic_algorithm::ParentSelectionConfig;
using neuroevolution::genetic_algorithm::Population;
using neuroevolution::genetic_algorithm::SelectionRandomEngine;
using neuroevolution::genetic_algorithm::TryBreedChildGenome;
using neuroevolution::genetic_algorithm::TryBreedChildGenomeFromPopulation;
using neuroevolution::genetic_algorithm::TryFindBestIndividualIndex;
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

bool TestConfigValidationAndPopulationBookkeeping() {
    constexpr std::size_t kPopulationSize = 3;

    GeneticAlgorithmConfig valid_config{};
    valid_config.elite_count = 1;
    valid_config.mutation_probability = 0.1f;
    valid_config.mutation_sigma = 0.25f;

    GeneticAlgorithmConfig invalid_config = valid_config;
    invalid_config.elite_count = 4;

    Population<ModelGenome<2>, kPopulationSize> population{};
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
    ok &= ExpectTrue(IsValidGeneticAlgorithmConfig<kPopulationSize>(valid_config), "Expected valid GA config");
    ok &= ExpectTrue(!IsValidGeneticAlgorithmConfig<kPopulationSize>(invalid_config),
                     "Expected elite count above population size to be rejected");
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

    BreedingConfig second_parent_only_config{};
    second_parent_only_config.first_parent_probability = 0.0f;

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

} // namespace

int main() {
    if (!TestConfigValidationAndPopulationBookkeeping()) {
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

    std::cout << "PASS: genetic_algorithm_test\n";
    return 0;
}
