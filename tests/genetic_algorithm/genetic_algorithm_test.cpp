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
using neuroevolution::genetic_algorithm::GenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::IsValidBreedingConfig;
using neuroevolution::genetic_algorithm::IsValidGenerationAssemblyConfig;
using neuroevolution::genetic_algorithm::IsValidMutationConfig;
using neuroevolution::genetic_algorithm::IsValidParentSelectionConfig;
using neuroevolution::genetic_algorithm::ModelGenome;
using neuroevolution::genetic_algorithm::MutationConfig;
using neuroevolution::genetic_algorithm::ParentSelectionConfig;
using neuroevolution::genetic_algorithm::Population;
using neuroevolution::genetic_algorithm::TryFindBestIndividualIndex;
using neuroevolution::genetic_algorithm::TryMaterializeActionEmbeddings;
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

bool TestGenerationAssemblyConfigValidationKeepsOnlySharedCudaConfig() {
    BreedingConfig valid_breeding_config{};
    MutationConfig valid_mutation_config{};
    GenerationAssemblyConfig valid_assembly_config{};

    BreedingConfig invalid_breeding_config{};
    invalid_breeding_config.crossover_temperature_level1 = 1.5f;

    MutationConfig invalid_mutation_config{};
    invalid_mutation_config.mutation_probability = 1.5f;

    GenerationAssemblyConfig invalid_assembly_config{};
    invalid_assembly_config.parent_selection.cellular_breeding_radius = 0;

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

} // namespace

int main() {
    if (!TestPopulationBookkeeping()) {
        return 1;
    }

    if (!TestGenomeMaterializesOutputEmbeddingTrainableParametersAgainstFixedWords()) {
        return 1;
    }

    if (!TestGenerationAssemblyConfigValidationKeepsOnlySharedCudaConfig()) {
        return 1;
    }

    std::cout << "PASS: genetic_algorithm_test\n";
    return 0;
}
