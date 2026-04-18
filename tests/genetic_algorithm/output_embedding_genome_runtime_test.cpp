#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "common/fixed_buffer.hpp"
#include "common/float16.hpp"
#include "genetic_algorithm/genetic_algorithm.hpp"

namespace {

using neuroevolution::common::FixedBuffer;
using neuroevolution::common::ToFloat;
using neuroevolution::genetic_algorithm::ActiveOutputEmbeddingCount;
using neuroevolution::genetic_algorithm::BreedChildGenome;
using neuroevolution::genetic_algorithm::BreedingConfig;
using neuroevolution::genetic_algorithm::BreedingRandomEngine;
using neuroevolution::genetic_algorithm::ModelGenome;
using neuroevolution::genetic_algorithm::MutateGenome;
using neuroevolution::genetic_algorithm::MutationConfig;
using neuroevolution::genetic_algorithm::MutationRandomEngine;
using neuroevolution::genetic_algorithm::output_tail_ops::kRowBlendMaximumLambda;
using neuroevolution::genetic_algorithm::output_tail_ops::kRowBlendMinimumLambda;
using neuroevolution::model::output_embedding::kTrainableFeatureDimension;
using neuroevolution::genetic_algorithm::TryMaterializeActionEmbeddings;
using neuroevolution::genetic_algorithm::TryResizeOutputEmbeddingGenome;
using neuroevolution::model::output_embedding::ActionEmbedding;
using neuroevolution::wordle::Word;

constexpr float kTolerance = 1.0e-6f;

Word MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    Word word{};

    if (!neuroevolution::wordle::TryMakeWordFromAscii(letters, word)) {
        throw std::invalid_argument(
            "Output-embedding runtime test word literal must contain exactly five uppercase ASCII letters.");
    }

    return word;
}

Word MakeInvalidWord() {
    Word word{};
    word.letter_indices[neuroevolution::wordle::kWordLength - 1] = neuroevolution::wordle::kAlphabetSize;
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

bool TestMaterializationUsesRuntimeActiveCount() {
    ModelGenome<4> genome{};
    genome.output_embedding.active_count = 2;
    genome.output_embedding.trainable_tails[0][0] = 0.25f;
    genome.output_embedding.trainable_tails[1][1] = -0.75f;

    FixedBuffer<Word, 4> action_words{};
    action_words[0] = MakeWord("CRANE");
    action_words[1] = MakeWord("SLATE");
    action_words[2] = MakeInvalidWord();
    action_words[3] = MakeInvalidWord();

    FixedBuffer<ActionEmbedding, 4> action_embeddings{};
    const bool default_materialize_ok =
        TryMaterializeActionEmbeddings(genome.output_embedding, action_words, action_embeddings);
    const bool too_many_materialize_ok =
        TryMaterializeActionEmbeddings(genome.output_embedding, action_words, action_embeddings, 3);

    bool ok = true;
    ok &= ExpectTrue(default_materialize_ok, "Expected runtime active count to limit materialization");
    ok &= ExpectTrue(!too_many_materialize_ok, "Expected materialization past the active count to be rejected");
    ok &= ExpectNear(ToFloat(action_embeddings[0].trainable_tail[0]), 0.25f, "active embedding tail[0][0]");
    ok &= ExpectNear(ToFloat(action_embeddings[1].trainable_tail[1]), -0.75f, "active embedding tail[1][1]");
    ok &= ExpectTrue(ActiveOutputEmbeddingCount(genome.output_embedding) == 2,
                     "Expected runtime active count helper to report two active embeddings");
    return ok;
}

bool TestMutationLeavesDormantEmbeddingRowsUntouched() {
    ModelGenome<4> genome{};
    genome.output_embedding.active_count = 1;
    genome.output_embedding.trainable_tails[0][0] = 1.0f;
    genome.output_embedding.trainable_tails[1][0] = 2.0f;
    genome.output_embedding.trainable_tails[2][0] = 3.0f;

    MutationConfig config{};
    config.mutation_probability = 1.0f;
    config.mutation_sigma = 0.5f;
    config.output_tail_row_scale_mutation_probability = 0.0f;

    MutationRandomEngine random_engine(17);
    MutateGenome(genome, random_engine, config);

    bool ok = true;
    ok &= ExpectTrue(ToFloat(genome.output_embedding.trainable_tails[0][0]) != 1.0f,
                     "Expected the active embedding row to mutate");
    ok &= ExpectNear(ToFloat(genome.output_embedding.trainable_tails[1][0]), 2.0f,
                     "Expected first dormant embedding row to remain unchanged");
    ok &= ExpectNear(ToFloat(genome.output_embedding.trainable_tails[2][0]), 3.0f,
                     "Expected second dormant embedding row to remain unchanged");
    return ok;
}

bool TestBreedingKeepsOnlySharedActiveEmbeddingRows() {
    ModelGenome<4> first_parent{};
    ModelGenome<4> second_parent{};

    first_parent.output_embedding.active_count = 1;
    first_parent.output_embedding.trainable_tails[0][0] = 5.0f;

    second_parent.output_embedding.active_count = 3;
    second_parent.output_embedding.trainable_tails[0][0] = 10.0f;
    second_parent.output_embedding.trainable_tails[1][0] = 20.0f;
    second_parent.output_embedding.trainable_tails[2][0] = 30.0f;

    BreedingConfig config{};
    config.first_parent_probability = 1.0f;
    config.output_tail_row_arithmetic_recombination_probability = 0.0f;

    BreedingRandomEngine random_engine(9);
    const ModelGenome<4> child = BreedChildGenome(first_parent, second_parent, random_engine, config);

    bool ok = true;
    ok &= ExpectTrue(child.output_embedding.active_count == 1,
                     "Expected child embedding genome to keep only the shared active rows");
    ok &= ExpectNear(ToFloat(child.output_embedding.trainable_tails[0][0]), 5.0f,
                     "Expected active child embedding row to come from the selected parent");
    ok &= ExpectNear(ToFloat(child.output_embedding.trainable_tails[1][0]), 0.0f,
                     "Expected first dormant child embedding row to be cleared");
    ok &= ExpectNear(ToFloat(child.output_embedding.trainable_tails[2][0]), 0.0f,
                     "Expected second dormant child embedding row to be cleared");
    return ok;
}

bool TestOutputTailArithmeticRecombinationUsesOneBoundedBlendPerRow() {
    ModelGenome<2> first_parent{};
    ModelGenome<2> second_parent{};
    first_parent.output_embedding.active_count = 1;
    second_parent.output_embedding.active_count = 1;

    first_parent.output_embedding.trainable_tails[0][0] = 10.0f;
    first_parent.output_embedding.trainable_tails[0][1] = -6.0f;
    first_parent.output_embedding.trainable_tails[0][2] = 4.0f;
    second_parent.output_embedding.trainable_tails[0][0] = 30.0f;
    second_parent.output_embedding.trainable_tails[0][1] = 14.0f;
    second_parent.output_embedding.trainable_tails[0][2] = 24.0f;

    BreedingConfig config{};
    config.first_parent_probability = 1.0f;
    config.output_tail_row_arithmetic_recombination_probability = 1.0f;

    BreedingRandomEngine random_engine(123);
    const ModelGenome<2> child = BreedChildGenome(first_parent, second_parent, random_engine, config);

    const float blended0 = ToFloat(child.output_embedding.trainable_tails[0][0]);
    const float blended1 = ToFloat(child.output_embedding.trainable_tails[0][1]);
    const float blended2 = ToFloat(child.output_embedding.trainable_tails[0][2]);
    const float lambda = (blended0 - 30.0f) / (10.0f - 30.0f);
    const float lambda1 = (blended1 - 14.0f) / (-6.0f - 14.0f);
    const float lambda2 = (blended2 - 24.0f) / (4.0f - 24.0f);

    bool ok = true;
    ok &= ExpectTrue(lambda >= kRowBlendMinimumLambda,
                     "Expected arithmetic recombination lambda to stay above the configured minimum");
    ok &= ExpectTrue(lambda <= kRowBlendMaximumLambda,
                     "Expected arithmetic recombination lambda to stay below the configured maximum");
    ok &= ExpectTrue(std::fabs(lambda1 - lambda) <= 1.0e-3f,
                     "Expected arithmetic recombination to reuse one lambda across the row");
    ok &= ExpectTrue(std::fabs(lambda2 - lambda) <= 1.0e-3f,
                     "Expected arithmetic recombination to reuse the same lambda for later row features");
    return ok;
}

bool TestOutputTailRowScaleMutationAppliesRareUniformRowScaling() {
    ModelGenome<2> genome{};
    genome.output_embedding.active_count = 1;
    for (std::size_t feature_index = 0; feature_index < kTrainableFeatureDimension; ++feature_index) {
        genome.output_embedding.trainable_tails[0][feature_index] = 1.0f + static_cast<float>(feature_index);
    }

    MutationConfig config{};
    config.mutation_probability = 0.0f;
    config.mutation_sigma = 0.0f;
    config.output_tail_row_scale_mutation_probability = 1.0f;

    MutationRandomEngine random_engine(17);
    MutateGenome(genome, random_engine, config);

    const float scale = ToFloat(genome.output_embedding.trainable_tails[0][0]) / 1.0f;
    const bool scale_is_expected = (std::fabs(scale - 0.98f) <= 5.0e-3f) || (std::fabs(scale - 1.02f) <= 5.0e-3f);

    bool ok = true;
    ok &= ExpectTrue(scale_is_expected, "Expected row-scale mutation to use a +/-2% multiplier");
    for (std::size_t feature_index = 0; feature_index < kTrainableFeatureDimension; ++feature_index) {
        const float expected_baseline = 1.0f + static_cast<float>(feature_index);
        const float feature_scale = ToFloat(genome.output_embedding.trainable_tails[0][feature_index]) / expected_baseline;
        ok &= ExpectTrue(std::fabs(feature_scale - scale) <= 5.0e-3f,
                         "Expected row-scale mutation to apply the same multiplier to every row feature");
    }
    return ok;
}

bool TestResizeRejectsCountsAboveCapacity() {
    neuroevolution::genetic_algorithm::OutputEmbeddingGenome<4> genome{};

    const bool resize_ok = TryResizeOutputEmbeddingGenome(genome, 3);
    const bool resize_too_large_ok = TryResizeOutputEmbeddingGenome(genome, 5);

    bool ok = true;
    ok &= ExpectTrue(resize_ok, "Expected resizing within capacity to succeed");
    ok &= ExpectTrue(!resize_too_large_ok, "Expected resizing above capacity to be rejected");
    ok &= ExpectTrue(genome.active_count == 3, "Expected successful resize to update the active count");
    return ok;
}

} // namespace

int main() {
    if (!TestMaterializationUsesRuntimeActiveCount()) {
        return 1;
    }

    if (!TestMutationLeavesDormantEmbeddingRowsUntouched()) {
        return 1;
    }

    if (!TestBreedingKeepsOnlySharedActiveEmbeddingRows()) {
        return 1;
    }

    if (!TestOutputTailArithmeticRecombinationUsesOneBoundedBlendPerRow()) {
        return 1;
    }

    if (!TestOutputTailRowScaleMutationAppliesRareUniformRowScaling()) {
        return 1;
    }

    if (!TestResizeRejectsCountsAboveCapacity()) {
        return 1;
    }

    std::cout << "PASS: output_embedding_genome_runtime_test\n";
    return 0;
}
