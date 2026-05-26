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
using neuroevolution::genetic_algorithm::ModelGenome;
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

    if (!TestResizeRejectsCountsAboveCapacity()) {
        return 1;
    }

    std::cout << "PASS: output_embedding_genome_runtime_test\n";
    return 0;
}
