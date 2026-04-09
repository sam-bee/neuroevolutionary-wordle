#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "common/fixed_buffer.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "wordle/word.hpp"

namespace {

using neuroevolution::common::FixedBuffer;
using neuroevolution::model::output_embedding::ActionEmbedding;
using neuroevolution::model::output_embedding::ActionEmbeddingVector;
using neuroevolution::model::output_embedding::kOutputEmbeddingDimension;
using neuroevolution::model::output_embedding::kTrainableFeatureDimension;
using neuroevolution::model::output_embedding::kWordFeatureDimension;
using neuroevolution::model::output_embedding::MaterializeActionEmbedding;
using neuroevolution::model::output_embedding::PolicyVector;
using neuroevolution::model::output_embedding::ScoreActionEmbedding;
using neuroevolution::model::output_embedding::SelectBestActionWord;
using neuroevolution::model::output_embedding::SelectBestLegalActionWord;
using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::model::output_embedding::TrySelectBestAction;
using neuroevolution::model::output_embedding::TrySelectBestLegalAction;
using neuroevolution::wordle::LetterIndexFromAscii;
using neuroevolution::wordle::MakeWordleGrid;
using neuroevolution::wordle::TryAppendGuess;
using neuroevolution::wordle::Word;
using neuroevolution::wordle::WordleGrid;

constexpr float kTolerance = 1.0e-6f;

Word MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    Word word{};

    if (!neuroevolution::wordle::TryMakeWordFromAscii(letters, word)) {
        throw std::invalid_argument(
            "Output-embedding test word literal must contain exactly five uppercase ASCII letters.");
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

ActionEmbedding MakeActionEmbedding(const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    ActionEmbedding action_embedding{};
    action_embedding.word = MakeWord(letters);
    return action_embedding;
}

bool TestMaterializeActionEmbeddingUsesLetterCountsAndTrainableTail() {
    ActionEmbedding action_embedding = MakeActionEmbedding("CRASS");
    action_embedding.trainable_tail[0] = 0.5f;
    action_embedding.trainable_tail[1] = -1.5f;

    ActionEmbeddingVector embedding_vector{};
    MaterializeActionEmbedding(action_embedding, embedding_vector);

    bool ok = true;
    ok &= ExpectNear(embedding_vector[static_cast<std::size_t>(LetterIndexFromAscii('A'))], 1.0f, "A count feature");
    ok &= ExpectNear(embedding_vector[static_cast<std::size_t>(LetterIndexFromAscii('B'))], -1.0f, "B count feature");
    ok &= ExpectNear(embedding_vector[static_cast<std::size_t>(LetterIndexFromAscii('C'))], 1.0f, "C count feature");
    ok &= ExpectNear(embedding_vector[static_cast<std::size_t>(LetterIndexFromAscii('R'))], 1.0f, "R count feature");
    ok &= ExpectNear(embedding_vector[static_cast<std::size_t>(LetterIndexFromAscii('S'))], 2.0f, "S count feature");
    ok &= ExpectNear(embedding_vector[kWordFeatureDimension + 0], 0.5f, "first trainable feature");
    ok &= ExpectNear(embedding_vector[kWordFeatureDimension + 1], -1.5f, "second trainable feature");
    return ok;
}

bool TestSelectBestActionUsesDotProductAcrossFixedAndTrainableDimensions() {
    FixedBuffer<ActionEmbedding, 3> action_embeddings{};
    action_embeddings[0] = MakeActionEmbedding("APPLE");
    action_embeddings[1] = MakeActionEmbedding("BEEFY");
    action_embeddings[2] = MakeActionEmbedding("MUMMY");

    action_embeddings[0].trainable_tail[0] = 0.5f;
    action_embeddings[0].trainable_tail[1] = -1.0f;
    action_embeddings[1].trainable_tail[0] = 1.0f;
    action_embeddings[1].trainable_tail[1] = 0.25f;
    action_embeddings[2].trainable_tail[0] = -0.5f;
    action_embeddings[2].trainable_tail[1] = 2.0f;

    PolicyVector policy_vector{};
    policy_vector[static_cast<std::size_t>(LetterIndexFromAscii('A'))] = 1.0f;
    policy_vector[static_cast<std::size_t>(LetterIndexFromAscii('B'))] = 2.0f;
    policy_vector[static_cast<std::size_t>(LetterIndexFromAscii('E'))] = 3.0f;
    policy_vector[static_cast<std::size_t>(LetterIndexFromAscii('M'))] = -1.0f;
    policy_vector[kWordFeatureDimension + 0] = 4.0f;
    policy_vector[kWordFeatureDimension + 1] = -2.0f;

    const float apple_score = ScoreActionEmbedding(policy_vector, action_embeddings[0]);
    const float beefy_score = ScoreActionEmbedding(policy_vector, action_embeddings[1]);
    const float mummy_score = ScoreActionEmbedding(policy_vector, action_embeddings[2]);

    SelectedAction selected_action{};
    const bool select_ok = TrySelectBestAction(policy_vector, action_embeddings.values, 3, selected_action);
    const Word selected_word = SelectBestActionWord(policy_vector, action_embeddings.values, 3);

    bool ok = true;
    ok &= ExpectTrue(select_ok, "Expected best-action selection to succeed for a non-empty valid table");
    ok &= ExpectNear(apple_score, 7.0f, "APPLE score");
    ok &= ExpectNear(beefy_score, 11.5f, "BEEFY score");
    ok &= ExpectNear(mummy_score, -15.0f, "MUMMY score");
    ok &= ExpectTrue(selected_action.action_index == 1, "Expected BEEFY to have the highest score");
    ok &= ExpectNear(selected_action.score, 11.5f, "selected action score");
    ok &= ExpectWordEquals(selected_action.word, MakeWord("BEEFY"), "selected action word");
    ok &= ExpectWordEquals(selected_word, MakeWord("BEEFY"), "selected word wrapper");
    return ok;
}

bool TestTrySelectBestActionRejectsEmptyTables() {
    PolicyVector policy_vector{};
    SelectedAction selected_action{};

    const bool select_ok = TrySelectBestAction(policy_vector, nullptr, 0, selected_action);
    return ExpectTrue(!select_ok, "Expected empty output-embedding table to be rejected");
}

bool TestTrySelectBestLegalActionSkipsPreviouslyGuessedWords() {
    FixedBuffer<ActionEmbedding, 3> action_embeddings{};
    action_embeddings[0] = MakeActionEmbedding("APPLE");
    action_embeddings[1] = MakeActionEmbedding("BEEFY");
    action_embeddings[2] = MakeActionEmbedding("MUMMY");

    action_embeddings[0].trainable_tail[0] = 0.5f;
    action_embeddings[0].trainable_tail[1] = -1.0f;
    action_embeddings[1].trainable_tail[0] = 1.0f;
    action_embeddings[1].trainable_tail[1] = 0.25f;
    action_embeddings[2].trainable_tail[0] = -0.5f;
    action_embeddings[2].trainable_tail[1] = 2.0f;

    PolicyVector policy_vector{};
    policy_vector[static_cast<std::size_t>(LetterIndexFromAscii('A'))] = 1.0f;
    policy_vector[static_cast<std::size_t>(LetterIndexFromAscii('B'))] = 2.0f;
    policy_vector[static_cast<std::size_t>(LetterIndexFromAscii('E'))] = 3.0f;
    policy_vector[static_cast<std::size_t>(LetterIndexFromAscii('M'))] = -1.0f;
    policy_vector[kWordFeatureDimension + 0] = 4.0f;
    policy_vector[kWordFeatureDimension + 1] = -2.0f;

    WordleGrid grid = MakeWordleGrid(MakeWord("SOLAR"));
    if (!TryAppendGuess(grid, MakeWord("BEEFY"))) {
        throw std::invalid_argument("Legal-selection test fixture could not append the repeated guess.");
    }

    SelectedAction selected_action{};
    const bool select_ok = TrySelectBestLegalAction(policy_vector, grid, action_embeddings.values, 3, selected_action);
    const Word selected_word = SelectBestLegalActionWord(policy_vector, grid, action_embeddings.values, 3);

    bool ok = true;
    ok &= ExpectTrue(select_ok, "Expected legal selection to succeed when at least one action remains legal");
    ok &= ExpectTrue(selected_action.action_index == 0,
                     "Expected APPLE to be selected after masking the repeated BEEFY guess");
    ok &= ExpectNear(selected_action.score, 7.0f, "selected legal action score");
    ok &= ExpectWordEquals(selected_action.word, MakeWord("APPLE"), "selected legal action word");
    ok &= ExpectWordEquals(selected_word, MakeWord("APPLE"), "selected legal wrapper word");
    return ok;
}

bool TestTrySelectBestLegalActionRejectsTablesWithNoLegalActions() {
    FixedBuffer<ActionEmbedding, 1> action_embeddings{};
    action_embeddings[0] = MakeActionEmbedding("BEEFY");

    PolicyVector policy_vector{};
    WordleGrid grid = MakeWordleGrid(MakeWord("SOLAR"));
    if (!TryAppendGuess(grid, MakeWord("BEEFY"))) {
        throw std::invalid_argument("No-legal-action test fixture could not append the repeated guess.");
    }

    SelectedAction selected_action{};
    const bool select_ok = TrySelectBestLegalAction(policy_vector, grid, action_embeddings.values, 1, selected_action);
    return ExpectTrue(!select_ok, "Expected legal selection to fail when every action is a repeated guess");
}

} // namespace

int main() {
    static_assert(kOutputEmbeddingDimension == neuroevolution::model::dense_trunk::kDenseTrunkOutputSize);
    static_assert(kWordFeatureDimension == neuroevolution::wordle::kAlphabetSize);
    static_assert(kTrainableFeatureDimension == 38);

    if (!TestMaterializeActionEmbeddingUsesLetterCountsAndTrainableTail()) {
        return 1;
    }

    if (!TestSelectBestActionUsesDotProductAcrossFixedAndTrainableDimensions()) {
        return 1;
    }

    if (!TestTrySelectBestActionRejectsEmptyTables()) {
        return 1;
    }

    if (!TestTrySelectBestLegalActionSkipsPreviouslyGuessedWords()) {
        return 1;
    }

    if (!TestTrySelectBestLegalActionRejectsTablesWithNoLegalActions()) {
        return 1;
    }

    std::cout << "PASS: output_embedding_test\n";
    return 0;
}
