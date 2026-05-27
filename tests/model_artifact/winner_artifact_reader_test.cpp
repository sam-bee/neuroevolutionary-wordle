#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string_view>

#include "model_artifact/winner_artifact.hpp"
#include "model_artifact/winner_artifact_reader.hpp"
#include "training_folder/training_data.hpp"
#include "wordle/word.hpp"

namespace {

using neuroevolution::model_artifact::LoadedWinnerArtifact;
using neuroevolution::model_artifact::TryReadWinnerArtifact;
using neuroevolution::model_artifact::TryWriteWinnerArtifact;
using neuroevolution::model_artifact::WinnerArtifactMetadata;
using neuroevolution::model_artifact::WinnerArtifactPaths;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::wordle::TryMakeWordFromAscii;

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

bool MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1], neuroevolution::wordle::Word &word_out) {
    return TryMakeWordFromAscii(letters, word_out);
}

bool TestWinnerArtifactReaderRoundTripsSavedArtifacts() {
    const std::filesystem::path output_directory =
        std::filesystem::current_path() / "Testing" / "winner-artifact-reader-test-output";
    std::filesystem::remove_all(output_directory);

    TrainingWordCatalog action_space_words{};
    bool ok = MakeWord("CRANE", action_space_words.words[0]);
    ok &= MakeWord("SLATE", action_space_words.words[1]);
    ok &= MakeWord("TRACE", action_space_words.words[2]);
    action_space_words.word_count = 3;

    const std::size_t genome_byte_count = 512;
    std::unique_ptr<std::uint8_t[]> genome_bytes(new std::uint8_t[genome_byte_count]);
    for (std::size_t byte_index = 0; byte_index < genome_byte_count; ++byte_index) {
        genome_bytes[byte_index] = static_cast<std::uint8_t>((byte_index * 7U) % 251U);
    }

    WinnerArtifactMetadata metadata{};
    metadata.best_fitness = 0.25f;
    metadata.generation_index = 12;
    metadata.best_index = 4;
    metadata.best_slot_index = 19U;
    metadata.action_count = action_space_words.word_count;
    metadata.genome_byte_count = genome_byte_count;
    metadata.seed = 99U;
    metadata.action_space_path = "data/action-space-randomised.txt";

    WinnerArtifactPaths paths{};
    ok &= TryWriteWinnerArtifact(output_directory, genome_bytes.get(), action_space_words, metadata, paths);
    if (!ok) {
        std::filesystem::remove_all(output_directory);
        return false;
    }

    LoadedWinnerArtifact artifact{};
    ok &= TryReadWinnerArtifact(paths.binary_path, paths.metadata_path, artifact);
    ok &= ExpectTrue(artifact.metadata.generation_index == metadata.generation_index,
                     "Expected reader to recover generation index");
    ok &= ExpectTrue(artifact.metadata.best_index == metadata.best_index, "Expected reader to recover best index");
    ok &= ExpectTrue(artifact.metadata.best_slot_index == metadata.best_slot_index,
                     "Expected reader to recover best slot index");
    ok &=
        ExpectTrue(artifact.metadata.action_count == metadata.action_count, "Expected reader to recover action count");
    ok &= ExpectTrue(artifact.metadata.genome_byte_count == metadata.genome_byte_count,
                     "Expected reader to recover genome byte count");
    ok &= ExpectTrue(artifact.action_space_words.word_count == action_space_words.word_count,
                     "Expected reader to recover embedded action-space words");
    ok &= ExpectTrue(artifact.genome_bytes != nullptr, "Expected reader to load the binary genome payload");
    ok &= ExpectTrue(std::memcmp(artifact.genome_bytes.get(), genome_bytes.get(), genome_byte_count) == 0,
                     "Expected reader to preserve the saved genome payload exactly");
    ok &= ExpectTrue(artifact.action_space_words.words[0] == action_space_words.words[0],
                     "Expected reader to preserve the first action-space word");
    ok &= ExpectTrue(artifact.action_space_words.words[2] == action_space_words.words[2],
                     "Expected reader to preserve the last action-space word");

    std::filesystem::remove_all(output_directory);
    return ok;
}

} // namespace

int main() {
    if (!TestWinnerArtifactReaderRoundTripsSavedArtifacts()) {
        return 1;
    }

    std::cout << "PASS: winner_artifact_reader_test\n";
    return 0;
}
