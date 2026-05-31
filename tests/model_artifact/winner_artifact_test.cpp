#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include "model_artifact/winner_artifact.hpp"
#include "training_folder/training_data.hpp"
#include "wordle/word.hpp"

namespace {

using neuroevolution::model_artifact::TryWriteWinnerArtifact;
using neuroevolution::model_artifact::TryWriteWinnerArtifactToPaths;
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

bool ReadWholeFile(const std::filesystem::path &path, std::string &contents_out) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }

    contents_out.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    return stream.good() || stream.eof();
}

bool MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1], neuroevolution::wordle::Word &word_out) {
    return TryMakeWordFromAscii(letters, word_out);
}

bool TestWinnerArtifactWriterPersistsBinaryAndMetadata() {
    const std::filesystem::path output_directory =
        std::filesystem::current_path() / "Testing" / "winner-artifact-test-output";
    std::filesystem::remove_all(output_directory);

    TrainingWordCatalog action_space_words{};
    bool ok = MakeWord("CRANE", action_space_words.words[0]);
    ok &= MakeWord("SLATE", action_space_words.words[1]);
    action_space_words.word_count = 2;

    const std::size_t genome_byte_count = 384;
    std::unique_ptr<std::uint8_t[]> genome_bytes(new std::uint8_t[genome_byte_count]);
    for (std::size_t byte_index = 0; byte_index < genome_byte_count; ++byte_index) {
        genome_bytes[byte_index] = static_cast<std::uint8_t>(byte_index % 251U);
    }

    WinnerArtifactMetadata metadata{};
    metadata.best_fitness = 0.75f;
    metadata.generation_index = 7;
    metadata.best_index = 3;
    metadata.best_slot_index = 11U;
    metadata.action_count = 2;
    metadata.genome_byte_count = genome_byte_count;
    metadata.seed = 12345U;
    metadata.action_space_path = "data/action-space-randomised.txt";

    WinnerArtifactPaths paths{};
    ok &= TryWriteWinnerArtifact(output_directory, genome_bytes.get(), action_space_words, metadata, paths);
    ok &= ExpectTrue(!paths.timestamp_local.empty(), "Expected writer to return a human-readable local timestamp");
    ok &= ExpectTrue(paths.binary_path.extension() == ".bin", "Expected writer to create a .bin artifact");
    ok &= ExpectTrue(paths.metadata_path.extension() == ".json", "Expected writer to create a .json sidecar");
    ok &= ExpectTrue(paths.binary_path.filename().string().find("winner-") == 0,
                     "Expected binary artifact name to start with winner-");
    ok &= ExpectTrue(paths.binary_path.filename().string().find("-seed12345") != std::string::npos,
                     "Expected binary artifact name to include the seed");
    if (!ok) {
        std::filesystem::remove_all(output_directory);
        return false;
    }

    std::string binary_contents{};
    std::string metadata_contents{};
    ok &= ReadWholeFile(paths.binary_path, binary_contents);
    ok &= ReadWholeFile(paths.metadata_path, metadata_contents);
    ok &= ExpectTrue(binary_contents.size() == genome_byte_count,
                     "Expected binary artifact size to match the saved genome payload");
    ok &= ExpectTrue(binary_contents ==
                         std::string(reinterpret_cast<const char *>(genome_bytes.get()), genome_byte_count),
                     "Expected binary artifact contents to match the saved genome payload");
    ok &= ExpectTrue(metadata_contents.find("\"generation_index\": 7") != std::string::npos,
                     "Expected metadata sidecar to record the generation index");
    ok &= ExpectTrue(metadata_contents.find("\"best_slot_index\": 11") != std::string::npos,
                     "Expected metadata sidecar to record the winning slot index");
    ok &= ExpectTrue(metadata_contents.find("\"seed\": 12345") != std::string::npos,
                     "Expected metadata sidecar to record the seed");
    ok &= ExpectTrue(metadata_contents.find("data/action-space-randomised.txt") != std::string::npos,
                     "Expected metadata sidecar to record the action-space path");
    ok &= ExpectTrue(metadata_contents.find("\"action_space_words\": [\"CRANE\", \"SLATE\"]") != std::string::npos,
                     "Expected metadata sidecar to embed the active action-space words");

    std::filesystem::remove_all(output_directory);
    return ok;
}

bool TestWinnerArtifactWriterPersistsExactSidecarPaths() {
    const std::filesystem::path output_directory =
        std::filesystem::current_path() / "Testing" / "winner-artifact-exact-path-test-output";
    std::filesystem::remove_all(output_directory);

    TrainingWordCatalog action_space_words{};
    bool ok = MakeWord("CRANE", action_space_words.words[0]);
    action_space_words.word_count = 1;

    const std::size_t genome_byte_count = 128;
    std::unique_ptr<std::uint8_t[]> genome_bytes(new std::uint8_t[genome_byte_count]);
    for (std::size_t byte_index = 0; byte_index < genome_byte_count; ++byte_index) {
        genome_bytes[byte_index] = static_cast<std::uint8_t>(17U + (byte_index % 53U));
    }

    WinnerArtifactMetadata metadata{};
    metadata.best_fitness = 0.42f;
    metadata.generation_index = 19;
    metadata.best_index = 5;
    metadata.best_slot_index = 31U;
    metadata.action_count = 1;
    metadata.genome_byte_count = genome_byte_count;
    metadata.seed = 987U;
    metadata.action_space_path = "data/action-space-randomised.txt";

    const std::filesystem::path binary_path = output_directory / "ga-runtime.best.bin";
    const std::filesystem::path metadata_path = output_directory / "ga-runtime.best.json";
    WinnerArtifactPaths paths{};
    ok &= TryWriteWinnerArtifactToPaths(binary_path, metadata_path, genome_bytes.get(), action_space_words, metadata,
                                        paths);
    ok &= ExpectTrue(paths.binary_path == binary_path, "Expected exact-path writer to use the requested binary path");
    ok &=
        ExpectTrue(paths.metadata_path == metadata_path, "Expected exact-path writer to use the requested JSON path");
    if (!ok) {
        std::filesystem::remove_all(output_directory);
        return false;
    }

    std::string binary_contents{};
    std::string metadata_contents{};
    ok &= ReadWholeFile(binary_path, binary_contents);
    ok &= ReadWholeFile(metadata_path, metadata_contents);
    ok &= ExpectTrue(binary_contents ==
                         std::string(reinterpret_cast<const char *>(genome_bytes.get()), genome_byte_count),
                     "Expected exact-path writer to persist the requested genome payload");
    ok &= ExpectTrue(metadata_contents.find("\"generation_index\": 19") != std::string::npos,
                     "Expected exact-path metadata to record the generation index");
    ok &= ExpectTrue(metadata_contents.find("\"best_fitness\": 0.42") != std::string::npos,
                     "Expected exact-path metadata to record the fitness");

    std::filesystem::remove_all(output_directory);
    return ok;
}

} // namespace

int main() {
    if (!TestWinnerArtifactWriterPersistsBinaryAndMetadata()) {
        return 1;
    }

    if (!TestWinnerArtifactWriterPersistsExactSidecarPaths()) {
        return 1;
    }

    std::cout << "PASS: winner_artifact_test\n";
    return 0;
}
