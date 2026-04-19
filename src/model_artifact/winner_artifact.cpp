#include "model_artifact/winner_artifact.hpp"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string_view>

namespace neuroevolution::model_artifact {

namespace {

bool IsValidWinnerArtifactInputs(const WinnerArtifactMetadata &metadata,
                                 const training_folder::TrainingWordCatalog &action_space_words) noexcept {
    return (metadata.action_count > 0) && (metadata.genome_byte_count > 0) &&
           training_folder::IsValidTrainingWordCatalog(action_space_words) &&
           (metadata.action_count <= action_space_words.word_count);
}

std::string EscapeJsonString(const std::string_view input) {
    std::string escaped{};
    escaped.reserve(input.size());

    for (const char character : input) {
        switch (character) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped += character;
            break;
        }
    }

    return escaped;
}

bool TryGetLocalTimestamp(std::tm &local_time_out) {
    const std::time_t now = std::time(nullptr);
#if defined(_WIN32)
    return localtime_s(&local_time_out, &now) == 0;
#else
    return localtime_r(&now, &local_time_out) != nullptr;
#endif
}

std::string FormatTimestampForFilename(const std::tm &local_time) {
    std::ostringstream stream{};
    stream << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S");
    return stream.str();
}

std::string FormatTimestampForJson(const std::tm &local_time) {
    std::ostringstream stream{};
    stream << std::put_time(&local_time, "%Y-%m-%dT%H:%M:%S");
    return stream.str();
}

std::string WordToAsciiString(const wordle::Word &word) {
    std::string text(wordle::kWordLength, 'A');
    for (std::size_t position = 0; position < wordle::kWordLength; ++position) {
        text[position] = static_cast<char>('A' + word.letter_indices[position]);
    }

    return text;
}

bool TryOpenUniqueArtifactFiles(const std::filesystem::path &directory, const std::string &stem,
                                std::ofstream &binary_stream_out, std::ofstream &metadata_stream_out,
                                WinnerArtifactPaths &paths_out) {
    for (std::size_t suffix = 0; suffix < 1000; ++suffix) {
        const std::string candidate_stem =
            (suffix == 0) ? stem : (stem + "-" + std::to_string(static_cast<unsigned long long>(suffix + 1)));

        const std::filesystem::path binary_path = directory / (candidate_stem + ".bin");
        const std::filesystem::path metadata_path = directory / (candidate_stem + ".json");
        if (std::filesystem::exists(binary_path) || std::filesystem::exists(metadata_path)) {
            continue;
        }

        std::ofstream binary_stream(binary_path, std::ios::binary);
        if (!binary_stream.is_open()) {
            return false;
        }

        std::ofstream metadata_stream(metadata_path, std::ios::binary);
        if (!metadata_stream.is_open()) {
            binary_stream.close();
            std::filesystem::remove(binary_path);
            return false;
        }

        binary_stream_out = std::move(binary_stream);
        metadata_stream_out = std::move(metadata_stream);
        paths_out.binary_path = binary_path;
        paths_out.metadata_path = metadata_path;
        return true;
    }

    return false;
}

} // namespace

bool TryWriteWinnerArtifact(const std::filesystem::path &directory, const std::uint8_t *genome_bytes,
                            const training_folder::TrainingWordCatalog &action_space_words,
                            const WinnerArtifactMetadata &metadata, WinnerArtifactPaths &paths_out) {
    paths_out = {};
    if ((genome_bytes == nullptr) || !IsValidWinnerArtifactInputs(metadata, action_space_words)) {
        return false;
    }

    std::error_code filesystem_error{};
    if (!std::filesystem::create_directories(directory, filesystem_error) && filesystem_error) {
        return false;
    }

    std::tm local_time{};
    if (!TryGetLocalTimestamp(local_time)) {
        return false;
    }

    const std::string filename_timestamp = FormatTimestampForFilename(local_time);
    paths_out.timestamp_local = FormatTimestampForJson(local_time);

    std::ostringstream stem_stream{};
    stem_stream << "winner-" << filename_timestamp << "-g" << metadata.generation_index << "-seed" << metadata.seed;

    std::ofstream binary_stream{};
    std::ofstream metadata_stream{};
    if (!TryOpenUniqueArtifactFiles(directory, stem_stream.str(), binary_stream, metadata_stream, paths_out)) {
        return false;
    }

    binary_stream.write(reinterpret_cast<const char *>(genome_bytes),
                        static_cast<std::streamsize>(metadata.genome_byte_count));
    if (!binary_stream.good()) {
        binary_stream.close();
        metadata_stream.close();
        std::filesystem::remove(paths_out.binary_path);
        std::filesystem::remove(paths_out.metadata_path);
        paths_out = {};
        return false;
    }
    binary_stream.close();

    metadata_stream << "{\n"
                    << "  \"format_version\": 2,\n"
                    << "  \"timestamp_local\": \"" << EscapeJsonString(paths_out.timestamp_local) << "\",\n"
                    << "  \"generation_index\": " << metadata.generation_index << ",\n"
                    << "  \"best_fitness\": " << metadata.best_fitness << ",\n"
                    << "  \"best_index\": " << metadata.best_index << ",\n"
                    << "  \"best_slot_index\": " << metadata.best_slot_index << ",\n"
                    << "  \"action_count\": " << metadata.action_count << ",\n"
                    << "  \"genome_byte_count\": " << metadata.genome_byte_count << ",\n"
                    << "  \"seed\": " << metadata.seed << ",\n"
                    << "  \"action_space_path\": \"" << EscapeJsonString(metadata.action_space_path.string())
                    << "\",\n"
                    << "  \"action_space_words\": [";
    for (std::size_t word_index = 0; word_index < metadata.action_count; ++word_index) {
        if (word_index != 0) {
            metadata_stream << ", ";
        }

        metadata_stream << '"' << EscapeJsonString(WordToAsciiString(action_space_words.words[word_index])) << '"';
    }
    metadata_stream << "]\n}\n";
    if (!metadata_stream.good()) {
        metadata_stream.close();
        std::filesystem::remove(paths_out.binary_path);
        std::filesystem::remove(paths_out.metadata_path);
        paths_out = {};
        return false;
    }

    metadata_stream.close();
    return true;
}

} // namespace neuroevolution::model_artifact
