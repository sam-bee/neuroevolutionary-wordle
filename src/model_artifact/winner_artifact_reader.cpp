#include "model_artifact/winner_artifact_reader.hpp"

#include <charconv>
#include <fstream>
#include <iterator>
#include <new>
#include <string>
#include <string_view>

namespace neuroevolution::model_artifact {

namespace {

constexpr int kSupportedWinnerArtifactFormatVersion = 2;

bool ReadWholeFile(const std::filesystem::path &path, std::string &contents_out) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }

    contents_out.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    return stream.good() || stream.eof();
}

bool ReadWholeBinaryFile(const std::filesystem::path &path, std::unique_ptr<std::uint8_t[]> &bytes_out,
                         std::size_t &byte_count_out) {
    bytes_out.reset();
    byte_count_out = 0;

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream.is_open()) {
        return false;
    }

    const std::streamsize size = stream.tellg();
    if (size <= 0) {
        return false;
    }

    stream.seekg(0, std::ios::beg);
    bytes_out.reset(new (std::nothrow) std::uint8_t[static_cast<std::size_t>(size)]);
    if (bytes_out == nullptr) {
        return false;
    }

    if (!stream.read(reinterpret_cast<char *>(bytes_out.get()), size)) {
        bytes_out.reset();
        return false;
    }

    byte_count_out = static_cast<std::size_t>(size);
    return true;
}

bool LocateJsonValue(const std::string &json, const std::string_view key, std::size_t &value_begin_out) {
    const std::string needle = "\"" + std::string(key) + "\"";
    const std::size_t key_position = json.find(needle);
    if (key_position == std::string::npos) {
        return false;
    }

    const std::size_t colon_position = json.find(':', key_position + needle.size());
    if (colon_position == std::string::npos) {
        return false;
    }

    value_begin_out = colon_position + 1;
    while ((value_begin_out < json.size()) &&
           ((json[value_begin_out] == ' ') || (json[value_begin_out] == '\n') || (json[value_begin_out] == '\r') ||
            (json[value_begin_out] == '\t'))) {
        ++value_begin_out;
    }

    return value_begin_out < json.size();
}

bool TryReadJsonString(const std::string &json, const std::string_view key, std::string &value_out) {
    std::size_t position = 0;
    if (!LocateJsonValue(json, key, position) || (json[position] != '"')) {
        return false;
    }

    ++position;
    value_out.clear();
    while (position < json.size()) {
        const char current = json[position];
        if (current == '"') {
            return true;
        }

        if (current == '\\') {
            ++position;
            if (position >= json.size()) {
                return false;
            }

            switch (json[position]) {
            case '\\':
            case '"':
                value_out += json[position];
                break;
            case 'n':
                value_out += '\n';
                break;
            case 'r':
                value_out += '\r';
                break;
            case 't':
                value_out += '\t';
                break;
            default:
                return false;
            }
        } else {
            value_out += current;
        }

        ++position;
    }

    return false;
}

template <typename NumericType>
bool TryReadJsonNumber(const std::string &json, const std::string_view key, NumericType &value_out) {
    std::size_t position = 0;
    if (!LocateJsonValue(json, key, position)) {
        return false;
    }

    const char *begin = json.data() + position;
    const char *end = begin;
    while ((end < (json.data() + json.size())) &&
           ((*end == '-') || (*end == '+') || (*end == '.') || ((*end >= '0') && (*end <= '9')) || (*end == 'e') ||
            (*end == 'E'))) {
        ++end;
    }

    if (begin == end) {
        return false;
    }

    const auto result = std::from_chars(begin, end, value_out);
    return result.ec == std::errc{};
}

bool TryReadJsonWordArray(const std::string &json, const std::string_view key,
                          training_folder::TrainingWordCatalog &catalog_out) {
    std::size_t position = 0;
    if (!LocateJsonValue(json, key, position) || (json[position] != '[')) {
        return false;
    }

    ++position;
    catalog_out = {};

    while (position < json.size()) {
        while ((position < json.size()) &&
               ((json[position] == ' ') || (json[position] == '\n') || (json[position] == '\r') ||
                (json[position] == '\t') || (json[position] == ','))) {
            ++position;
        }

        if (position >= json.size()) {
            return false;
        }

        if (json[position] == ']') {
            return catalog_out.word_count > 0;
        }

        if (json[position] != '"' || (catalog_out.word_count >= training_folder::kTrainingWordCatalogCapacity)) {
            return false;
        }

        ++position;
        std::string word_text{};
        while ((position < json.size()) && (json[position] != '"')) {
            word_text += json[position];
            ++position;
        }

        if ((position >= json.size()) || (word_text.size() != wordle::kWordLength)) {
            return false;
        }

        ++position;
        wordle::Word word{};
        const char letters[wordle::kWordLength + 1] = {word_text[0], word_text[1], word_text[2], word_text[3],
                                                       word_text[4], '\0'};
        if (!wordle::TryMakeWordFromAscii(letters, word)) {
            return false;
        }

        catalog_out.words[catalog_out.word_count] = word;
        ++catalog_out.word_count;
    }

    return false;
}

bool IsValidLoadedWinnerArtifact(const LoadedWinnerArtifact &artifact) noexcept {
    return training_folder::IsValidTrainingWordCatalog(artifact.action_space_words) &&
           (artifact.metadata.action_count == artifact.action_space_words.word_count) &&
           (artifact.metadata.action_count > 0) && (artifact.metadata.genome_byte_count > 0) &&
           (artifact.genome_bytes != nullptr);
}

} // namespace

bool TryReadWinnerArtifact(const std::filesystem::path &binary_path, const std::filesystem::path &metadata_path,
                           LoadedWinnerArtifact &artifact_out) {
    artifact_out = {};

    std::string metadata_json{};
    std::string action_space_path_text{};
    std::size_t format_version = 0;
    if (!ReadWholeFile(metadata_path, metadata_json) ||
        !TryReadJsonNumber(metadata_json, "format_version", format_version) ||
        (static_cast<int>(format_version) != kSupportedWinnerArtifactFormatVersion)) {
        return false;
    }

    if (!TryReadJsonString(metadata_json, "timestamp_local", artifact_out.timestamp_local) ||
        !TryReadJsonNumber(metadata_json, "generation_index", artifact_out.metadata.generation_index) ||
        !TryReadJsonNumber(metadata_json, "best_fitness", artifact_out.metadata.best_fitness) ||
        !TryReadJsonNumber(metadata_json, "best_index", artifact_out.metadata.best_index) ||
        !TryReadJsonNumber(metadata_json, "best_slot_index", artifact_out.metadata.best_slot_index) ||
        !TryReadJsonNumber(metadata_json, "action_count", artifact_out.metadata.action_count) ||
        !TryReadJsonNumber(metadata_json, "genome_byte_count", artifact_out.metadata.genome_byte_count) ||
        !TryReadJsonNumber(metadata_json, "seed", artifact_out.metadata.seed) ||
        !TryReadJsonString(metadata_json, "action_space_path", action_space_path_text) ||
        !TryReadJsonWordArray(metadata_json, "action_space_words", artifact_out.action_space_words)) {
        artifact_out = {};
        return false;
    }
    artifact_out.metadata.action_space_path = action_space_path_text;

    std::size_t binary_byte_count = 0;
    if (!ReadWholeBinaryFile(binary_path, artifact_out.genome_bytes, binary_byte_count) ||
        (binary_byte_count != artifact_out.metadata.genome_byte_count) || !IsValidLoadedWinnerArtifact(artifact_out)) {
        artifact_out = {};
        return false;
    }

    return true;
}

} // namespace neuroevolution::model_artifact
