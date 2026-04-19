#pragma once

#include <string>
#include <string_view>

#include "training_folder/training_data.hpp"
#include "wordle/wordle_grid.hpp"

namespace neuroevolution::play_wordle {

enum class SolutionPromptParseStatus {
    kSolutionAccepted = 0,
    kExitRequested = 1,
    kInvalid = 2,
};

struct SolutionPromptParseResult {
    SolutionPromptParseStatus status = SolutionPromptParseStatus::kInvalid;
    wordle::Word solution{};
    std::string message{};
};

std::string WordToAsciiString(const wordle::Word &word);

bool DoesCatalogContainWord(const training_folder::TrainingWordCatalog &catalog, const wordle::Word &word) noexcept;

SolutionPromptParseResult ParseSolutionPrompt(std::string_view input,
                                              const training_folder::TrainingWordCatalog &action_space_words);

std::string RenderWordleGridAnsi(const wordle::WordleGrid &grid);

} // namespace neuroevolution::play_wordle
