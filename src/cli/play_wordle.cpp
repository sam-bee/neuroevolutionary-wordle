#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "inference/single_model_device_runtime.hpp"
#include "model_artifact/winner_artifact_reader.hpp"
#include "play_wordle/interactive_ui.hpp"
#include "wordle/wordle_grid.hpp"

namespace {

using neuroevolution::inference::DestroySingleModelDeviceRuntime;
using neuroevolution::inference::SingleModelDeviceRuntime;
using neuroevolution::inference::SingleModelDeviceRuntimeStatusCode;
using neuroevolution::inference::TryCreateSingleModelDeviceRuntime;
using neuroevolution::inference::TrySelectNextGuessWithSingleModelDeviceRuntime;
using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::model_artifact::LoadedWinnerArtifact;
using neuroevolution::model_artifact::TryReadWinnerArtifact;
using neuroevolution::play_wordle::ParseSolutionPrompt;
using neuroevolution::play_wordle::RenderWordleGridAnsi;
using neuroevolution::play_wordle::SolutionPromptParseStatus;
using neuroevolution::play_wordle::WordToAsciiString;
using neuroevolution::wordle::MakeWordleGrid;
using neuroevolution::wordle::TryAppendGuess;
using neuroevolution::wordle::Word;
using neuroevolution::wordle::WordleGrid;

struct PlayWordleCliConfig {
    std::filesystem::path binary_path{};
    std::filesystem::path metadata_path{};
};

void PrintUsage(const char *program_name) { std::cerr << "Usage: " << program_name << " <winner.bin> <winner.json>\n"; }

bool TryParseCliArguments(const int argc, char **argv, PlayWordleCliConfig &config_out) {
    config_out = {};
    if (argc != 3) {
        return false;
    }

    config_out.binary_path = argv[1];
    config_out.metadata_path = argv[2];
    return true;
}

std::string_view DeviceRuntimeStatusMessage(const SingleModelDeviceRuntimeStatusCode status) {
    switch (status) {
    case SingleModelDeviceRuntimeStatusCode::kOk:
        return "ok";
    case SingleModelDeviceRuntimeStatusCode::kInvalidRuntime:
        return "invalid runtime";
    case SingleModelDeviceRuntimeStatusCode::kCudaMemcpyFailed:
        return "CUDA memcpy failed";
    case SingleModelDeviceRuntimeStatusCode::kKernelLaunchFailed:
        return "CUDA kernel launch failed";
    case SingleModelDeviceRuntimeStatusCode::kKernelExecutionFailed:
        return "CUDA kernel execution failed";
    case SingleModelDeviceRuntimeStatusCode::kPolicyForwardFailed:
        return "policy forward failed";
    case SingleModelDeviceRuntimeStatusCode::kActionSelectionFailed:
        return "action selection failed";
    }

    return "unknown runtime status";
}

bool TryPlaySingleGame(const SingleModelDeviceRuntime &runtime, const Word &solution) {
    WordleGrid grid = MakeWordleGrid(solution);

    std::cout << '\n' << RenderWordleGridAnsi(grid) << "\n";
    while (!grid.IsFinished()) {
        SelectedAction selected_action{};
        SingleModelDeviceRuntimeStatusCode status = SingleModelDeviceRuntimeStatusCode::kInvalidRuntime;
        if (!TrySelectNextGuessWithSingleModelDeviceRuntime(runtime, grid, selected_action, &status)) {
            std::cerr << "Device inference failed: " << DeviceRuntimeStatusMessage(status) << '\n';
            return false;
        }

        if (!TryAppendGuess(grid, selected_action.word)) {
            std::cerr << "Could not append model guess " << WordToAsciiString(selected_action.word)
                      << " to the current board.\n";
            return false;
        }

        std::cout << "\nGuess " << grid.turn_count << ": " << WordToAsciiString(selected_action.word) << '\n';
        std::cout << RenderWordleGridAnsi(grid) << "\n";
    }

    if (grid.IsWon()) {
        std::cout << "\nSolved in " << grid.turn_count << " turns.\n";
    } else {
        std::cout << "\nFailed to solve within " << neuroevolution::wordle::kMaxTurnCount << " turns.\n";
    }

    return true;
}

} // namespace

int main(int argc, char **argv) {
    PlayWordleCliConfig cli_config{};
    if (!TryParseCliArguments(argc, argv, cli_config)) {
        PrintUsage(argv[0]);
        return 1;
    }

    LoadedWinnerArtifact artifact{};
    if (!TryReadWinnerArtifact(cli_config.binary_path, cli_config.metadata_path, artifact)) {
        std::cerr << "Could not read winner artifact from " << cli_config.binary_path << " and "
                  << cli_config.metadata_path << ".\n";
        return 1;
    }

    SingleModelDeviceRuntime runtime{};
    if (!TryCreateSingleModelDeviceRuntime(artifact.genome_bytes.get(), artifact.metadata.genome_byte_count,
                                           artifact.action_space_words, runtime)) {
        std::cerr << "Could not initialize the single-model device runtime.\n";
        return 1;
    }

    std::cout << "Loaded model artifact with " << artifact.action_space_words.word_count
              << " action words. Enter a five-letter solution or /exit.\n";

    bool ok = true;
    std::string input_line{};
    while (ok) {
        std::cout << "\nsolution> " << std::flush;
        if (!std::getline(std::cin, input_line)) {
            break;
        }

        const auto parse_result = ParseSolutionPrompt(input_line, artifact.action_space_words);
        if (parse_result.status == SolutionPromptParseStatus::kExitRequested) {
            break;
        }

        if (parse_result.status != SolutionPromptParseStatus::kSolutionAccepted) {
            std::cout << parse_result.message << '\n';
            continue;
        }

        ok = TryPlaySingleGame(runtime, parse_result.solution);
    }

    DestroySingleModelDeviceRuntime(runtime);
    return ok ? 0 : 1;
}
