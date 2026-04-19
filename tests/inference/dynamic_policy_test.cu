#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "inference/dynamic_policy.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "model/policy_model/policy_model.hpp"
#include "training_folder/training_data.hpp"
#include "wordle/word.hpp"
#include "wordle/wordle_grid.hpp"

namespace {

using neuroevolution::inference::dynamic_policy::DynamicInferenceStatusCode;
using neuroevolution::inference::dynamic_policy::HasGridAlreadyGuessedWord;
using neuroevolution::inference::dynamic_policy::SelectNextGuessFromDynamicGenome;
using neuroevolution::inference::dynamic_policy::TrySelectBestDynamicAction;
using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::model::policy_model::PolicyVector;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::wordle::LetterIndexFromAscii;
using neuroevolution::wordle::MakeWordleGrid;
using neuroevolution::wordle::TryAppendGuess;
using neuroevolution::wordle::TryMakeWordFromAscii;
using neuroevolution::wordle::Word;
using neuroevolution::wordle::WordleGrid;

constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr std::size_t kActionCount = 2;
constexpr int kStatusBestActionSelectionFailed = 1;
constexpr int kStatusNextGuessSelectionFailed = 2;
constexpr int kStatusRepeatMaskCheckFailed = 3;

Word MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    Word word{};
    if (!TryMakeWordFromAscii(letters, word)) {
        throw std::invalid_argument("Dynamic-policy test words must contain exactly five uppercase letters.");
    }

    return word;
}

bool CheckCuda(const cudaError_t error, const std::string_view action) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA failure during " << action << ": " << cudaGetErrorString(error) << '\n';
        return false;
    }

    return true;
}

bool SelectVisibleCudaDevice() {
    int device_count = 0;
    if (!CheckCuda(cudaGetDeviceCount(&device_count), "querying visible CUDA device count")) {
        return false;
    }

    if (device_count <= kSelectedVisibleDeviceIndex) {
        std::cerr << "FAIL: selected logical device index " << kSelectedVisibleDeviceIndex
                  << " is not available in this process\n";
        return false;
    }

    return CheckCuda(cudaSetDevice(kSelectedVisibleDeviceIndex), "selecting visible CUDA device");
}

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

bool ExpectWordEquals(const Word &actual, const Word &expected, const std::string_view label) {
    for (std::size_t position = 0; position < neuroevolution::wordle::kWordLength; ++position) {
        if (actual.letter_indices[position] != expected.letter_indices[position]) {
            std::cerr << "FAIL: " << label << " mismatch at position " << position << '\n';
            return false;
        }
    }

    return true;
}

__global__ void DynamicPolicyKernel(const TrainingWordCatalog *catalog, const std::uint8_t *genome_bytes,
                                    SelectedAction *best_action_out, SelectedAction *next_action_out,
                                    int *status_out) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    PolicyVector policy_vector{};
    policy_vector[LetterIndexFromAscii('S')] = 1.0f;

    if (!TrySelectBestDynamicAction(policy_vector, *catalog, genome_bytes, kActionCount, *best_action_out)) {
        *status_out = kStatusBestActionSelectionFailed;
        return;
    }

    WordleGrid grid = MakeWordleGrid(catalog->words[1]);
    if (!TryAppendGuess(grid, catalog->words[0]) || !HasGridAlreadyGuessedWord(grid, catalog->words[0])) {
        *status_out = kStatusRepeatMaskCheckFailed;
        return;
    }

    const DynamicInferenceStatusCode inference_status =
        SelectNextGuessFromDynamicGenome(grid, *catalog, genome_bytes, kActionCount, *next_action_out);
    if (inference_status != DynamicInferenceStatusCode::kOk) {
        *status_out = kStatusNextGuessSelectionFailed;
        return;
    }

    *status_out = 0;
}

bool TestDynamicPolicyHelpersOnDevice() {
    TrainingWordCatalog catalog{};
    catalog.words[0] = MakeWord("CRANE");
    catalog.words[1] = MakeWord("SLATE");
    catalog.word_count = kActionCount;

    const std::size_t genome_byte_count =
        neuroevolution::genetic_algorithm::genome::ComputeDynamicGenomeStrideBytes(kActionCount);
    std::unique_ptr<std::uint8_t[]> host_genome_bytes(new std::uint8_t[genome_byte_count]());

    TrainingWordCatalog *device_catalog = nullptr;
    std::uint8_t *device_genome_bytes = nullptr;
    SelectedAction *device_best_action = nullptr;
    SelectedAction *device_next_action = nullptr;
    int *device_status = nullptr;

    bool ok = true;

    ok &= CheckCuda(cudaMalloc(&device_catalog, sizeof(catalog)), "allocating device action catalog");
    ok &= CheckCuda(cudaMalloc(&device_genome_bytes, genome_byte_count), "allocating device genome buffer");
    ok &= CheckCuda(cudaMalloc(&device_best_action, sizeof(SelectedAction)), "allocating device best-action buffer");
    ok &= CheckCuda(cudaMalloc(&device_next_action, sizeof(SelectedAction)), "allocating device next-action buffer");
    ok &= CheckCuda(cudaMalloc(&device_status, sizeof(int)), "allocating device status buffer");

    if (ok) {
        ok &= CheckCuda(cudaMemcpy(device_catalog, &catalog, sizeof(catalog), cudaMemcpyHostToDevice),
                        "copying action catalog to device");
        ok &= CheckCuda(cudaMemcpy(device_genome_bytes, host_genome_bytes.get(), genome_byte_count, cudaMemcpyHostToDevice),
                        "copying genome bytes to device");
    }

    if (ok) {
        DynamicPolicyKernel<<<1, 1>>>(device_catalog, device_genome_bytes, device_best_action, device_next_action,
                                      device_status);
        ok &= CheckCuda(cudaGetLastError(), "launching dynamic-policy kernel");
        ok &= CheckCuda(cudaDeviceSynchronize(), "waiting for dynamic-policy kernel completion");
    }

    SelectedAction host_best_action{};
    SelectedAction host_next_action{};
    int host_status = -1;
    if (ok) {
        ok &= CheckCuda(cudaMemcpy(&host_best_action, device_best_action, sizeof(SelectedAction), cudaMemcpyDeviceToHost),
                        "copying best action back to host");
        ok &= CheckCuda(cudaMemcpy(&host_next_action, device_next_action, sizeof(SelectedAction), cudaMemcpyDeviceToHost),
                        "copying next action back to host");
        ok &= CheckCuda(cudaMemcpy(&host_status, device_status, sizeof(int), cudaMemcpyDeviceToHost),
                        "copying status back to host");
    }

    if (ok) {
        ok &= ExpectTrue(host_status == 0, "Expected dynamic-policy kernel to succeed");
        ok &= ExpectTrue(host_best_action.action_index == 1, "Expected best-action helper to pick the second word");
        ok &= ExpectWordEquals(host_best_action.word, catalog.words[1],
                               "Expected best-action helper to select SLATE");
        ok &= ExpectTrue(host_next_action.action_index == 1,
                         "Expected repeat-guess masking to skip the previously guessed first word");
        ok &= ExpectWordEquals(host_next_action.word, catalog.words[1],
                               "Expected next-guess helper to select SLATE after masking CRANE");
    }

    if (device_status != nullptr) {
        cudaFree(device_status);
    }
    if (device_next_action != nullptr) {
        cudaFree(device_next_action);
    }
    if (device_best_action != nullptr) {
        cudaFree(device_best_action);
    }
    if (device_genome_bytes != nullptr) {
        cudaFree(device_genome_bytes);
    }
    if (device_catalog != nullptr) {
        cudaFree(device_catalog);
    }

    return ok;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    if (!TestDynamicPolicyHelpersOnDevice()) {
        return 1;
    }

    std::cout << "PASS: dynamic_policy_test\n";
    return 0;
}
