#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "../policy_model/policy_model_fixture.hpp"
#include "common/fixed_buffer.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "model/policy_model/policy_model.hpp"
#include "wordle/word.hpp"
#include "wordle/wordle_grid.hpp"

namespace {

using neuroevolution::common::FixedBuffer;
using neuroevolution::model::output_embedding::ActionEmbedding;
using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::model::output_embedding::TrySelectBestAction;
using neuroevolution::model::policy_model::PolicyModelParameters;
using neuroevolution::model::policy_model::PolicyVector;
using neuroevolution::model::policy_model::TryForwardPolicyModel;
using neuroevolution::tests::policy_model::PolicyModelGoldenFixture;
using neuroevolution::wordle::MakeWordleGrid;
using neuroevolution::wordle::TryAppendGuess;
using neuroevolution::wordle::TryMakeWordFromAscii;
using neuroevolution::wordle::Word;
using neuroevolution::wordle::WordleGrid;

constexpr float kTolerance = 1.0e-6f;
constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr int kStatusAppendGuessFailed = 1;
constexpr int kStatusPolicyForwardFailed = 2;
constexpr int kStatusActionSelectionFailed = 3;
constexpr std::size_t kActionCount = 3;

Word MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    Word word{};

    if (!TryMakeWordFromAscii(letters, word)) {
        throw std::invalid_argument(
            "Output-embedding device end-to-end test word literal must contain exactly five uppercase letters.");
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

    std::cout << "cudaGetDeviceCount(): " << device_count << '\n';

    for (int device_index = 0; device_index < device_count; ++device_index) {
        cudaDeviceProp properties{};
        if (!CheckCuda(cudaGetDeviceProperties(&properties, device_index), "querying visible CUDA device properties")) {
            return false;
        }

        std::cout << "Visible CUDA device " << device_index << ": " << properties.name << " (CC " << properties.major
                  << '.' << properties.minor << ")\n";
    }

    if (device_count <= kSelectedVisibleDeviceIndex) {
        std::cerr << "FAIL: selected logical device index " << kSelectedVisibleDeviceIndex
                  << " is not available in this process\n";
        return false;
    }

    if (!CheckCuda(cudaSetDevice(kSelectedVisibleDeviceIndex), "selecting visible CUDA device")) {
        return false;
    }

    cudaDeviceProp selected_properties{};
    if (!CheckCuda(cudaGetDeviceProperties(&selected_properties, kSelectedVisibleDeviceIndex),
                   "querying selected CUDA device properties")) {
        return false;
    }

    std::cout << "Selected logical CUDA device: " << kSelectedVisibleDeviceIndex << '\n';
    std::cout << "Selected device properties: " << selected_properties.name << " (CC " << selected_properties.major
              << '.' << selected_properties.minor << ")\n";

    return true;
}

FixedBuffer<ActionEmbedding, kActionCount> MakeActionEmbeddings() {
    FixedBuffer<ActionEmbedding, kActionCount> action_embeddings{};
    action_embeddings[0].word = MakeWord("CABBY");
    action_embeddings[1].word = MakeWord("CACAO");
    action_embeddings[2].word = MakeWord("FUZZY");
    return action_embeddings;
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

__global__ void OutputEmbeddingDeviceEndToEndKernel(const PolicyModelParameters *parameters, const Word *solution,
                                                    const Word *guess, const ActionEmbedding *action_embeddings,
                                                    SelectedAction *selected_action, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    WordleGrid grid = MakeWordleGrid(*solution);
    if (!TryAppendGuess(grid, *guess)) {
        *status = kStatusAppendGuessFailed;
        return;
    }

    PolicyVector policy_output{};
    if (!TryForwardPolicyModel(*parameters, grid, policy_output)) {
        *status = kStatusPolicyForwardFailed;
        return;
    }

    SelectedAction action{};
    if (!TrySelectBestAction(policy_output, action_embeddings, kActionCount, action)) {
        *status = kStatusActionSelectionFailed;
        return;
    }

    *selected_action = action;
    *status = 0;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    const PolicyModelGoldenFixture fixture{};
    const Word solution = fixture.solution;
    const Word guess = fixture.guess;
    const auto action_embeddings = MakeActionEmbeddings();
    const Word expected_word = MakeWord("CACAO");
    constexpr std::size_t kExpectedActionIndex = 1;
    constexpr float kExpectedScore = 24.5f;

    PolicyModelParameters *device_parameters = nullptr;
    Word *device_solution = nullptr;
    Word *device_guess = nullptr;
    ActionEmbedding *device_action_embeddings = nullptr;
    SelectedAction *device_selected_action = nullptr;
    int *device_status = nullptr;

    bool ok = true;

    ok &= CheckCuda(cudaMalloc(&device_parameters, sizeof(fixture.parameters)), "allocating policy-model parameters");
    ok &= CheckCuda(cudaMalloc(&device_solution, sizeof(solution)), "allocating solution input");
    ok &= CheckCuda(cudaMalloc(&device_guess, sizeof(guess)), "allocating guess input");
    ok &= CheckCuda(cudaMalloc(&device_action_embeddings, sizeof(ActionEmbedding) * kActionCount),
                    "allocating action-embedding table");
    ok &= CheckCuda(cudaMalloc(&device_selected_action, sizeof(SelectedAction)), "allocating selected-action buffer");
    ok &= CheckCuda(cudaMalloc(&device_status, sizeof(int)), "allocating status buffer");

    if (ok) {
        ok &= CheckCuda(
            cudaMemcpy(device_parameters, &fixture.parameters, sizeof(fixture.parameters), cudaMemcpyHostToDevice),
            "copying policy-model parameters to device");
        ok &= CheckCuda(cudaMemcpy(device_solution, &solution, sizeof(solution), cudaMemcpyHostToDevice),
                        "copying solution input to device");
        ok &= CheckCuda(cudaMemcpy(device_guess, &guess, sizeof(guess), cudaMemcpyHostToDevice),
                        "copying guess input to device");
        ok &= CheckCuda(cudaMemcpy(device_action_embeddings, action_embeddings.values,
                                   sizeof(ActionEmbedding) * kActionCount, cudaMemcpyHostToDevice),
                        "copying action-embedding table to device");
    }

    if (ok) {
        OutputEmbeddingDeviceEndToEndKernel<<<1, 1>>>(device_parameters, device_solution, device_guess,
                                                      device_action_embeddings, device_selected_action, device_status);
        ok &= CheckCuda(cudaGetLastError(), "launching output-embedding device end-to-end kernel");
        ok &= CheckCuda(cudaDeviceSynchronize(), "waiting for output-embedding device end-to-end kernel completion");
    }

    SelectedAction host_selected_action{};
    int host_status = -1;

    if (ok) {
        ok &= CheckCuda(
            cudaMemcpy(&host_selected_action, device_selected_action, sizeof(SelectedAction), cudaMemcpyDeviceToHost),
            "copying selected action back to host");
        ok &= CheckCuda(cudaMemcpy(&host_status, device_status, sizeof(int), cudaMemcpyDeviceToHost),
                        "copying kernel status back to host");
    }

    if (ok && (host_status == kStatusAppendGuessFailed)) {
        std::cerr << "FAIL: device kernel could not append the test guess to a freshly created grid\n";
        ok = false;
    }

    if (ok && (host_status == kStatusPolicyForwardFailed)) {
        std::cerr << "FAIL: device kernel could not forward the device-built grid through the policy model\n";
        ok = false;
    }

    if (ok && (host_status == kStatusActionSelectionFailed)) {
        std::cerr << "FAIL: device kernel could not select an action from the output embedding table\n";
        ok = false;
    }

    if (ok && (host_status != 0)) {
        std::cerr << "FAIL: device kernel returned unexpected status " << host_status << '\n';
        ok = false;
    }

    if (ok) {
        ok &= (host_selected_action.action_index == kExpectedActionIndex);
        if (!ok) {
            std::cerr << "FAIL: expected selected action index " << kExpectedActionIndex << ", got "
                      << host_selected_action.action_index << '\n';
        }
    }

    if (ok) {
        ok &= ExpectNear(host_selected_action.score, kExpectedScore, "selected action score");
        ok &= ExpectWordEquals(host_selected_action.word, expected_word, "selected action word");
    }

    if (device_status != nullptr) {
        cudaFree(device_status);
    }
    if (device_selected_action != nullptr) {
        cudaFree(device_selected_action);
    }
    if (device_action_embeddings != nullptr) {
        cudaFree(device_action_embeddings);
    }
    if (device_guess != nullptr) {
        cudaFree(device_guess);
    }
    if (device_solution != nullptr) {
        cudaFree(device_solution);
    }
    if (device_parameters != nullptr) {
        cudaFree(device_parameters);
    }

    if (!ok) {
        return 1;
    }

    std::cout << "PASS: output_embedding_device_e2e_test\n";
    return 0;
}
