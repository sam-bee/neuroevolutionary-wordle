#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "../model/policy_model/policy_model_fixture.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "inference/single_model_device_runtime.hpp"
#include "training_folder/training_data.hpp"
#include "wordle/word.hpp"

namespace {

using neuroevolution::inference::DestroySingleModelDeviceRuntime;
using neuroevolution::inference::SingleModelDeviceRuntime;
using neuroevolution::inference::SingleModelDeviceRuntimeStatusCode;
using neuroevolution::inference::TryCreateSingleModelDeviceRuntime;
using neuroevolution::inference::TrySelectNextGuessWithSingleModelDeviceRuntime;
using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::tests::policy_model::PolicyModelGoldenFixture;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::wordle::TryMakeWordFromAscii;
using neuroevolution::wordle::Word;

constexpr float kTolerance = 1.0e-6f;
constexpr int kSelectedVisibleDeviceIndex = 0;

Word MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    Word word{};
    if (!TryMakeWordFromAscii(letters, word)) {
        throw std::invalid_argument("Single-model runtime test words must contain exactly five uppercase letters.");
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

bool ExpectNear(const float actual, const float expected, const std::string_view label) {
    if (std::fabs(actual - expected) > kTolerance) {
        std::cerr << "FAIL: " << label << " expected " << expected << ", got " << actual << '\n';
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

bool TestSingleModelDeviceRuntimeSelectsBestAction() {
    const PolicyModelGoldenFixture fixture{};

    TrainingWordCatalog action_space_words{};
    action_space_words.words[0] = MakeWord("CABBY");
    action_space_words.words[1] = MakeWord("CACAO");
    action_space_words.words[2] = MakeWord("FUZZY");
    action_space_words.word_count = 3;

    const std::size_t genome_byte_count =
        neuroevolution::genetic_algorithm::genome::ComputeDynamicGenomeStrideBytes(action_space_words.word_count);
    std::unique_ptr<std::uint8_t[]> genome_bytes(new std::uint8_t[genome_byte_count]());
    neuroevolution::genetic_algorithm::genome::GenomePolicyModelParameters(genome_bytes.get()) = fixture.parameters;

    SingleModelDeviceRuntime runtime{};
    bool ok = TryCreateSingleModelDeviceRuntime(genome_bytes.get(), genome_byte_count, action_space_words, runtime);
    ok &= ExpectTrue(ok, "Expected single-model runtime creation to succeed");
    if (!ok) {
        DestroySingleModelDeviceRuntime(runtime);
        return false;
    }

    SelectedAction selected_action{};
    SingleModelDeviceRuntimeStatusCode status = SingleModelDeviceRuntimeStatusCode::kInvalidRuntime;
    ok &= TrySelectNextGuessWithSingleModelDeviceRuntime(runtime, fixture.MakeSingleTurnGrid(), selected_action, &status);
    ok &= ExpectTrue(status == SingleModelDeviceRuntimeStatusCode::kOk,
                     "Expected runtime to report successful device inference");
    ok &= ExpectTrue(selected_action.action_index == 1, "Expected runtime to pick the CACAO action index");
    ok &= ExpectNear(selected_action.score, 24.5f, "selected action score");
    ok &= ExpectWordEquals(selected_action.word, action_space_words.words[1], "selected action word");

    DestroySingleModelDeviceRuntime(runtime);
    return ok;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    if (!TestSingleModelDeviceRuntimeSelectsBestAction()) {
        return 1;
    }

    std::cout << "PASS: single_model_device_runtime_test\n";
    return 0;
}
