#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/genetic_algorithm.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::common::ToFloat16;
using neuroevolution::genetic_algorithm::ModelGenome;
using neuroevolution::genetic_algorithm::TryInjectNewOutputEmbedding;
using neuroevolution::model::initialization::MakeRandomPolicyModelParameters;
using neuroevolution::model::output_embedding::TrainableActionEmbeddingTail;
using neuroevolution::model::policy_model::PolicyVector;
using neuroevolution::model::policy_model::TryForwardPolicyModel;
using neuroevolution::wordle::HintGridGroup;
using neuroevolution::wordle::TryBuildHintGridGroup;
using neuroevolution::wordle::Word;

constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr int kStatusSuccess = 0;
constexpr int kStatusInjectionFailed = 1;
constexpr float kTolerance = 1.0e-5f;

Word MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    Word word{};
    if (!neuroevolution::wordle::TryMakeWordFromAscii(letters, word)) {
        throw std::invalid_argument(
            "Output-embedding injection test word literal must contain exactly five uppercase ASCII letters.");
    }

    return word;
}

Word MakeInvalidWord() {
    Word word = MakeWord("SPARE");
    word.letter_indices[neuroevolution::wordle::kWordLength - 1] = neuroevolution::wordle::kAlphabetSize;
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

bool ExpectTailEquals(const TrainableActionEmbeddingTail &actual, const TrainableActionEmbeddingTail &expected,
                      const std::string_view label_prefix) {
    bool ok = true;

    for (std::size_t feature_index = 0;
         feature_index < neuroevolution::model::output_embedding::kTrainableFeatureDimension; ++feature_index) {
        ok &= ExpectNear(ToFloat(actual[feature_index]), ToFloat(expected[feature_index]),
                         std::string(label_prefix) + " feature " + std::to_string(feature_index));
    }

    return ok;
}

template <std::size_t ActionCapacity>
void PopulateSentinelGenome(ModelGenome<ActionCapacity> &genome, const std::size_t active_count) {
    genome = {};
    genome.policy_model = MakeRandomPolicyModelParameters(123U);
    genome.output_embedding.active_count = active_count;

    for (std::size_t action_index = 0; action_index < ActionCapacity; ++action_index) {
        for (std::size_t feature_index = 0;
             feature_index < neuroevolution::model::output_embedding::kTrainableFeatureDimension; ++feature_index) {
            const float value =
                (static_cast<float>(action_index) * 10.0f) + (static_cast<float>(feature_index) * 0.125f) + 0.5f;
            genome.output_embedding.trainable_tails[action_index][feature_index] = ToFloat16(value);
        }
    }
}

template <std::size_t ActionCapacity>
bool ComputeExpectedInjectedTail(const ModelGenome<ActionCapacity> &genome, const Word &target_word,
                                 TrainableActionEmbeddingTail &tail_out) {
    tail_out = {};

    HintGridGroup hint_grid_group{};
    if (!TryBuildHintGridGroup(target_word, hint_grid_group)) {
        return false;
    }

    float feature_sums[neuroevolution::model::output_embedding::kTrainableFeatureDimension]{};

    for (std::size_t grid_index = 0; grid_index < neuroevolution::wordle::kHintGridGroupSize; ++grid_index) {
        PolicyVector policy_vector{};
        if (!TryForwardPolicyModel(genome.policy_model, hint_grid_group.grids[grid_index], policy_vector)) {
            return false;
        }

        for (std::size_t feature_index = 0;
             feature_index < neuroevolution::model::output_embedding::kTrainableFeatureDimension; ++feature_index) {
            feature_sums[feature_index] +=
                policy_vector[neuroevolution::model::output_embedding::kWordFeatureDimension + feature_index];
        }
    }

    constexpr float kReciprocalHintGridCount = 1.0f / static_cast<float>(neuroevolution::wordle::kHintGridGroupSize);
    for (std::size_t feature_index = 0;
         feature_index < neuroevolution::model::output_embedding::kTrainableFeatureDimension; ++feature_index) {
        tail_out[feature_index] = ToFloat16(feature_sums[feature_index] * kReciprocalHintGridCount);
    }

    return true;
}

template <std::size_t ActionCapacity>
__global__ void InjectOutputEmbeddingKernel(ModelGenome<ActionCapacity> *genome, const Word target_word, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    *status = TryInjectNewOutputEmbedding(*genome, target_word) ? kStatusSuccess : kStatusInjectionFailed;
}

template <std::size_t ActionCapacity>
bool RunInjectionKernel(const ModelGenome<ActionCapacity> &input_genome, const Word &target_word,
                        ModelGenome<ActionCapacity> &output_genome, int &status_out) {
    ModelGenome<ActionCapacity> *device_genome = nullptr;
    int *device_status = nullptr;
    bool ok = true;

    ok &= CheckCuda(cudaMalloc(&device_genome, sizeof(ModelGenome<ActionCapacity>)), "allocating device genome");
    ok &= CheckCuda(cudaMalloc(&device_status, sizeof(int)), "allocating device status");

    if (ok) {
        ok &= CheckCuda(
            cudaMemcpy(device_genome, &input_genome, sizeof(ModelGenome<ActionCapacity>), cudaMemcpyHostToDevice),
            "copying genome to device");
        ok &= CheckCuda(cudaMemset(device_status, 0, sizeof(int)), "clearing device status");
    }

    if (ok) {
        InjectOutputEmbeddingKernel<<<1, 1>>>(device_genome, target_word, device_status);
        ok &= CheckCuda(cudaGetLastError(), "launching output-embedding injection kernel");
        ok &= CheckCuda(cudaDeviceSynchronize(), "waiting for output-embedding injection kernel");
    }

    if (ok) {
        ok &= CheckCuda(
            cudaMemcpy(&output_genome, device_genome, sizeof(ModelGenome<ActionCapacity>), cudaMemcpyDeviceToHost),
            "copying injected genome to host");
        ok &= CheckCuda(cudaMemcpy(&status_out, device_status, sizeof(int), cudaMemcpyDeviceToHost),
                        "copying injection status to host");
    }

    cudaFree(device_genome);
    cudaFree(device_status);
    return ok;
}

bool TestDeviceInjectionAppendsExpectedTail() {
    constexpr std::size_t kActionCapacity = 4;

    ModelGenome<kActionCapacity> input_genome{};
    PopulateSentinelGenome(input_genome, 3);

    const TrainableActionEmbeddingTail original_first_tail = input_genome.output_embedding.trainable_tails[0];
    const TrainableActionEmbeddingTail original_second_tail = input_genome.output_embedding.trainable_tails[1];
    const TrainableActionEmbeddingTail original_third_tail = input_genome.output_embedding.trainable_tails[2];
    const Word injected_word = MakeWord("SPARE");

    TrainableActionEmbeddingTail expected_tail{};
    bool ok = ComputeExpectedInjectedTail(input_genome, injected_word, expected_tail);
    ok &= ExpectTrue(ok, "Expected host-side injected tail synthesis to succeed");

    ModelGenome<kActionCapacity> output_genome{};
    int status = -1;
    ok &= RunInjectionKernel(input_genome, injected_word, output_genome, status);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(status == kStatusSuccess, "Expected device injection to report success");
    ok &= ExpectTrue(output_genome.output_embedding.active_count == 4,
                     "Expected successful injection to increment the active embedding count");
    ok &= ExpectTailEquals(output_genome.output_embedding.trainable_tails[0], original_first_tail,
                           "preserved first tail");
    ok &= ExpectTailEquals(output_genome.output_embedding.trainable_tails[1], original_second_tail,
                           "preserved second tail");
    ok &= ExpectTailEquals(output_genome.output_embedding.trainable_tails[2], original_third_tail,
                           "preserved third tail");
    ok &= ExpectTailEquals(output_genome.output_embedding.trainable_tails[3], expected_tail, "injected tail");
    return ok;
}

bool TestDeviceInjectionRejectsFullEmbeddingGenome() {
    constexpr std::size_t kActionCapacity = 4;

    ModelGenome<kActionCapacity> input_genome{};
    PopulateSentinelGenome(input_genome, kActionCapacity);
    const TrainableActionEmbeddingTail original_last_tail = input_genome.output_embedding.trainable_tails[3];

    ModelGenome<kActionCapacity> output_genome{};
    int status = -1;
    bool ok = RunInjectionKernel(input_genome, MakeWord("SPARE"), output_genome, status);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(status == kStatusInjectionFailed, "Expected full embedding genome injection to fail");
    ok &= ExpectTrue(output_genome.output_embedding.active_count == kActionCapacity,
                     "Expected failed injection to keep the active embedding count unchanged");
    ok &= ExpectTailEquals(output_genome.output_embedding.trainable_tails[3], original_last_tail,
                           "preserved last tail after full-capacity failure");
    return ok;
}

bool TestDeviceInjectionRejectsInvalidWord() {
    constexpr std::size_t kActionCapacity = 4;

    ModelGenome<kActionCapacity> input_genome{};
    PopulateSentinelGenome(input_genome, 3);
    const TrainableActionEmbeddingTail original_dormant_tail = input_genome.output_embedding.trainable_tails[3];

    ModelGenome<kActionCapacity> output_genome{};
    int status = -1;
    bool ok = RunInjectionKernel(input_genome, MakeInvalidWord(), output_genome, status);
    if (!ok) {
        return false;
    }

    ok &= ExpectTrue(status == kStatusInjectionFailed, "Expected invalid-word injection to fail");
    ok &= ExpectTrue(output_genome.output_embedding.active_count == 3,
                     "Expected failed invalid-word injection to keep the active embedding count unchanged");
    ok &= ExpectTailEquals(output_genome.output_embedding.trainable_tails[3], original_dormant_tail,
                           "preserved dormant tail after invalid-word failure");
    return ok;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    if (!TestDeviceInjectionAppendsExpectedTail()) {
        return 1;
    }

    if (!TestDeviceInjectionRejectsFullEmbeddingGenome()) {
        return 1;
    }

    if (!TestDeviceInjectionRejectsInvalidWord()) {
        return 1;
    }

    std::cout << "PASS: output_embedding_injection_test\n";
    return 0;
}
