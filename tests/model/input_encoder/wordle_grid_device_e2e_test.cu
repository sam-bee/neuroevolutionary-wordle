#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "model/model_input/wordle_grid_state.hpp"
#include "shared_encoder_fixture.hpp"
#include "wordle/word.hpp"
#include "wordle/wordle_grid.hpp"

namespace {

using neuroevolution::model::input_encoder::kEncoderOutputSize;
using neuroevolution::model::input_encoder::SharedEncoderParameters;
using neuroevolution::model::model_input::kModelInputVectorSize;
using neuroevolution::model::model_input::ModelInputStateVector;
using neuroevolution::model::model_input::ModelInputTurnOffset;
using neuroevolution::model::model_input::TryEncodeWordleGridState;
using neuroevolution::tests::input_encoder::SharedEncoderGoldenFixture;
using neuroevolution::wordle::MakeWordleGrid;
using neuroevolution::wordle::TryAppendGuess;
using neuroevolution::wordle::TryMakeWordFromAscii;
using neuroevolution::wordle::Word;
using neuroevolution::wordle::WordleGrid;

constexpr float kTolerance = 1.0e-6f;
constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr int kStatusAppendGuessFailed = 1;
constexpr int kStatusEncodeFailed = 2;

Word MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    Word word{};

    if (!TryMakeWordFromAscii(letters, word)) {
        throw std::invalid_argument(
            "Device end-to-end test word literal must contain exactly five uppercase ASCII letters.");
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

bool ExpectVectorNear(const ModelInputStateVector &actual, const ModelInputStateVector &expected) {
    bool ok = true;

    for (std::size_t index = 0; index < kModelInputVectorSize; ++index) {
        const float delta = std::fabs(actual[index] - expected[index]);
        if (delta > kTolerance) {
            std::cerr << "FAIL: output mismatch at index " << index << ", expected " << expected[index] << ", got "
                      << actual[index] << '\n';
            ok = false;
        }
    }

    return ok;
}

ModelInputStateVector MakeExpectedOutput(const SharedEncoderGoldenFixture &fixture) {
    ModelInputStateVector expected{};

    const std::size_t first_turn_offset = ModelInputTurnOffset(0);
    for (std::size_t value_index = 0; value_index < kEncoderOutputSize; ++value_index) {
        expected[first_turn_offset + value_index] = fixture.expected_output[value_index];
    }

    return expected;
}

__global__ void WordleGridDeviceEndToEndKernel(const SharedEncoderParameters *parameters, const Word *solution,
                                               const Word *guess, ModelInputStateVector *output, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    WordleGrid grid = MakeWordleGrid(*solution);
    if (!TryAppendGuess(grid, *guess)) {
        *status = kStatusAppendGuessFailed;
        return;
    }

    ModelInputStateVector encoded_state{};
    if (!TryEncodeWordleGridState(*parameters, grid, encoded_state)) {
        *status = kStatusEncodeFailed;
        return;
    }

    *output = encoded_state;
    *status = 0;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    const SharedEncoderGoldenFixture fixture{};
    const Word solution = MakeWord("AEBDF");
    const Word guess = MakeWord("ABCDE");
    const ModelInputStateVector expected_output = MakeExpectedOutput(fixture);

    SharedEncoderParameters *device_parameters = nullptr;
    Word *device_solution = nullptr;
    Word *device_guess = nullptr;
    ModelInputStateVector *device_output = nullptr;
    int *device_status = nullptr;

    bool ok = true;

    ok &= CheckCuda(cudaMalloc(&device_parameters, sizeof(fixture.parameters)), "allocating encoder parameters");
    ok &= CheckCuda(cudaMalloc(&device_solution, sizeof(solution)), "allocating solution input");
    ok &= CheckCuda(cudaMalloc(&device_guess, sizeof(guess)), "allocating guess input");
    ok &= CheckCuda(cudaMalloc(&device_output, sizeof(ModelInputStateVector)), "allocating output buffer");
    ok &= CheckCuda(cudaMalloc(&device_status, sizeof(int)), "allocating status buffer");

    if (ok) {
        ok &= CheckCuda(
            cudaMemcpy(device_parameters, &fixture.parameters, sizeof(fixture.parameters), cudaMemcpyHostToDevice),
            "copying encoder parameters to device");
        ok &= CheckCuda(cudaMemcpy(device_solution, &solution, sizeof(solution), cudaMemcpyHostToDevice),
                        "copying solution input to device");
        ok &= CheckCuda(cudaMemcpy(device_guess, &guess, sizeof(guess), cudaMemcpyHostToDevice),
                        "copying guess input to device");
    }

    if (ok) {
        WordleGridDeviceEndToEndKernel<<<1, 1>>>(device_parameters, device_solution, device_guess, device_output,
                                                 device_status);
        ok &= CheckCuda(cudaGetLastError(), "launching wordle-grid device end-to-end kernel");
        ok &= CheckCuda(cudaDeviceSynchronize(), "waiting for wordle-grid device end-to-end kernel completion");
    }

    ModelInputStateVector host_output{};
    int host_status = -1;

    if (ok) {
        ok &= CheckCuda(cudaMemcpy(&host_output, device_output, sizeof(ModelInputStateVector), cudaMemcpyDeviceToHost),
                        "copying encoded output back to host");
        ok &= CheckCuda(cudaMemcpy(&host_status, device_status, sizeof(int), cudaMemcpyDeviceToHost),
                        "copying kernel status back to host");
    }

    if (ok && (host_status == kStatusAppendGuessFailed)) {
        std::cerr << "FAIL: device kernel could not append the test guess to a freshly created grid\n";
        ok = false;
    }

    if (ok && (host_status == kStatusEncodeFailed)) {
        std::cerr << "FAIL: device kernel could not encode the device-built grid\n";
        ok = false;
    }

    if (ok && (host_status != 0)) {
        std::cerr << "FAIL: device kernel returned unexpected status " << host_status << '\n';
        ok = false;
    }

    if (ok) {
        ok &= ExpectVectorNear(host_output, expected_output);
    }

    if (device_status != nullptr) {
        cudaFree(device_status);
    }
    if (device_output != nullptr) {
        cudaFree(device_output);
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

    std::cout << "PASS: wordle_grid_device_e2e_test\n";
    return 0;
}
