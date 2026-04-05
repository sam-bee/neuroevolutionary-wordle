#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>

#include "model/input_encoder/encoder_spec.hpp"
#include "model/input_encoder/shared_encoder.hpp"
#include "wordle/turn.hpp"

namespace {

using neuroevolution::model::input_encoder::EncodedTurnVector;
using neuroevolution::model::input_encoder::FeedbackFeatureOffset;
using neuroevolution::model::input_encoder::GuessLetterFeatureOffset;
using neuroevolution::model::input_encoder::SharedEncoderParameters;
using neuroevolution::model::input_encoder::TryForwardOccupiedTurn;
using neuroevolution::model::input_encoder::kEncoderOutputSize;
using neuroevolution::model::input_encoder::kEncoderHiddenSize;
using neuroevolution::model::input_encoder::kTurnFeatureCount;
using neuroevolution::wordle::TileFeedback;
using neuroevolution::wordle::Turn;

constexpr float kTolerance = 1.0e-6f;
constexpr int kSelectedVisibleDeviceIndex = 0;

__global__ void InputEncoderSmokeKernel(const SharedEncoderParameters *parameters,
                                        const Turn *turn,
                                        EncodedTurnVector *output,
                                        int *status) {
  if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
    return;
  }

  EncodedTurnVector encoded_turn{};
  if (!TryForwardOccupiedTurn(*parameters, *turn, encoded_turn)) {
    *status = 1;
    return;
  }

  *output = encoded_turn;
  *status = 0;
}

bool CheckCuda(const cudaError_t error, const std::string_view action) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA failure during " << action << ": "
              << cudaGetErrorString(error) << '\n';
    return false;
  }

  return true;
}

bool SelectVisibleCudaDevice() {
  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count),
                 "querying visible CUDA device count")) {
    return false;
  }

  std::cout << "cudaGetDeviceCount(): " << device_count << '\n';

  for (int device_index = 0; device_index < device_count; ++device_index) {
    cudaDeviceProp properties{};
    if (!CheckCuda(cudaGetDeviceProperties(&properties, device_index),
                   "querying visible CUDA device properties")) {
      return false;
    }

    std::cout << "Visible CUDA device " << device_index << ": "
              << properties.name << " (CC " << properties.major << '.'
              << properties.minor << ")\n";
  }

  if (device_count <= kSelectedVisibleDeviceIndex) {
    std::cerr << "FAIL: selected logical device index "
              << kSelectedVisibleDeviceIndex
              << " is not available in this process\n";
    return false;
  }

  if (!CheckCuda(cudaSetDevice(kSelectedVisibleDeviceIndex),
                 "selecting visible CUDA device")) {
    return false;
  }

  cudaDeviceProp selected_properties{};
  if (!CheckCuda(cudaGetDeviceProperties(&selected_properties,
                                         kSelectedVisibleDeviceIndex),
                 "querying selected CUDA device properties")) {
    return false;
  }

  std::cout << "Selected logical CUDA device: "
            << kSelectedVisibleDeviceIndex << '\n';
  std::cout << "Selected device properties: " << selected_properties.name
            << " (CC " << selected_properties.major << '.'
            << selected_properties.minor << ")\n";

  return true;
}

bool ExpectVectorNear(const EncodedTurnVector &actual,
                      const EncodedTurnVector &expected) {
  bool ok = true;

  for (std::size_t index = 0; index < kEncoderOutputSize; ++index) {
    const float delta = std::fabs(actual[index] - expected[index]);
    if (delta > kTolerance) {
      std::cerr << "FAIL: output mismatch at index " << index
                << ", expected " << expected[index]
                << ", got " << actual[index] << '\n';
      ok = false;
    }
  }

  return ok;
}

void PopulateSmokeTestParameters(SharedEncoderParameters &parameters) {
  constexpr std::size_t kGuessA0 = GuessLetterFeatureOffset(0, 0);
  constexpr std::size_t kGuessB1 = GuessLetterFeatureOffset(1, 1);
  constexpr std::size_t kGuessC2 = GuessLetterFeatureOffset(2, 2);
  constexpr std::size_t kGuessD3 = GuessLetterFeatureOffset(3, 3);
  constexpr std::size_t kGuessE4 = GuessLetterFeatureOffset(4, 4);
  constexpr std::size_t kGreen0 = FeedbackFeatureOffset(0, 0);
  constexpr std::size_t kYellow1 = FeedbackFeatureOffset(1, 1);
  constexpr std::size_t kGrey2 = FeedbackFeatureOffset(2, 2);
  constexpr std::size_t kGreen3 = FeedbackFeatureOffset(3, 0);
  constexpr std::size_t kYellow4 = FeedbackFeatureOffset(4, 1);

  parameters.input_to_hidden.biases[0] = -2.0f;
  parameters.input_to_hidden.biases[1] = 1.0f;
  parameters.input_to_hidden.biases[2] = -1.0f;

  parameters.input_to_hidden.weights[(0 * kTurnFeatureCount) + kGuessA0] =
      1.5f;
  parameters.input_to_hidden.weights[(0 * kTurnFeatureCount) + kGuessB1] =
      0.5f;
  parameters.input_to_hidden.weights[(0 * kTurnFeatureCount) + kGreen0] = 4.0f;

  parameters.input_to_hidden.weights[(1 * kTurnFeatureCount) + kGuessC2] =
      2.0f;
  parameters.input_to_hidden.weights[(1 * kTurnFeatureCount) + kGrey2] = -5.0f;
  parameters.input_to_hidden.weights[(1 * kTurnFeatureCount) + kYellow4] =
      0.5f;

  parameters.input_to_hidden.weights[(2 * kTurnFeatureCount) + kGuessD3] =
      1.0f;
  parameters.input_to_hidden.weights[(2 * kTurnFeatureCount) + kGuessE4] =
      2.0f;
  parameters.input_to_hidden.weights[(2 * kTurnFeatureCount) + kYellow1] =
      0.25f;
  parameters.input_to_hidden.weights[(2 * kTurnFeatureCount) + kGreen3] =
      0.75f;

  parameters.hidden_to_output.biases[0] = 0.5f;
  parameters.hidden_to_output.biases[1] = -2.0f;
  parameters.hidden_to_output.biases[2] = 1.25f;

  parameters.hidden_to_output.weights[(0 * kEncoderHiddenSize) + 0] = 2.0f;
  parameters.hidden_to_output.weights[(0 * kEncoderHiddenSize) + 2] = -1.0f;

  parameters.hidden_to_output.weights[(1 * kEncoderHiddenSize) + 0] = -0.5f;
  parameters.hidden_to_output.weights[(1 * kEncoderHiddenSize) + 2] = 4.0f;

  parameters.hidden_to_output.weights[(2 * kEncoderHiddenSize) + 1] = 7.0f;
}

Turn MakeSmokeTestTurn() {
  return Turn{
      .letter_indices = {{0, 1, 2, 3, 4}},
      .feedback = {{
          TileFeedback::green,
          TileFeedback::yellow,
          TileFeedback::grey,
          TileFeedback::green,
          TileFeedback::yellow,
      }},
  };
}

} // namespace

int main() {
  if (!SelectVisibleCudaDevice()) {
    return 1;
  }

  SharedEncoderParameters host_parameters{};
  PopulateSmokeTestParameters(host_parameters);

  const Turn host_turn = MakeSmokeTestTurn();

  EncodedTurnVector expected{};
  expected[0] = 5.5f;
  expected[1] = 8.0f;
  expected[2] = 1.25f;

  SharedEncoderParameters *device_parameters = nullptr;
  Turn *device_turn = nullptr;
  EncodedTurnVector *device_output = nullptr;
  int *device_status = nullptr;

  bool ok = true;

  ok &= CheckCuda(cudaMalloc(&device_parameters, sizeof(SharedEncoderParameters)),
                  "allocating encoder parameters");
  ok &= CheckCuda(cudaMalloc(&device_turn, sizeof(Turn)),
                  "allocating turn input");
  ok &= CheckCuda(cudaMalloc(&device_output, sizeof(EncodedTurnVector)),
                  "allocating output buffer");
  ok &= CheckCuda(cudaMalloc(&device_status, sizeof(int)),
                  "allocating status buffer");

  if (ok) {
    ok &= CheckCuda(cudaMemcpy(device_parameters, &host_parameters,
                               sizeof(SharedEncoderParameters),
                               cudaMemcpyHostToDevice),
                    "copying encoder parameters to device");
    ok &= CheckCuda(cudaMemcpy(device_turn, &host_turn, sizeof(Turn),
                               cudaMemcpyHostToDevice),
                    "copying turn input to device");
  }

  if (ok) {
    InputEncoderSmokeKernel<<<1, 1>>>(device_parameters, device_turn,
                                      device_output, device_status);
    ok &= CheckCuda(cudaGetLastError(), "launching smoke-test kernel");
    ok &= CheckCuda(cudaDeviceSynchronize(),
                    "waiting for smoke-test kernel completion");
  }

  EncodedTurnVector host_output{};
  int host_status = -1;

  if (ok) {
    ok &= CheckCuda(cudaMemcpy(&host_output, device_output,
                               sizeof(EncodedTurnVector),
                               cudaMemcpyDeviceToHost),
                    "copying encoded output back to host");
    ok &= CheckCuda(cudaMemcpy(&host_status, device_status, sizeof(int),
                               cudaMemcpyDeviceToHost),
                    "copying kernel status back to host");
  }

  if (ok && (host_status != 0)) {
    std::cerr << "FAIL: device kernel reported invalid turn input\n";
    ok = false;
  }

  if (ok) {
    ok &= ExpectVectorNear(host_output, expected);
  }

  if (device_status != nullptr) {
    cudaFree(device_status);
  }
  if (device_output != nullptr) {
    cudaFree(device_output);
  }
  if (device_turn != nullptr) {
    cudaFree(device_turn);
  }
  if (device_parameters != nullptr) {
    cudaFree(device_parameters);
  }

  if (!ok) {
    return 1;
  }

  std::cout << "PASS: input_encoder_device_smoke_test\n";
  return 0;
}
