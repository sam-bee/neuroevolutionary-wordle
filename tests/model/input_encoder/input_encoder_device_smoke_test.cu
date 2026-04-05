#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>

#include "model/input_encoder/shared_encoder.hpp"
#include "shared_encoder_fixture.hpp"

namespace {

using neuroevolution::model::input_encoder::EncodedTurnVector;
using neuroevolution::model::input_encoder::kEncoderOutputSize;
using neuroevolution::model::input_encoder::SharedEncoderParameters;
using neuroevolution::model::input_encoder::detail::TryForwardOccupiedTurn;
using neuroevolution::tests::input_encoder::SharedEncoderGoldenFixture;
using neuroevolution::wordle::Turn;

constexpr float kTolerance = 1.0e-6f;
constexpr int kSelectedVisibleDeviceIndex = 0;

__global__ void InputEncoderSmokeKernel(const SharedEncoderParameters *parameters, const Turn *turn,
                                        EncodedTurnVector *output, int *status) {
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

bool ExpectVectorNear(const EncodedTurnVector &actual, const EncodedTurnVector &expected) {
    bool ok = true;

    for (std::size_t index = 0; index < kEncoderOutputSize; ++index) {
        const float delta = std::fabs(actual[index] - expected[index]);
        if (delta > kTolerance) {
            std::cerr << "FAIL: output mismatch at index " << index << ", expected " << expected[index] << ", got "
                      << actual[index] << '\n';
            ok = false;
        }
    }

    return ok;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    const SharedEncoderGoldenFixture fixture{};

    SharedEncoderParameters *device_parameters = nullptr;
    Turn *device_turn = nullptr;
    EncodedTurnVector *device_output = nullptr;
    int *device_status = nullptr;

    bool ok = true;

    ok &= CheckCuda(cudaMalloc(&device_parameters, sizeof(fixture.parameters)), "allocating encoder parameters");
    ok &= CheckCuda(cudaMalloc(&device_turn, sizeof(fixture.turn)), "allocating turn input");
    ok &= CheckCuda(cudaMalloc(&device_output, sizeof(EncodedTurnVector)), "allocating output buffer");
    ok &= CheckCuda(cudaMalloc(&device_status, sizeof(int)), "allocating status buffer");

    if (ok) {
        ok &= CheckCuda(
            cudaMemcpy(device_parameters, &fixture.parameters, sizeof(fixture.parameters), cudaMemcpyHostToDevice),
            "copying encoder parameters to device");
        ok &= CheckCuda(cudaMemcpy(device_turn, &fixture.turn, sizeof(fixture.turn), cudaMemcpyHostToDevice),
                        "copying turn input to device");
    }

    if (ok) {
        InputEncoderSmokeKernel<<<1, 1>>>(device_parameters, device_turn, device_output, device_status);
        ok &= CheckCuda(cudaGetLastError(), "launching smoke-test kernel");
        ok &= CheckCuda(cudaDeviceSynchronize(), "waiting for smoke-test kernel completion");
    }

    EncodedTurnVector host_output{};
    int host_status = -1;

    if (ok) {
        ok &= CheckCuda(cudaMemcpy(&host_output, device_output, sizeof(EncodedTurnVector), cudaMemcpyDeviceToHost),
                        "copying encoded output back to host");
        ok &= CheckCuda(cudaMemcpy(&host_status, device_status, sizeof(int), cudaMemcpyDeviceToHost),
                        "copying kernel status back to host");
    }

    if (ok && (host_status != 0)) {
        std::cerr << "FAIL: device kernel reported invalid turn input\n";
        ok = false;
    }

    if (ok) {
        ok &= ExpectVectorNear(host_output, fixture.expected_output);
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
