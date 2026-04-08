#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::training_folder::DeviceTrainingDataShard;
using neuroevolution::training_folder::LoadInitialTrainingDataShardFromActionSpace;
using neuroevolution::training_folder::TrainingDataShard;
using neuroevolution::training_folder::UploadTrainingDataShardToDeviceConstantMemory;
using neuroevolution::wordle::TryMakeWordFromAscii;
using neuroevolution::wordle::Word;

constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr int kStatusInvalidConstantShard = 1;

Word MakeWord(const char (&letters)[neuroevolution::wordle::kWordLength + 1]) {
    Word word{};

    if (!TryMakeWordFromAscii(letters, word)) {
        throw std::invalid_argument(
            "Training-data device test word literal must contain exactly five uppercase ASCII letters.");
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
    bool ok = true;

    for (std::size_t position = 0; position < neuroevolution::wordle::kWordLength; ++position) {
        if (actual.letter_indices[position] != expected.letter_indices[position]) {
            std::cerr << "FAIL: " << label << " mismatch at position " << position << '\n';
            ok = false;
        }
    }

    return ok;
}

__global__ void CopyTrainingDataShardFromConstantMemoryKernel(TrainingDataShard *shard_out, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    const TrainingDataShard &constant_shard = DeviceTrainingDataShard();
    if (!neuroevolution::training_folder::IsValidTrainingDataShard(constant_shard)) {
        *status = kStatusInvalidConstantShard;
        return;
    }

    *shard_out = constant_shard;
    *status = 0;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    const TrainingDataShard shard = LoadInitialTrainingDataShardFromActionSpace();
    if (!UploadTrainingDataShardToDeviceConstantMemory(shard)) {
        std::cerr << "FAIL: could not upload training-data shard to device constant memory\n";
        return 1;
    }

    TrainingDataShard *device_shard = nullptr;
    int *device_status = nullptr;
    bool ok = true;

    ok &= CheckCuda(cudaMalloc(&device_shard, sizeof(TrainingDataShard)), "allocating training-data shard output");
    ok &= CheckCuda(cudaMalloc(&device_status, sizeof(int)), "allocating training-data status output");

    if (ok) {
        CopyTrainingDataShardFromConstantMemoryKernel<<<1, 1>>>(device_shard, device_status);
        ok &= CheckCuda(cudaGetLastError(), "launching training-data constant-memory kernel");
        ok &= CheckCuda(cudaDeviceSynchronize(), "waiting for training-data constant-memory kernel");
    }

    TrainingDataShard host_shard{};
    int host_status = -1;

    if (ok) {
        ok &= CheckCuda(cudaMemcpy(&host_shard, device_shard, sizeof(TrainingDataShard), cudaMemcpyDeviceToHost),
                        "copying training-data shard back to host");
        ok &= CheckCuda(cudaMemcpy(&host_status, device_status, sizeof(int), cudaMemcpyDeviceToHost),
                        "copying training-data status back to host");
    }

    if (ok && (host_status == kStatusInvalidConstantShard)) {
        std::cerr << "FAIL: constant-memory training-data shard was invalid on device\n";
        ok = false;
    }

    if (ok && (host_status != 0)) {
        std::cerr << "FAIL: training-data constant-memory kernel returned unexpected status " << host_status << '\n';
        ok = false;
    }

    if (ok) {
        ok &= ExpectTrue(host_shard.entry_count == 5, "Expected constant-memory shard to contain five entries");
        ok &= ExpectWordEquals(host_shard.entries[0].word, MakeWord("AARGH"), "constant-memory first word");
        ok &= ExpectWordEquals(host_shard.entries[1].word, MakeWord("ABACK"), "constant-memory second word");
        ok &= ExpectWordEquals(host_shard.entries[2].word, MakeWord("ABASE"), "constant-memory third word");
        ok &= ExpectWordEquals(host_shard.entries[3].word, MakeWord("ABATE"), "constant-memory fourth word");
        ok &= ExpectWordEquals(host_shard.entries[4].word, MakeWord("ABBAS"), "constant-memory fifth word");
    }

    cudaFree(device_shard);
    cudaFree(device_status);

    if (!ok) {
        return 1;
    }

    std::cout << "PASS: training_data_device_test\n";
    return 0;
}
