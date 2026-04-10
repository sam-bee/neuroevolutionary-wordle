#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::training_folder::DeviceTrainingDataShard;
using neuroevolution::training_folder::kInitialTrainingDataShardEntryCount;
using neuroevolution::training_folder::LoadInitialTrainingDataShardFromActionSpace;
using neuroevolution::training_folder::TrainingDataShard;
using neuroevolution::training_folder::UploadTrainingDataShardToDeviceConstantMemory;
using neuroevolution::wordle::Word;

constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr int kStatusInvalidConstantShard = 1;
constexpr std::array<const char *, kInitialTrainingDataShardEntryCount> kExpectedTrainingWords = {
    "MINOS", "VODKA", "RAZOR", "GRADS", "CURLS", "BILGE", "GREET", "PYLON", "ENTER", "READY",
    "VERDE", "AUGER", "FOOTS", "BRACE", "PURTY", "SPORT", "TIRES", "FRISK", "AFFIX", "CHUMS",
};

Word MakeWord(const std::string_view letters) {
    if (letters.size() != neuroevolution::wordle::kWordLength) {
        throw std::invalid_argument("Training-data device test word view must contain exactly five characters.");
    }

    Word word{};
    for (std::size_t position = 0; position < neuroevolution::wordle::kWordLength; ++position) {
        const char value = letters[position];
        if (!neuroevolution::wordle::IsAsciiUppercaseLetter(value)) {
            throw std::invalid_argument(
                "Training-data device test word view must contain only uppercase ASCII letters.");
        }

        word.letter_indices[position] = neuroevolution::wordle::LetterIndexFromAscii(value);
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

bool ExpectTrainingShardMatchesExpectedWords(const TrainingDataShard &shard, const std::string_view label_prefix) {
    bool ok = true;

    ok &= ExpectTrue(shard.entry_count == kExpectedTrainingWords.size(),
                     std::string(label_prefix) + " should contain the expected number of entries");

    const std::size_t comparison_count =
        (shard.entry_count < kExpectedTrainingWords.size()) ? shard.entry_count : kExpectedTrainingWords.size();

    for (std::size_t entry_index = 0; entry_index < comparison_count; ++entry_index) {
        ok &= ExpectWordEquals(shard.entries[entry_index].word, MakeWord(kExpectedTrainingWords[entry_index]),
                               std::string(label_prefix) + " word " + std::to_string(entry_index));
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
        ok &= ExpectTrainingShardMatchesExpectedWords(host_shard, "constant-memory training-data shard");
    }

    cudaFree(device_shard);
    cudaFree(device_status);

    if (!ok) {
        return 1;
    }

    std::cout << "PASS: training_data_device_test\n";
    return 0;
}
