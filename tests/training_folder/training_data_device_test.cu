#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "training_folder/training_data.hpp"

namespace {

using neuroevolution::training_folder::DeviceTrainingWordCatalog;
using neuroevolution::training_folder::kDefaultInitialActiveWordCount;
using neuroevolution::training_folder::kTrainingWordCatalogCapacity;
using neuroevolution::training_folder::LoadTrainingWordCatalogFromActionSpace;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::training_folder::UploadTrainingWordCatalogToDeviceConstantMemory;
using neuroevolution::wordle::Word;

constexpr int kSelectedVisibleDeviceIndex = 0;
constexpr int kStatusInvalidConstantShard = 1;
constexpr std::array<const char *, kDefaultInitialActiveWordCount> kExpectedTrainingWords = {
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

bool ExpectTrainingWordCatalogMatchesExpectedPrefix(const TrainingWordCatalog &catalog,
                                                    const std::string_view label_prefix) {
    bool ok = true;

    ok &= ExpectTrue(catalog.word_count == kTrainingWordCatalogCapacity,
                     std::string(label_prefix) + " should contain the full training-word catalog");

    for (std::size_t entry_index = 0; entry_index < kExpectedTrainingWords.size(); ++entry_index) {
        ok &= ExpectWordEquals(catalog.words[entry_index], MakeWord(kExpectedTrainingWords[entry_index]),
                               std::string(label_prefix) + " word " + std::to_string(entry_index));
    }

    return ok;
}

__global__ void CopyTrainingWordCatalogFromConstantMemoryKernel(TrainingWordCatalog *catalog_out, int *status) {
    if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
        return;
    }

    const TrainingWordCatalog &constant_catalog = DeviceTrainingWordCatalog();
    if (!neuroevolution::training_folder::IsValidTrainingWordCatalog(constant_catalog)) {
        *status = kStatusInvalidConstantShard;
        return;
    }

    *catalog_out = constant_catalog;
    *status = 0;
}

} // namespace

int main() {
    if (!SelectVisibleCudaDevice()) {
        return 1;
    }

    const TrainingWordCatalog catalog = LoadTrainingWordCatalogFromActionSpace();
    if (!UploadTrainingWordCatalogToDeviceConstantMemory(catalog)) {
        std::cerr << "FAIL: could not upload training-word catalog to device constant memory\n";
        return 1;
    }

    TrainingWordCatalog *device_catalog = nullptr;
    int *device_status = nullptr;
    bool ok = true;

    ok &=
        CheckCuda(cudaMalloc(&device_catalog, sizeof(TrainingWordCatalog)), "allocating training-word catalog output");
    ok &= CheckCuda(cudaMalloc(&device_status, sizeof(int)), "allocating training-data status output");

    if (ok) {
        CopyTrainingWordCatalogFromConstantMemoryKernel<<<1, 1>>>(device_catalog, device_status);
        ok &= CheckCuda(cudaGetLastError(), "launching training-data constant-memory kernel");
        ok &= CheckCuda(cudaDeviceSynchronize(), "waiting for training-data constant-memory kernel");
    }

    TrainingWordCatalog host_catalog{};
    int host_status = -1;

    if (ok) {
        ok &= CheckCuda(cudaMemcpy(&host_catalog, device_catalog, sizeof(TrainingWordCatalog), cudaMemcpyDeviceToHost),
                        "copying training-word catalog back to host");
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
        ok &= ExpectTrainingWordCatalogMatchesExpectedPrefix(host_catalog, "constant-memory training-word catalog");
    }

    cudaFree(device_catalog);
    cudaFree(device_status);

    if (!ok) {
        return 1;
    }

    std::cout << "PASS: training_data_device_test\n";
    return 0;
}
