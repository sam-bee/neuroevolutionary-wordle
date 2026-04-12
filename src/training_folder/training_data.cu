#include "training_folder/training_data.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace neuroevolution::training_folder {

__constant__ TrainingWordCatalog kDeviceTrainingWordCatalog{};

__device__ const TrainingWordCatalog &DeviceTrainingWordCatalog() noexcept { return kDeviceTrainingWordCatalog; }

bool UploadTrainingWordCatalogToDeviceConstantMemory(const TrainingWordCatalog &catalog) {
    if (!IsValidTrainingWordCatalog(catalog)) {
        return false;
    }

    const cudaError_t copy_error = cudaMemcpyToSymbol(kDeviceTrainingWordCatalog, &catalog, sizeof(TrainingWordCatalog));
    return copy_error == cudaSuccess;
}

void UploadTrainingWordCatalogToDeviceConstantMemoryOrThrow(const TrainingWordCatalog &catalog) {
    if (!IsValidTrainingWordCatalog(catalog)) {
        throw std::invalid_argument(
            "Training-word catalog must be structurally valid before upload to constant memory.");
    }

    const cudaError_t copy_error =
        cudaMemcpyToSymbol(kDeviceTrainingWordCatalog, &catalog, sizeof(TrainingWordCatalog));
    if (copy_error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(copy_error));
    }
}

} // namespace neuroevolution::training_folder
