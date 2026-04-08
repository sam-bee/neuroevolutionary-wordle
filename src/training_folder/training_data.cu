#include "training_folder/training_data.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace neuroevolution::training_folder {

__constant__ TrainingDataShard kDeviceTrainingDataShard{};

__device__ const TrainingDataShard &DeviceTrainingDataShard() noexcept { return kDeviceTrainingDataShard; }

bool UploadTrainingDataShardToDeviceConstantMemory(const TrainingDataShard &shard) {
    if (!IsValidTrainingDataShard(shard)) {
        return false;
    }

    const cudaError_t copy_error = cudaMemcpyToSymbol(kDeviceTrainingDataShard, &shard, sizeof(TrainingDataShard));
    return copy_error == cudaSuccess;
}

void UploadTrainingDataShardToDeviceConstantMemoryOrThrow(const TrainingDataShard &shard) {
    if (!IsValidTrainingDataShard(shard)) {
        throw std::invalid_argument("Training-data shard must be structurally valid before upload to constant memory.");
    }

    const cudaError_t copy_error = cudaMemcpyToSymbol(kDeviceTrainingDataShard, &shard, sizeof(TrainingDataShard));
    if (copy_error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(copy_error));
    }
}

} // namespace neuroevolution::training_folder
