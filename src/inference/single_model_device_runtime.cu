#include "inference/single_model_device_runtime.hpp"

#include <cuda_runtime.h>

#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "inference/dynamic_policy.hpp"

namespace neuroevolution::inference {

namespace {

using neuroevolution::inference::dynamic_policy::DynamicPolicyBlockScratch;
using neuroevolution::inference::dynamic_policy::DynamicInferenceStatusCode;
using neuroevolution::inference::dynamic_policy::kDynamicPolicyThreadsPerBlock;
using neuroevolution::inference::dynamic_policy::SelectNextGuessFromDynamicGenomeConcurrently;
using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::training_folder::IsValidTrainingWordCatalog;
using neuroevolution::training_folder::TrainingWordCatalog;
using neuroevolution::wordle::WordleGrid;

constexpr SingleModelDeviceRuntimeStatusCode kOk = SingleModelDeviceRuntimeStatusCode::kOk;

constexpr bool IsValidCreateInputs(const std::uint8_t *host_genome_bytes, const std::size_t genome_byte_count,
                                   const TrainingWordCatalog &action_space_words) noexcept {
    return (host_genome_bytes != nullptr) && IsValidTrainingWordCatalog(action_space_words) &&
           (action_space_words.word_count > 0) && (genome_byte_count > 0) &&
           (genome_byte_count ==
            genetic_algorithm::genome::ComputeDynamicGenomeStrideBytes(action_space_words.word_count));
}

void FreeDeviceRuntimeBuffers(SingleModelDeviceRuntime &runtime) noexcept {
    if (runtime.device_status != nullptr) {
        cudaFree(runtime.device_status);
    }
    if (runtime.device_selected_action != nullptr) {
        cudaFree(runtime.device_selected_action);
    }
    if (runtime.device_grid != nullptr) {
        cudaFree(runtime.device_grid);
    }
    if (runtime.device_action_space_words != nullptr) {
        cudaFree(runtime.device_action_space_words);
    }
    if (runtime.device_genome_bytes != nullptr) {
        cudaFree(runtime.device_genome_bytes);
    }

    runtime = {};
}

bool CheckCuda(const cudaError_t error) noexcept { return error == cudaSuccess; }

__global__ void SelectNextGuessKernel(const TrainingWordCatalog *action_space_words, const std::uint8_t *genome_bytes,
                                      const std::size_t action_count, const WordleGrid *grid,
                                      SelectedAction *selected_action_out, int *status_out) {
    if ((action_space_words == nullptr) || (genome_bytes == nullptr) || (grid == nullptr) ||
        (selected_action_out == nullptr) || (status_out == nullptr) || (action_count == 0) ||
        (action_count > action_space_words->word_count)) {
        if (threadIdx.x == 0) {
            *status_out = static_cast<int>(SingleModelDeviceRuntimeStatusCode::kInvalidRuntime);
        }
        return;
    }

    __shared__ DynamicPolicyBlockScratch<kDynamicPolicyThreadsPerBlock> scratch;
    SelectedAction selected_action{};
    const auto inference_status = SelectNextGuessFromDynamicGenomeConcurrently<kDynamicPolicyThreadsPerBlock>(
            *grid, *action_space_words, genome_bytes, action_count, scratch, selected_action);

    if (threadIdx.x == 0) {
        if (inference_status == DynamicInferenceStatusCode::kPolicyForwardFailed) {
            *status_out = static_cast<int>(SingleModelDeviceRuntimeStatusCode::kPolicyForwardFailed);
            return;
        }

        if (inference_status != DynamicInferenceStatusCode::kOk) {
            *status_out = static_cast<int>(SingleModelDeviceRuntimeStatusCode::kActionSelectionFailed);
            return;
        }

        *selected_action_out = selected_action;
        *status_out = static_cast<int>(kOk);
    }
}

} // namespace

bool TryCreateSingleModelDeviceRuntime(const std::uint8_t *host_genome_bytes, const std::size_t genome_byte_count,
                                       const TrainingWordCatalog &action_space_words,
                                       SingleModelDeviceRuntime &runtime_out) {
    runtime_out = {};
    if (!IsValidCreateInputs(host_genome_bytes, genome_byte_count, action_space_words)) {
        return false;
    }

    SingleModelDeviceRuntime runtime{};
    runtime.action_count = action_space_words.word_count;
    runtime.genome_byte_count = genome_byte_count;

    if (!CheckCuda(cudaMalloc(&runtime.device_genome_bytes, genome_byte_count)) ||
        !CheckCuda(cudaMalloc(&runtime.device_action_space_words, sizeof(action_space_words))) ||
        !CheckCuda(cudaMalloc(&runtime.device_grid, sizeof(WordleGrid))) ||
        !CheckCuda(cudaMalloc(&runtime.device_selected_action, sizeof(SelectedAction))) ||
        !CheckCuda(cudaMalloc(&runtime.device_status, sizeof(int)))) {
        FreeDeviceRuntimeBuffers(runtime);
        return false;
    }

    if (!CheckCuda(cudaMemcpy(runtime.device_genome_bytes, host_genome_bytes, genome_byte_count, cudaMemcpyHostToDevice)) ||
        !CheckCuda(cudaMemcpy(runtime.device_action_space_words, &action_space_words, sizeof(action_space_words),
                              cudaMemcpyHostToDevice))) {
        FreeDeviceRuntimeBuffers(runtime);
        return false;
    }

    runtime_out = runtime;
    return true;
}

void DestroySingleModelDeviceRuntime(SingleModelDeviceRuntime &runtime) noexcept { FreeDeviceRuntimeBuffers(runtime); }

bool TrySelectNextGuessWithSingleModelDeviceRuntime(
    const SingleModelDeviceRuntime &runtime, const WordleGrid &grid, SelectedAction &selected_action_out,
    SingleModelDeviceRuntimeStatusCode *status_out) {
    if (status_out != nullptr) {
        *status_out = SingleModelDeviceRuntimeStatusCode::kInvalidRuntime;
    }

    if (!IsValidSingleModelDeviceRuntime(runtime)) {
        return false;
    }

    if (!CheckCuda(cudaMemcpy(runtime.device_grid, &grid, sizeof(grid), cudaMemcpyHostToDevice))) {
        if (status_out != nullptr) {
            *status_out = SingleModelDeviceRuntimeStatusCode::kCudaMemcpyFailed;
        }
        return false;
    }

    SelectNextGuessKernel<<<1, kDynamicPolicyThreadsPerBlock>>>(
        runtime.device_action_space_words, runtime.device_genome_bytes, runtime.action_count, runtime.device_grid,
        runtime.device_selected_action, runtime.device_status);
    if (!CheckCuda(cudaGetLastError())) {
        if (status_out != nullptr) {
            *status_out = SingleModelDeviceRuntimeStatusCode::kKernelLaunchFailed;
        }
        return false;
    }

    if (!CheckCuda(cudaDeviceSynchronize())) {
        if (status_out != nullptr) {
            *status_out = SingleModelDeviceRuntimeStatusCode::kKernelExecutionFailed;
        }
        return false;
    }

    int raw_status = static_cast<int>(SingleModelDeviceRuntimeStatusCode::kInvalidRuntime);
    if (!CheckCuda(cudaMemcpy(&raw_status, runtime.device_status, sizeof(raw_status), cudaMemcpyDeviceToHost))) {
        if (status_out != nullptr) {
            *status_out = SingleModelDeviceRuntimeStatusCode::kCudaMemcpyFailed;
        }
        return false;
    }

    const auto runtime_status = static_cast<SingleModelDeviceRuntimeStatusCode>(raw_status);
    if ((runtime_status == SingleModelDeviceRuntimeStatusCode::kOk) &&
        !CheckCuda(cudaMemcpy(&selected_action_out, runtime.device_selected_action, sizeof(selected_action_out),
                              cudaMemcpyDeviceToHost))) {
        if (status_out != nullptr) {
            *status_out = SingleModelDeviceRuntimeStatusCode::kCudaMemcpyFailed;
        }
        return false;
    }

    if (status_out != nullptr) {
        *status_out = runtime_status;
    }

    return runtime_status == SingleModelDeviceRuntimeStatusCode::kOk;
}

} // namespace neuroevolution::inference
