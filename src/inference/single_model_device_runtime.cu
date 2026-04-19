#include "inference/single_model_device_runtime.hpp"

#include <cuda_runtime.h>

#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "inference/dynamic_policy.hpp"

namespace neuroevolution::inference {

namespace {

using neuroevolution::inference::dynamic_policy::ScoreDynamicActionEmbedding;
using neuroevolution::model::output_embedding::SelectedAction;
using neuroevolution::model::policy_model::PolicyVector;
using neuroevolution::model::policy_model::TryForwardPolicyModel;
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
    __shared__ PolicyVector shared_policy_vector;
    __shared__ SelectedAction shared_best_actions[kSingleModelInferenceThreadsPerBlock];
    __shared__ int shared_has_candidate[kSingleModelInferenceThreadsPerBlock];
    __shared__ int shared_status;

    if ((action_space_words == nullptr) || (genome_bytes == nullptr) || (grid == nullptr) ||
        (selected_action_out == nullptr) || (status_out == nullptr) || (action_count == 0) ||
        (action_count > action_space_words->word_count)) {
        if (threadIdx.x == 0) {
            *status_out = static_cast<int>(SingleModelDeviceRuntimeStatusCode::kInvalidRuntime);
        }
        return;
    }

    if (threadIdx.x == 0) {
        if (!TryForwardPolicyModel(genetic_algorithm::genome::GenomePolicyModelParameters(genome_bytes), *grid,
                                   shared_policy_vector)) {
            shared_status = static_cast<int>(SingleModelDeviceRuntimeStatusCode::kPolicyForwardFailed);
        } else {
            shared_status = static_cast<int>(SingleModelDeviceRuntimeStatusCode::kOk);
        }
    }

    __syncthreads();

    if (shared_status != static_cast<int>(kOk)) {
        return;
    }

    const genetic_algorithm::genome::TrainableActionEmbeddingTail *tail_rows =
        genetic_algorithm::genome::GenomeTailRows(genome_bytes);

    SelectedAction local_best_action{};
    bool has_local_candidate = false;
    for (std::size_t action_index = static_cast<std::size_t>(threadIdx.x); action_index < action_count;
         action_index += static_cast<std::size_t>(blockDim.x)) {
        const float score = ScoreDynamicActionEmbedding(shared_policy_vector, action_space_words->words[action_index],
                                                        tail_rows[action_index]);
        if (!has_local_candidate || (score > local_best_action.score)) {
            local_best_action.action_index = action_index;
            local_best_action.word = action_space_words->words[action_index];
            local_best_action.score = score;
            has_local_candidate = true;
        }
    }

    shared_has_candidate[threadIdx.x] = has_local_candidate ? 1 : 0;
    if (has_local_candidate) {
        shared_best_actions[threadIdx.x] = local_best_action;
    }

    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            const int peer_index = threadIdx.x + offset;
            if ((shared_has_candidate[peer_index] != 0) &&
                ((shared_has_candidate[threadIdx.x] == 0) ||
                 (shared_best_actions[peer_index].score > shared_best_actions[threadIdx.x].score))) {
                shared_best_actions[threadIdx.x] = shared_best_actions[peer_index];
                shared_has_candidate[threadIdx.x] = 1;
            }
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (shared_has_candidate[0] == 0) {
            *status_out = static_cast<int>(SingleModelDeviceRuntimeStatusCode::kActionSelectionFailed);
            return;
        }

        *selected_action_out = shared_best_actions[0];
        *status_out = static_cast<int>(SingleModelDeviceRuntimeStatusCode::kOk);
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

    SelectNextGuessKernel<<<1, kSingleModelInferenceThreadsPerBlock>>>(
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
