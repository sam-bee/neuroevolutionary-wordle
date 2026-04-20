#pragma once

#include <cstddef>
#include <cstdint>

#include "model/output_embedding/output_embedding.hpp"
#include "training_folder/training_data.hpp"
#include "wordle/wordle_grid.hpp"

namespace neuroevolution::inference {

enum class SingleModelDeviceRuntimeStatusCode : int {
    kOk = 0,
    kInvalidRuntime = 1,
    kCudaMemcpyFailed = 2,
    kKernelLaunchFailed = 3,
    kKernelExecutionFailed = 4,
    kPolicyForwardFailed = 5,
    kActionSelectionFailed = 6,
};

struct SingleModelDeviceRuntime {
    std::uint8_t *device_genome_bytes = nullptr;
    training_folder::TrainingWordCatalog *device_action_space_words = nullptr;
    wordle::WordleGrid *device_grid = nullptr;
    model::output_embedding::SelectedAction *device_selected_action = nullptr;
    int *device_status = nullptr;
    std::size_t action_count = 0;
    std::size_t genome_byte_count = 0;
};

constexpr bool IsValidSingleModelDeviceRuntime(const SingleModelDeviceRuntime &runtime) noexcept {
    return (runtime.device_genome_bytes != nullptr) && (runtime.device_action_space_words != nullptr) &&
           (runtime.device_grid != nullptr) && (runtime.device_selected_action != nullptr) &&
           (runtime.device_status != nullptr) && (runtime.action_count > 0) && (runtime.genome_byte_count > 0);
}

bool TryCreateSingleModelDeviceRuntime(const std::uint8_t *host_genome_bytes, std::size_t genome_byte_count,
                                       const training_folder::TrainingWordCatalog &action_space_words,
                                       SingleModelDeviceRuntime &runtime_out);

void DestroySingleModelDeviceRuntime(SingleModelDeviceRuntime &runtime) noexcept;

bool TrySelectNextGuessWithSingleModelDeviceRuntime(
    const SingleModelDeviceRuntime &runtime, const wordle::WordleGrid &grid,
    model::output_embedding::SelectedAction &selected_action_out,
    SingleModelDeviceRuntimeStatusCode *status_out = nullptr);

} // namespace neuroevolution::inference
