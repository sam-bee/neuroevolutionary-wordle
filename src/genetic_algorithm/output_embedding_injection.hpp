#pragma once

#include <cmath>
#include <cstddef>

#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"
#include "common/float16.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "model/policy_model/policy_model.hpp"
#include "wordle/hint_grid.hpp"

namespace neuroevolution::genetic_algorithm {

namespace output_embedding_injection {

constexpr float kTailNormEpsilon = 1.0e-6f;

inline NEUROEVOLUTION_HOST_DEVICE void SwapFloat(float &left, float &right) noexcept {
    const float temporary = left;
    left = right;
    right = temporary;
}

template <std::size_t Capacity>
inline NEUROEVOLUTION_HOST_DEVICE float SelectKth(common::FixedBuffer<float, Capacity> &values, const std::size_t count,
                                                  const std::size_t kth) noexcept {
    std::size_t left = 0;
    std::size_t right = count - 1;

    while (left < right) {
        const float pivot = values[(left + right) / 2];
        std::size_t lower = left;
        std::size_t upper = right;

        while (lower <= upper) {
            while (values[lower] < pivot) {
                ++lower;
            }
            while (values[upper] > pivot) {
                if (upper == 0) {
                    break;
                }
                --upper;
            }
            if (lower <= upper) {
                SwapFloat(values[lower], values[upper]);
                ++lower;
                if (upper == 0) {
                    break;
                }
                --upper;
            }
        }

        if (kth <= upper) {
            right = upper;
        } else if (kth >= lower) {
            left = lower;
        } else {
            return values[kth];
        }
    }

    return values[left];
}

inline NEUROEVOLUTION_HOST_DEVICE float
TrainableTailNorm(const model::output_embedding::TrainableActionEmbeddingTail &tail) noexcept {
    float sum_of_squares = 0.0f;
    for (std::size_t feature_index = 0; feature_index < model::output_embedding::kTrainableFeatureDimension;
         ++feature_index) {
        const float value = common::ToFloat(tail[feature_index]);
        sum_of_squares += value * value;
    }

    return sqrtf(sum_of_squares);
}

template <std::size_t NormCapacity>
inline NEUROEVOLUTION_HOST_DEVICE bool
TryComputeMedianTailNorm(const model::output_embedding::TrainableActionEmbeddingTail *tail_rows,
                         const std::size_t tail_count, common::FixedBuffer<float, NormCapacity> &norms,
                         float &median_norm_out) noexcept {
    median_norm_out = 0.0f;
    if ((tail_rows == nullptr) || (tail_count == 0) || (tail_count > NormCapacity)) {
        return false;
    }

    for (std::size_t action_index = 0; action_index < tail_count; ++action_index) {
        norms[action_index] = TrainableTailNorm(tail_rows[action_index]);
    }

    const std::size_t upper_middle_index = tail_count / 2;
    const float upper_middle = SelectKth(norms, tail_count, upper_middle_index);
    if ((tail_count % 2) != 0) {
        median_norm_out = upper_middle;
        return true;
    }

    const float lower_middle = SelectKth(norms, tail_count, upper_middle_index - 1);
    median_norm_out = 0.5f * (lower_middle + upper_middle);
    return true;
}

} // namespace output_embedding_injection

#if defined(__CUDACC__)
template <int WarpWidth> struct OutputEmbeddingInjectionWarpScratch {
    wordle::HintGridGroup hint_grid_group{};
    common::FixedBuffer<float, model::output_embedding::kTrainableFeatureDimension> feature_sums{};
    model::policy_model::PolicyModelWarpScratch<WarpWidth> policy_model{};
    float target_norm = 0.0f;
    float current_norm = 0.0f;
    float scale = 0.0f;
    int status = 0;
};

template <int WarpWidth>
inline __device__ bool
TrySeedOutputEmbeddingTailFromHintGridsConcurrently(const model::policy_model::PolicyModelParameters &policy_model,
                                                    const wordle::Word &target_word,
                                                    model::output_embedding::TrainableActionEmbeddingTail &tail_out,
                                                    OutputEmbeddingInjectionWarpScratch<WarpWidth> &scratch) noexcept {
    const std::size_t lane_index = static_cast<std::size_t>(threadIdx.x % WarpWidth);

    if (lane_index == 0) {
        tail_out = {};
        scratch.status =
            wordle::IsValidWord(target_word) && wordle::TryBuildHintGridGroup(target_word, scratch.hint_grid_group);
    }
    __syncwarp();

    if (scratch.status == 0) {
        return false;
    }

    for (std::size_t feature_index = lane_index; feature_index < model::output_embedding::kTrainableFeatureDimension;
         feature_index += WarpWidth) {
        scratch.feature_sums[feature_index] = 0.0f;
    }
    __syncwarp();

    for (std::size_t grid_index = 0; grid_index < wordle::kHintGridGroupSize; ++grid_index) {
        const bool forward_ok = model::policy_model::TryForwardPolicyModelConcurrently<WarpWidth>(
            policy_model, scratch.hint_grid_group.grids[grid_index], scratch.policy_model);
        if (lane_index == 0) {
            scratch.status = forward_ok ? 1 : 0;
        }
        __syncwarp();

        if (scratch.status == 0) {
            return false;
        }

        for (std::size_t feature_index = lane_index;
             feature_index < model::output_embedding::kTrainableFeatureDimension; feature_index += WarpWidth) {
            scratch.feature_sums[feature_index] +=
                scratch.policy_model.policy_vector[model::output_embedding::kWordFeatureDimension + feature_index];
        }
        __syncwarp();
    }

    constexpr float kReciprocalHintGridCount = 1.0f / static_cast<float>(wordle::kHintGridGroupSize);
    for (std::size_t feature_index = lane_index; feature_index < model::output_embedding::kTrainableFeatureDimension;
         feature_index += WarpWidth) {
        tail_out[feature_index] = common::ToFloat16(scratch.feature_sums[feature_index] * kReciprocalHintGridCount);
    }
    __syncwarp();

    return true;
}

template <int WarpWidth>
inline __device__ bool
TryScaleTrainableTailToNormConcurrently(model::output_embedding::TrainableActionEmbeddingTail &tail,
                                        const float target_norm,
                                        OutputEmbeddingInjectionWarpScratch<WarpWidth> &scratch) noexcept {
    const std::size_t lane_index = static_cast<std::size_t>(threadIdx.x % WarpWidth);

    if (lane_index == 0) {
        if (target_norm <= output_embedding_injection::kTailNormEpsilon) {
            scratch.current_norm = 0.0f;
            scratch.scale = 0.0f;
            scratch.status = 1;
        } else {
            scratch.current_norm = output_embedding_injection::TrainableTailNorm(tail);
            scratch.status = (scratch.current_norm > output_embedding_injection::kTailNormEpsilon) ? 1 : 0;
            scratch.scale = (scratch.status != 0) ? (target_norm / scratch.current_norm) : 0.0f;
        }
    }
    __syncwarp();

    if (scratch.status == 0) {
        return false;
    }

    for (std::size_t feature_index = lane_index; feature_index < model::output_embedding::kTrainableFeatureDimension;
         feature_index += WarpWidth) {
        tail[feature_index] = common::ToFloat16(common::ToFloat(tail[feature_index]) * scratch.scale);
    }
    __syncwarp();

    return true;
}
#endif

} // namespace neuroevolution::genetic_algorithm
