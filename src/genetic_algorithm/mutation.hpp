#pragma once

#include "common/cuda_compat.hpp"

namespace neuroevolution::genetic_algorithm {

constexpr float kDefaultMutationProbability = 0.0001f;
constexpr float kDefaultMutationSigma = 0.02f;
constexpr float kDefaultOutputTailRowScaleMutationProbability = 0.0f;

struct MutationConfig {
    float mutation_probability = kDefaultMutationProbability;
    float mutation_sigma = kDefaultMutationSigma;
    float output_tail_row_scale_mutation_probability = kDefaultOutputTailRowScaleMutationProbability;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidMutationConfig(const MutationConfig &config) noexcept {
    return (config.mutation_probability >= 0.0f) && (config.mutation_probability <= 1.0f) &&
           (config.mutation_sigma >= 0.0f) && (config.output_tail_row_scale_mutation_probability >= 0.0f) &&
           (config.output_tail_row_scale_mutation_probability <= 1.0f);
}

} // namespace neuroevolution::genetic_algorithm
