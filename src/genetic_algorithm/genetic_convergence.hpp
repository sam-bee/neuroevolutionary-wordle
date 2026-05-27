#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

#include "common/float16.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"
#include "model/output_embedding/output_embedding.hpp"

namespace neuroevolution::genetic_algorithm::genetic_convergence {

constexpr std::size_t kDefaultSampleOrganismCount = 256;
constexpr std::size_t kDefaultSampleWeightCount = 8192;
constexpr std::size_t kDefaultPairCount = 512;
constexpr std::size_t kDefaultIntervalGenerations = 10;

struct GeneticConvergencePair {
    std::size_t first = 0;
    std::size_t second = 0;
};

struct GeneticConvergenceSamplePlan {
    std::vector<std::size_t> organism_indices{};
    std::vector<std::size_t> weight_indices{};
    std::vector<GeneticConvergencePair> pairs{};
};

struct GeneticConvergenceMetrics {
    float centroid_distance_mean = 0.0f;
    float centroid_distance_min = 0.0f;
    float centroid_distance_max = 0.0f;
    float pairwise_distance_mean = 0.0f;
    float pairwise_distance_min = 0.0f;
    float pairwise_distance_max = 0.0f;
};

constexpr std::size_t PolicyModelTrainableScalarCount() noexcept {
    static_assert(sizeof(genome::PolicyModelParameters) % sizeof(common::Float16) == 0,
                  "Policy model parameters are expected to be contiguous fp16 trainable scalars.");
    return sizeof(genome::PolicyModelParameters) / sizeof(common::Float16);
}

constexpr std::size_t TrainableScalarCountForActionCount(const std::size_t action_count) noexcept {
    return PolicyModelTrainableScalarCount() + (action_count * model::output_embedding::kTrainableFeatureDimension);
}

inline std::uint64_t MixSamplingSeed(const std::uint32_t run_seed, const std::size_t generation) noexcept {
    std::uint64_t value = 0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(run_seed);
    value ^= (static_cast<std::uint64_t>(generation) + 0xbf58476d1ce4e5b9ULL + (value << 6U) + (value >> 2U));
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

inline std::vector<std::size_t> SampleUniqueIndices(const std::size_t universe_size, const std::size_t requested_count,
                                                    std::mt19937_64 &random_engine) {
    const std::size_t sample_count = (requested_count < universe_size) ? requested_count : universe_size;
    std::vector<std::size_t> indices{};
    indices.reserve(sample_count);
    if (sample_count == universe_size) {
        indices.resize(sample_count);
        std::iota(indices.begin(), indices.end(), std::size_t{0});
        return indices;
    }

    std::unordered_set<std::size_t> selected{};
    selected.reserve(sample_count * 2U);
    for (std::size_t cursor = universe_size - sample_count; cursor < universe_size; ++cursor) {
        std::uniform_int_distribution<std::size_t> distribution(0, cursor);
        const std::size_t candidate = distribution(random_engine);
        const std::size_t value = (selected.find(candidate) != selected.end()) ? cursor : candidate;
        selected.insert(value);
        indices.push_back(value);
    }
    return indices;
}

inline GeneticConvergenceSamplePlan MakeSamplePlan(const std::size_t live_organism_count,
                                                   const std::size_t trainable_weight_count,
                                                   const std::uint32_t run_seed, const std::size_t generation) {
    GeneticConvergenceSamplePlan plan{};
    if ((live_organism_count == 0) || (trainable_weight_count == 0)) {
        return plan;
    }

    std::mt19937_64 random_engine(MixSamplingSeed(run_seed, generation));
    plan.organism_indices = SampleUniqueIndices(live_organism_count, kDefaultSampleOrganismCount, random_engine);
    plan.weight_indices = SampleUniqueIndices(trainable_weight_count, kDefaultSampleWeightCount, random_engine);

    if (plan.organism_indices.size() < 2) {
        return plan;
    }

    plan.pairs.reserve(kDefaultPairCount);
    std::uniform_int_distribution<std::size_t> first_distribution(0, plan.organism_indices.size() - 1U);
    std::uniform_int_distribution<std::size_t> second_distribution(0, plan.organism_indices.size() - 2U);
    for (std::size_t pair_index = 0; pair_index < kDefaultPairCount; ++pair_index) {
        const std::size_t first = first_distribution(random_engine);
        std::size_t second = second_distribution(random_engine);
        if (second >= first) {
            ++second;
        }
        plan.pairs.push_back(GeneticConvergencePair{first, second});
    }
    return plan;
}

inline bool TryComputeMetrics(const std::vector<float> &sampled_values, const std::size_t sample_organism_count,
                              const std::size_t sample_weight_count, const std::vector<GeneticConvergencePair> &pairs,
                              GeneticConvergenceMetrics &metrics_out) {
    metrics_out = {};
    if ((sample_organism_count == 0) || (sample_weight_count == 0) ||
        (sampled_values.size() != (sample_organism_count * sample_weight_count))) {
        return false;
    }

    std::vector<float> centroid(sample_weight_count, 0.0f);
    for (std::size_t organism_index = 0; organism_index < sample_organism_count; ++organism_index) {
        const std::size_t organism_offset = organism_index * sample_weight_count;
        for (std::size_t weight_index = 0; weight_index < sample_weight_count; ++weight_index) {
            centroid[weight_index] += sampled_values[organism_offset + weight_index];
        }
    }

    const float inverse_organism_count = 1.0f / static_cast<float>(sample_organism_count);
    for (float &value : centroid) {
        value *= inverse_organism_count;
    }

    float centroid_distance_sum = 0.0f;
    metrics_out.centroid_distance_min = std::numeric_limits<float>::max();
    for (std::size_t organism_index = 0; organism_index < sample_organism_count; ++organism_index) {
        const std::size_t organism_offset = organism_index * sample_weight_count;
        float squared_distance = 0.0f;
        for (std::size_t weight_index = 0; weight_index < sample_weight_count; ++weight_index) {
            const float delta = sampled_values[organism_offset + weight_index] - centroid[weight_index];
            squared_distance += delta * delta;
        }

        const float distance = std::sqrt(squared_distance);
        centroid_distance_sum += distance;
        if (distance < metrics_out.centroid_distance_min) {
            metrics_out.centroid_distance_min = distance;
        }
        if (distance > metrics_out.centroid_distance_max) {
            metrics_out.centroid_distance_max = distance;
        }
    }
    metrics_out.centroid_distance_mean = centroid_distance_sum * inverse_organism_count;

    if (pairs.empty()) {
        metrics_out.pairwise_distance_min = 0.0f;
        return true;
    }

    float pairwise_distance_sum = 0.0f;
    metrics_out.pairwise_distance_min = std::numeric_limits<float>::max();
    for (const GeneticConvergencePair &pair : pairs) {
        if ((pair.first >= sample_organism_count) || (pair.second >= sample_organism_count) ||
            (pair.first == pair.second)) {
            metrics_out = {};
            return false;
        }

        const std::size_t first_offset = pair.first * sample_weight_count;
        const std::size_t second_offset = pair.second * sample_weight_count;
        float squared_distance = 0.0f;
        for (std::size_t weight_index = 0; weight_index < sample_weight_count; ++weight_index) {
            const float delta =
                sampled_values[first_offset + weight_index] - sampled_values[second_offset + weight_index];
            squared_distance += delta * delta;
        }

        const float distance = std::sqrt(squared_distance);
        pairwise_distance_sum += distance;
        if (distance < metrics_out.pairwise_distance_min) {
            metrics_out.pairwise_distance_min = distance;
        }
        if (distance > metrics_out.pairwise_distance_max) {
            metrics_out.pairwise_distance_max = distance;
        }
    }
    metrics_out.pairwise_distance_mean = pairwise_distance_sum / static_cast<float>(pairs.size());
    return true;
}

} // namespace neuroevolution::genetic_algorithm::genetic_convergence
