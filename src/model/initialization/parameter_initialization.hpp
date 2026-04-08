#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

#include "common/fixed_buffer.hpp"
#include "common/float16.hpp"
#include "model/output_embedding/output_embedding.hpp"
#include "model/policy_model/policy_model.hpp"

namespace neuroevolution::model::initialization {

using RandomEngine = std::mt19937;

struct ParameterInitializationConfig {
    float dense_weight_gain = 1.0f;
    float output_embedding_tail_stddev = 0.05f;
};

constexpr bool IsValidParameterInitializationConfig(const ParameterInitializationConfig &config) noexcept {
    return (config.dense_weight_gain > 0.0f) && (config.output_embedding_tail_stddev >= 0.0f);
}

namespace detail {

template <typename Scalar, std::size_t Size>
inline void FillBufferWithConstant(common::FixedBuffer<Scalar, Size> &buffer, const float value) {
    for (std::size_t index = 0; index < Size; ++index) {
        buffer[index] = common::ToFloat16(value);
    }
}

template <typename Scalar, std::size_t Size>
inline void FillBufferWithNormal(common::FixedBuffer<Scalar, Size> &buffer, RandomEngine &random_engine,
                                 const float stddev) {
    std::normal_distribution<float> distribution(0.0f, stddev);

    for (std::size_t index = 0; index < Size; ++index) {
        buffer[index] = common::ToFloat16(distribution(random_engine));
    }
}

inline float HeNormalStddev(const std::size_t fan_in, const float gain) {
    return gain * std::sqrt(2.0f / static_cast<float>(fan_in));
}

template <typename DenseLayer>
inline void InitializeDenseLayer(DenseLayer &layer, RandomEngine &random_engine, const float weight_stddev) {
    FillBufferWithNormal(layer.weights, random_engine, weight_stddev);
    FillBufferWithConstant(layer.biases, 0.0f);
}

} // namespace detail

void InitializeRandomPolicyModelParameters(policy_model::PolicyModelParameters &parameters, RandomEngine &random_engine,
                                           const ParameterInitializationConfig &config = {});

policy_model::PolicyModelParameters MakeRandomPolicyModelParameters(std::uint32_t seed,
                                                                    const ParameterInitializationConfig &config = {});

template <std::size_t ActionCount>
inline void InitializeRandomOutputEmbeddingTrainableTails(
    common::FixedBuffer<output_embedding::TrainableActionEmbeddingTail, ActionCount> &trainable_tails,
    RandomEngine &random_engine, const ParameterInitializationConfig &config = {}) {
    for (std::size_t action_index = 0; action_index < ActionCount; ++action_index) {
        detail::FillBufferWithNormal(trainable_tails[action_index], random_engine, config.output_embedding_tail_stddev);
    }
}

template <std::size_t ActionCount>
inline common::FixedBuffer<output_embedding::TrainableActionEmbeddingTail, ActionCount>
MakeRandomOutputEmbeddingTrainableTails(const std::uint32_t seed, const ParameterInitializationConfig &config = {}) {
    RandomEngine random_engine(seed);
    common::FixedBuffer<output_embedding::TrainableActionEmbeddingTail, ActionCount> trainable_tails{};
    InitializeRandomOutputEmbeddingTrainableTails(trainable_tails, random_engine, config);
    return trainable_tails;
}

} // namespace neuroevolution::model::initialization
