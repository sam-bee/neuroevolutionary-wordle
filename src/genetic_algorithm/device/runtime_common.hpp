#pragma once

#include <cstddef>
#include <cstdint>

#include "training_folder/training_data.hpp"

namespace neuroevolution::genetic_algorithm::device_common {

struct RuntimeWordCounts {
    std::size_t training_word_count = training_folder::kDefaultInitialActiveWordCount;
    std::size_t action_space_word_count = training_folder::kDefaultInitialActiveWordCount;
    training_folder::WordCountSchedule training_word_schedule{};
    std::size_t shard_initial_radius = training_folder::kDefaultTrainingShardInitialRadius;
    std::size_t shard_radius_growth_period_generations = training_folder::kDefaultShardRadiusGrowthPeriodGenerations;
};

struct PopulationFitnessSummary {
    float best_fitness = 0.0f;
    float average_fitness = 0.0f;
    std::size_t best_index = 0;
    std::uint32_t best_slot_index = static_cast<std::uint32_t>(-1);
    std::size_t generation_index = 0;
    std::size_t action_count = 0;
    std::size_t population_size = 0;
};

} // namespace neuroevolution::genetic_algorithm::device_common
