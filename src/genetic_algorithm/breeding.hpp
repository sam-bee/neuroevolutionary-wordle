#pragma once

#include "common/cuda_compat.hpp"
namespace neuroevolution::genetic_algorithm {

constexpr float kDefaultCrossoverTemperatureLevel1 = 0.02f;
constexpr float kDefaultCrossoverTemperatureLevel2 = 0.01f;
constexpr float kDefaultCrossoverTemperatureLevel3 = 0.005f;

struct BreedingConfig {
    float crossover_temperature_level1 = kDefaultCrossoverTemperatureLevel1;
    float crossover_temperature_level2 = kDefaultCrossoverTemperatureLevel2;
    float crossover_temperature_level3 = kDefaultCrossoverTemperatureLevel3;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidBreedingConfig(const BreedingConfig &config) noexcept {
    return (config.crossover_temperature_level1 >= 0.0f) && (config.crossover_temperature_level1 <= 1.0f) &&
           (config.crossover_temperature_level2 >= 0.0f) && (config.crossover_temperature_level2 <= 1.0f) &&
           (config.crossover_temperature_level3 >= 0.0f) && (config.crossover_temperature_level3 <= 1.0f);
}

} // namespace neuroevolution::genetic_algorithm
