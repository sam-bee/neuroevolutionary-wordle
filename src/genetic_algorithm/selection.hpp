#pragma once

#include <cstddef>
#include <limits>

#include "genetic_algorithm/spatial/grid.hpp"

namespace neuroevolution::genetic_algorithm {

constexpr std::size_t kNoIndividualIndex = std::numeric_limits<std::size_t>::max();
constexpr float kDefaultParentSelectionRankExponent = 0.5f;
constexpr float kMaximumParentSelectionRankExponent = 4.0f;

struct ParentSelectionConfig {
    std::size_t cellular_breeding_radius = spatial::kCellularBreedingRadius;
    float rank_exponent = kDefaultParentSelectionRankExponent;
};

struct ParentPair {
    std::size_t first_parent_index = 0;
    std::size_t second_parent_index = 0;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidParentSelectionConfig(const ParentSelectionConfig &config) noexcept {
    return (config.cellular_breeding_radius > 0) && (config.rank_exponent >= 0.0f) &&
           (config.rank_exponent <= kMaximumParentSelectionRankExponent);
}

} // namespace neuroevolution::genetic_algorithm
