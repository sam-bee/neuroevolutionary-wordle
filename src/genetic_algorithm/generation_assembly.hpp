#pragma once

#include "genetic_algorithm/breeding.hpp"
#include "genetic_algorithm/mutation.hpp"
#include "genetic_algorithm/selection.hpp"

namespace neuroevolution::genetic_algorithm {

struct GenerationAssemblyConfig {
    ParentSelectionConfig parent_selection{};
    BreedingConfig breeding{};
    MutationConfig mutation{};
};

constexpr bool IsValidGenerationAssemblyConfig(const GenerationAssemblyConfig &config) noexcept {
    return IsValidParentSelectionConfig(config.parent_selection) && IsValidBreedingConfig(config.breeding) &&
           IsValidMutationConfig(config.mutation);
}

} // namespace neuroevolution::genetic_algorithm
