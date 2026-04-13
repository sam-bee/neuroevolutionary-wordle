#include "genetic_algorithm/genome/dynamic_layout.hpp"

#include <cstring>
#include <new>
#include <random>

#include "common/float16.hpp"
#include "model/initialization/parameter_initialization.hpp"

namespace neuroevolution::genetic_algorithm::genome {

namespace {

void FillTailRowsWithNormal(TrainableActionEmbeddingTail *tail_rows, const std::size_t action_count,
                            model::initialization::RandomEngine &random_engine,
                            const model::initialization::ParameterInitializationConfig &config) {
    std::normal_distribution<float> distribution(0.0f, config.output_embedding_tail_stddev);

    for (std::size_t action_index = 0; action_index < action_count; ++action_index) {
        for (std::size_t feature_index = 0; feature_index < model::output_embedding::kTrainableFeatureDimension;
             ++feature_index) {
            tail_rows[action_index][feature_index] = common::ToFloat16(distribution(random_engine));
        }
    }
}

} // namespace

void AlignedGenomeStorageDeleter::operator()(std::uint8_t *pointer) const noexcept {
    if (pointer != nullptr) {
        ::operator delete[](pointer, std::align_val_t(DynamicGenomeAlignment()));
    }
}

bool TryAllocateHostGenomeStorage(HostPopulation &population) {
    population.genomes.reset();

    if (population.layout.genotype_bytes == 0) {
        return false;
    }

    try {
        population.genomes.reset(static_cast<std::uint8_t *>(
            ::operator new[](population.layout.genotype_bytes, std::align_val_t(DynamicGenomeAlignment()))));
    } catch (...) {
        return false;
    }

    std::memset(population.genomes.get(), 0, population.layout.genotype_bytes);
    return true;
}

bool TryInitializeRandomHostPopulation(HostPopulation &population, const std::size_t population_size,
                                       const std::size_t action_count, const std::uint32_t seed,
                                       const PopulationInitializationConfig &config) {
    if (!IsValidPopulationInitializationConfig(config)) {
        return false;
    }

    population = {};
    population.layout.active_individual_count = population_size;
    population.layout.generation_index = 0;
    population.layout.action_count = action_count;
    population.layout.genome_stride_bytes = ComputeDynamicGenomeStrideBytes(action_count);
    population.layout.genotype_bytes = population_size * population.layout.genome_stride_bytes;
    if (!IsValidDynamicPopulationLayout(population.layout) || !TryAllocateHostGenomeStorage(population)) {
        return false;
    }

    model::initialization::RandomEngine random_engine(seed);
    for (std::size_t individual_index = 0; individual_index < population.layout.active_individual_count; ++individual_index) {
        std::uint8_t *genome_bytes = HostGenomeBytesAt(population, individual_index);
        model::initialization::InitializeRandomPolicyModelParameters(GenomePolicyModelParameters(genome_bytes),
                                                                     random_engine, config.parameter_initialization);
        FillTailRowsWithNormal(GenomeTailRows(genome_bytes), population.layout.action_count, random_engine,
                               config.parameter_initialization);
    }

    return true;
}

} // namespace neuroevolution::genetic_algorithm::genome
