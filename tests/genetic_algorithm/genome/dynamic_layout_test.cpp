#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>

#include "common/float16.hpp"
#include "genetic_algorithm/genome/dynamic_layout.hpp"

namespace {

using neuroevolution::common::ToFloat;
using neuroevolution::common::ToFloat16;
using neuroevolution::genetic_algorithm::genome::ActiveActionCountInTailChunk;
using neuroevolution::genetic_algorithm::genome::DynamicTailSchema;
using neuroevolution::genetic_algorithm::genome::DynamicTailSchemaForLayout;
using neuroevolution::genetic_algorithm::genome::GenomeBodyParameters;
using neuroevolution::genetic_algorithm::genome::GenomePolicyModelParameters;
using neuroevolution::genetic_algorithm::genome::GenomeTailChunk;
using neuroevolution::genetic_algorithm::genome::GenomeTailRows;
using neuroevolution::genetic_algorithm::genome::GenomeTailRow;
using neuroevolution::genetic_algorithm::genome::HostGenomeBytesAt;
using neuroevolution::genetic_algorithm::genome::HostGenomeViewAt;
using neuroevolution::genetic_algorithm::genome::HostPopulation;
using neuroevolution::genetic_algorithm::genome::IsValidDynamicPopulationLayout;
using neuroevolution::genetic_algorithm::genome::IsValidDynamicTailSchema;
using neuroevolution::genetic_algorithm::genome::MakeDynamicPopulationLayout;
using neuroevolution::genetic_algorithm::genome::MakeDynamicTailSchema;
using neuroevolution::genetic_algorithm::genome::TailRowLocationForActionIndex;
using neuroevolution::genetic_algorithm::genome::TailRowStorageIndex;
using neuroevolution::genetic_algorithm::genome::TryAllocateHostGenomeStorage;

constexpr float kTolerance = 1.0e-6f;

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

bool ExpectNear(const float actual, const float expected, const std::string_view label) {
    if (std::fabs(actual - expected) > kTolerance) {
        std::cerr << "FAIL: " << label << " expected " << expected << ", got " << actual << '\n';
        return false;
    }

    return true;
}

bool TestDynamicTailSchemaMapsActionsIntoChunks() {
    const DynamicTailSchema schema = MakeDynamicTailSchema(7, 3);

    bool ok = true;
    ok &= ExpectTrue(IsValidDynamicTailSchema(schema), "Expected chunked tail schema to be valid");
    ok &= ExpectTrue(schema.action_count == 7, "Expected tail schema to preserve action count");
    ok &= ExpectTrue(schema.chunk_action_capacity == 3, "Expected tail schema to preserve chunk capacity");
    ok &= ExpectTrue(schema.chunk_count == 3, "Expected tail schema to report three chunks");
    ok &= ExpectTrue(ActiveActionCountInTailChunk(schema, 0) == 3, "Expected first chunk to be full");
    ok &= ExpectTrue(ActiveActionCountInTailChunk(schema, 1) == 3, "Expected second chunk to be full");
    ok &= ExpectTrue(ActiveActionCountInTailChunk(schema, 2) == 1, "Expected final chunk to contain one action");

    const auto first_location = TailRowLocationForActionIndex(schema, 0);
    ok &= ExpectTrue(first_location.chunk_index == 0, "Expected action zero to live in chunk zero");
    ok &= ExpectTrue(first_location.action_offset_in_chunk == 0,
                     "Expected action zero to be the first row in its chunk");

    const auto middle_location = TailRowLocationForActionIndex(schema, 5);
    ok &= ExpectTrue(middle_location.chunk_index == 1, "Expected action five to live in chunk one");
    ok &= ExpectTrue(middle_location.action_offset_in_chunk == 2,
                     "Expected action five to be the third row in chunk one");
    ok &= ExpectTrue(TailRowStorageIndex(schema, middle_location) == 5,
                     "Expected chunked storage index to map back to the original row");

    const auto final_location = TailRowLocationForActionIndex(schema, 6);
    ok &= ExpectTrue(final_location.chunk_index == 2, "Expected final action to live in the last chunk");
    ok &= ExpectTrue(final_location.action_offset_in_chunk == 0,
                     "Expected final action to be the first row of the final chunk");
    return ok;
}

bool TestGenomeViewReadsContiguousStorageThroughChunkedSchema() {
    HostPopulation population{};
    population.layout = MakeDynamicPopulationLayout(1, 0, 5, 2);

    bool ok = true;
    ok &= ExpectTrue(IsValidDynamicPopulationLayout(population.layout), "Expected chunked population layout to be valid");
    ok &= TryAllocateHostGenomeStorage(population);
    ok &= ExpectTrue(ok, "Expected host genome storage allocation to succeed");
    if (!ok) {
        return false;
    }

    const auto genome_view = HostGenomeViewAt(population, 0);
    ok &= ExpectTrue(&GenomeBodyParameters(genome_view) == &GenomePolicyModelParameters(HostGenomeBytesAt(population, 0)),
                     "Expected genome body view to point at the policy-model storage");
    ok &= ExpectTrue(DynamicTailSchemaForLayout(population.layout).chunk_count == 3,
                     "Expected chunked population layout to expose three logical chunks");

    for (std::size_t action_index = 0; action_index < population.layout.action_count; ++action_index) {
        GenomeTailRow(genome_view, action_index)[0] = ToFloat16(static_cast<float>(action_index) + 0.5f);
    }

    const auto middle_chunk = GenomeTailChunk(genome_view, 1);
    ok &= ExpectTrue(middle_chunk.chunk_index == 1, "Expected middle chunk index to be preserved");
    ok &= ExpectTrue(middle_chunk.first_action_index == 2,
                     "Expected middle chunk to start at action index two");
    ok &= ExpectTrue(middle_chunk.active_action_count == 2, "Expected middle chunk to contain two active actions");
    ok &= ExpectNear(ToFloat(middle_chunk.rows[0][0]), 2.5f, "Expected middle chunk row zero to map to action two");
    ok &= ExpectNear(ToFloat(middle_chunk.rows[1][0]), 3.5f, "Expected middle chunk row one to map to action three");

    const auto final_chunk = GenomeTailChunk(genome_view, 2);
    ok &= ExpectTrue(final_chunk.first_action_index == 4, "Expected final chunk to start at action index four");
    ok &= ExpectTrue(final_chunk.active_action_count == 1, "Expected final chunk to contain one active action");
    ok &= ExpectNear(ToFloat(final_chunk.rows[0][0]), 4.5f, "Expected final chunk row zero to map to action four");

    const auto contiguous_tail_rows = GenomeTailRows(HostGenomeBytesAt(population, 0));
    ok &= ExpectNear(ToFloat(contiguous_tail_rows[4][0]), 4.5f,
                     "Expected contiguous compatibility storage to match chunked view access");
    return ok;
}

} // namespace

int main() {
    if (!TestDynamicTailSchemaMapsActionsIntoChunks()) {
        return 1;
    }

    if (!TestGenomeViewReadsContiguousStorageThroughChunkedSchema()) {
        return 1;
    }

    std::cout << "PASS: dynamic_layout_test\n";
    return 0;
}
