#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "genetic_algorithm/spatial/grid.hpp"

namespace {

using neuroevolution::genetic_algorithm::spatial::CellularGridShape;
using neuroevolution::genetic_algorithm::spatial::CellularNeighborList;
using neuroevolution::genetic_algorithm::spatial::ContainsNeighborIndex;
using neuroevolution::genetic_algorithm::spatial::FloorSquarePopulationSize;
using neuroevolution::genetic_algorithm::spatial::GridColumnFromIndex;
using neuroevolution::genetic_algorithm::spatial::GridIndexFromRowColumn;
using neuroevolution::genetic_algorithm::spatial::GridRowFromIndex;
using neuroevolution::genetic_algorithm::spatial::IsValidCellularGridShape;
using neuroevolution::genetic_algorithm::spatial::TryCollectCellularSecondParentCandidates;
using neuroevolution::genetic_algorithm::spatial::TryMakeCellularGridShape;
using neuroevolution::genetic_algorithm::spatial::TryProjectCellIndexBetweenSquareGrids;
using neuroevolution::genetic_algorithm::spatial::WrapToroidalCoordinate;

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

bool TestPopulationFlooringRoundsDownToSquares() {
    bool ok = true;
    ok &= ExpectTrue(FloorSquarePopulationSize(1) == 1, "Expected 1 to remain 1");
    ok &= ExpectTrue(FloorSquarePopulationSize(2) == 1, "Expected 2 to floor to 1");
    ok &= ExpectTrue(FloorSquarePopulationSize(15) == 9, "Expected 15 to floor to 9");
    ok &= ExpectTrue(FloorSquarePopulationSize(16) == 16, "Expected 16 to remain 16");
    ok &= ExpectTrue(FloorSquarePopulationSize(17) == 16, "Expected 17 to floor to 16");
    ok &= ExpectTrue(FloorSquarePopulationSize(26) == 25, "Expected 26 to floor to 25");
    return ok;
}

bool TestGridShapeAndIndexRoundTrips() {
    CellularGridShape shape{};
    bool ok = true;
    ok &= TryMakeCellularGridShape(16, shape);
    ok &= ExpectTrue(IsValidCellularGridShape(shape), "Expected 4x4 shape to be valid");
    ok &= ExpectTrue(shape.side_length == 4, "Expected 16 cells to produce side length 4");
    ok &= ExpectTrue(GridIndexFromRowColumn(shape, 2, 3) == 11, "Expected row/column to map to linear index");
    ok &= ExpectTrue(GridRowFromIndex(shape, 11) == 2, "Expected linear index to map back to row");
    ok &= ExpectTrue(GridColumnFromIndex(shape, 11) == 3, "Expected linear index to map back to column");

    CellularGridShape invalid_shape{};
    ok &= ExpectTrue(!TryMakeCellularGridShape(15, invalid_shape),
                     "Expected non-square cell counts to be rejected as grid shapes");
    return ok;
}

bool TestToroidalWrappingWrapsBothDirections() {
    bool ok = true;
    ok &= ExpectTrue(WrapToroidalCoordinate(-1, 5) == 4, "Expected -1 to wrap to the final coordinate");
    ok &= ExpectTrue(WrapToroidalCoordinate(5, 5) == 0, "Expected side length to wrap back to zero");
    ok &= ExpectTrue(WrapToroidalCoordinate(6, 5) == 1, "Expected values above side length to wrap");
    return ok;
}

bool TestRadiusTwoMooreNeighborhoodHasTwentyFourDistinctCandidatesOnFiveByFiveGrid() {
    CellularGridShape shape{};
    CellularNeighborList neighbors{};
    bool ok = true;
    ok &= TryMakeCellularGridShape(25, shape);
    ok &= TryCollectCellularSecondParentCandidates(shape, 12, neighbors);
    ok &= ExpectTrue(neighbors.count == 24, "Expected 5x5 radius-two Moore neighborhood to have 24 candidates");
    ok &= ExpectTrue(!ContainsNeighborIndex(neighbors, 12), "Expected focal cell to be excluded from candidates");

    for (std::size_t cell_index = 0; cell_index < shape.cell_count; ++cell_index) {
        if (cell_index == 12) {
            continue;
        }

        ok &= ExpectTrue(ContainsNeighborIndex(neighbors, cell_index),
                         "Expected every other 5x5 cell to appear in the radius-two neighborhood");
    }

    return ok;
}

bool TestRadiusTwoMooreNeighborhoodDeduplicatesOnSmallToroidalGrids() {
    CellularGridShape shape{};
    CellularNeighborList neighbors{};
    bool ok = true;
    ok &= TryMakeCellularGridShape(9, shape);
    ok &= TryCollectCellularSecondParentCandidates(shape, 4, neighbors);
    ok &= ExpectTrue(neighbors.count == 8,
                     "Expected a 3x3 toroidal grid to deduplicate radius-two neighbors down to the other 8 cells");
    ok &= ExpectTrue(!ContainsNeighborIndex(neighbors, 4), "Expected focal cell exclusion to survive deduplication");
    return ok;
}

bool TestGridProjectionMapsShrunkenChildCellsAcrossTheCurrentGrid() {
    CellularGridShape source_shape{};
    CellularGridShape target_shape{};
    bool ok = true;
    ok &= TryMakeCellularGridShape(25, source_shape);
    ok &= TryMakeCellularGridShape(9, target_shape);

    std::size_t source_cell_index = 0;
    ok &= TryProjectCellIndexBetweenSquareGrids(source_shape, target_shape, 0, source_cell_index);
    ok &= ExpectTrue(source_cell_index == 0, "Expected the top-left child cell to project to the top-left parent");
    ok &= TryProjectCellIndexBetweenSquareGrids(source_shape, target_shape, 4, source_cell_index);
    ok &= ExpectTrue(source_cell_index == 12, "Expected the center child cell to project to the center parent");
    ok &= TryProjectCellIndexBetweenSquareGrids(source_shape, target_shape, 8, source_cell_index);
    ok &= ExpectTrue(source_cell_index == 24,
                     "Expected the bottom-right child cell to project to the bottom-right parent");
    return ok;
}

} // namespace

int main() {
    if (!TestPopulationFlooringRoundsDownToSquares()) {
        return 1;
    }

    if (!TestGridShapeAndIndexRoundTrips()) {
        return 1;
    }

    if (!TestToroidalWrappingWrapsBothDirections()) {
        return 1;
    }

    if (!TestRadiusTwoMooreNeighborhoodHasTwentyFourDistinctCandidatesOnFiveByFiveGrid()) {
        return 1;
    }

    if (!TestRadiusTwoMooreNeighborhoodDeduplicatesOnSmallToroidalGrids()) {
        return 1;
    }

    if (!TestGridProjectionMapsShrunkenChildCellsAcrossTheCurrentGrid()) {
        return 1;
    }

    std::cout << "PASS: spatial_grid_test\n";
    return 0;
}
