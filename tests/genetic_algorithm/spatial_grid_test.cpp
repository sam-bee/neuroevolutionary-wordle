#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "genetic_algorithm/spatial/grid.hpp"

namespace {

using neuroevolution::genetic_algorithm::spatial::CellularGridShape;
using neuroevolution::genetic_algorithm::spatial::CellularNeighborList;
using neuroevolution::genetic_algorithm::spatial::ContainsNeighborIndex;
using neuroevolution::genetic_algorithm::spatial::FloorRowPreservingPopulationSize;
using neuroevolution::genetic_algorithm::spatial::FloorSquarePopulationSize;
using neuroevolution::genetic_algorithm::spatial::GridColumnFromIndex;
using neuroevolution::genetic_algorithm::spatial::GridIndexFromRowColumn;
using neuroevolution::genetic_algorithm::spatial::GridRowFromIndex;
using neuroevolution::genetic_algorithm::spatial::IsValidCellularGridShape;
using neuroevolution::genetic_algorithm::spatial::TryCollectCellularSecondParentCandidates;
using neuroevolution::genetic_algorithm::spatial::TryMakeCellularGridShape;
using neuroevolution::genetic_algorithm::spatial::TryMakeRectangularCellularGridShape;
using neuroevolution::genetic_algorithm::spatial::WrapToroidalCoordinate;

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

bool TestPopulationFlooringRoundsDownToSquaresForStartup() {
    bool ok = true;
    ok &= ExpectTrue(FloorSquarePopulationSize(1) == 1, "Expected 1 to remain 1");
    ok &= ExpectTrue(FloorSquarePopulationSize(2) == 1, "Expected 2 to floor to 1");
    ok &= ExpectTrue(FloorSquarePopulationSize(15) == 9, "Expected 15 to floor to 9");
    ok &= ExpectTrue(FloorSquarePopulationSize(16) == 16, "Expected 16 to remain 16");
    ok &= ExpectTrue(FloorSquarePopulationSize(17) == 16, "Expected 17 to floor to 16");
    ok &= ExpectTrue(FloorSquarePopulationSize(26) == 25, "Expected 26 to floor to 25");
    return ok;
}

bool TestPopulationFlooringRoundsDownToWholeRows() {
    bool ok = true;
    ok &= ExpectTrue(FloorRowPreservingPopulationSize(24, 5) == 20, "Expected 24 cells to floor to four rows of five");
    ok &= ExpectTrue(FloorRowPreservingPopulationSize(25, 5) == 25, "Expected 25 cells to remain five rows of five");
    ok &= ExpectTrue(FloorRowPreservingPopulationSize(4, 5) == 0, "Expected too-small capacity to fit no full row");
    return ok;
}

bool TestGridShapeAndIndexRoundTrips() {
    CellularGridShape shape{};
    bool ok = true;
    ok &= TryMakeRectangularCellularGridShape(3, 5, shape);
    ok &= ExpectTrue(IsValidCellularGridShape(shape), "Expected 3x5 shape to be valid");
    ok &= ExpectTrue(shape.row_count == 3, "Expected rectangular shape to preserve row count");
    ok &= ExpectTrue(shape.column_count == 5, "Expected rectangular shape to preserve column count");
    ok &= ExpectTrue(shape.cell_count == 15, "Expected rectangular shape to compute cell count");
    ok &= ExpectTrue(GridIndexFromRowColumn(shape, 2, 3) == 13, "Expected row/column to map to linear index");
    ok &= ExpectTrue(GridRowFromIndex(shape, 13) == 2, "Expected linear index to map back to row");
    ok &= ExpectTrue(GridColumnFromIndex(shape, 13) == 3, "Expected linear index to map back to column");

    CellularGridShape square_shape{};
    ok &= TryMakeCellularGridShape(16, square_shape);
    ok &= ExpectTrue(square_shape.row_count == 4, "Expected square helper to derive row count");
    ok &= ExpectTrue(square_shape.column_count == 4, "Expected square helper to derive column count");
    return ok;
}

bool TestToroidalWrappingWrapsBothDirections() {
    bool ok = true;
    ok &= ExpectTrue(WrapToroidalCoordinate(-1, 5) == 4, "Expected -1 to wrap to the final coordinate");
    ok &= ExpectTrue(WrapToroidalCoordinate(5, 5) == 0, "Expected side length to wrap back to zero");
    ok &= ExpectTrue(WrapToroidalCoordinate(6, 5) == 1, "Expected values above side length to wrap");
    return ok;
}

bool TestRadiusTwoMooreNeighborhoodHasTwentyFiveDistinctCandidatesOnFiveByFiveGrid() {
    CellularGridShape shape{};
    CellularNeighborList neighbors{};
    bool ok = true;
    ok &= TryMakeCellularGridShape(25, shape);
    ok &= TryCollectCellularSecondParentCandidates(shape, 12, neighbors);
    ok &= ExpectTrue(neighbors.count == 25, "Expected 5x5 radius-two Moore neighborhood to have 25 candidates");
    ok &= ExpectTrue(ContainsNeighborIndex(neighbors, 12), "Expected focal cell to be included in candidates");

    for (std::size_t cell_index = 0; cell_index < shape.cell_count; ++cell_index) {
        ok &= ExpectTrue(ContainsNeighborIndex(neighbors, cell_index),
                         "Expected every 5x5 cell to appear in the radius-two neighborhood");
    }

    return ok;
}

bool TestRadiusTwoMooreNeighborhoodCanReachDeletedRowParents() {
    CellularGridShape parent_shape{};
    CellularNeighborList neighbors{};
    bool ok = true;
    ok &= TryMakeRectangularCellularGridShape(5, 5, parent_shape);
    ok &= TryCollectCellularSecondParentCandidates(parent_shape, GridIndexFromRowColumn(parent_shape, 3, 2), neighbors);
    ok &= ExpectTrue(ContainsNeighborIndex(neighbors, GridIndexFromRowColumn(parent_shape, 4, 2)),
                     "Expected bottom-row parents to stay eligible from the last surviving child row");
    return ok;
}

bool TestRadiusTwoMooreNeighborhoodDeduplicatesOnSmallToroidalGrids() {
    CellularGridShape shape{};
    CellularNeighborList neighbors{};
    bool ok = true;
    ok &= TryMakeCellularGridShape(9, shape);
    ok &= TryCollectCellularSecondParentCandidates(shape, 4, neighbors);
    ok &= ExpectTrue(neighbors.count == 9,
                     "Expected a 3x3 toroidal grid to deduplicate radius-two candidates down to all 9 cells");
    ok &= ExpectTrue(ContainsNeighborIndex(neighbors, 4), "Expected focal cell inclusion to survive deduplication");
    return ok;
}

} // namespace

int main() {
    if (!TestPopulationFlooringRoundsDownToSquaresForStartup()) {
        return 1;
    }

    if (!TestPopulationFlooringRoundsDownToWholeRows()) {
        return 1;
    }

    if (!TestGridShapeAndIndexRoundTrips()) {
        return 1;
    }

    if (!TestToroidalWrappingWrapsBothDirections()) {
        return 1;
    }

    if (!TestRadiusTwoMooreNeighborhoodHasTwentyFiveDistinctCandidatesOnFiveByFiveGrid()) {
        return 1;
    }

    if (!TestRadiusTwoMooreNeighborhoodCanReachDeletedRowParents()) {
        return 1;
    }

    if (!TestRadiusTwoMooreNeighborhoodDeduplicatesOnSmallToroidalGrids()) {
        return 1;
    }

    std::cout << "PASS: spatial_grid_test\n";
    return 0;
}
