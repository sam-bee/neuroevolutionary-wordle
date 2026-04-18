#pragma once

#include <cstddef>
#include <cstdint>
#include "common/cuda_compat.hpp"
#include "common/fixed_buffer.hpp"

namespace neuroevolution::genetic_algorithm::spatial {

constexpr std::size_t kCellularBreedingRadius = 2;
constexpr std::size_t kMaxCellularSecondParentCandidateCount =
    ((2 * kCellularBreedingRadius) + 1) * ((2 * kCellularBreedingRadius) + 1) - 1;
constexpr float kPositiveSelectionFitnessFloor = 1.40129846e-45f;
constexpr std::size_t kMaxSizeT = static_cast<std::size_t>(-1);

struct CellularGridShape {
    std::size_t side_length = 0;
    std::size_t cell_count = 0;
};

struct CellularNeighborList {
    common::FixedBuffer<std::size_t, kMaxCellularSecondParentCandidateCount> indices{};
    std::size_t count = 0;
};

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidCellularGridShape(const CellularGridShape &shape) noexcept {
    return (shape.side_length > 0) && (shape.cell_count > 0) &&
           (shape.side_length <= (kMaxSizeT / shape.side_length)) &&
           ((shape.side_length * shape.side_length) == shape.cell_count);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t FloorSquareRoot(const std::size_t value) noexcept {
    std::size_t low = 0;
    std::size_t high = (value < 3037000499ULL) ? value : 3037000499ULL;
    std::size_t best = 0;

    while (low <= high) {
        const std::size_t mid = low + ((high - low) / 2);
        if ((mid != 0) && (mid > (value / mid))) {
            if (mid == 0) {
                break;
            }
            high = mid - 1;
            continue;
        }

        best = mid;
        low = mid + 1;
    }

    return best;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t FloorSquarePopulationSize(const std::size_t nominal_population_size)
    noexcept {
    const std::size_t side_length = FloorSquareRoot(nominal_population_size);
    return side_length * side_length;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool TryMakeCellularGridShape(const std::size_t cell_count,
                                                                   CellularGridShape &shape_out) noexcept {
    shape_out = {};
    if (cell_count == 0) {
        return false;
    }

    const std::size_t side_length = FloorSquareRoot(cell_count);
    if ((side_length == 0) || (side_length > (kMaxSizeT / side_length)) || ((side_length * side_length) != cell_count)) {
        return false;
    }

    shape_out.side_length = side_length;
    shape_out.cell_count = cell_count;
    return true;
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t GridRowFromIndex(const CellularGridShape &shape,
                                                                  const std::size_t cell_index) noexcept {
    return (!IsValidCellularGridShape(shape) || (shape.side_length == 0)) ? 0 : (cell_index / shape.side_length);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t GridColumnFromIndex(const CellularGridShape &shape,
                                                                     const std::size_t cell_index) noexcept {
    return (!IsValidCellularGridShape(shape) || (shape.side_length == 0)) ? 0 : (cell_index % shape.side_length);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t GridIndexFromRowColumn(const CellularGridShape &shape,
                                                                        const std::size_t row,
                                                                        const std::size_t column) noexcept {
    return (!IsValidCellularGridShape(shape) || (row >= shape.side_length) || (column >= shape.side_length))
               ? 0
               : ((row * shape.side_length) + column);
}

constexpr NEUROEVOLUTION_HOST_DEVICE std::size_t WrapToroidalCoordinate(const std::ptrdiff_t coordinate,
                                                                        const std::size_t side_length) noexcept {
    if (side_length == 0) {
        return 0;
    }

    const std::ptrdiff_t modulus = static_cast<std::ptrdiff_t>(side_length);
    std::ptrdiff_t wrapped = coordinate % modulus;
    if (wrapped < 0) {
        wrapped += modulus;
    }

    return static_cast<std::size_t>(wrapped);
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool ContainsNeighborIndex(const CellularNeighborList &neighbors,
                                                                const std::size_t candidate_index) noexcept {
    for (std::size_t neighbor_offset = 0; neighbor_offset < neighbors.count; ++neighbor_offset) {
        if (neighbors.indices[neighbor_offset] == candidate_index) {
            return true;
        }
    }

    return false;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool TryCollectMooreRadiusNeighbors(const CellularGridShape &shape,
                                                                         const std::size_t focal_cell_index,
                                                                         const std::size_t radius,
                                                                         CellularNeighborList &neighbors_out) noexcept {
    neighbors_out = {};
    if (!IsValidCellularGridShape(shape) || (focal_cell_index >= shape.cell_count) ||
        (radius > shape.side_length + shape.side_length)) {
        return false;
    }

    const std::size_t focal_row = GridRowFromIndex(shape, focal_cell_index);
    const std::size_t focal_column = GridColumnFromIndex(shape, focal_cell_index);

    for (std::ptrdiff_t row_offset = -static_cast<std::ptrdiff_t>(radius);
         row_offset <= static_cast<std::ptrdiff_t>(radius); ++row_offset) {
        for (std::ptrdiff_t column_offset = -static_cast<std::ptrdiff_t>(radius);
             column_offset <= static_cast<std::ptrdiff_t>(radius); ++column_offset) {
            if ((row_offset == 0) && (column_offset == 0)) {
                continue;
            }

            const std::size_t neighbor_row =
                WrapToroidalCoordinate(static_cast<std::ptrdiff_t>(focal_row) + row_offset, shape.side_length);
            const std::size_t neighbor_column =
                WrapToroidalCoordinate(static_cast<std::ptrdiff_t>(focal_column) + column_offset, shape.side_length);
            const std::size_t neighbor_index = GridIndexFromRowColumn(shape, neighbor_row, neighbor_column);

            if ((neighbor_index == focal_cell_index) || ContainsNeighborIndex(neighbors_out, neighbor_index)) {
                continue;
            }

            if (neighbors_out.count >= kMaxCellularSecondParentCandidateCount) {
                return false;
            }

            neighbors_out.indices[neighbors_out.count] = neighbor_index;
            ++neighbors_out.count;
        }
    }

    return neighbors_out.count > 0;
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool
TryCollectCellularSecondParentCandidates(const CellularGridShape &shape, const std::size_t focal_cell_index,
                                         CellularNeighborList &neighbors_out) noexcept {
    return TryCollectMooreRadiusNeighbors(shape, focal_cell_index, kCellularBreedingRadius, neighbors_out);
}

constexpr NEUROEVOLUTION_HOST_DEVICE bool
TryProjectCellIndexBetweenSquareGrids(const CellularGridShape &source_shape, const CellularGridShape &target_shape,
                                      const std::size_t target_cell_index, std::size_t &source_cell_index_out) noexcept {
    source_cell_index_out = 0;
    if (!IsValidCellularGridShape(source_shape) || !IsValidCellularGridShape(target_shape) ||
        (target_cell_index >= target_shape.cell_count)) {
        return false;
    }

    const std::size_t target_row = GridRowFromIndex(target_shape, target_cell_index);
    const std::size_t target_column = GridColumnFromIndex(target_shape, target_cell_index);
    const std::size_t source_row = (target_shape.side_length == 1)
                                       ? (source_shape.side_length / 2)
                                       : ((target_row * (source_shape.side_length - 1)) /
                                          (target_shape.side_length - 1));
    const std::size_t source_column = (target_shape.side_length == 1)
                                          ? (source_shape.side_length / 2)
                                          : ((target_column * (source_shape.side_length - 1)) /
                                             (target_shape.side_length - 1));
    source_cell_index_out = GridIndexFromRowColumn(source_shape, source_row, source_column);
    return source_cell_index_out < source_shape.cell_count;
}

} // namespace neuroevolution::genetic_algorithm::spatial
