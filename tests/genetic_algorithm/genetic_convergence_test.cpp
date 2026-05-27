#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

#include "genetic_algorithm/genetic_convergence.hpp"

namespace {

using neuroevolution::genetic_algorithm::genetic_convergence::GeneticConvergenceMetrics;
using neuroevolution::genetic_algorithm::genetic_convergence::GeneticConvergencePair;
using neuroevolution::genetic_algorithm::genetic_convergence::TryComputeMetrics;

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

bool TestIdenticalSyntheticGenotypesHaveZeroDistance() {
    const std::vector<float> sampled_values{
        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
    };
    const std::vector<GeneticConvergencePair> pairs{
        {.first = 0, .second = 1},
        {.first = 1, .second = 2},
        {.first = 2, .second = 0},
    };

    GeneticConvergenceMetrics metrics{};
    bool ok = true;
    ok &= ExpectTrue(TryComputeMetrics(sampled_values, 3, 4, pairs, metrics),
                     "Expected identical synthetic genotypes to produce metrics");
    ok &= ExpectNear(metrics.centroid_distance_mean, 0.0f, "identical centroid mean distance");
    ok &= ExpectNear(metrics.centroid_distance_min, 0.0f, "identical centroid min distance");
    ok &= ExpectNear(metrics.centroid_distance_max, 0.0f, "identical centroid max distance");
    ok &= ExpectNear(metrics.pairwise_distance_mean, 0.0f, "identical pairwise mean distance");
    ok &= ExpectNear(metrics.pairwise_distance_min, 0.0f, "identical pairwise min distance");
    ok &= ExpectNear(metrics.pairwise_distance_max, 0.0f, "identical pairwise max distance");
    return ok;
}

bool TestDifferentSyntheticGenotypesHavePositiveDistance() {
    const std::vector<float> sampled_values{
        0.0f, 0.0f, 0.0f, 3.0f, 4.0f, 0.0f,
    };
    const std::vector<GeneticConvergencePair> pairs{
        {.first = 0, .second = 1},
        {.first = 1, .second = 0},
    };

    GeneticConvergenceMetrics metrics{};
    bool ok = true;
    ok &= ExpectTrue(TryComputeMetrics(sampled_values, 2, 3, pairs, metrics),
                     "Expected different synthetic genotypes to produce metrics");
    ok &= ExpectNear(metrics.centroid_distance_mean, 2.5f, "different centroid mean distance");
    ok &= ExpectNear(metrics.centroid_distance_min, 2.5f, "different centroid min distance");
    ok &= ExpectNear(metrics.centroid_distance_max, 2.5f, "different centroid max distance");
    ok &= ExpectNear(metrics.pairwise_distance_mean, 5.0f, "different pairwise mean distance");
    ok &= ExpectNear(metrics.pairwise_distance_min, 5.0f, "different pairwise min distance");
    ok &= ExpectNear(metrics.pairwise_distance_max, 5.0f, "different pairwise max distance");
    ok &= ExpectTrue(metrics.centroid_distance_mean > 0.0f,
                     "Expected different genotypes to have positive centroid distance");
    ok &= ExpectTrue(metrics.pairwise_distance_mean > 0.0f,
                     "Expected different genotypes to have positive pairwise distance");
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= TestIdenticalSyntheticGenotypesHaveZeroDistance();
    ok &= TestDifferentSyntheticGenotypesHavePositiveDistance();
    return ok ? 0 : 1;
}
