#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <string_view>

#include "model/input_encoder/encoder_spec.hpp"
#include "model/input_encoder/turn_features.hpp"
#include "wordle/turn.hpp"

namespace {

using neuroevolution::model::input_encoder::EncodeTurnFeatures;
using neuroevolution::model::input_encoder::FeedbackFeatureOffset;
using neuroevolution::model::input_encoder::GuessLetterFeatureOffset;
using neuroevolution::model::input_encoder::kTurnFeatureCount;
using neuroevolution::model::input_encoder::TurnInputVector;
using neuroevolution::model::input_encoder::detail::MaterializeTurnInputInPlace;
using neuroevolution::wordle::TileFeedback;
using neuroevolution::wordle::Turn;

bool ExpectTrue(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    return true;
}

bool ExpectDiscreteFeaturesSet(const auto &discrete, const std::initializer_list<std::size_t> expected_indices) {
    std::array<bool, kTurnFeatureCount> expected{};

    for (const std::size_t index : expected_indices) {
        expected[index] = true;
    }

    bool ok = true;

    for (std::size_t index = 0; index < kTurnFeatureCount; ++index) {
        const std::uint8_t expected_value = expected[index] ? 1u : 0u;
        if (discrete[index] != expected_value) {
            std::cerr << "FAIL: discrete feature mismatch at index " << index << ", expected "
                      << static_cast<int>(expected_value) << ", got " << static_cast<int>(discrete[index]) << '\n';
            ok = false;
        }
    }

    return ok;
}

bool ExpectMaterializedFeaturesSet(const neuroevolution::model::input_encoder::TurnInputVector &materialized,
                                   const std::initializer_list<std::size_t> expected_indices) {
    std::array<bool, kTurnFeatureCount> expected{};

    for (const std::size_t index : expected_indices) {
        expected[index] = true;
    }

    bool ok = true;

    for (std::size_t index = 0; index < kTurnFeatureCount; ++index) {
        const float expected_value = expected[index] ? 1.0f : 0.0f;
        if (materialized[index] != expected_value) {
            std::cerr << "FAIL: materialized feature mismatch at index " << index << ", expected " << expected_value
                      << ", got " << materialized[index] << '\n';
            ok = false;
        }
    }

    return ok;
}

bool TestEncodeTurnFeaturesGoldenCase() {
    const Turn turn{
        .letter_indices = {{0, 1, 2, 3, 4}},
        .feedback = {{
            TileFeedback::green,
            TileFeedback::yellow,
            TileFeedback::grey,
            TileFeedback::green,
            TileFeedback::yellow,
        }},
    };

    const auto features = EncodeTurnFeatures(turn);
    TurnInputVector materialized{};
    MaterializeTurnInputInPlace(features, materialized);

    constexpr std::array<std::size_t, 10> kExpectedActiveIndices{
        GuessLetterFeatureOffset(0, 0), GuessLetterFeatureOffset(1, 1), GuessLetterFeatureOffset(2, 2),
        GuessLetterFeatureOffset(3, 3), GuessLetterFeatureOffset(4, 4), FeedbackFeatureOffset(0, 0),
        FeedbackFeatureOffset(1, 1),    FeedbackFeatureOffset(2, 2),    FeedbackFeatureOffset(3, 0),
        FeedbackFeatureOffset(4, 1),
    };

    bool ok = true;

    ok &= ExpectTrue(kExpectedActiveIndices[0] == 0, "Expected A at position 0 to map to index 0");
    ok &= ExpectTrue(kExpectedActiveIndices[1] == 27, "Expected B at position 1 to map to index 27");
    ok &= ExpectTrue(kExpectedActiveIndices[2] == 54, "Expected C at position 2 to map to index 54");
    ok &= ExpectTrue(kExpectedActiveIndices[3] == 81, "Expected D at position 3 to map to index 81");
    ok &= ExpectTrue(kExpectedActiveIndices[4] == 108, "Expected E at position 4 to map to index 108");
    ok &= ExpectTrue(kExpectedActiveIndices[5] == 130, "Expected green at position 0 to map to index 130");
    ok &= ExpectTrue(kExpectedActiveIndices[6] == 134, "Expected yellow at position 1 to map to index 134");
    ok &= ExpectTrue(kExpectedActiveIndices[7] == 138, "Expected grey at position 2 to map to index 138");
    ok &= ExpectTrue(kExpectedActiveIndices[8] == 139, "Expected green at position 3 to map to index 139");
    ok &= ExpectTrue(kExpectedActiveIndices[9] == 143, "Expected yellow at position 4 to map to index 143");

    ok &= ExpectDiscreteFeaturesSet(features.discrete, {
                                                           kExpectedActiveIndices[0],
                                                           kExpectedActiveIndices[1],
                                                           kExpectedActiveIndices[2],
                                                           kExpectedActiveIndices[3],
                                                           kExpectedActiveIndices[4],
                                                           kExpectedActiveIndices[5],
                                                           kExpectedActiveIndices[6],
                                                           kExpectedActiveIndices[7],
                                                           kExpectedActiveIndices[8],
                                                           kExpectedActiveIndices[9],
                                                       });

    ok &= ExpectMaterializedFeaturesSet(materialized, {
                                                          kExpectedActiveIndices[0],
                                                          kExpectedActiveIndices[1],
                                                          kExpectedActiveIndices[2],
                                                          kExpectedActiveIndices[3],
                                                          kExpectedActiveIndices[4],
                                                          kExpectedActiveIndices[5],
                                                          kExpectedActiveIndices[6],
                                                          kExpectedActiveIndices[7],
                                                          kExpectedActiveIndices[8],
                                                          kExpectedActiveIndices[9],
                                                      });

    return ok;
}

} // namespace

int main() {
    if (!TestEncodeTurnFeaturesGoldenCase()) {
        return 1;
    }

    std::cout << "PASS: turn_features_test\n";
    return 0;
}
