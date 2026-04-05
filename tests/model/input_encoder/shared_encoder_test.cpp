#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>

#include "model/input_encoder/encoder_spec.hpp"
#include "model/input_encoder/shared_encoder.hpp"
#include "model/input_encoder/turn_features.hpp"
#include "wordle/turn.hpp"

namespace {

using neuroevolution::model::input_encoder::EncodedTurnVector;
using neuroevolution::model::input_encoder::FeedbackFeatureOffset;
using neuroevolution::model::input_encoder::ForwardOccupiedTurn;
using neuroevolution::model::input_encoder::ForwardSharedEncoder;
using neuroevolution::model::input_encoder::GuessLetterFeatureOffset;
using neuroevolution::model::input_encoder::MaterializeTurnInput;
using neuroevolution::model::input_encoder::SharedEncoderParameters;
using neuroevolution::model::input_encoder::kEncoderOutputSize;
using neuroevolution::wordle::TileFeedback;
using neuroevolution::wordle::Turn;

constexpr float kTolerance = 1.0e-6f;

bool ExpectTrue(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    return false;
  }

  return true;
}

bool ExpectVectorNear(const EncodedTurnVector& actual,
                      const EncodedTurnVector& expected,
                      const std::string_view label) {
  bool ok = true;

  for (std::size_t index = 0; index < kEncoderOutputSize; ++index) {
    const float delta = std::fabs(actual[index] - expected[index]);
    if (delta > kTolerance) {
      std::cerr << "FAIL: " << label << " mismatch at index " << index
                << ", expected " << expected[index]
                << ", got " << actual[index] << '\n';
      ok = false;
    }
  }

  return ok;
}

bool TestSharedEncoderForwardPassGoldenCase() {
  SharedEncoderParameters parameters{};

  constexpr std::size_t kGuessA0 = GuessLetterFeatureOffset(0, 0);
  constexpr std::size_t kGuessB1 = GuessLetterFeatureOffset(1, 1);
  constexpr std::size_t kGuessC2 = GuessLetterFeatureOffset(2, 2);
  constexpr std::size_t kGuessD3 = GuessLetterFeatureOffset(3, 3);
  constexpr std::size_t kGuessE4 = GuessLetterFeatureOffset(4, 4);
  constexpr std::size_t kGreen0 = FeedbackFeatureOffset(0, 0);
  constexpr std::size_t kYellow1 = FeedbackFeatureOffset(1, 1);
  constexpr std::size_t kGrey2 = FeedbackFeatureOffset(2, 2);
  constexpr std::size_t kGreen3 = FeedbackFeatureOffset(3, 0);
  constexpr std::size_t kYellow4 = FeedbackFeatureOffset(4, 1);

  parameters.input_to_hidden.biases[0] = -2.0f;
  parameters.input_to_hidden.biases[1] = 1.0f;
  parameters.input_to_hidden.biases[2] = -1.0f;

  parameters.input_to_hidden.weights[(0 * neuroevolution::model::input_encoder::kTurnFeatureCount) + kGuessA0] = 1.5f;
  parameters.input_to_hidden.weights[(0 * neuroevolution::model::input_encoder::kTurnFeatureCount) + kGuessB1] = 0.5f;
  parameters.input_to_hidden.weights[(0 * neuroevolution::model::input_encoder::kTurnFeatureCount) + kGreen0] = 4.0f;

  parameters.input_to_hidden.weights[(1 * neuroevolution::model::input_encoder::kTurnFeatureCount) + kGuessC2] = 2.0f;
  parameters.input_to_hidden.weights[(1 * neuroevolution::model::input_encoder::kTurnFeatureCount) + kGrey2] = -5.0f;
  parameters.input_to_hidden.weights[(1 * neuroevolution::model::input_encoder::kTurnFeatureCount) + kYellow4] = 0.5f;

  parameters.input_to_hidden.weights[(2 * neuroevolution::model::input_encoder::kTurnFeatureCount) + kGuessD3] = 1.0f;
  parameters.input_to_hidden.weights[(2 * neuroevolution::model::input_encoder::kTurnFeatureCount) + kGuessE4] = 2.0f;
  parameters.input_to_hidden.weights[(2 * neuroevolution::model::input_encoder::kTurnFeatureCount) + kYellow1] = 0.25f;
  parameters.input_to_hidden.weights[(2 * neuroevolution::model::input_encoder::kTurnFeatureCount) + kGreen3] = 0.75f;

  parameters.hidden_to_output.biases[0] = 0.5f;
  parameters.hidden_to_output.biases[1] = -2.0f;
  parameters.hidden_to_output.biases[2] = 1.25f;

  parameters.hidden_to_output.weights[(0 * neuroevolution::model::input_encoder::kEncoderHiddenSize) + 0] = 2.0f;
  parameters.hidden_to_output.weights[(0 * neuroevolution::model::input_encoder::kEncoderHiddenSize) + 2] = -1.0f;

  parameters.hidden_to_output.weights[(1 * neuroevolution::model::input_encoder::kEncoderHiddenSize) + 0] = -0.5f;
  parameters.hidden_to_output.weights[(1 * neuroevolution::model::input_encoder::kEncoderHiddenSize) + 2] = 4.0f;

  parameters.hidden_to_output.weights[(2 * neuroevolution::model::input_encoder::kEncoderHiddenSize) + 1] = 7.0f;

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

  const auto encoded_features = neuroevolution::model::input_encoder::EncodeTurnFeatures(turn);
  const auto input_vector = MaterializeTurnInput(encoded_features);

  const EncodedTurnVector direct_output = ForwardSharedEncoder(parameters, input_vector);
  const EncodedTurnVector occupied_turn_output = ForwardOccupiedTurn(parameters, turn);

  EncodedTurnVector expected{};
  expected[0] = 5.5f;
  expected[1] = 8.0f;
  expected[2] = 1.25f;

  bool ok = true;
  ok &= ExpectTrue(direct_output[0] == 5.5f, "Expected output neuron 0 to equal 5.5");
  ok &= ExpectTrue(direct_output[1] == 8.0f, "Expected output neuron 1 to equal 8.0");
  ok &= ExpectTrue(direct_output[2] == 1.25f, "Expected output neuron 2 to equal 1.25");
  ok &= ExpectVectorNear(direct_output, expected, "ForwardSharedEncoder");
  ok &= ExpectVectorNear(occupied_turn_output, expected, "ForwardOccupiedTurn");

  return ok;
}

}  // namespace

int main() {
  if (!TestSharedEncoderForwardPassGoldenCase()) {
    return 1;
  }

  std::cout << "PASS: shared_encoder_test\n";
  return 0;
}
