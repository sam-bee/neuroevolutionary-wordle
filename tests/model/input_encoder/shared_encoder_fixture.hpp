#pragma once

#include <cstddef>
#include <stdexcept>

#include "model/input_encoder/encoder_spec.hpp"
#include "model/input_encoder/shared_encoder.hpp"
#include "model/input_encoder/turn_features.hpp"
#include "wordle/turn.hpp"

namespace neuroevolution::tests::input_encoder {

class SharedEncoderGoldenFixture {
  public:
    SharedEncoderGoldenFixture() {
        PopulateParameters(parameters);
        PopulateTurn(turn);
        PopulateExpectedOutput(expected_output);
    }

    model::input_encoder::TurnInputVector MaterializedInput() const {
        const auto features = model::input_encoder::EncodeTurnFeatures(turn);
        model::input_encoder::TurnInputVector materialized{};
        model::input_encoder::detail::MaterializeTurnInputInPlace(features, materialized);
        return materialized;
    }

    model::input_encoder::SharedEncoderParameters parameters{};
    wordle::Turn turn{};
    model::input_encoder::EncodedTurnVector expected_output{};

  private:
    static wordle::Word MakeWord(const char (&letters)[wordle::kWordLength + 1]) {
        wordle::Word word{};

        if (!wordle::TryMakeWordFromAscii(letters, word)) {
            throw std::invalid_argument("Word fixture literal must contain exactly five uppercase ASCII letters.");
        }

        return word;
    }

    static void PopulateParameters(model::input_encoder::SharedEncoderParameters &parameters) {
        using model::input_encoder::FeedbackFeatureOffset;
        using model::input_encoder::GuessLetterFeatureOffset;
        using model::input_encoder::kEncoderHiddenSize;
        using model::input_encoder::kTurnFeatureCount;

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

        parameters.input_to_hidden.weights[(0 * kTurnFeatureCount) + kGuessA0] = 1.5f;
        parameters.input_to_hidden.weights[(0 * kTurnFeatureCount) + kGuessB1] = 0.5f;
        parameters.input_to_hidden.weights[(0 * kTurnFeatureCount) + kGreen0] = 4.0f;

        parameters.input_to_hidden.weights[(1 * kTurnFeatureCount) + kGuessC2] = 2.0f;
        parameters.input_to_hidden.weights[(1 * kTurnFeatureCount) + kGrey2] = -5.0f;
        parameters.input_to_hidden.weights[(1 * kTurnFeatureCount) + kYellow4] = 0.5f;

        parameters.input_to_hidden.weights[(2 * kTurnFeatureCount) + kGuessD3] = 1.0f;
        parameters.input_to_hidden.weights[(2 * kTurnFeatureCount) + kGuessE4] = 2.0f;
        parameters.input_to_hidden.weights[(2 * kTurnFeatureCount) + kYellow1] = 0.25f;
        parameters.input_to_hidden.weights[(2 * kTurnFeatureCount) + kGreen3] = 0.75f;

        parameters.hidden_to_output.biases[0] = 0.5f;
        parameters.hidden_to_output.biases[1] = -2.0f;
        parameters.hidden_to_output.biases[2] = 1.25f;

        parameters.hidden_to_output.weights[(0 * kEncoderHiddenSize) + 0] = 2.0f;
        parameters.hidden_to_output.weights[(0 * kEncoderHiddenSize) + 2] = -1.0f;

        parameters.hidden_to_output.weights[(1 * kEncoderHiddenSize) + 0] = -0.5f;
        parameters.hidden_to_output.weights[(1 * kEncoderHiddenSize) + 2] = 4.0f;

        parameters.hidden_to_output.weights[(2 * kEncoderHiddenSize) + 1] = 7.0f;
    }

    static void PopulateTurn(wordle::Turn &turn) {
        turn = wordle::Turn{
            .guess = MakeWord("ABCDE"),
            .feedback = wordle::Feedback{{'G', 'Y', '-', 'G', 'Y'}},
        };
    }

    static void PopulateExpectedOutput(model::input_encoder::EncodedTurnVector &expected_output) {
        expected_output[0] = 5.5f;
        expected_output[1] = 8.0f;
        expected_output[2] = 1.25f;
    }
};

} // namespace neuroevolution::tests::input_encoder
