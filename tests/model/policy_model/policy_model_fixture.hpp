#pragma once

#include <stdexcept>

#include "../input_encoder/shared_encoder_fixture.hpp"
#include "model/model_input/model_input_spec.hpp"
#include "model/policy_model/policy_model.hpp"
#include "wordle/word.hpp"
#include "wordle/wordle_grid.hpp"

namespace neuroevolution::tests::policy_model {

class PolicyModelGoldenFixture {
  public:
    PolicyModelGoldenFixture() : solution(MakeWord("AEBDF")), guess(MakeWord("ABCDE")) {
        parameters.input_encoder = shared_encoder_fixture.parameters;
        PopulateDenseTrunkParameters(parameters.dense_trunk);
        PopulateExpectedSingleTurnOutput(shared_encoder_fixture.expected_output, expected_single_turn_output);
        PopulateExpectedVirginOutput(expected_virgin_output);
    }

    wordle::WordleGrid MakeSingleTurnGrid() const {
        wordle::WordleGrid grid = wordle::MakeWordleGrid(solution);
        if (!wordle::TryAppendGuess(grid, guess)) {
            throw std::invalid_argument("Policy-model fixture failed to append its single test guess.");
        }

        return grid;
    }

    model::policy_model::PolicyModelParameters parameters{};
    wordle::Word solution{};
    wordle::Word guess{};
    model::policy_model::PolicyVector expected_single_turn_output{};
    model::policy_model::PolicyVector expected_virgin_output{};

  private:
    static wordle::Word MakeWord(const char (&letters)[wordle::kWordLength + 1]) {
        wordle::Word word{};

        if (!wordle::TryMakeWordFromAscii(letters, word)) {
            throw std::invalid_argument(
                "Policy-model fixture word literal must contain exactly five uppercase letters.");
        }

        return word;
    }

    static void PopulateDenseTrunkParameters(model::dense_trunk::DenseTrunkParameters &parameters) {
        constexpr std::size_t kVirginFlag = model::model_input::kModelInputVirginFlagOffset;
        constexpr std::size_t kTurn0Value0 = model::model_input::ModelInputTurnOffset(0) + 0;
        constexpr std::size_t kTurn0Value1 = model::model_input::ModelInputTurnOffset(0) + 1;
        constexpr std::size_t kTurn0Value2 = model::model_input::ModelInputTurnOffset(0) + 2;

        parameters.input_to_hidden0.biases[0] = -1.0f;
        parameters.input_to_hidden0.weights[(0 * model::dense_trunk::kDenseTrunkInputSize) + kTurn0Value0] = 1.0f;
        parameters.input_to_hidden0.weights[(0 * model::dense_trunk::kDenseTrunkInputSize) + kTurn0Value1] = 0.5f;

        parameters.input_to_hidden0.biases[1] = 0.0f;
        parameters.input_to_hidden0.weights[(1 * model::dense_trunk::kDenseTrunkInputSize) + kTurn0Value1] = -0.25f;
        parameters.input_to_hidden0.weights[(1 * model::dense_trunk::kDenseTrunkInputSize) + kTurn0Value2] = 2.0f;

        parameters.input_to_hidden0.biases[2] = -1.5f;
        parameters.input_to_hidden0.weights[(2 * model::dense_trunk::kDenseTrunkInputSize) + kVirginFlag] = 2.0f;

        parameters.hidden0_to_hidden1.biases[0] = 0.25f;
        parameters.hidden0_to_hidden1.weights[(0 * model::dense_trunk::kDenseTrunkHiddenSize0) + 0] = 0.5f;
        parameters.hidden0_to_hidden1.weights[(0 * model::dense_trunk::kDenseTrunkHiddenSize0) + 1] = 2.0f;

        parameters.hidden0_to_hidden1.biases[1] = -1.0f;
        parameters.hidden0_to_hidden1.weights[(1 * model::dense_trunk::kDenseTrunkHiddenSize0) + 0] = 0.25f;
        parameters.hidden0_to_hidden1.weights[(1 * model::dense_trunk::kDenseTrunkHiddenSize0) + 2] = 4.0f;

        parameters.hidden1_to_output.biases[0] = -2.0f;
        parameters.hidden1_to_output.weights[(0 * model::dense_trunk::kDenseTrunkHiddenSize1) + 0] = 2.0f;

        parameters.hidden1_to_output.biases[1] = 0.5f;
        parameters.hidden1_to_output.weights[(1 * model::dense_trunk::kDenseTrunkHiddenSize1) + 0] = -0.5f;
        parameters.hidden1_to_output.weights[(1 * model::dense_trunk::kDenseTrunkHiddenSize1) + 1] = 4.0f;

        parameters.hidden1_to_output.biases[2] = 1.0f;
        parameters.hidden1_to_output.weights[(2 * model::dense_trunk::kDenseTrunkHiddenSize1) + 1] = 3.0f;
    }

    static void PopulateExpectedSingleTurnOutput(const model::input_encoder::EncodedTurnVector &encoded_turn,
                                                 model::policy_model::PolicyVector &expected_output) {
        const float turn_value0 = encoded_turn[0];
        const float turn_value1 = encoded_turn[1];
        const float turn_value2 = encoded_turn[2];

        const float hidden0_0 = -1.0f + turn_value0 + (0.5f * turn_value1);
        const float hidden0_1 = (-0.25f * turn_value1) + (2.0f * turn_value2);
        const float hidden0_2 = 0.0f;

        const float hidden1_0 = 0.25f + (0.5f * hidden0_0) + (2.0f * hidden0_1);
        const float hidden1_1 = -1.0f + (0.25f * hidden0_0) + (4.0f * hidden0_2);

        expected_output[0] = -2.0f + (2.0f * hidden1_0);
        expected_output[1] = 0.5f + (-0.5f * hidden1_0) + (4.0f * hidden1_1);
        expected_output[2] = 1.0f + (3.0f * hidden1_1);
    }

    static void PopulateExpectedVirginOutput(model::policy_model::PolicyVector &expected_output) {
        expected_output[0] = -1.5f;
        expected_output[1] = 4.375f;
        expected_output[2] = 4.0f;
    }

    neuroevolution::tests::input_encoder::SharedEncoderGoldenFixture shared_encoder_fixture{};
};

} // namespace neuroevolution::tests::policy_model
