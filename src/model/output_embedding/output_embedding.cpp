#include "model/output_embedding/output_embedding.hpp"

#include <stdexcept>

namespace neuroevolution::model::output_embedding {

wordle::Word SelectBestActionWord(const PolicyVector &policy_vector, const ActionEmbedding *action_embeddings,
                                  const std::size_t action_count) {
    SelectedAction selected_action{};
    if (!TrySelectBestAction(policy_vector, action_embeddings, action_count, selected_action)) {
        throw std::invalid_argument("Output embedding selection requires a non-empty table of valid action words.");
    }

    return selected_action.word;
}

} // namespace neuroevolution::model::output_embedding
