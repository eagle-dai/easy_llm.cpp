#ifndef EASY_LLM_MODELS_EMBEDDING_HPP
#define EASY_LLM_MODELS_EMBEDDING_HPP

#include <vector>
#include <memory>
#include <string>

#include "tensor.hpp"
#include "models/loader.hpp"

namespace easy_llm {

class Embedding {
public:
    Embedding();
    Embedding(int vocab_size, int embedding_dim);
    Embedding(int vocab_size, int embedding_dim, std::vector<data_type>& weights);

    Tensor forward(const std::vector<int>& input) const;
    Tensor forward(const std::vector<std::vector<int>>& input) const;
    Tensor forward(const Tensor& input) const;
    void load_param(ModelParam& model_param);

private:
    void apply_rope(Tensor& word_embed);

    Tensor weights_;
    int vocab_size_;
    int embedding_dim_;
    const float rope_theta_{10000.0f};
};

} // namespace easy_llm

#endif // EASY_LLM_MODELS_EMBEDDING_HPP
