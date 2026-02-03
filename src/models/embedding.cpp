#include "models/embedding.hpp"

#include <random>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include "utils.hpp"
#include "ops.hpp"

namespace easy_llm {

extern std::mt19937 gen;
extern std::uniform_real_distribution<> dis;

Embedding::Embedding() {}

Embedding::Embedding(int vocab_size, int embedding_dim): vocab_size_(vocab_size), embedding_dim_(embedding_dim) {
    weights_ = Tensor(vocab_size * embedding_dim);
    for (int i = 0; i < weights_.size(); ++i) {
        weights_.at(i) = data_type(dis(gen));
    }
}

Embedding::Embedding(int vocab_size, int embedding_dim, std::vector<data_type>& weights) {
    vocab_size_ = vocab_size;
    embedding_dim_ = embedding_dim;
    weights_ = Tensor{weights, {vocab_size, embedding_dim}};
}

void Embedding::load_param(ModelParam& model_param) {
    weights_ = model_param.get_param("model.embed_tokens.weight");
    vocab_size_ = weights_.shape()[0];
    embedding_dim_ = weights_.shape()[1];
    spdlog::info("Embedding layer loaded: vocab_size = {}, embedding_dim = {}", vocab_size_, embedding_dim_);
}

Tensor Embedding::forward(const std::vector<int>& input) const {
    Tensor embed;
    embed.reserve(input.size() * embedding_dim_);
    for (int idx : input) {
        if (idx < 0 || idx >= vocab_size_) {
            throw std::out_of_range(
                "Embedding::forward: word id out of range. idx=" + std::to_string(idx) +
                ", vocab_size=" + std::to_string(vocab_size_));
        }
        for (int i = 0; i < embedding_dim_; ++i) {
            embed.emplace_back(weights_.at(idx * embedding_dim_ + i));
        }
    }
    std::vector<int> shape{static_cast<int>(input.size()), embedding_dim_};
    embed.reshape(shape);
    return embed;
}

Tensor Embedding::forward(const std::vector<std::vector<int>>& inputs) const {
    Tensor output;
    output.reserve(inputs.size() * inputs[0].size() * embedding_dim_);
    // spdlog::debug("weights_.size() == {}, weights_.shape() == {}", weights_.size(), fmt::join(weights_.shape(), ", "));
    // spdlog::debug("vocab_size_ == {}, embedding_dim_ == {}", vocab_size_, embedding_dim_);
    // spdlog::debug("inputs_batch_size == {}, inputs_seq_len == {}", inputs.size(), inputs[0].size());
    for (const auto& input : inputs) {
        for (const int& idx : input) {
            // spdlog::debug("idx == {}", idx);
            if (idx < 0 || idx >= vocab_size_) {
                throw std::out_of_range(
                    "Embedding::forward: word id out of range. idx=" + std::to_string(idx) +
                    ", vocab_size=" + std::to_string(vocab_size_));
            }
            for (int i = 0; i < embedding_dim_; ++i) {
                output.emplace_back(weights_.at(idx * embedding_dim_ + i));
            }
        }
    }
    std::vector<int> shape{static_cast<int>(inputs.size()), static_cast<int>(inputs[0].size()), embedding_dim_};
    output.reshape(shape);
    return output;
}

Tensor Embedding::forward(const Tensor& input) const {
    Tensor result = ops::matmul_3d(input, weights_);
    return result;
}

} // namespace easy_llm
