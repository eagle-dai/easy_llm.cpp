#include "models/linear.hpp"

#include <random>
#include <spdlog/spdlog.h>
#include "utils.hpp"
#include "ops.hpp"

namespace easy_llm {

extern std::mt19937 gen;
extern std::uniform_real_distribution<> dis;

Linear::Linear() {}

Linear::Linear(int out_dim, int in_dim): out_dim_(out_dim), in_dim_(in_dim) {
    weights_.resize(out_dim_ * in_dim_);
    bias_.resize(in_dim_);
    for (int i = 0; i < weights_.size(); ++i) {
        weights_.at(i) = data_type(dis(gen));
    }
    for (int i = 0; i < bias_.size(); ++i) {
        bias_.at(i) = data_type(0.0f);
    }
}

void Linear::load_param(const std::string& key, ModelParam& model_param) {
    weights_ = model_param.take_param(key + ".weight");
    out_dim_ = weights_.shape()[0];
    in_dim_ = weights_.shape()[1];
    if (model_param.contains(key + ".bias")) {
        bias_ = model_param.take_param(key + ".bias");
        bias_.reshape({bias_.size()});  // ori shape is {bias_.size(), 1}; due to GQA, bias_.size() might not equal in_dim_
    } else {
        bias_.resize(out_dim_);
        bias_.reshape({out_dim_});
        for (int i = 0; i < bias_.size(); ++i) {
            bias_.at(i) = data_type(0.0f);
        }
    }
}

Tensor Linear::forward(const Tensor& input) const {
    return ops::matmul_3d(input, weights_, bias_);
}

} // namespace easy_llm
