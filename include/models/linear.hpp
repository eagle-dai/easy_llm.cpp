#ifndef EASY_LLM_MODELS_LINEAR_HPP
#define EASY_LLM_MODELS_LINEAR_HPP

#include <vector>
#include <memory>
#include <string>

#include "tensor.hpp"
#include "models/loader.hpp"

namespace easy_llm {

class Linear {
public:
    Linear();
    Linear(int out_dim, int in_dim);

    Tensor forward(const Tensor& input) const;
    void load_param(const std::string& key, ModelParam& model_param);
    int get_out_dim() const { return out_dim_; }
    int get_in_dim() const { return in_dim_; }
    const Tensor& weights() const { return weights_; }
    const Tensor& bias() const { return bias_; }

private:
    Tensor weights_;
    Tensor bias_;
    int out_dim_;
    int in_dim_;
};

} // namespace easy_llm

#endif // EASY_LLM_MODELS_LINEAR_HPP
