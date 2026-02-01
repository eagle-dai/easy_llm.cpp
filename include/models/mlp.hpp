#ifndef EASY_GPT_MODELS_MLP_HPP
#define EASY_GPT_MODELS_MLP_HPP

#include <vector>
#include <memory>
#include <string>

#include "tensor.hpp"
#include "models/loader.hpp"
#include "models/linear.hpp"
#include "models/norm.hpp"

namespace easy_gpt {

class MLP {
public:
    MLP();
    MLP(int hidden_dim);

    Tensor forward(const Tensor& input) const;
    void load_param(const std::string& key, ModelParam& model_param);

private:
    Linear up_proj_;
    Linear gate_proj_;
    Linear down_proj_;
    int hidden_dim_;
    RMSNorm norm_{};
};

} // namespace easy_gpt

#endif // EASY_GPT_MODELS_MLP_HPP
