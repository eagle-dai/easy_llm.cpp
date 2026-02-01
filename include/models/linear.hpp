#ifndef EASY_GPT_MODELS_LINEAR_HPP
#define EASY_GPT_MODELS_LINEAR_HPP

#include <vector>
#include <memory>
#include <string>

#include "tensor.hpp"
#include "models/loader.hpp"

namespace easy_gpt {

class Linear {
public:
    Linear();
    Linear(int out_dim, int in_dim);

    Tensor forward(const Tensor& input) const;
    void load_param(const std::string& key, ModelParam& model_param);
    int get_out_dim() const { return out_dim_; }

private:
    Tensor weights_;
    Tensor bias_;
    int out_dim_;
    int in_dim_;
};

} // namespace easy_gpt

#endif // EASY_GPT_MODELS_LINEAR_HPP
