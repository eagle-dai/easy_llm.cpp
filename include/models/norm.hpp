#ifndef EASY_GPT_NORM_HPP
#define EASY_GPT_NORM_HPP

#include <array>
#include <vector>
#include <memory>
#include <unordered_map>

#include "tensor.hpp"
#include "config.hpp"
#include "models/loader.hpp"

namespace easy_gpt {

class RMSNorm {
public:
    RMSNorm();
    void load_param(const std::string& key, ModelParam& model_param);
    Tensor forward(const Tensor& input) const;

private:
    Tensor weight_;
    int dim_;
};

} // namespace easy_gpt

#endif // EASY_GPT_NORM_HPP
