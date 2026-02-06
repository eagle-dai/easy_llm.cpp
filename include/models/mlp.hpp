#ifndef EASY_LLM_MODELS_MLP_HPP
#define EASY_LLM_MODELS_MLP_HPP

#include <vector>
#include <memory>
#include <string>

#include "tensor.hpp"
#include "models/loader.hpp"
#include "models/linear.hpp"
#include "models/norm.hpp"
#ifdef USE_CUDA
    #include "cuda/ops/mlp.hpp"
#endif

namespace easy_llm {

class LayerKeyPrefix;

class MLP {
public:
    MLP();
    MLP(int hidden_dim);

    Tensor forward(const Tensor& input) const;
    void load_param(const LayerKeyPrefix& key_prefix, const std::string& key, ModelParam& model_param);

private:
    Tensor forward_cpu(const Tensor& input) const;
#ifdef USE_CUDA
    Tensor forward_cuda(const Tensor& input) const;
    mutable bool cuda_enabled_{true};
#endif

    Linear up_proj_;
    Linear gate_proj_;
    Linear down_proj_;
    int hidden_dim_;
    RMSNorm norm_{};
};

} // namespace easy_llm

#endif // EASY_LLM_MODELS_MLP_HPP
