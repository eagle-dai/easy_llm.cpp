#ifndef EASY_LLM_CUDA_OPS_MLP_HPP
#define EASY_LLM_CUDA_OPS_MLP_HPP

#include "tensor.hpp"

namespace easy_llm {
namespace cuda {
namespace ops {

Tensor mlp_forward_cuda(
    const Tensor& input,
    const Tensor& norm_weight,
    const Tensor& up_weight, const Tensor& up_bias,
    const Tensor& gate_weight, const Tensor& gate_bias,
    const Tensor& down_weight, const Tensor& down_bias);

} // namespace ops
} // namespace cuda
} // namespace easy_llm

#endif // EASY_LLM_CUDA_OPS_MLP_HPP
