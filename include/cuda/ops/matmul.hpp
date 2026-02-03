#ifndef EASY_LLM_CUDA_OPS_MATMUL_HPP
#define EASY_LLM_CUDA_OPS_MATMUL_HPP

#include "tensor.hpp"

namespace easy_llm {
namespace ops {

Tensor matmul_3d_cuda(const Tensor& input, const Tensor& weights);

} // namespace ops
} // namespace easy_llm

#endif // EASY_LLM_CUDA_OPS_MATMUL_HPP
