#ifndef EASY_GPT_CUDA_OPS_MATMUL_HPP
#define EASY_GPT_CUDA_OPS_MATMUL_HPP

#include "tensor.hpp"

namespace easy_gpt {
namespace ops {

Tensor matmul_3d_cuda(const Tensor& input, const Tensor& weights);

} // namespace ops
} // namespace easy_gpt

#endif // EASY_GPT_CUDA_OPS_MATMUL_HPP
