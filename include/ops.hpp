#ifndef EASY_LLM_OPS_HPP
#define EASY_LLM_OPS_HPP

#include <vector>
#include "tensor.hpp"

namespace easy_llm {
namespace ops {

Tensor matmul_3d(const Tensor& input, const Tensor& weights);
Tensor matmul_3d(const Tensor& input, const Tensor& weights, const Tensor& bias);
Tensor matmul_4d(const Tensor& input, const Tensor& weights);
void add_inplace(Tensor& target, const Tensor& source);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor concat(const std::vector<Tensor>& inputs, int axis = 0);
void apply_rope(Tensor& input, int offset = 0, float rope_theta = 10000.f);
void apply_rope(Tensor& input, const std::vector<int>& offsets, float rope_theta = 10000.f);
std::pair<std::vector<float>, std::vector<int>> softmax(const Tensor& logits);

} // namespace ops
} // namespace easy_llm

#endif // EASY_LLM_OPS_HPP
