#include "ops.hpp"

#include <numeric>
#include <iostream>

#include "spdlog/spdlog.h"

#ifdef USE_CUDA
    #include "cuda/ops/matmul.hpp"
    #include "cuda/runtime.hpp"
#endif

namespace easy_llm {
namespace ops {

using std::vector;

namespace {

struct Matmul3dShape {
    int batch;
    int seq_len;
    int height;
    int width;
};

Matmul3dShape validate_matmul_3d_shapes(const Tensor& input, const Tensor& weights) {
    const int height = weights.shape()[0];
    const int width = weights.shape()[1];
    if (input.shape().back() != width) {
        spdlog::error("matmul_3d input.shape().back() == {}, width == {}",
                      input.shape().back(), width);
        throw std::invalid_argument("The last dimension of input must be equal to width.");
    }
    const int batch = input.shape()[0];
    const int seq_len = input.shape()[1];
    return Matmul3dShape{batch, seq_len, height, width};
}

Tensor matmul_3d_cpu(const Tensor& input, const Tensor& weights,
                     int batch, int seq_len, int height, int width) {
    Tensor output(batch * seq_len * height);

    // Pre-convert weights -> float (once per call)
    std::vector<float> weights_f(height * width);
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < height * width; ++i) {
#ifdef USE_BF16
        weights_f[i] = bf16_to_float(weights[i].data);
#else
        weights_f[i] = static_cast<float>(weights[i]);
#endif
    }

    // Parallelize over (b, s) rows.
#ifdef USE_OPENMP
    #pragma omp parallel
#endif
    {
        std::vector<float> input_row_f(width);
#ifdef USE_OPENMP
        #pragma omp for collapse(2) schedule(static)
#endif
        for (int b = 0; b < batch; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                const int row_id = b * seq_len + s;
                const int input_base_offset = row_id * width;

                for (int w = 0; w < width; ++w) {
#ifdef USE_BF16
                    input_row_f[w] = bf16_to_float(input[input_base_offset + w].data);
#else
                    input_row_f[w] = static_cast<float>(input[input_base_offset + w]);
#endif
                }

                for (int h = 0; h < height; ++h) {
                    float sum = 0.0f;
                    const float* wf = &weights_f[h * width];
#ifdef USE_OPENMP
                    #pragma omp simd reduction(+:sum)
#endif
                    for (int w = 0; w < width; ++w) {
                        sum += input_row_f[w] * wf[w];
                    }
#ifdef USE_BF16
                    output[row_id * height + h].data = float_to_bf16(sum);
#else
                    output[row_id * height + h] = data_type(sum);
#endif
                }
            }
        }
    }
    output.reshape({batch, seq_len, height});
    return output;
}

} // namespace

Tensor matmul_3d(const Tensor& input, const Tensor& weights) {
    // spdlog::trace("matmul_3d input shape == {},{}", input.shape()[1], input.shape()[2]);
    // spdlog::trace("matmul_3d weights_ shape == {},{}", weights.shape()[0], weights.shape()[1]);
    const auto shape = validate_matmul_3d_shapes(input, weights);

#if defined(USE_CUDA)
    if (::easy_llm::cuda::available()) {
        try {
            return matmul_3d_cuda(input, weights);
        } catch (const std::exception& e) {
            spdlog::error("matmul_3d CUDA failed, falling back to CPU: {}", e.what());
        }
    }
#endif

    return matmul_3d_cpu(input, weights, shape.batch, shape.seq_len, shape.height, shape.width);
}

Tensor matmul_4d(const Tensor& input, const Tensor& weights) {
    if (weights.shape().size() != 4 || input.shape().size() != 4) {
        throw std::invalid_argument("matmul_4d expects 4D tensors.");
    }
    int height = weights.shape()[2];
    int width = weights.shape()[3];
    int batch = input.shape()[0];
    int num_heads = input.shape()[1];
    int seq_len = input.shape()[2];
    if (weights.shape()[0] != batch || weights.shape()[1] != num_heads) {
        throw std::invalid_argument("matmul_4d batch/head mismatch between input and weights.");
    }
    if (input.size() != batch * num_heads * seq_len * height) {
        throw std::invalid_argument("matmul_4d input size does not match shape.");
    }
    if (weights.size() != batch * num_heads * height * width) {
        throw std::invalid_argument("matmul_4d weights size does not match shape.");
    }
    if (input.shape().back() != height) {
        spdlog::error("matmul_4d input.shape().back() == {}, height == {}", input.shape().back(), height);
        throw std::invalid_argument("The last dimension of input must be equal to height.");
    }
    Tensor output(batch * num_heads * seq_len * width);

    // Pre-convert weights -> float (once per call)
    std::vector<float> weights_f(weights.size());
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
#ifdef USE_BF16
        weights_f[i] = bf16_to_float(weights[i].data);
#else
        weights_f[i] = static_cast<float>(weights[i]);
#endif
    }

    const int BLOCK_SIZE = 256;
    int num_blocks = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;

#ifdef USE_OPENMP
    #pragma omp parallel
#endif
    {
        vector<float> acc(BLOCK_SIZE);
        vector<float> input_row_f(height);
#ifdef USE_OPENMP
        #pragma omp for collapse(3) schedule(static)
#endif
        for(int b = 0; b < batch; ++b) {
            for(int nh = 0; nh < num_heads; ++nh) {
                for(int s = 0; s < seq_len; ++s) {
                    int input_base = (((b * num_heads) + nh) * seq_len + s) * height;
                    int weights_base = ((b * num_heads) + nh) * height * width;
                    int output_row_base = (((b * num_heads) + nh) * seq_len + s) * width;

                    for(int h = 0; h < height; ++h) {
#ifdef USE_BF16
                        input_row_f[h] = bf16_to_float(input[input_base + h].data);
#else
                        input_row_f[h] = static_cast<float>(input[input_base + h]);
#endif
                    }

                    for(int wb = 0; wb < num_blocks; ++wb) {
                        int w_start = wb * BLOCK_SIZE;
                        int w_count = std::min(BLOCK_SIZE, width - w_start);
                        std::fill(acc.begin(), acc.begin() + w_count, 0.0f);
                        for(int h = 0; h < height; ++h) {
                            float val_in = input_row_f[h];
                            const float* wf = &weights_f[weights_base + h * width + w_start];
#ifdef USE_OPENMP
                            #pragma omp simd
#endif
                            for(int k = 0; k < w_count; ++k) {
                                acc[k] += val_in * wf[k];
                            }
                        }
                        int output_base = output_row_base + w_start;
                        for(int k = 0; k < w_count; ++k) {
#ifdef USE_BF16
                            output[output_base + k].data = float_to_bf16(acc[k]);
#else
                            output[output_base + k] = data_type(acc[k]);
#endif
                        }
                    }
                }
            }
        }
    }
    output.reshape({batch, num_heads, seq_len, width});
    return output;
}

void add_inplace(Tensor& target, const Tensor& source) {
    if (target.size() != source.size()) {
        throw std::invalid_argument("Tensors must have the same size for add_inplace.");
    }
#ifdef USE_OPENMP
    #pragma omp simd
#endif
    for(int i = 0; i < target.size(); ++i){
        target[i] += source[i];
    }
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Tensors must have the same size for multiplication.");
    }
    Tensor output(a.size(), a.shape());
#ifdef USE_OPENMP
    #pragma omp simd
#endif
    for(int i = 0; i < a.size(); ++i){
        output[i] = a[i] * b[i];
    }
    return output;
}

Tensor concat(const vector<Tensor>& inputs, int axis) {
    if (inputs.empty()) {
        throw std::invalid_argument("inputs must not be empty.");
    }
    const auto& base_shape = inputs[0].shape();
    int ndim = static_cast<int>(base_shape.size());
    if (axis < 0 || axis >= ndim) {
        spdlog::error("concat axis out of range: axis = {}, ndim = {}", axis, ndim);
        throw std::out_of_range("axis out of range.");
    }

    int concat_dim = 0;
    for (const auto& t : inputs) {
        if (t.shape().size() != base_shape.size()) {
            throw std::invalid_argument("All tensors must have the same number of dimensions.");
        }
        for (int d = 0; d < ndim; ++d) {
            if (d == axis) continue;
            if (t.shape()[d] != base_shape[d]) {
                spdlog::error("Tensor shape mismatch at dimension {}: {} vs {}", d, t.shape()[d], base_shape[d]);
                throw std::invalid_argument("Tensor shapes must match except at the concat axis.");
            }
        }
        concat_dim += t.shape()[axis];
    }
    vector<int> out_shape = base_shape;
    out_shape[axis] = concat_dim;
    int outer_block = 1;
    for (int d = 0; d < axis; ++d) {
        outer_block *= base_shape[d];
    }
    int inner_block = 1;
    for (int d = axis + 1; d < ndim; ++d) {
        inner_block *= base_shape[d];
    }
    int out_size = outer_block * concat_dim * inner_block;
    Tensor output(out_size, out_shape);
    int axis_offset = 0;

    for (const auto& t : inputs) {
        int t_dim = t.shape()[axis];
#ifdef USE_OPENMP
        #pragma omp parallel for collapse(2)
#endif
        for (int outer_idx = 0; outer_idx < outer_block; ++outer_idx) {
            for (int axis_idx = 0; axis_idx < t_dim; ++axis_idx) {
                for (int inner_idx = 0; inner_idx < inner_block; ++inner_idx) {
                    int in_index  = outer_idx * (t_dim * inner_block) + axis_idx * inner_block + inner_idx;
                    int out_index = outer_idx * (concat_dim * inner_block)
                                  + (axis_offset + axis_idx) * inner_block
                                  + inner_idx;
                    output[out_index] = t[in_index];
                }
            }
        }
        axis_offset += t_dim;
    }
    return output;
}

void apply_rope(Tensor& input, int offset, float rope_theta) {
    // RoPE: Relative Positional Encoding for 4D Tensor [batch, head, seq, dim]
    if (input.shape().size() != 4) {
        throw std::invalid_argument("apply_rope expects a 4D tensor [batch, head, seq, dim]. Got shape: " + fmt::format("{}", fmt::join(input.shape(), ", ")));
    }
    int batch_size = input.shape()[0];
    int num_heads = input.shape()[1];
    int seq_len = input.shape()[2];
    int head_dim = input.shape()[3];
    int half_dim = head_dim / 2;
    if (head_dim % 2 != 0) throw std::invalid_argument("head_dim must be even for RoPE.");
    vector<float> freq_factors(half_dim);  // Precompute frequency factors so we do not repeat the calculation each outer loop
    for (int k = 0; k < half_dim; ++k) {
        freq_factors[k] = std::pow(rope_theta, (2.0f * k) / static_cast<float>(head_dim));
    }

    auto &data = input.data();
#ifdef USE_OPENMP
    #pragma omp parallel for collapse(4)
#endif
    for (int i = 0; i < batch_size; ++i) {
        for (int h = 0; h < num_heads; ++h) {
            for (int j = 0; j < seq_len; ++j) {
                for (int k = 0; k < half_dim; ++k) {
                    float freq = freq_factors[k];
                    float theta = static_cast<float>(j + offset) / freq;
                    float cos_t = std::cos(theta);
                    float sin_t = std::sin(theta);
                    
                    int idx1 = i * (num_heads * seq_len * head_dim) +
                               h * (seq_len * head_dim) +
                               j * head_dim +
                               k;
                    int idx2 = idx1 + half_dim;
                    data_type x1 = data[idx1];
                    data_type x2 = data[idx2];
                    data[idx1] = x1 * data_type(cos_t) - x2 * data_type(sin_t);
                    data[idx2] = x1 * data_type(sin_t) + x2 * data_type(cos_t);
                }
            }
        }
    }
}

std::pair<vector<float>, vector<int>> softmax(const Tensor& logits) {
    auto data_ = logits.data();
    vector<float> softmax_probs(data_.size());
    vector<int> softmax_shape = logits.shape();
    int last_dim = softmax_shape.back();
    int outer_size = data_.size() / last_dim;
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < outer_size; ++i){
        float max_val = -std::numeric_limits<float>::infinity();
        for(int j = 0; j < last_dim; ++j){
            if(float(data_[i * last_dim + j]) > max_val){
                max_val = float(data_[i * last_dim + j]);
            }
        }
        float sum = 0.0f;
        for(int j = 0; j < last_dim; ++j){
            softmax_probs[i * last_dim + j] = ::exp(float(data_[i * last_dim + j]) - max_val);
            sum += softmax_probs[i * last_dim + j];
        }
        const float epsilon = std::numeric_limits<float>::epsilon() * 10.0f;
        sum = std::max(sum, epsilon);
        for(int j = 0; j < last_dim; ++j){
            softmax_probs[i * last_dim + j] /= sum;
        }
    }
    return std::make_pair(softmax_probs, softmax_shape);
}

} // namespace ops
} // namespace easy_llm
