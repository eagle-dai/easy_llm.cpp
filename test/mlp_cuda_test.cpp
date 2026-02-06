#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "cuda/ops/mlp.hpp"
#include "cuda/runtime.hpp"
#include "tensor.hpp"

namespace easy_llm {
namespace {

Tensor make_tensor(std::mt19937& rng, const std::vector<int>& shape, float scale = 0.1f) {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<data_type> data;
    data.reserve(size);
    for (int i = 0; i < size; ++i) {
        data.emplace_back(data_type(dist(rng)));
    }
    return Tensor(data, shape);
}

Tensor rms_norm_ref(const Tensor& input, const Tensor& weight) {
    if (input.shape().empty()) {
        throw std::invalid_argument("rms_norm_ref: input shape is empty.");
    }
    if (weight.size() != input.shape().back()) {
        throw std::invalid_argument("rms_norm_ref: weight size mismatch.");
    }
    Tensor output = input;
    const int hidden_dim = input.shape().back();
    const int rows = input.size() / hidden_dim;
    for (int row = 0; row < rows; ++row) {
        float sum_sq = 0.0f;
        const int base = row * hidden_dim;
        for (int i = 0; i < hidden_dim; ++i) {
            float v = static_cast<float>(input[base + i]);
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_dim) + 1e-6f);
        for (int i = 0; i < hidden_dim; ++i) {
            float v = static_cast<float>(input[base + i]);
            float w = static_cast<float>(weight[i]);
            output[base + i] = data_type(v * w * inv_rms);
        }
    }
    return output;
}

Tensor linear_ref(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    if (input.shape().size() != 3 || weight.shape().size() != 2) {
        throw std::invalid_argument("linear_ref: invalid rank.");
    }
    const int batch = input.shape()[0];
    const int seq_len = input.shape()[1];
    const int in_dim = input.shape()[2];
    const int out_dim = weight.shape()[0];
    if (weight.shape()[1] != in_dim) {
        throw std::invalid_argument("linear_ref: in_dim mismatch.");
    }
    if (bias.size() != out_dim) {
        throw std::invalid_argument("linear_ref: bias mismatch.");
    }
    Tensor output(batch * seq_len * out_dim, {batch, seq_len, out_dim});
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            const int row = b * seq_len + s;
            for (int o = 0; o < out_dim; ++o) {
                float sum = static_cast<float>(bias[o]);
                for (int i = 0; i < in_dim; ++i) {
                    const float x = static_cast<float>(input[row * in_dim + i]);
                    const float w = static_cast<float>(weight[o * in_dim + i]);
                    sum += x * w;
                }
                output[row * out_dim + o] = data_type(sum);
            }
        }
    }
    return output;
}

Tensor silu_mul_ref(const Tensor& up, const Tensor& gate) {
    if (up.shape() != gate.shape()) {
        throw std::invalid_argument("silu_mul_ref: shape mismatch.");
    }
    Tensor out = up;
    for (int i = 0; i < out.size(); ++i) {
        float g = static_cast<float>(gate[i]);
        float silu = g / (1.0f + std::exp(-g));
        float u = static_cast<float>(up[i]);
        out[i] = data_type(u * silu);
    }
    return out;
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape() || a.size() != b.size()) {
        throw std::invalid_argument("max_abs_diff: shape mismatch.");
    }
    float max_v = 0.0f;
    for (int i = 0; i < a.size(); ++i) {
        float diff = std::fabs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
        if (diff > max_v) {
            max_v = diff;
        }
    }
    return max_v;
}

} // namespace
} // namespace easy_llm

int main() {
    using namespace easy_llm;

    if (!cuda::initialize()) {
        std::cout << "SKIP: CUDA unavailable\n";
        return 0;
    }

    std::mt19937 rng(1234);
    const int batch = 2;
    const int seq_len = 3;
    const int hidden_dim = 8;
    const int inter_dim = 16;

    Tensor input = make_tensor(rng, {batch, seq_len, hidden_dim});
    Tensor norm_weight = make_tensor(rng, {hidden_dim});
    Tensor up_weight = make_tensor(rng, {inter_dim, hidden_dim});
    Tensor up_bias = make_tensor(rng, {inter_dim});
    Tensor gate_weight = make_tensor(rng, {inter_dim, hidden_dim});
    Tensor gate_bias = make_tensor(rng, {inter_dim});
    Tensor down_weight = make_tensor(rng, {hidden_dim, inter_dim});
    Tensor down_bias = make_tensor(rng, {hidden_dim});

    Tensor input_norm = rms_norm_ref(input, norm_weight);
    Tensor up = linear_ref(input_norm, up_weight, up_bias);
    Tensor gate = linear_ref(input_norm, gate_weight, gate_bias);
    Tensor activated = silu_mul_ref(up, gate);
    Tensor cpu_out = linear_ref(activated, down_weight, down_bias);

    Tensor cuda_out = cuda::ops::mlp_forward_cuda(input,
                                                  norm_weight,
                                                  up_weight, up_bias,
                                                  gate_weight, gate_bias,
                                                  down_weight, down_bias);

    float diff = max_abs_diff(cpu_out, cuda_out);
    std::cout << "max_abs_diff=" << diff << "\n";
    if (diff > 8e-2f) {
        std::cerr << "FAIL: CUDA MLP parity diff too large\n";
        return 1;
    }

    std::vector<data_type> norm_weight_column_data = norm_weight.data();
    Tensor norm_weight_column(norm_weight_column_data, {hidden_dim, 1});
    Tensor cpu_out_column = linear_ref(
        silu_mul_ref(
            linear_ref(rms_norm_ref(input, norm_weight_column), up_weight, up_bias),
            linear_ref(rms_norm_ref(input, norm_weight_column), gate_weight, gate_bias)),
        down_weight, down_bias);

    Tensor cuda_out_column = cuda::ops::mlp_forward_cuda(input,
                                                         norm_weight_column,
                                                         up_weight, up_bias,
                                                         gate_weight, gate_bias,
                                                         down_weight, down_bias);
    float diff_column = max_abs_diff(cpu_out_column, cuda_out_column);
    std::cout << "max_abs_diff_column_norm=" << diff_column << "\n";
    if (diff_column > 8e-2f) {
        std::cerr << "FAIL: CUDA MLP parity diff too large for column norm weight\n";
        return 1;
    }

    return 0;
}
