#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "ops.hpp"
#include "tensor.hpp"

namespace easy_llm {
namespace {

Tensor make_tensor(const std::vector<float>& values, const std::vector<int>& shape) {
    std::vector<data_type> data;
    data.reserve(values.size());
    for (float v : values) {
        data.emplace_back(data_type(v));
    }
    Tensor tensor(data, shape);
    if (tensor.size() != static_cast<int>(values.size())) {
        throw std::invalid_argument("make_tensor: size mismatch.");
    }
    return tensor;
}

Tensor add_bias_ref(const Tensor& input, const Tensor& bias) {
    Tensor out = input;
    if (out.shape().size() != 3 || bias.shape().size() != 1 || bias.size() != out.shape()[2]) {
        return out;
    }
    const int rows = out.shape()[0] * out.shape()[1];
    const int cols = out.shape()[2];
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out[r * cols + c] = out[r * cols + c] + bias[c];
        }
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

    Tensor input = make_tensor({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    }, {1, 2, 3});
    Tensor weights = make_tensor({
        1.0f, 0.0f, -1.0f,
        2.0f, 3.0f, 4.0f,
    }, {2, 3});
    Tensor bias = make_tensor({0.5f, -1.0f}, {2});

    Tensor without_bias = ops::matmul_3d(input, weights);
    Tensor with_bias = ops::matmul_3d(input, weights, bias);
    Tensor expected = add_bias_ref(without_bias, bias);

    float diff = max_abs_diff(expected, with_bias);
    if (diff > 6e-2f) {
        std::cerr << "FAIL: matmul_3d bias mismatch, max_abs_diff=" << diff << "\n";
        return 1;
    }

    Tensor invalid_bias = make_tensor({1.0f}, {1});
    Tensor with_invalid_bias = ops::matmul_3d(input, weights, invalid_bias);
    float invalid_diff = max_abs_diff(without_bias, with_invalid_bias);
    if (invalid_diff > 6e-2f) {
        std::cerr << "FAIL: matmul_3d invalid bias behavior changed, max_abs_diff="
                  << invalid_diff << "\n";
        return 1;
    }

    std::cout << "PASS: matmul_3d bias test\n";
    return 0;
}
