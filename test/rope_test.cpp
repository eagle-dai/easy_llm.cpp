#include <algorithm>
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

Tensor slice_batch_4d(const Tensor& input, int batch_index) {
    const auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("slice_batch_4d expects 4D tensor.");
    }
    int batch = shape[0];
    if (batch_index < 0 || batch_index >= batch) {
        throw std::out_of_range("slice_batch_4d batch index out of range.");
    }
    int block = shape[1] * shape[2] * shape[3];
    std::vector<data_type> data(static_cast<size_t>(block));
    int start = batch_index * block;
    std::copy(input.data().begin() + start, input.data().begin() + start + block, data.begin());
    return Tensor(data, {1, shape[1], shape[2], shape[3]});
}

void write_batch_4d(Tensor& output, int batch_index, const Tensor& slice) {
    const auto shape = output.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("write_batch_4d expects 4D tensor.");
    }
    int batch = shape[0];
    if (batch_index < 0 || batch_index >= batch) {
        throw std::out_of_range("write_batch_4d batch index out of range.");
    }
    int block = shape[1] * shape[2] * shape[3];
    int start = batch_index * block;
    std::copy(slice.data().begin(), slice.data().end(), output.data().begin() + start);
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

void expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace
} // namespace easy_llm

int main() {
    using namespace easy_llm;

    const int batch = 2;
    const int heads = 2;
    const int seq_len = 3;
    const int head_dim = 4;
    const float rope_theta = 1000000.0f;

    std::vector<float> values;
    values.reserve(batch * heads * seq_len * head_dim);
    for (int i = 0; i < batch * heads * seq_len * head_dim; ++i) {
        values.push_back(static_cast<float>((i % 17) - 8) / 7.0f);
    }
    Tensor base = make_tensor(values, {batch, heads, seq_len, head_dim});

    Tensor expected_non_uniform = base;
    std::vector<int> offsets_non_uniform{-1, 2};
    for (int b = 0; b < batch; ++b) {
        Tensor slice = slice_batch_4d(expected_non_uniform, b);
        ops::apply_rope(slice, offsets_non_uniform[b], rope_theta);
        write_batch_4d(expected_non_uniform, b, slice);
    }
    Tensor actual_non_uniform = base;
    ops::apply_rope(actual_non_uniform, offsets_non_uniform, rope_theta);
    const float diff_non_uniform = max_abs_diff(actual_non_uniform, expected_non_uniform);
    expect(diff_non_uniform < 1e-3f, "FAIL: non-uniform offsets mismatch.");

    Tensor expected_uniform = base;
    const int uniform_offset = 3;
    ops::apply_rope(expected_uniform, uniform_offset, rope_theta);
    Tensor actual_uniform = base;
    std::vector<int> offsets_uniform(batch, uniform_offset);
    ops::apply_rope(actual_uniform, offsets_uniform, rope_theta);
    const float diff_uniform = max_abs_diff(actual_uniform, expected_uniform);
    expect(diff_uniform < 1e-3f, "FAIL: uniform offsets mismatch.");

    bool throw_on_bad_offset_size = false;
    try {
        std::vector<int> bad_offsets{0};
        Tensor bad_case = base;
        ops::apply_rope(bad_case, bad_offsets, rope_theta);
    } catch (const std::invalid_argument&) {
        throw_on_bad_offset_size = true;
    }
    expect(throw_on_bad_offset_size, "FAIL: expected invalid_argument for bad offsets size.");

    std::cout << "PASS: rope offsets test\n";
    return 0;
}
