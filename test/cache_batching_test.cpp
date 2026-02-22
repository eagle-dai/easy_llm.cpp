#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "models/cache_batching.hpp"
#include "tensor.hpp"

namespace easy_llm {
namespace {

Tensor make_tensor(const std::vector<float>& values, const std::vector<int>& shape) {
    std::vector<data_type> data;
    data.reserve(values.size());
    for (float v : values) {
        data.emplace_back(data_type(v));
    }
    return Tensor(data, shape);
}

bool nearly_equal(float a, float b, float eps = 6e-2f) {
    return std::fabs(a - b) <= eps;
}

} // namespace
} // namespace easy_llm

int main() {
    using namespace easy_llm;

    Tensor cache0 = make_tensor({
        1, 2,
        3, 4
    }, {1, 1, 2, 2});

    Tensor cache1 = make_tensor({
        10, 11,
        12, 13,
        14, 15,
        16, 17
    }, {1, 1, 4, 2});

    std::vector<Tensor> cache_by_sample = {cache0, cache1};
    std::vector<int> sample_ids = {0, 1};

    BatchedCacheView batched = build_padded_active_cache(cache_by_sample, sample_ids);
    const auto out_shape = batched.cache.shape();
    if (out_shape != std::vector<int>({2, 1, 4, 2})) {
        std::cerr << "FAIL: unexpected batched cache shape\n";
        return 1;
    }
    if (batched.valid_lens != std::vector<int>({2, 4})) {
        std::cerr << "FAIL: unexpected valid lens\n";
        return 1;
    }

    const auto& out = batched.cache.data();
    // sample0 occupies first 8 elements, first two positions should be copied.
    if (!nearly_equal(static_cast<float>(out[0]), 1.0f) ||
        !nearly_equal(static_cast<float>(out[1]), 2.0f) ||
        !nearly_equal(static_cast<float>(out[2]), 3.0f) ||
        !nearly_equal(static_cast<float>(out[3]), 4.0f)) {
        std::cerr << "FAIL: sample0 data copy mismatch\n";
        return 1;
    }
    if (!nearly_equal(static_cast<float>(out[4]), 0.0f) ||
        !nearly_equal(static_cast<float>(out[5]), 0.0f) ||
        !nearly_equal(static_cast<float>(out[6]), 0.0f) ||
        !nearly_equal(static_cast<float>(out[7]), 0.0f)) {
        std::cerr << "FAIL: sample0 padded tail should be zero\n";
        return 1;
    }

    Tensor scores = make_tensor({
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f
    }, {2, 1, 1, 4});
    apply_valid_length_mask(scores, batched.valid_lens);
    const auto& masked = scores.data();

    const float neg_inf = -std::numeric_limits<float>::infinity();
    if (!std::isinf(static_cast<float>(masked[2])) ||
        !std::isinf(static_cast<float>(masked[3])) ||
        static_cast<float>(masked[2]) > neg_inf ||
        static_cast<float>(masked[3]) > neg_inf) {
        std::cerr << "FAIL: sample0 tail is not masked\n";
        return 1;
    }
    if (!nearly_equal(static_cast<float>(masked[4]), 0.5f) ||
        !nearly_equal(static_cast<float>(masked[7]), 0.8f)) {
        std::cerr << "FAIL: sample1 valid positions changed unexpectedly\n";
        return 1;
    }

    std::cout << "PASS: cache batching test\n";
    return 0;
}
