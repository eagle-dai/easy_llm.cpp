#include <iostream>
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

template <typename ExceptionType, typename Fn>
bool expect_throws(Fn&& fn, const char* case_name) {
    try {
        fn();
    } catch (const ExceptionType&) {
        return true;
    }
    std::cerr << "FAIL: " << case_name << " should throw\n";
    return false;
}

} // namespace
} // namespace easy_llm

int main() {
    using namespace easy_llm;

    Tensor cache_ok = make_tensor({
        1, 2,
        3, 4
    }, {1, 1, 2, 2});

    Tensor cache_bad_heads = make_tensor({
        5, 6,
        7, 8,
        9, 10,
        11, 12
    }, {1, 2, 2, 1});

    Tensor cache_bad_dim = make_tensor({
        5, 6, 7,
        8, 9, 10
    }, {1, 1, 2, 3});

    if (!expect_throws<std::invalid_argument>([&]() {
            (void)build_padded_active_cache({cache_ok, cache_bad_heads}, {0, 1});
        }, "heads mismatch")) {
        return 1;
    }

    if (!expect_throws<std::invalid_argument>([&]() {
            (void)build_padded_active_cache({cache_ok, cache_bad_dim}, {0, 1});
        }, "head_dim mismatch")) {
        return 1;
    }

    if (!expect_throws<std::invalid_argument>([&]() {
            (void)build_padded_active_cache({cache_ok}, {});
        }, "empty sample_ids")) {
        return 1;
    }

    if (!expect_throws<std::out_of_range>([&]() {
            (void)build_padded_active_cache({cache_ok}, {2});
        }, "sample_id out of range")) {
        return 1;
    }

    Tensor scores = make_tensor({
        0.1f, 0.2f,
        0.3f, 0.4f
    }, {1, 1, 2, 2});
    if (!expect_throws<std::invalid_argument>([&]() {
            apply_valid_length_mask(scores, {});
        }, "valid_lens size mismatch")) {
        return 1;
    }

    std::cout << "PASS: cache batching invariants test\n";
    return 0;
}
