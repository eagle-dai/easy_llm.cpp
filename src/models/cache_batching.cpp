#include "models/cache_batching.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace easy_llm {

namespace {

void validate_per_sample_cache_shape(const Tensor& cache, const char* fn_name) {
    const auto& shape = cache.shape();
    if (shape.size() != 4 || shape[0] != 1) {
        throw std::invalid_argument(std::string(fn_name) +
                                    ": per-sample cache must have shape [1, heads, seq, dim].");
    }
    const int expected = shape[0] * shape[1] * shape[2] * shape[3];
    if (cache.size() != expected) {
        throw std::invalid_argument(std::string(fn_name) + ": tensor size does not match shape.");
    }
}

} // namespace

BatchedCacheView build_padded_active_cache(const std::vector<Tensor>& cache_by_sample,
                                           const std::vector<int>& sample_ids) {
    if (sample_ids.empty()) {
        throw std::invalid_argument("build_padded_active_cache: sample_ids is empty.");
    }
    int first_sample_id = sample_ids[0];
    if (first_sample_id < 0 || first_sample_id >= static_cast<int>(cache_by_sample.size())) {
        throw std::out_of_range("build_padded_active_cache: sample_id out of range.");
    }
    const Tensor& first = cache_by_sample[first_sample_id];
    validate_per_sample_cache_shape(first, "build_padded_active_cache");
    const auto first_shape = first.shape();
    const int num_heads = first_shape[1];
    const int head_dim = first_shape[3];

    std::vector<int> valid_lens;
    valid_lens.reserve(sample_ids.size());
    int max_seq_len = 0;
    for (int sample_id : sample_ids) {
        if (sample_id < 0 || sample_id >= static_cast<int>(cache_by_sample.size())) {
            throw std::out_of_range("build_padded_active_cache: sample_id out of range.");
        }
        const Tensor& src_tensor = cache_by_sample[sample_id];
        validate_per_sample_cache_shape(src_tensor, "build_padded_active_cache");
        const auto src_shape = src_tensor.shape();
        if (src_shape[1] != num_heads || src_shape[3] != head_dim) {
            throw std::invalid_argument("build_padded_active_cache: heads/head_dim mismatch.");
        }
        valid_lens.push_back(src_shape[2]);
        max_seq_len = std::max(max_seq_len, src_shape[2]);
    }
    if (max_seq_len <= 0) {
        throw std::invalid_argument("build_padded_active_cache: max_seq_len must be > 0.");
    }

    const int batch = static_cast<int>(sample_ids.size());
    const int block = num_heads * max_seq_len * head_dim;
    Tensor output(batch * block, {batch, num_heads, max_seq_len, head_dim});

    for (int b = 0; b < batch; ++b) {
        const int sample_id = sample_ids[b];
        const Tensor& src_tensor = cache_by_sample[sample_id];
        const auto src_shape = src_tensor.shape();
        const int seq_len = src_shape[2];
        const auto& src = src_tensor.data();
        auto& dst = output.data();
        for (int h = 0; h < num_heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                const int src_base = (h * seq_len + s) * head_dim;
                const int dst_base = ((b * num_heads + h) * max_seq_len + s) * head_dim;
                std::copy(src.begin() + src_base, src.begin() + src_base + head_dim, dst.begin() + dst_base);
            }
        }
    }

    return {std::move(output), std::move(valid_lens)};
}

void apply_valid_length_mask(Tensor& scores, const std::vector<int>& valid_lens) {
    const auto shape = scores.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("apply_valid_length_mask: scores must be 4D.");
    }
    const int batch = shape[0];
    const int heads = shape[1];
    const int seq_len = shape[2];
    const int total_len = shape[3];
    if (batch != static_cast<int>(valid_lens.size())) {
        throw std::invalid_argument("apply_valid_length_mask: valid_lens size must match batch.");
    }
    auto& data = scores.data();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int b = 0; b < batch; ++b) {
        int valid_len = std::clamp(valid_lens[b], 0, total_len);
        if (valid_len >= total_len) {
            continue;
        }
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                const int row_offset = ((b * heads + h) * seq_len + s) * total_len;
                for (int k = valid_len; k < total_len; ++k) {
                    data[row_offset + k] = data_type(neg_inf);
                }
            }
        }
    }
}

} // namespace easy_llm
