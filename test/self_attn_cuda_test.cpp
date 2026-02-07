#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "cuda/ops/self_attn.hpp"
#include "cuda/runtime.hpp"
#include "ops.hpp"
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

Tensor add_bias(const Tensor& input, const Tensor& bias) {
    Tensor out = input;
    if (out.shape().size() != 3 || bias.shape().size() != 1 || bias.size() != out.shape()[2]) {
        return out;
    }
    int batch = out.shape()[0];
    int seq_len = out.shape()[1];
    int hidden = out.shape()[2];
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < hidden; ++h) {
                int idx = (b * seq_len + s) * hidden + h;
                out[idx] = out[idx] + bias[h];
            }
        }
    }
    return out;
}

Tensor rms_norm_ref(const Tensor& input, const Tensor& weight) {
    if (input.shape().empty() || input.shape().back() != weight.size()) {
        throw std::invalid_argument("rms_norm_ref: invalid shape.");
    }
    Tensor out = input;
    const int features = input.shape().back();
    const int vectors = input.size() / features;
    for (int v = 0; v < vectors; ++v) {
        float sum_sq = 0.0f;
        const int base = v * features;
        for (int i = 0; i < features; ++i) {
            float val = static_cast<float>(input[base + i]);
            sum_sq += val * val;
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / features + 1e-6f);
        for (int i = 0; i < features; ++i) {
            out[base + i] = data_type(static_cast<float>(weight[i]) * static_cast<float>(input[base + i]) * inv_rms);
        }
    }
    return out;
}

Tensor slice_batch_4d(const Tensor& input, int batch_index) {
    const auto shape = input.shape();
    int block = shape[1] * shape[2] * shape[3];
    std::vector<data_type> data(block);
    int start = batch_index * block;
    std::copy(input.data().begin() + start, input.data().begin() + start + block, data.begin());
    return Tensor(data, {1, shape[1], shape[2], shape[3]});
}

void write_batch_4d(Tensor& output, int batch_index, const Tensor& slice) {
    const auto shape = output.shape();
    int block = shape[1] * shape[2] * shape[3];
    int start = batch_index * block;
    std::copy(slice.data().begin(), slice.data().end(), output.data().begin() + start);
}

void apply_causal_mask(Tensor& scores, int seq_len, int total_len) {
    int batch = scores.shape()[0];
    int heads = scores.shape()[1];
    float neg_inf = -std::numeric_limits<float>::infinity();
    auto& data = scores.data();
    int start_pos = total_len - seq_len;
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                int mask_start = start_pos + s + 1;
                if (mask_start >= total_len) {
                    continue;
                }
                int row_offset = ((b * heads + h) * seq_len + s) * total_len;
                for (int k = mask_start; k < total_len; ++k) {
                    data[row_offset + k] = data_type(neg_inf);
                }
            }
        }
    }
}

void apply_padding_mask(Tensor& scores, const std::vector<int>& sample_ids, const std::vector<int>& pad_lens) {
    int batch = scores.shape()[0];
    int heads = scores.shape()[1];
    int seq_len = scores.shape()[2];
    int total_len = scores.shape()[3];
    float neg_inf = -std::numeric_limits<float>::infinity();
    auto& data = scores.data();
    for (int b = 0; b < batch; ++b) {
        int sample_id = sample_ids[b];
        if (sample_id < 0 || sample_id >= static_cast<int>(pad_lens.size())) {
            continue;
        }
        int pad_len = std::max(0, std::min(pad_lens[sample_id], total_len));
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                int row_offset = ((b * heads + h) * seq_len + s) * total_len;
                for (int k = 0; k < pad_len; ++k) {
                    data[row_offset + k] = data_type(neg_inf);
                }
            }
        }
    }
}

Tensor build_active_cache(const std::vector<Tensor>& cache, const std::vector<int>& sample_ids) {
    if (sample_ids.size() == 1) {
        return cache[sample_ids[0]];
    }
    const Tensor& first = cache[sample_ids[0]];
    const auto shape = first.shape();
    int batch = static_cast<int>(sample_ids.size());
    int block = shape[1] * shape[2] * shape[3];
    Tensor out(batch * block, {batch, shape[1], shape[2], shape[3]});
    for (int i = 0; i < batch; ++i) {
        const Tensor& src = cache[sample_ids[i]];
        std::copy(src.data().begin(), src.data().end(), out.data().begin() + i * block);
    }
    return out;
}

Tensor run_cpu_reference(const Tensor& input,
                         const std::vector<int>& sample_ids,
                         const std::vector<int>& offsets,
                         const std::vector<int>& pad_lens,
                         const cuda::ops::SelfAttnCudaParams& params,
                         const Tensor& norm_weight,
                         const Tensor& q_weight, const Tensor& q_bias,
                         const Tensor& k_weight, const Tensor& k_bias,
                         const Tensor& v_weight, const Tensor& v_bias,
                         const Tensor& o_weight, const Tensor& o_bias,
                         std::vector<Tensor>& cache_k_by_sample,
                         std::vector<Tensor>& cache_v_by_sample,
                         std::vector<int>& cache_len_by_sample) {
    Tensor input_norm = rms_norm_ref(input, norm_weight);
    Tensor q = add_bias(ops::matmul_3d(input_norm, q_weight), q_bias);
    Tensor k = add_bias(ops::matmul_3d(input_norm, k_weight), k_bias);
    Tensor v = add_bias(ops::matmul_3d(input_norm, v_weight), v_bias);

    q.split_head(params.num_heads).transpose(1, 2);
    k.split_head(params.num_heads_kv).transpose(1, 2);
    v.split_head(params.num_heads_kv).transpose(1, 2);

    bool uniform_offset = true;
    for (int i = 1; i < static_cast<int>(offsets.size()); ++i) {
        if (offsets[i] != offsets[0]) {
            uniform_offset = false;
            break;
        }
    }
    if (uniform_offset) {
        int offset = offsets.empty() ? 0 : offsets[0];
        ops::apply_rope(q, offset, params.rope_theta);
        ops::apply_rope(k, offset, params.rope_theta);
    } else {
        for (int i = 0; i < static_cast<int>(sample_ids.size()); ++i) {
            Tensor q_slice = slice_batch_4d(q, i);
            Tensor k_slice = slice_batch_4d(k, i);
            ops::apply_rope(q_slice, offsets[i], params.rope_theta);
            ops::apply_rope(k_slice, offsets[i], params.rope_theta);
            write_batch_4d(q, i, q_slice);
            write_batch_4d(k, i, k_slice);
        }
    }

    int repeat_factor = params.num_heads / params.num_heads_kv;
    k.repeat(repeat_factor, 1);
    v.repeat(repeat_factor, 1);

    int batch = static_cast<int>(sample_ids.size());
    int seq_len = input.shape()[1];
    for (int i = 0; i < batch; ++i) {
        int sample_id = sample_ids[i];
        Tensor k_slice = slice_batch_4d(k, i);
        Tensor v_slice = slice_batch_4d(v, i);
        if (cache_len_by_sample[sample_id] == 0) {
            cache_k_by_sample[sample_id] = std::move(k_slice);
            cache_v_by_sample[sample_id] = std::move(v_slice);
        } else {
            cache_k_by_sample[sample_id] = ops::concat({cache_k_by_sample[sample_id], k_slice}, 2);
            cache_v_by_sample[sample_id] = ops::concat({cache_v_by_sample[sample_id], v_slice}, 2);
        }
        cache_len_by_sample[sample_id] += seq_len;
    }

    Tensor cache_k_batch = build_active_cache(cache_k_by_sample, sample_ids);
    Tensor cache_v_batch = build_active_cache(cache_v_by_sample, sample_ids);
    cache_k_batch.transpose(2, 3);

    Tensor scores = ops::matmul_4d(q, cache_k_batch);
    int total_len = scores.shape()[3];
    if (seq_len > 1) {
        apply_causal_mask(scores, seq_len, total_len);
    }
    if (!pad_lens.empty()) {
        apply_padding_mask(scores, sample_ids, pad_lens);
    }
    scores.scale_inplace(1.0f / std::sqrt(static_cast<float>(params.head_dim)));

    Tensor attn = scores.softmax();
    Tensor out = ops::matmul_4d(attn, cache_v_batch);
    out.transpose(1, 2).reshape(input.shape());
    return add_bias(ops::matmul_3d(out, o_weight), o_bias);
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
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

    std::mt19937 rng(42);
    cuda::ops::SelfAttnCudaParams params{};
    params.hidden_dim = 8;
    params.num_heads = 2;
    params.num_heads_kv = 1;
    params.head_dim = 4;
    params.rope_theta = 10000.0f;

    const int batch_size = 2;
    const int seq_len = 3;
    std::vector<int> sample_ids{0, 1};
    std::vector<int> offsets{-1, 2};
    std::vector<int> pad_lens{1, 0};

    Tensor input = make_tensor(rng, {batch_size, seq_len, params.hidden_dim});
    Tensor norm_weight = make_tensor(rng, {params.hidden_dim});
    Tensor q_weight = make_tensor(rng, {params.hidden_dim, params.hidden_dim});
    Tensor q_bias = make_tensor(rng, {params.hidden_dim});
    Tensor k_weight = make_tensor(rng, {params.num_heads_kv * params.head_dim, params.hidden_dim});
    Tensor k_bias = make_tensor(rng, {params.num_heads_kv * params.head_dim});
    Tensor v_weight = make_tensor(rng, {params.num_heads_kv * params.head_dim, params.hidden_dim});
    Tensor v_bias = make_tensor(rng, {params.num_heads_kv * params.head_dim});
    Tensor o_weight = make_tensor(rng, {params.hidden_dim, params.hidden_dim});
    Tensor o_bias = make_tensor(rng, {params.hidden_dim});

    std::vector<Tensor> cpu_cache_k(2);
    std::vector<Tensor> cpu_cache_v(2);
    std::vector<int> cpu_cache_len(2, 0);
    Tensor cpu_out = run_cpu_reference(input, sample_ids, offsets, pad_lens, params,
                                       norm_weight, q_weight, q_bias, k_weight, k_bias,
                                       v_weight, v_bias, o_weight, o_bias,
                                       cpu_cache_k, cpu_cache_v, cpu_cache_len);

    cuda::ops::SelfAttnCudaState cuda_state;
    cuda_state.init_kv_cache(2);
    Tensor cuda_out = cuda::ops::self_attn_forward_cuda(input, sample_ids, offsets, pad_lens, params,
                                                         norm_weight,
                                                         q_weight, q_bias,
                                                         k_weight, k_bias,
                                                         v_weight, v_bias,
                                                         o_weight, o_bias,
                                                         cuda_state);

    const float diff = max_abs_diff(cpu_out, cuda_out);
    std::cout << "max_abs_diff=" << diff << "\n";
    if (diff > 8e-2f) {
        std::cerr << "FAIL: CUDA parity diff too large\n";
        return 1;
    }

    // Regression: loader stores 1D tensors as [N, 1]; CUDA path should accept this norm weight layout.
    std::vector<data_type> norm_weight_column_data = norm_weight.data();
    Tensor norm_weight_column(norm_weight_column_data, {params.hidden_dim, 1});

    std::vector<Tensor> cpu_cache_k_column(2);
    std::vector<Tensor> cpu_cache_v_column(2);
    std::vector<int> cpu_cache_len_column(2, 0);
    Tensor cpu_out_column = run_cpu_reference(input, sample_ids, offsets, pad_lens, params,
                                              norm_weight_column, q_weight, q_bias, k_weight, k_bias,
                                              v_weight, v_bias, o_weight, o_bias,
                                              cpu_cache_k_column, cpu_cache_v_column, cpu_cache_len_column);

    cuda::ops::SelfAttnCudaState cuda_state_column;
    cuda_state_column.init_kv_cache(2);
    Tensor cuda_out_column;
    try {
        cuda_out_column = cuda::ops::self_attn_forward_cuda(input, sample_ids, offsets, pad_lens, params,
                                                            norm_weight_column,
                                                            q_weight, q_bias,
                                                            k_weight, k_bias,
                                                            v_weight, v_bias,
                                                            o_weight, o_bias,
                                                            cuda_state_column);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: column norm_weight should be accepted, got exception: " << e.what() << "\n";
        return 1;
    }

    const float diff_column = max_abs_diff(cpu_out_column, cuda_out_column);
    std::cout << "max_abs_diff_column_norm=" << diff_column << "\n";
    if (diff_column > 8e-2f) {
        std::cerr << "FAIL: CUDA parity diff too large for column norm weight\n";
        return 1;
    }

    cuda_state.clear_kv_cache(0);
    if (cuda_state.cache_len(0) != 0) {
        std::cerr << "FAIL: clear_kv_cache did not reset length\n";
        return 1;
    }

    // Performance regression guard (decode-like): avoid per-step realloc/copy churn.
    cuda::ops::SelfAttnCudaState cuda_state_perf;
    cuda_state_perf.init_kv_cache(1);
    cuda_state_perf.reset_stats();
    std::vector<int> perf_sample_ids{0};
    std::vector<int> perf_pad_lens{0};
    const int perf_steps = 64;
    for (int step = 0; step < perf_steps; ++step) {
        Tensor perf_input = make_tensor(rng, {1, 1, params.hidden_dim});
        std::vector<int> perf_offsets{cuda_state_perf.cache_len(0)};
        (void)cuda::ops::self_attn_forward_cuda(perf_input, perf_sample_ids, perf_offsets, perf_pad_lens, params,
                                                norm_weight,
                                                q_weight, q_bias,
                                                k_weight, k_bias,
                                                v_weight, v_bias,
                                                o_weight, o_bias,
                                                cuda_state_perf);
    }
    const auto perf_stats = cuda_state_perf.stats();
    std::cout << "scratch_reallocations=" << perf_stats.scratch_reallocations
              << ", pad_lens_uploads=" << perf_stats.pad_lens_uploads
              << ", decode_seq1_path_hits=" << perf_stats.decode_seq1_path_hits
              << ", decode_graph_captures=" << perf_stats.decode_graph_captures
              << ", decode_graph_launches=" << perf_stats.decode_graph_launches << "\n";
    if (perf_stats.scratch_reallocations > 40) {
        std::cerr << "FAIL: scratch reallocations too high in decode loop\n";
        return 1;
    }
    if (perf_stats.pad_lens_uploads > 2) {
        std::cerr << "FAIL: pad_lens uploaded too many times\n";
        return 1;
    }
    if (perf_stats.decode_seq1_path_hits < static_cast<std::uint64_t>(perf_steps)) {
        std::cerr << "FAIL: decode seq=1 path was not consistently selected\n";
        return 1;
    }
    if (perf_stats.decode_graph_captures == 0 || perf_stats.decode_graph_launches == 0) {
        std::cerr << "FAIL: decode CUDA graph was not used\n";
        return 1;
    }
    if (perf_stats.decode_graph_launches < perf_stats.decode_graph_captures) {
        std::cerr << "FAIL: decode graph launches must be >= captures\n";
        return 1;
    }

    cuda_state.reset_kv_cache();
    return 0;
}
