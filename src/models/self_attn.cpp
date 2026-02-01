#include "models/self_attn.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include "ops.hpp"

#include "config.hpp"

namespace easy_gpt {

namespace {

int shape_product(const std::vector<int>& shape) {
    int total = 1;
    for (int dim : shape) {
        total *= dim;
    }
    return total;
}

void validate_tensor_size(const Tensor& tensor, const char* name) {
    if (tensor.shape().empty()) {
        throw std::invalid_argument(std::string("Tensor has empty shape: ") + name);
    }
    int expected = shape_product(tensor.shape());
    if (tensor.size() != expected) {
        throw std::invalid_argument(std::string("Tensor size mismatch: ") + name);
    }
}

Tensor slice_batch_4d(const Tensor& input, int batch_index) {
    const auto& shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("slice_batch_4d expects a 4D tensor.");
    }
    int batch = shape[0];
    if (batch_index < 0 || batch_index >= batch) {
        throw std::out_of_range("slice_batch_4d batch index out of range.");
    }
    int num_heads = shape[1];
    int seq_len = shape[2];
    int head_dim = shape[3];
    int block = num_heads * seq_len * head_dim;
    if (input.size() != batch * block) {
        throw std::invalid_argument("slice_batch_4d: input size does not match shape.");
    }
    int start = batch_index * block;
    std::vector<data_type> data(static_cast<size_t>(block));
    const auto& src = input.data();
    std::copy(src.begin() + start, src.begin() + start + block, data.begin());
    return Tensor{data, {1, num_heads, seq_len, head_dim}};
}

void write_batch_4d(Tensor& target, int batch_index, const Tensor& slice) {
    const auto& target_shape = target.shape();
    const auto& slice_shape = slice.shape();
    if (target_shape.size() != 4 || slice_shape.size() != 4) {
        throw std::invalid_argument("write_batch_4d expects 4D tensors.");
    }
    if (slice_shape[0] != 1 || slice_shape[1] != target_shape[1] ||
        slice_shape[2] != target_shape[2] || slice_shape[3] != target_shape[3]) {
        throw std::invalid_argument("write_batch_4d slice shape mismatch.");
    }
    int batch = target_shape[0];
    if (batch_index < 0 || batch_index >= batch) {
        throw std::out_of_range("write_batch_4d batch index out of range.");
    }
    int block = target_shape[1] * target_shape[2] * target_shape[3];
    if (target.size() != batch * block) {
        throw std::invalid_argument("write_batch_4d: target size does not match shape.");
    }
    int start = batch_index * block;
    const auto& src = slice.data();
    auto& dst = target.data();
    std::copy(src.begin(), src.end(), dst.begin() + start);
}

bool is_uniform_offset(const std::vector<int>& offsets) {
    if (offsets.empty()) {
        return true;
    }
    int offset0 = offsets[0];
    for (int offset : offsets) {
        if (offset != offset0) {
            return false;
        }
    }
    return true;
}

void apply_causal_mask(Tensor& scores, int seq_len, int total_len) {
    int batch = scores.shape()[0];
    int heads = scores.shape()[1];
    float neg_inf = -std::numeric_limits<float>::infinity();
    auto& data = scores.data();
    int start_pos = total_len - seq_len;

#ifdef USE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                int mask_start = start_pos + s + 1;
                if (mask_start < total_len) {
                    int row_offset = ((b * heads + h) * seq_len + s) * total_len;
                    for (int k = mask_start; k < total_len; ++k) {
                        data[row_offset + k] = data_type(neg_inf);
                    }
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
        int pad_len = pad_lens[sample_id];
        if (pad_len <= 0) {
            continue;
        }
        if (pad_len > total_len) {
            pad_len = total_len;
        }
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

} // namespace

struct SelfAttn::ForwardContext {
    const Tensor& input;
    const std::vector<int>& sample_ids;
    const std::vector<int>* pos_offsets;
    std::vector<int> offsets;
    int seq_len{0};
    int max_sample_id{-1};

    ForwardContext(const Tensor& input,
                   const std::vector<int>& sample_ids,
                   const std::vector<int>* pos_offsets)
        : input(input), sample_ids(sample_ids), pos_offsets(pos_offsets) {}
};

SelfAttn::SelfAttn() {}

SelfAttn::SelfAttn(const Config& config) {
    num_heads_ = config.num_heads;
    num_heads_kv_ = config.num_heads_kv;
    rope_theta_ = config.rope_theta;
}

SelfAttn::SelfAttn(int hidden_dim, int num_heads)
    : hidden_dim_(hidden_dim), num_heads_(num_heads),
      head_dim_(hidden_dim / num_heads),
      q_proj_(hidden_dim, hidden_dim),
      k_proj_(hidden_dim, hidden_dim),
      v_proj_(hidden_dim, hidden_dim),
      o_proj_(hidden_dim, hidden_dim) {
}

void SelfAttn::load_param(const std::string& key, ModelParam& model_param) {
    q_proj_.load_param(key + ".self_attn.q_proj", model_param);
    k_proj_.load_param(key + ".self_attn.k_proj", model_param);
    v_proj_.load_param(key + ".self_attn.v_proj", model_param);
    o_proj_.load_param(key + ".self_attn.o_proj", model_param);
    hidden_dim_ = q_proj_.get_out_dim();
    if (num_heads_ <= 0) {
        spdlog::error("num_heads_ is invalid: {}", num_heads_);
        return;
    }
    head_dim_ = hidden_dim_ / num_heads_;
    norm_.load_param(key + ".input_layernorm", model_param);
}

Tensor SelfAttn::forward(const Tensor& input, const std::vector<int>& sample_ids, const std::vector<int>* pos_offsets) {
    if (num_heads_ <= 0 || num_heads_kv_ <= 0 || rope_theta_ <= 0.0f) {
        spdlog::error("SelfAttn not configured: num_heads={}, num_heads_kv={}, rope_theta={}", num_heads_, num_heads_kv_, rope_theta_);
        return Tensor();
    }
    ForwardContext ctx(input, sample_ids, pos_offsets);
    validate_forward_inputs(ctx);
    ctx.max_sample_id = *std::max_element(ctx.sample_ids.begin(), ctx.sample_ids.end());
    ensure_cache_capacity(ctx.max_sample_id + 1);

    auto input_norm = norm_.forward(input);
    auto q = q_proj_.forward(input_norm);  // [batch, seq, hidden_dim]
    auto k = k_proj_.forward(input_norm);
    auto v = v_proj_.forward(input_norm);
    q.split_head(num_heads_).transpose(1, 2);  // [batch, num_head, seq, head_dim]
    k.split_head(num_heads_kv_).transpose(1, 2);  // [batch, num_head_kv, seq, head_dim]
    v.split_head(num_heads_kv_).transpose(1, 2);  // [batch, num_head_kv, seq, head_dim]
    validate_tensor_size(q, "q");
    validate_tensor_size(k, "k");
    validate_tensor_size(v, "v");
    compute_offsets(ctx);
    apply_rope_offsets(q, k, ctx);

    expand_kv_heads(k, v);
    validate_tensor_size(k, "k_repeat");
    validate_tensor_size(v, "v_repeat");
    append_kv_cache(k, v, ctx);
    auto cache_k_batch = build_active_cache(cache_k_by_sample_, ctx);
    auto cache_v_batch = build_active_cache(cache_v_by_sample_, ctx);
    validate_tensor_size(cache_k_batch, "cache_k_batch");
    validate_tensor_size(cache_v_batch, "cache_v_batch");
    cache_k_batch.transpose(2, 3);  // [batch, num_head, head_dim, seq]

    int seq_len = q.shape()[2];
    ctx.seq_len = seq_len;
    auto scores = ops::matmul_4d(q, cache_k_batch);  // [batch, num_head, seq, seq]
    apply_attention_masks(scores, ctx);
    scores.scale_inplace(1.0f / std::sqrt(head_dim_));  // scale scores
    auto attention = scores.softmax();
    auto attn_output = ops::matmul_4d(attention, cache_v_batch);  // [batch, num_head, seq, head_dim]
    auto input_shape = input.shape();
    attn_output.transpose(1, 2).reshape(input_shape);
    auto output = o_proj_.forward(attn_output);
    return output;
}

void SelfAttn::validate_forward_inputs(const ForwardContext& ctx) const {
    if (ctx.input.shape()[0] != static_cast<int>(ctx.sample_ids.size())) {
        throw std::invalid_argument("SelfAttn forward: input batch size must match sample_ids size.");
    }
}

void SelfAttn::compute_offsets(ForwardContext& ctx) const {
    ctx.offsets.clear();
    ctx.offsets.reserve(ctx.sample_ids.size());
    if (ctx.pos_offsets && ctx.pos_offsets->size() == ctx.sample_ids.size()) {
        ctx.offsets.assign(ctx.pos_offsets->begin(), ctx.pos_offsets->end());
        return;
    }
    for (int sample_id : ctx.sample_ids) {
        ctx.offsets.push_back(cache_len_by_sample_[sample_id]);
    }
}

void SelfAttn::apply_rope_offsets(Tensor& q, Tensor& k, const ForwardContext& ctx) {
    if (is_uniform_offset(ctx.offsets)) {
        int offset0 = ctx.offsets.empty() ? 0 : ctx.offsets[0];
        ops::apply_rope(q, offset0, rope_theta_);
        ops::apply_rope(k, offset0, rope_theta_);
        return;
    }
    for (int i = 0; i < static_cast<int>(ctx.sample_ids.size()); ++i) {
        int sample_offset = ctx.offsets[i];
        Tensor q_slice = slice_batch_4d(q, i);
        Tensor k_slice = slice_batch_4d(k, i);
        ops::apply_rope(q_slice, sample_offset, rope_theta_);
        ops::apply_rope(k_slice, sample_offset, rope_theta_);
        write_batch_4d(q, i, q_slice);
        write_batch_4d(k, i, k_slice);
    }
}

void SelfAttn::expand_kv_heads(Tensor& k, Tensor& v) {
    k.repeat(num_heads_ / num_heads_kv_, 1);  // [batch, num_head_kv, seq, head_dim]
    v.repeat(num_heads_ / num_heads_kv_, 1);  // [batch, num_head_kv, seq, head_dim]
}

void SelfAttn::apply_attention_masks(Tensor& scores, const ForwardContext& ctx) const {
    int total_len = scores.shape()[3];
    if (ctx.seq_len > 1) {
        apply_causal_mask(scores, ctx.seq_len, total_len);
    }
    if (!pad_lens_by_sample_.empty()) {
        apply_padding_mask(scores, ctx.sample_ids, pad_lens_by_sample_);
    }
}

void SelfAttn::init_kv_cache(int batch_size) {
    cache_k_by_sample_.assign(batch_size, Tensor());
    cache_v_by_sample_.assign(batch_size, Tensor());
    cache_len_by_sample_.assign(batch_size, 0);
}

void SelfAttn::clear_kv_cache(int sample_id) {
    if (sample_id < 0 || sample_id >= static_cast<int>(cache_len_by_sample_.size())) {
        return;
    }
    cache_k_by_sample_[sample_id] = Tensor();
    cache_v_by_sample_[sample_id] = Tensor();
    cache_len_by_sample_[sample_id] = 0;
}

void SelfAttn::append_kv_cache(const Tensor& k, const Tensor& v, const ForwardContext& ctx) {
    int batch = k.shape()[0];
    if (batch != static_cast<int>(ctx.sample_ids.size())) {
        throw std::invalid_argument("append_kv_cache: k batch size mismatch with sample_ids.");
    }
    int seq_len = k.shape()[2];
    for (int i = 0; i < batch; ++i) {
        int sample_id = ctx.sample_ids[i];
        ensure_cache_capacity(sample_id + 1);
        Tensor k_slice = slice_batch_4d(k, i);
        Tensor v_slice = slice_batch_4d(v, i);
        if (cache_len_by_sample_[sample_id] == 0) {
            cache_k_by_sample_[sample_id] = std::move(k_slice);
            cache_v_by_sample_[sample_id] = std::move(v_slice);
        } else {
            cache_k_by_sample_[sample_id] = ops::concat({cache_k_by_sample_[sample_id], k_slice}, 2);
            cache_v_by_sample_[sample_id] = ops::concat({cache_v_by_sample_[sample_id], v_slice}, 2);
        }
        cache_len_by_sample_[sample_id] += seq_len;
    }
}

Tensor SelfAttn::build_active_cache(const std::vector<Tensor>& cache_by_sample, const ForwardContext& ctx) const {
    if (ctx.sample_ids.size() == 1) {
        int sample_id = ctx.sample_ids[0];
        if (sample_id < 0 || sample_id >= static_cast<int>(cache_by_sample.size())) {
            throw std::out_of_range("build_active_cache: sample_id out of range.");
        }
        const Tensor& single = cache_by_sample[sample_id];
        return single;
    }
    int first_sample_id = ctx.sample_ids[0];
    if (first_sample_id < 0 || first_sample_id >= static_cast<int>(cache_by_sample.size())) {
        throw std::out_of_range("build_active_cache: sample_id out of range.");
    }

    const Tensor& first = cache_by_sample[first_sample_id];
    const auto& shape = first.shape();
    if (shape.size() != 4 || shape[0] != 1) {
        throw std::invalid_argument("build_active_cache expects per-sample cache with shape [1, heads, seq, dim].");
    }
    int batch = static_cast<int>(ctx.sample_ids.size());
    int num_heads = shape[1];
    int seq_len = shape[2];
    int head_dim = shape[3];
    int block = num_heads * seq_len * head_dim;

    Tensor output(batch * block, {batch, num_heads, seq_len, head_dim});
    auto& dst = output.data();
    for (int i = 0; i < batch; ++i) {
        int sample_id = ctx.sample_ids[i];
        if (sample_id < 0 || sample_id >= static_cast<int>(cache_by_sample.size())) {
            throw std::out_of_range("build_active_cache: sample_id out of range.");
        }
        const Tensor& src_tensor = cache_by_sample[sample_id];
        if (src_tensor.shape() != shape) {
            throw std::invalid_argument("build_active_cache: per-sample cache shape mismatch.");
        }
        const auto& src = src_tensor.data();
        int start = i * block;
        std::copy(src.begin(), src.end(), dst.begin() + start);
    }
    return output;
}

void SelfAttn::ensure_cache_capacity(int min_size) {
    if (min_size <= 0) {
        return;
    }
    if (static_cast<int>(cache_k_by_sample_.size()) >= min_size) {
        return;
    }
    cache_k_by_sample_.resize(min_size);
    cache_v_by_sample_.resize(min_size);
    cache_len_by_sample_.resize(min_size, 0);
}

void SelfAttn::reset_kv_cache() {
    cache_k_by_sample_.clear();
    cache_v_by_sample_.clear();
    cache_len_by_sample_.clear();
}

void SelfAttn::set_pad_lens(const std::vector<int>& pad_lens) {
    pad_lens_by_sample_ = pad_lens;
}

} // namespace easy_gpt
