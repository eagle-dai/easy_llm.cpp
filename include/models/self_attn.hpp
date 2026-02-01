#ifndef EASY_GPT_MODELS_SELF_ATTN_HPP
#define EASY_GPT_MODELS_SELF_ATTN_HPP

#include <vector>
#include <memory>
#include <string>

#include "tensor.hpp"
#include "config.hpp"
#include "models/loader.hpp"
#include "models/linear.hpp"
#include "models/norm.hpp"

namespace easy_gpt {

class SelfAttn {
public:
    SelfAttn();
    explicit SelfAttn(const Config& config);
    SelfAttn(int hidden_dim, int num_heads);

    Tensor forward(const Tensor& input, const std::vector<int>& sample_ids, const std::vector<int>* pos_offsets = nullptr);
    void load_param(const std::string& key, ModelParam& model_param);
    void init_kv_cache(int batch_size);
    void clear_kv_cache(int sample_id);
    void reset_kv_cache();
    void set_pad_lens(const std::vector<int>& pad_lens);

private:
    struct ForwardContext;

    void validate_forward_inputs(const ForwardContext& ctx) const;
    void compute_offsets(ForwardContext& ctx) const;
    void apply_rope_offsets(Tensor& q, Tensor& k, const ForwardContext& ctx);
    void expand_kv_heads(Tensor& k, Tensor& v);
    void apply_attention_masks(Tensor& scores, const ForwardContext& ctx) const;
    void append_kv_cache(const Tensor& k, const Tensor& v, const ForwardContext& ctx);
    Tensor build_active_cache(const std::vector<Tensor>& cache_by_sample, const ForwardContext& ctx) const;
    void ensure_cache_capacity(int min_size);

    int hidden_dim_;
    int num_heads_{0};
    int num_heads_kv_{0};
    int head_dim_;
    float rope_theta_{0.0f};
    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear o_proj_;
    RMSNorm norm_{};
    // Per-sample KV cache: each entry is [1, num_heads, cache_len, head_dim]
    std::vector<Tensor> cache_k_by_sample_;
    std::vector<Tensor> cache_v_by_sample_;
    std::vector<int> cache_len_by_sample_;
    std::vector<int> pad_lens_by_sample_;

};

} // namespace easy_gpt

#endif // EASY_GPT_MODELS_SELF_ATTN_HPP
