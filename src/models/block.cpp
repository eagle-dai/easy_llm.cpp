#include "models/block.hpp"

#include <spdlog/spdlog.h>
#include "ops.hpp"

#include "config.hpp"
#include "models/layer_key_prefix.hpp"

namespace easy_llm {

Block::Block() {
}

Block::Block(const Config& config)
    : self_attn_(config) {
}

void Block::load_param(const LayerKeyPrefix& key_prefix, const std::string& key, ModelParam& model_param) {
    self_attn_.load_param(key_prefix, key, model_param);
    mlp_.load_param(key_prefix, key, model_param);
}

Tensor Block::forward(const Tensor& input, const std::vector<int>& sample_ids, const std::vector<int>* pos_offsets) {
    auto output_attn = self_attn_.forward(input, sample_ids, pos_offsets);
    ops::add_inplace(output_attn, input);
    auto output = mlp_.forward(output_attn);
    ops::add_inplace(output, output_attn);
    return output;
}

void Block::init_kv_cache(int batch_size) {
    self_attn_.init_kv_cache(batch_size);
}

void Block::clear_kv_cache(int sample_id) {
    self_attn_.clear_kv_cache(sample_id);
}

void Block::reset_kv_cache() {
    self_attn_.reset_kv_cache();
}

void Block::set_pad_lens(const std::vector<int>& pad_lens) {
    self_attn_.set_pad_lens(pad_lens);
}

void Block::set_self_attn_cuda_enabled(bool enabled) {
    self_attn_.set_cuda_enabled(enabled);
}

} // namespace easy_llm
