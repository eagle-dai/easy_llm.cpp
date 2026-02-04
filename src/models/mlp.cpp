#include "models/mlp.hpp"

#include <spdlog/spdlog.h>
#include "ops.hpp"
#include "models/layer_key_prefix.hpp"

namespace easy_llm {

MLP::MLP() {}

MLP::MLP(int hidden_dim)
    : up_proj_(4 * hidden_dim, hidden_dim),
      down_proj_(hidden_dim, 4 * hidden_dim),
      hidden_dim_(hidden_dim) {
}

void MLP::load_param(const LayerKeyPrefix& key_prefix, const std::string& key, ModelParam& model_param) {
    down_proj_.load_param(key_prefix.mlp_down_proj(key), model_param);
    gate_proj_.load_param(key_prefix.mlp_gate_proj(key), model_param);
    up_proj_.load_param(key_prefix.mlp_up_proj(key), model_param);
    norm_.load_param(key_prefix.post_attention_layer_norm(key), model_param);
}

Tensor MLP::forward(const Tensor& input) const {
    auto input_norm = norm_.forward(input);
    auto hidden = up_proj_.forward(input_norm);
    auto gate = gate_proj_.forward(input_norm);
    gate.silu();
    hidden = ops::multiply(hidden, gate);
    auto result = down_proj_.forward(hidden);
    return result;
}

} // namespace easy_llm
