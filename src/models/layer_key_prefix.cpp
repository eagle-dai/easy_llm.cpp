#include "models/layer_key_prefix.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include <spdlog/spdlog.h>

#include "config.hpp"

namespace easy_llm {

namespace {

std::string to_lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

bool is_qwen2_family(const Config& config) {
    std::string architecture = to_lower(config.architecture);
    std::string model_type = to_lower(config.model_type);
    return architecture.find("qwen2") != std::string::npos || model_type == "qwen2";
}

class Qwen2_5_LayerKeyPrefix final : public LayerKeyPrefix {
public:
    std::string layer(int layer_idx) const override {
        return std::string(kModelLayers) + std::to_string(layer_idx);
    }

    std::string model_norm() const override {
        return kModelNorm;
    }

    std::string self_attn_q_proj(const std::string& layer_key) const override {
        return layer_key + kSelfAttnQProj;
    }

    std::string self_attn_k_proj(const std::string& layer_key) const override {
        return layer_key + kSelfAttnKProj;
    }

    std::string self_attn_v_proj(const std::string& layer_key) const override {
        return layer_key + kSelfAttnVProj;
    }

    std::string self_attn_o_proj(const std::string& layer_key) const override {
        return layer_key + kSelfAttnOProj;
    }

    std::string input_layer_norm(const std::string& layer_key) const override {
        return layer_key + kInputLayerNorm;
    }

    std::string mlp_down_proj(const std::string& layer_key) const override {
        return layer_key + kMlpDownProj;
    }

    std::string mlp_gate_proj(const std::string& layer_key) const override {
        return layer_key + kMlpGateProj;
    }

    std::string mlp_up_proj(const std::string& layer_key) const override {
        return layer_key + kMlpUpProj;
    }

    std::string post_attention_layer_norm(const std::string& layer_key) const override {
        return layer_key + kPostAttentionLayerNorm;
    }

private:
    static constexpr const char* kModelLayers = "model.layers.";
    static constexpr const char* kModelNorm = "model.norm";
    static constexpr const char* kSelfAttnQProj = ".self_attn.q_proj";
    static constexpr const char* kSelfAttnKProj = ".self_attn.k_proj";
    static constexpr const char* kSelfAttnVProj = ".self_attn.v_proj";
    static constexpr const char* kSelfAttnOProj = ".self_attn.o_proj";
    static constexpr const char* kInputLayerNorm = ".input_layernorm";
    static constexpr const char* kMlpDownProj = ".mlp.down_proj";
    static constexpr const char* kMlpGateProj = ".mlp.gate_proj";
    static constexpr const char* kMlpUpProj = ".mlp.up_proj";
    static constexpr const char* kPostAttentionLayerNorm = ".post_attention_layernorm";
};

} // namespace

std::unique_ptr<LayerKeyPrefix> create_layer_key_prefix(const Config& config) {
    if (is_qwen2_family(config)) {
        spdlog::info("LayerKeyPrefix selected: Qwen2 family");
        return std::make_unique<Qwen2_5_LayerKeyPrefix>();
    }
    spdlog::warn(
        "Unknown model architecture/model_type (architecture='{}', model_type='{}'), fallback to Qwen2 key prefix rules.",
        config.architecture,
        config.model_type
    );
    return std::make_unique<Qwen2_5_LayerKeyPrefix>();
}

} // namespace easy_llm
