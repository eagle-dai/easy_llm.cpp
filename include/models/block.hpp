#ifndef EASY_LLM_MODELS_BLOCK_HPP
#define EASY_LLM_MODELS_BLOCK_HPP

#include <vector>
#include <memory>
#include <string>

#include "tensor.hpp"
#include "config.hpp"
#include "models/loader.hpp"
#include "models/self_attn.hpp"
#include "models/mlp.hpp"

namespace easy_llm {

class LayerKeyPrefix;

class Block {
public:
    Block();
    explicit Block(const Config& config);
    Block(Block&& other) noexcept = default;
    Block& operator=(Block&& other) noexcept = default;

    Tensor forward(const Tensor& input, const std::vector<int>& sample_ids, const std::vector<int>* pos_offsets = nullptr);
    void load_param(const LayerKeyPrefix& key_prefix, const std::string& key, ModelParam& model_param);
    void init_kv_cache(int batch_size);
    void clear_kv_cache(int sample_id);
    void reset_kv_cache();
    void set_pad_lens(const std::vector<int>& pad_lens);

private:
    SelfAttn self_attn_;
    MLP mlp_;
};

} // namespace easy_llm

#endif // EASY_LLM_MODELS_BLOCK_HPP
