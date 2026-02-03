#ifndef EASY_LLM_MODELS_GPT_MODEL_HPP
#define EASY_LLM_MODELS_GPT_MODEL_HPP

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <random>

#include "tensor.hpp"
#include "config.hpp"
#include "data_manager.hpp"
#include "sampler.hpp"
#include "models/loader.hpp"
#include "models/embedding.hpp"
#include "models/block.hpp"
#include "models/norm.hpp"
#include "models/linear.hpp"

namespace easy_llm {

class GptModel {
public:
    static std::unique_ptr<GptModel> create(const Config& config, DataManager& data_manager, ModelParam& model_param);
    GptModel(const Config& config, DataManager& data_manager);
    ~GptModel();

    std::string forward(const std::vector<std::vector<int>>& input);

private:
    struct GenerationContext;

    void init_from_config();
    enum class ForwardLogMode {
        None,
        PrefillStep0
    };
    Tensor forward_logits(GenerationContext& ctx,
                          const std::vector<std::vector<int>>& input_tokens,
                          const std::vector<int>* pos_offsets,
                          ForwardLogMode log_mode);
    void prefill(GenerationContext& ctx, const std::vector<std::vector<int>>& input);
    std::vector<int> sample_and_record_last_token(GenerationContext& ctx,
                                                  const std::pair<std::vector<float>, std::vector<int>>& output_info,
                                                  int step);
    void decode(GenerationContext& ctx);
    void apply_eos_filter_and_update_state(GenerationContext& ctx);
    void filter_eos_samples(GenerationContext& ctx);
    
    void load_param(ModelParam& model_param);
    void init_kv_cache(int batch_size);
    void clear_kv_cache(int sample_id);
    void reset_kv_cache();

    const Config& config_;
    DataManager& data_manager_;
    std::unique_ptr<Embedding> embedding_{std::make_unique<Embedding>()};
    int num_blocks_{0};
    std::vector<std::unique_ptr<Block>> blocks_;
    RMSNorm norm_{};
    std::unique_ptr<Linear> out_linear_{std::make_unique<Linear>()};
    std::mt19937 rng_;
    std::unique_ptr<Sampler> sampler_;

};

} // namespace easy_llm

#endif // EASY_LLM_MODELS_GPT_MODEL_HPP
