#include "models/gpt_model.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <numeric>
#include <random>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include "ops.hpp"
#include "sampler.hpp"
#include "models/layer_key_prefix.hpp"
#include "models/generation_invariants.hpp"

namespace easy_llm {

using std::vector;
using std::string;
using std::make_unique;

struct GptModel::GenerationContext {
    int batch_size{0};
    int input_seq_len{0};
    int max_steps{0};
    int step{0};
    bool eos_enabled{false};
    std::vector<int> sample_ids;
    std::vector<int> next_generated_tokens;
    std::vector<int> pad_lens;
    std::vector<int> pos_lens_by_sample;

    std::vector<int> build_prefill_pos_offsets() const {
        return generation::build_prefill_pos_offsets(sample_ids, pad_lens);
    }

    std::vector<int> build_decode_pos_offsets() const {
        return generation::build_decode_pos_offsets(sample_ids, pos_lens_by_sample);
    }
};

namespace {

bool parse_env_bool(const char* value) {
    if (value == nullptr) {
        return false;
    }
    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text == "1" || text == "true" || text == "yes" || text == "on";
}

std::pair<std::vector<float>, std::vector<int>> build_token_info(const std::vector<float>& probs,
                                                                 int batch_size,
                                                                 int seq_len,
                                                                 int vocab_size,
                                                                 int token_index) {
    if (token_index < 0 || token_index >= seq_len) {
        throw std::invalid_argument("build_token_info: token_index out of range.");
    }
    std::vector<float> token_probs;
    token_probs.reserve(batch_size * vocab_size);
    for (int b = 0; b < batch_size; ++b) {
        int offset = (b * seq_len + token_index) * vocab_size;
        token_probs.insert(token_probs.end(), probs.begin() + offset, probs.begin() + offset + vocab_size);
    }
    std::vector<int> token_shape = {batch_size, 1, vocab_size};
    return {std::move(token_probs), std::move(token_shape)};
}

void log_prefill_topk_last_token(const std::vector<float>& probs, int input_seq_len, int vocab_size) {
    std::vector<std::pair<float, int>> token_probs;
    token_probs.reserve(vocab_size);
    int offset = (input_seq_len - 1) * vocab_size;
    for (int i = 0; i < vocab_size; ++i) {
        token_probs.push_back({probs[offset + i], i});
    }
    std::sort(token_probs.rbegin(), token_probs.rend());
    spdlog::trace("Prefill End Top 5 (Next Token Prediction):");
    for (int k = 0; k < 5; ++k) {
        spdlog::trace("  Token ID {}: Prob {:.4f}", token_probs[k].second, token_probs[k].first);
    }
}

std::vector<std::vector<int>> build_decode_step_tokens(const std::vector<int>& next_generated_tokens) {
    std::vector<std::vector<int>> input_single_token;
    input_single_token.reserve(next_generated_tokens.size());
    for (int token : next_generated_tokens) {
        input_single_token.push_back({token});
    }
    return input_single_token;
}

void log_decode_topk(const std::vector<float>& probs, int vocab_size, int step) {
    std::vector<std::pair<float, int>> token_probs;
    token_probs.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        token_probs.push_back({probs[i], i});
    }
    std::sort(token_probs.rbegin(), token_probs.rend());
    spdlog::trace("Step {} Top 5:", step);
    for (int k = 0; k < 5; ++k) {
        spdlog::trace("  Token ID {}: Prob {:.4f}", token_probs[k].second, token_probs[k].first);
    }
}

} // namespace

GptModel::~GptModel() = default;

std::unique_ptr<GptModel> GptModel::create(const Config& config, DataManager& data_manager, ModelParam& model_param) {
    auto model = make_unique<GptModel>(config, data_manager);
    model->init_from_config();
    model->load_param(model_param);
    spdlog::info("GptModel created");
    return model;
}

GptModel::GptModel(const Config& config, DataManager& data_manager)
    : config_(config),
      data_manager_(data_manager),
      layer_key_prefix_(create_layer_key_prefix(config)),
      rng_(static_cast<uint32_t>(config.seed)),
      sampler_(std::make_unique<TopKTopPSampler>(config.temperature, config.top_p, config.top_k, config.use_greedy, rng_)) {
}

void GptModel::init_from_config() {
    if (config_.num_layers <= 0) {
        spdlog::error("GptModel config missing num_layers");
        throw std::invalid_argument("GptModel config missing num_layers");
    }
    num_blocks_ = config_.num_layers;
    blocks_.clear();
    blocks_.reserve(num_blocks_);
    for (int i = 0; i < num_blocks_; ++i) {
        blocks_.emplace_back(make_unique<Block>(config_));
    }
}

void GptModel::load_param(ModelParam& model_param) {
    embedding_->load_param(model_param);
    for (int i = 0; i < num_blocks_; ++i) {
        blocks_[i]->load_param(*layer_key_prefix_, layer_key_prefix_->layer(i), model_param);
    }
    norm_.load_param(layer_key_prefix_->model_norm(), model_param);
}

string GptModel::forward(const vector<vector<int>>& input) {
    spdlog::info("GptModel forward (Parallel Prefill)");
    string result;

    if (input.empty()) {
        throw std::invalid_argument("forward: input batch is empty.");
    }
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i].empty()) {
            throw std::invalid_argument("forward: input sample is empty.");
        }
    }

    GenerationContext ctx;
    ctx.batch_size = static_cast<int>(input.size());
    ctx.max_steps = config_.max_steps;
    ctx.eos_enabled = config_.eos_token_id >= 0;

    init_kv_cache(ctx.batch_size);
    ctx.sample_ids.resize(ctx.batch_size);
    std::iota(ctx.sample_ids.begin(), ctx.sample_ids.end(), 0);
    ctx.pad_lens = data_manager_.get_pad_lens();
    ctx.pos_lens_by_sample = data_manager_.get_seq_lens();
    for (auto& block : blocks_) {
        block->set_pad_lens(ctx.pad_lens);
    }

    prefill(ctx, input);
    filter_eos_samples(ctx);
    spdlog::info("step == {} max_steps == {}", ctx.step, ctx.max_steps);
    if (!ctx.sample_ids.empty()) {
        decode(ctx);
    }
    reset_kv_cache();
    return result;
}

void GptModel::prefill(GenerationContext& ctx, const vector<vector<int>>& input) {
    if (ctx.batch_size != static_cast<int>(input.size())) {
        throw std::invalid_argument("prefill: input size must match batch size.");
    }
    if (static_cast<int>(ctx.sample_ids.size()) != ctx.batch_size) {
        throw std::invalid_argument("prefill: sample_ids size must match batch size.");
    }
    ctx.input_seq_len = 0;
    if (ctx.batch_size > 0) ctx.input_seq_len = input[0].size();
    spdlog::info("Starting Prefill with seq_len={}", ctx.input_seq_len);
    std::vector<int> pos_offsets = ctx.build_prefill_pos_offsets();
    Tensor logits = forward_logits(ctx, input, &pos_offsets, ForwardLogMode::PrefillStep0);
    auto output_info = ops::softmax(logits);
    auto& [probs, shape] = output_info;
    int vocab_size = shape.back();
    sampler_->set_params(config_.temperature, config_.top_p, config_.top_k, config_.use_greedy);
    if (spdlog::should_log(spdlog::level::trace)) {
        log_prefill_topk_last_token(probs, ctx.input_seq_len, vocab_size);
    }
    ctx.next_generated_tokens = sample_and_record_last_token(ctx, output_info, ctx.input_seq_len - 1);
    ctx.step = ctx.input_seq_len;
}

std::vector<int> GptModel::sample_and_record_last_token(GenerationContext& ctx,
                                                        const std::pair<std::vector<float>, std::vector<int>>& output_info,
                                                        int step) {
    const auto& shape = output_info.second;
    if (shape.size() < 3) {
        throw std::invalid_argument("sample_and_record_last_token: output shape must be [batch, seq, vocab].");
    }
    int seq_len = shape[1];
    int token_index = seq_len - 1;
    std::vector<int> next_generated_tokens = sample_tokens_from_output(output_info, token_index);
    int batch_size = shape[0];
    int vocab_size = shape[2];
    if (seq_len == 1) {
        data_manager_.add_output_token(output_info, step, ctx.sample_ids, next_generated_tokens);
    } else {
        const auto& probs = output_info.first;
        auto last_token_info = build_token_info(probs, batch_size, seq_len, vocab_size, token_index);
        data_manager_.add_output_token(last_token_info, step, ctx.sample_ids, next_generated_tokens);
    }
    return next_generated_tokens;
}

std::vector<int> GptModel::sample_tokens_from_output(
    const std::pair<std::vector<float>, std::vector<int>>& output_info,
    int token_index) {
    const auto& [probs, shape] = output_info;
    if (shape.size() < 3) {
        throw std::invalid_argument("sample_tokens_from_output: output shape must be [batch, seq, vocab].");
    }
    int batch_size = shape[0];
    int seq_len = shape[1];
    int vocab_size = shape[2];
    if (seq_len <= 0) {
        throw std::invalid_argument("sample_tokens_from_output: seq_len must be positive.");
    }
    if (token_index < 0 || token_index >= seq_len) {
        throw std::invalid_argument("sample_tokens_from_output: token_index out of range.");
    }
    std::vector<int> next_generated_tokens(batch_size, 0);
    for (int b = 0; b < batch_size; ++b) {
        int offset = (b * seq_len + token_index) * vocab_size;
        next_generated_tokens[b] = sampler_->sample_from_probs(probs.data() + offset, vocab_size);
    }
    return next_generated_tokens;
}

void GptModel::decode(GenerationContext& ctx) {
    if (ctx.next_generated_tokens.size() != ctx.sample_ids.size()) {
        throw std::invalid_argument("decode: next_generated_tokens size must match sample_ids size.");
    }
    sampler_->set_params(config_.temperature, config_.top_p, config_.top_k, config_.use_greedy);
    auto decode_start = std::chrono::steady_clock::now();
    int decode_steps = 0;
    while (ctx.step < ctx.max_steps && !ctx.sample_ids.empty()) {
        spdlog::info("----------Step {} : remaining batch size {}----------", ctx.step, ctx.sample_ids.size());
        std::vector<std::vector<int>> input_single_token = build_decode_step_tokens(ctx.next_generated_tokens);
        std::vector<int> pos_offsets = ctx.build_decode_pos_offsets();
        Tensor output = forward_logits(ctx, input_single_token, &pos_offsets, ForwardLogMode::None);
        auto output_info = ops::softmax(output);
        auto& [probs, shape] = output_info;
        int vocab_size = shape.back();
        if (spdlog::should_log(spdlog::level::trace)) {
            log_decode_topk(probs, vocab_size, ctx.step);
        }
        ctx.next_generated_tokens = sample_and_record_last_token(ctx, output_info, ctx.step);
        apply_eos_filter_and_update_state(ctx);
        decode_steps += 1;
    }
    if (decode_steps > 0) {
        auto decode_end = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count() / 1000.0;
        double avg_ms = total_ms / decode_steps;
        spdlog::info("Decode finished: steps={}, avg_ms_per_step={:.3f}, total_ms={:.3f}", decode_steps, avg_ms, total_ms);
    }
}

void GptModel::apply_eos_filter_and_update_state(GenerationContext& ctx) {
    generation::increment_pos_lens(ctx.sample_ids, ctx.pos_lens_by_sample);
    ctx.step += 1;
    filter_eos_samples(ctx);
}

void GptModel::filter_eos_samples(GenerationContext& ctx) {
    if (!ctx.eos_enabled) {
        return;
    }
    generation::EosFilterResult filtered =
        generation::filter_eos_samples(ctx.sample_ids, ctx.next_generated_tokens, config_.eos_token_id);
    for (int sample_id : filtered.cleared_sample_ids) {
        clear_kv_cache(sample_id);
    }
    ctx.sample_ids.swap(filtered.sample_ids);
    ctx.next_generated_tokens.swap(filtered.next_tokens);
}

Tensor GptModel::forward_logits(GenerationContext& ctx,
                                const std::vector<std::vector<int>>& input_tokens,
                                const std::vector<int>* pos_offsets,
                                ForwardLogMode log_mode) {
    Tensor block_output = embedding_->forward(input_tokens);
    if (log_mode == ForwardLogMode::PrefillStep0) {
        spdlog::info("Step 0 Embedding Output[:10]: {}", fmt::join(vector<float>(block_output.data().begin(), block_output.data().begin() + 10), ", "));
    }
    for (int i = 0; i < num_blocks_; ++i) {
        block_output = blocks_[i]->forward(block_output, ctx.sample_ids, pos_offsets);
        if (log_mode == ForwardLogMode::PrefillStep0 && i == 0) {
            spdlog::info("Step 0 Block 0 Output[:10]: {}", fmt::join(vector<float>(block_output.data().begin(), block_output.data().begin() + 10), ", "));
        }
    }
    auto output = norm_.forward(block_output);
    if (log_mode == ForwardLogMode::PrefillStep0) {
        spdlog::info("Prefill output shape: {}", fmt::join(output.shape(), ", "));
    }
    output = embedding_->forward(output);
    if (log_mode == ForwardLogMode::PrefillStep0) {
        spdlog::info("Step 0 Logits[:10]: {}", fmt::join(vector<float>(output.data().begin(), output.data().begin() + 10), ", "));
    }
    return output;
}

void GptModel::init_kv_cache(int batch_size) {
    for (auto& block : blocks_) {
        block->init_kv_cache(batch_size);
    }
}

void GptModel::clear_kv_cache(int sample_id) {
    for (auto& block : blocks_) {
        block->clear_kv_cache(sample_id);
    }
}

void GptModel::reset_kv_cache() {
    for (auto& block : blocks_) {
        block->reset_kv_cache();
    }
}

void GptModel::start_continuous(int initial_cache_slots) {
    if (initial_cache_slots < 0) {
        throw std::invalid_argument("start_continuous: initial_cache_slots must be >= 0.");
    }
    init_kv_cache(initial_cache_slots);
    const bool disable_cuda_self_attn = parse_env_bool(std::getenv("EASY_LLM_DISABLE_CONTINUOUS_CUDA_SELF_ATTN"));
    for (auto& block : blocks_) {
        block->set_self_attn_cuda_enabled(!disable_cuda_self_attn);
    }
    if (disable_cuda_self_attn) {
        spdlog::warn("Continuous mode enabled with CUDA SelfAttn disabled by env EASY_LLM_DISABLE_CONTINUOUS_CUDA_SELF_ATTN.");
    } else {
        spdlog::info("Continuous mode enabled: CUDA SelfAttn remains active.");
    }
}

void GptModel::set_continuous_pad_lens(const std::vector<int>& pad_lens) {
    for (auto& block : blocks_) {
        block->set_pad_lens(pad_lens);
    }
}

std::vector<int> GptModel::sample_prefill_continuous(
    const std::vector<int>& sample_ids,
    const std::vector<std::vector<int>>& input_tokens,
    const std::vector<int>& pos_offsets) {
    if (sample_ids.empty()) {
        throw std::invalid_argument("sample_prefill_continuous: sample_ids is empty.");
    }
    if (sample_ids.size() != input_tokens.size() || sample_ids.size() != pos_offsets.size()) {
        throw std::invalid_argument("sample_prefill_continuous: size mismatch.");
    }
    GenerationContext ctx;
    ctx.sample_ids = sample_ids;
    sampler_->set_params(config_.temperature, config_.top_p, config_.top_k, config_.use_greedy);
    Tensor logits = forward_logits(ctx, input_tokens, &pos_offsets, ForwardLogMode::None);
    auto output_info = ops::softmax(logits);
    const auto& shape = output_info.second;
    if (shape.size() < 3 || shape[1] <= 0) {
        throw std::invalid_argument("sample_prefill_continuous: invalid output shape.");
    }
    const int token_index = shape[1] - 1;
    return sample_tokens_from_output(output_info, token_index);
}

std::vector<int> GptModel::sample_decode_continuous(
    const std::vector<int>& sample_ids,
    const std::vector<std::vector<int>>& input_tokens,
    const std::vector<int>& pos_offsets) {
    if (sample_ids.empty()) {
        throw std::invalid_argument("sample_decode_continuous: sample_ids is empty.");
    }
    if (sample_ids.size() != input_tokens.size() || sample_ids.size() != pos_offsets.size()) {
        throw std::invalid_argument("sample_decode_continuous: size mismatch.");
    }
    for (const auto& token_ids : input_tokens) {
        if (token_ids.size() != 1) {
            throw std::invalid_argument("sample_decode_continuous: each sample must provide exactly one token.");
        }
    }
    GenerationContext ctx;
    ctx.sample_ids = sample_ids;
    sampler_->set_params(config_.temperature, config_.top_p, config_.top_k, config_.use_greedy);
    Tensor logits = forward_logits(ctx, input_tokens, &pos_offsets, ForwardLogMode::None);
    auto output_info = ops::softmax(logits);
    return sample_tokens_from_output(output_info, 0);
}

void GptModel::clear_continuous_sample(int sample_id) {
    clear_kv_cache(sample_id);
}

void GptModel::reset_continuous() {
    reset_kv_cache();
}

} // namespace easy_llm
