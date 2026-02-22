#include "continuous_batch_server.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>

#include "spdlog/spdlog.h"

namespace easy_llm {

double ContinuousBatchServer::elapsed_ms(TimePoint begin, TimePoint end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
}

ContinuousBatchServer::ContinuousBatchServer(const Config& config,
                                             GptModel& model,
                                             Tokenizer& tokenizer,
                                             const ContinuousBatchOptions& options)
    : config_(config),
      model_(model),
      tokenizer_(tokenizer),
      options_(options),
      pad_id_(tokenizer.get_pad_id()),
      eos_enabled_(config.eos_token_id >= 0) {
    if (options_.max_active_requests <= 0) {
        throw std::invalid_argument("ContinuousBatchServer: max_active_requests must be > 0.");
    }
    if (options_.prefill_batch_size <= 0) {
        throw std::invalid_argument("ContinuousBatchServer: prefill_batch_size must be > 0.");
    }
    if (options_.idle_sleep_ms < 0) {
        throw std::invalid_argument("ContinuousBatchServer: idle_sleep_ms must be >= 0.");
    }
    if (options_.stats_log_interval_ms < 0) {
        throw std::invalid_argument("ContinuousBatchServer: stats_log_interval_ms must be >= 0.");
    }
    if (pad_id_ < 0) {
        throw std::invalid_argument("ContinuousBatchServer: tokenizer pad_id is required.");
    }

    free_slots_.reserve(options_.max_active_requests);
    for (int i = options_.max_active_requests - 1; i >= 0; --i) {
        free_slots_.push_back(i);
    }
    pad_lens_by_slot_.assign(options_.max_active_requests, 0);
    const auto now = Clock::now();
    stats_started_at_ = now;
    last_stats_log_at_ = now;
    model_.start_continuous(options_.max_active_requests);
    model_.set_continuous_pad_lens(pad_lens_by_slot_);
}

int ContinuousBatchServer::submit_prompt(std::string prompt_text) {
    if (prompt_text.empty()) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(pending_mu_);
    const int request_id = next_request_id_++;
    pending_prompts_.push_back(PendingPrompt{request_id, std::move(prompt_text), Clock::now()});
    return request_id;
}

void ContinuousBatchServer::mark_input_closed() {
    std::lock_guard<std::mutex> lock(pending_mu_);
    input_closed_ = true;
}

void ContinuousBatchServer::run() {
    while (true) {
        bool progressed = false;
        progressed = admit_prefill_round() || progressed;
        progressed = decode_round() || progressed;
        maybe_log_runtime_stats();
        if (is_done()) {
            break;
        }
        if (!progressed) {
            std::this_thread::sleep_for(std::chrono::milliseconds(options_.idle_sleep_ms));
        }
    }
    log_final_runtime_stats();
    model_.reset_continuous();
}

bool ContinuousBatchServer::admit_prefill_round() {
    const int admission_limit = std::min(options_.prefill_batch_size,
                                         static_cast<int>(free_slots_.size()));
    if (admission_limit <= 0) {
        return false;
    }
    auto pending = pop_admission_candidates(admission_limit);
    if (pending.empty()) {
        return false;
    }
    auto prepared = prepare_admissions(std::move(pending));
    if (prepared.empty()) {
        return false;
    }
    const auto round_begin = Clock::now();

    std::vector<int> sample_ids;
    std::vector<int> pos_offsets;
    std::vector<std::vector<int>> padded_inputs;
    sample_ids.reserve(prepared.size());
    pos_offsets.reserve(prepared.size());
    padded_inputs.reserve(prepared.size());

    for (const auto& req : prepared) {
        pad_lens_by_slot_[req.slot_id] = req.pad_len;
        sample_ids.push_back(req.slot_id);
        pos_offsets.push_back(-req.pad_len);
        auto padded = req.token_ids;
        if (req.pad_len > 0) {
            padded.insert(padded.begin(), req.pad_len, pad_id_);
        }
        padded_inputs.push_back(std::move(padded));
    }
    model_.set_continuous_pad_lens(pad_lens_by_slot_);

    std::vector<int> sampled = model_.sample_prefill_continuous(sample_ids, padded_inputs, pos_offsets);
    if (sampled.size() != prepared.size()) {
        throw std::runtime_error("admit_prefill_round: sampled token size mismatch.");
    }
    const auto round_end = Clock::now();
    stats_.prefill_rounds += 1;
    stats_.prefill_requests += prepared.size();
    stats_.prefill_ms_total += elapsed_ms(round_begin, round_end);

    for (size_t i = 0; i < prepared.size(); ++i) {
        ActiveRequest request;
        request.request_id = prepared[i].request_id;
        request.slot_id = prepared[i].slot_id;
        request.pos_len = prepared[i].pos_len;
        request.max_generate_tokens = prepared[i].max_generate_tokens;
        request.next_token = sampled[i];
        request.submit_time = prepared[i].submit_time;
        request.admit_time = prepared[i].admit_time;
        if (sampled[i] != pad_id_) {
            request.generated_token_ids.push_back(sampled[i]);
            stats_.emitted_tokens_total += 1;
        }
        const bool hit_eos = eos_enabled_ && sampled[i] == config_.eos_token_id;
        const bool hit_limit = static_cast<int>(request.generated_token_ids.size()) >= request.max_generate_tokens;
        if (hit_eos || hit_limit) {
            finish_request(std::move(request), hit_eos, hit_limit);
            continue;
        }
        active_requests_.push_back(std::move(request));
    }
    spdlog::trace("[serve][prefill] admitted={}, round_ms={:.3f}, active_now={}",
                  prepared.size(),
                  elapsed_ms(round_begin, round_end),
                  active_requests_.size());
    return true;
}

bool ContinuousBatchServer::decode_round() {
    if (active_requests_.empty()) {
        return false;
    }
    const auto round_begin = Clock::now();
    std::vector<int> sample_ids;
    std::vector<int> pos_offsets;
    std::vector<std::vector<int>> input_tokens;
    sample_ids.reserve(active_requests_.size());
    pos_offsets.reserve(active_requests_.size());
    input_tokens.reserve(active_requests_.size());
    for (const auto& req : active_requests_) {
        sample_ids.push_back(req.slot_id);
        pos_offsets.push_back(req.pos_len);
        input_tokens.push_back({req.next_token});
    }
    std::vector<int> sampled = model_.sample_decode_continuous(sample_ids, input_tokens, pos_offsets);
    if (sampled.size() != active_requests_.size()) {
        throw std::runtime_error("decode_round: sampled token size mismatch.");
    }

    std::vector<ActiveRequest> next_active;
    next_active.reserve(active_requests_.size());
    for (size_t i = 0; i < active_requests_.size(); ++i) {
        ActiveRequest req = std::move(active_requests_[i]);
        const int token_id = sampled[i];
        req.pos_len += 1;
        req.decode_steps += 1;
        req.next_token = token_id;
        if (token_id != pad_id_) {
            req.generated_token_ids.push_back(token_id);
            stats_.emitted_tokens_total += 1;
        }
        const bool hit_eos = eos_enabled_ && token_id == config_.eos_token_id;
        const bool hit_limit = static_cast<int>(req.generated_token_ids.size()) >= req.max_generate_tokens;
        if (hit_eos || hit_limit) {
            finish_request(std::move(req), hit_eos, hit_limit);
            continue;
        }
        next_active.push_back(std::move(req));
    }
    const auto round_end = Clock::now();
    active_requests_.swap(next_active);
    stats_.decode_rounds += 1;
    stats_.decode_tokens_total += sampled.size();
    stats_.decode_ms_total += elapsed_ms(round_begin, round_end);
    spdlog::trace("[serve][decode] active_before={}, sampled_tokens={}, round_ms={:.3f}, active_after={}",
                  sample_ids.size(),
                  sampled.size(),
                  elapsed_ms(round_begin, round_end),
                  active_requests_.size());
    return true;
}

bool ContinuousBatchServer::is_done() const {
    if (!active_requests_.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(pending_mu_);
    return input_closed_ && pending_prompts_.empty();
}

void ContinuousBatchServer::finish_request(ActiveRequest&& request, bool hit_eos, bool hit_limit) {
    const auto now = Clock::now();
    const double queue_ms = elapsed_ms(request.submit_time, request.admit_time);
    const double service_ms = elapsed_ms(request.admit_time, now);
    const double total_ms = elapsed_ms(request.submit_time, now);
    const int output_tokens = static_cast<int>(request.generated_token_ids.size());
    const double tokens_per_sec = (service_ms > 0.0)
        ? (static_cast<double>(output_tokens) * 1000.0 / service_ms)
        : 0.0;

    stats_.completed_requests += 1;
    stats_.request_queue_ms += queue_ms;
    stats_.request_service_ms += service_ms;
    stats_.request_total_ms += total_ms;
    if (hit_eos) {
        stats_.finished_by_eos += 1;
    }
    if (hit_limit) {
        stats_.finished_by_limit += 1;
    }

    model_.clear_continuous_sample(request.slot_id);
    if (request.slot_id >= 0 && request.slot_id < static_cast<int>(pad_lens_by_slot_.size())) {
        pad_lens_by_slot_[request.slot_id] = 0;
        free_slots_.push_back(request.slot_id);
    }
    model_.set_continuous_pad_lens(pad_lens_by_slot_);
    spdlog::info("[serve][request {}] done reason={} tokens={} queue_ms={:.3f} service_ms={:.3f} total_ms={:.3f} tok_per_s={:.3f}",
                 request.request_id,
                 hit_eos ? "eos" : (hit_limit ? "max_tokens" : "other"),
                 output_tokens,
                 queue_ms,
                 service_ms,
                 total_ms,
                 tokens_per_sec);
    std::string text = tokenizer_.decode(request.generated_token_ids);
    std::cout << "[request " << request.request_id << "] " << text << "\n";
    std::cout.flush();
}

std::vector<ContinuousBatchServer::PendingPrompt> ContinuousBatchServer::pop_admission_candidates(int limit) {
    std::lock_guard<std::mutex> lock(pending_mu_);
    std::vector<PendingPrompt> out;
    const int count = std::min(limit, static_cast<int>(pending_prompts_.size()));
    out.reserve(count);
    for (int i = 0; i < count; ++i) {
        out.push_back(std::move(pending_prompts_.front()));
        pending_prompts_.pop_front();
    }
    return out;
}

std::vector<ContinuousBatchServer::PreparedAdmission> ContinuousBatchServer::prepare_admissions(
    std::vector<PendingPrompt>&& pending) {
    struct TokenizedPrompt {
        int request_id{0};
        TimePoint submit_time{};
        std::vector<int> token_ids;
    };

    std::vector<TokenizedPrompt> tokenized;
    tokenized.reserve(pending.size());
    int max_len = 0;
    for (auto& item : pending) {
        auto tokens = tokenizer_.tokenize(item.prompt_text);
        auto token_ids = tokenizer_.tokens_to_ids(tokens);
        if (token_ids.empty()) {
            std::cout << "[request " << item.request_id << "] " << "\n";
            continue;
        }
        max_len = std::max(max_len, static_cast<int>(token_ids.size()));
        tokenized.push_back(TokenizedPrompt{item.request_id, item.submit_time, std::move(token_ids)});
    }
    if (tokenized.empty()) {
        return {};
    }
    if (max_len <= 0) {
        throw std::runtime_error("prepare_admissions: max_len must be > 0.");
    }
    if (static_cast<int>(tokenized.size()) > static_cast<int>(free_slots_.size())) {
        throw std::runtime_error("prepare_admissions: not enough free slots.");
    }

    std::vector<PreparedAdmission> out;
    out.reserve(tokenized.size());
    const TimePoint admit_time = Clock::now();
    for (auto& item : tokenized) {
        const int slot_id = free_slots_.back();
        free_slots_.pop_back();
        const int seq_len = static_cast<int>(item.token_ids.size());
        const int pad_len = max_len - seq_len;
        const int max_generate_tokens = std::max(1, config_.max_steps - seq_len + 1);
        out.push_back(PreparedAdmission{
            item.request_id,
            slot_id,
            seq_len,
            pad_len,
            seq_len,
            max_generate_tokens,
            std::move(item.token_ids),
            item.submit_time,
            admit_time
        });
    }
    return out;
}

int ContinuousBatchServer::pending_size_unsafe() const {
    std::lock_guard<std::mutex> lock(pending_mu_);
    return static_cast<int>(pending_prompts_.size());
}

void ContinuousBatchServer::maybe_log_runtime_stats() {
    if (options_.stats_log_interval_ms <= 0) {
        return;
    }
    const auto now = Clock::now();
    const double since_last_ms = elapsed_ms(last_stats_log_at_, now);
    if (since_last_ms < static_cast<double>(options_.stats_log_interval_ms)) {
        return;
    }

    const double window_s = std::max(1e-6, since_last_ms / 1000.0);
    const std::uint64_t window_completed = stats_.completed_requests - last_logged_completed_requests_;
    const std::uint64_t window_emitted = stats_.emitted_tokens_total - last_logged_emitted_tokens_;
    const double req_per_s = static_cast<double>(window_completed) / window_s;
    const double tok_per_s = static_cast<double>(window_emitted) / window_s;
    const double avg_prefill_ms = stats_.prefill_rounds > 0
        ? (stats_.prefill_ms_total / static_cast<double>(stats_.prefill_rounds))
        : 0.0;
    const double avg_decode_ms = stats_.decode_rounds > 0
        ? (stats_.decode_ms_total / static_cast<double>(stats_.decode_rounds))
        : 0.0;
    const double avg_request_ms = stats_.completed_requests > 0
        ? (stats_.request_total_ms / static_cast<double>(stats_.completed_requests))
        : 0.0;
    const double avg_queue_ms = stats_.completed_requests > 0
        ? (stats_.request_queue_ms / static_cast<double>(stats_.completed_requests))
        : 0.0;
    const double avg_service_ms = stats_.completed_requests > 0
        ? (stats_.request_service_ms / static_cast<double>(stats_.completed_requests))
        : 0.0;

    spdlog::info("[serve][stats] active={} pending={} prefill_rounds={} decode_rounds={} done={} req_per_s={:.3f} tok_per_s={:.3f} avg_prefill_ms={:.3f} avg_decode_ms={:.3f} avg_req_ms={:.3f} avg_queue_ms={:.3f} avg_service_ms={:.3f}",
                 active_requests_.size(),
                 pending_size_unsafe(),
                 stats_.prefill_rounds,
                 stats_.decode_rounds,
                 stats_.completed_requests,
                 req_per_s,
                 tok_per_s,
                 avg_prefill_ms,
                 avg_decode_ms,
                 avg_request_ms,
                 avg_queue_ms,
                 avg_service_ms);

    last_stats_log_at_ = now;
    last_logged_completed_requests_ = stats_.completed_requests;
    last_logged_emitted_tokens_ = stats_.emitted_tokens_total;
}

void ContinuousBatchServer::log_final_runtime_stats() const {
    const auto now = Clock::now();
    const double uptime_ms = elapsed_ms(stats_started_at_, now);
    const double uptime_s = std::max(1e-6, uptime_ms / 1000.0);
    const double avg_prefill_ms = stats_.prefill_rounds > 0
        ? (stats_.prefill_ms_total / static_cast<double>(stats_.prefill_rounds))
        : 0.0;
    const double avg_decode_ms = stats_.decode_rounds > 0
        ? (stats_.decode_ms_total / static_cast<double>(stats_.decode_rounds))
        : 0.0;
    const double avg_request_ms = stats_.completed_requests > 0
        ? (stats_.request_total_ms / static_cast<double>(stats_.completed_requests))
        : 0.0;
    const double avg_queue_ms = stats_.completed_requests > 0
        ? (stats_.request_queue_ms / static_cast<double>(stats_.completed_requests))
        : 0.0;
    const double avg_service_ms = stats_.completed_requests > 0
        ? (stats_.request_service_ms / static_cast<double>(stats_.completed_requests))
        : 0.0;

    spdlog::info("[serve][summary] uptime_ms={:.3f} done={} eos={} limit={} emitted_tokens={} decode_tokens={} req_per_s={:.3f} tok_per_s={:.3f} avg_prefill_ms={:.3f} avg_decode_ms={:.3f} avg_req_ms={:.3f} avg_queue_ms={:.3f} avg_service_ms={:.3f}",
                 uptime_ms,
                 stats_.completed_requests,
                 stats_.finished_by_eos,
                 stats_.finished_by_limit,
                 stats_.emitted_tokens_total,
                 stats_.decode_tokens_total,
                 static_cast<double>(stats_.completed_requests) / uptime_s,
                 static_cast<double>(stats_.emitted_tokens_total) / uptime_s,
                 avg_prefill_ms,
                 avg_decode_ms,
                 avg_request_ms,
                 avg_queue_ms,
                 avg_service_ms);
}

} // namespace easy_llm
