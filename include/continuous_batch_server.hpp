#ifndef EASY_LLM_CONTINUOUS_BATCH_SERVER_HPP
#define EASY_LLM_CONTINUOUS_BATCH_SERVER_HPP

#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "config.hpp"
#include "models/gpt_model.hpp"
#include "tokenizer.hpp"

namespace easy_llm {

struct ContinuousBatchOptions {
    int max_active_requests{16};
    int prefill_batch_size{4};
    int idle_sleep_ms{2};
    int stats_log_interval_ms{1000};
};

class ContinuousBatchServer {
public:
    ContinuousBatchServer(const Config& config,
                          GptModel& model,
                          Tokenizer& tokenizer,
                          const ContinuousBatchOptions& options);

    int submit_prompt(std::string prompt_text);
    void mark_input_closed();
    void run();

private:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    struct PendingPrompt {
        int request_id{0};
        std::string prompt_text;
        TimePoint submit_time{};
    };

    struct ActiveRequest {
        int request_id{0};
        int slot_id{-1};
        int pos_len{0};
        int max_generate_tokens{0};
        int next_token{0};
        std::vector<int> generated_token_ids;
        TimePoint submit_time{};
        TimePoint admit_time{};
        int decode_steps{0};
    };

    struct PreparedAdmission {
        int request_id{0};
        int slot_id{-1};
        int seq_len{0};
        int pad_len{0};
        int pos_len{0};
        int max_generate_tokens{0};
        std::vector<int> token_ids;
        TimePoint submit_time{};
        TimePoint admit_time{};
    };

    struct RuntimeStats {
        std::uint64_t prefill_rounds{0};
        std::uint64_t decode_rounds{0};
        std::uint64_t prefill_requests{0};
        std::uint64_t decode_tokens_total{0};
        std::uint64_t emitted_tokens_total{0};
        std::uint64_t completed_requests{0};
        std::uint64_t finished_by_eos{0};
        std::uint64_t finished_by_limit{0};
        double prefill_ms_total{0.0};
        double decode_ms_total{0.0};
        double request_total_ms{0.0};
        double request_queue_ms{0.0};
        double request_service_ms{0.0};
    };

    bool admit_prefill_round();
    bool decode_round();
    bool is_done() const;

    void finish_request(ActiveRequest&& request, bool hit_eos, bool hit_limit);
    std::vector<PendingPrompt> pop_admission_candidates(int limit);
    std::vector<PreparedAdmission> prepare_admissions(std::vector<PendingPrompt>&& pending);
    void maybe_log_runtime_stats();
    void log_final_runtime_stats() const;
    static double elapsed_ms(TimePoint begin, TimePoint end);
    int pending_size_unsafe() const;

    const Config& config_;
    GptModel& model_;
    Tokenizer& tokenizer_;
    ContinuousBatchOptions options_;
    int pad_id_{-1};
    bool eos_enabled_{false};

    mutable std::mutex pending_mu_;
    std::deque<PendingPrompt> pending_prompts_;
    bool input_closed_{false};
    int next_request_id_{0};

    std::vector<int> free_slots_;
    std::vector<int> pad_lens_by_slot_;
    std::vector<ActiveRequest> active_requests_;

    RuntimeStats stats_;
    TimePoint stats_started_at_{};
    TimePoint last_stats_log_at_{};
    std::uint64_t last_logged_completed_requests_{0};
    std::uint64_t last_logged_emitted_tokens_{0};
};

} // namespace easy_llm

#endif // EASY_LLM_CONTINUOUS_BATCH_SERVER_HPP
