#ifndef EASY_LLM_MODELS_GENERATION_INVARIANTS_HPP
#define EASY_LLM_MODELS_GENERATION_INVARIANTS_HPP

#include <vector>

namespace easy_llm::generation {

struct EosFilterResult {
    std::vector<int> sample_ids;
    std::vector<int> next_tokens;
    std::vector<int> cleared_sample_ids;
};

std::vector<int> build_prefill_pos_offsets(const std::vector<int>& sample_ids,
                                           const std::vector<int>& pad_lens);

std::vector<int> build_decode_pos_offsets(const std::vector<int>& sample_ids,
                                          const std::vector<int>& pos_lens_by_sample);

void increment_pos_lens(const std::vector<int>& sample_ids,
                        std::vector<int>& pos_lens_by_sample);

EosFilterResult filter_eos_samples(const std::vector<int>& sample_ids,
                                   const std::vector<int>& next_tokens,
                                   int eos_token_id);

} // namespace easy_llm::generation

#endif // EASY_LLM_MODELS_GENERATION_INVARIANTS_HPP
