#include "models/generation_invariants.hpp"

#include <stdexcept>
#include <string>

namespace easy_llm::generation {

namespace {

void validate_sample_id(int sample_id,
                        int size,
                        const char* fn_name,
                        const char* source_name) {
    if (sample_id < 0 || sample_id >= size) {
        throw std::invalid_argument(std::string(fn_name) +
                                    ": sample_id out of range for " +
                                    source_name +
                                    " (sample_id=" + std::to_string(sample_id) +
                                    ", size=" + std::to_string(size) + ").");
    }
}

} // namespace

std::vector<int> build_prefill_pos_offsets(const std::vector<int>& sample_ids,
                                           const std::vector<int>& pad_lens) {
    std::vector<int> offsets;
    offsets.reserve(sample_ids.size());
    for (int sample_id : sample_ids) {
        validate_sample_id(sample_id,
                           static_cast<int>(pad_lens.size()),
                           "build_prefill_pos_offsets",
                           "pad_lens");
        offsets.push_back(-pad_lens[sample_id]);
    }
    return offsets;
}

std::vector<int> build_decode_pos_offsets(const std::vector<int>& sample_ids,
                                          const std::vector<int>& pos_lens_by_sample) {
    std::vector<int> offsets;
    offsets.reserve(sample_ids.size());
    for (int sample_id : sample_ids) {
        validate_sample_id(sample_id,
                           static_cast<int>(pos_lens_by_sample.size()),
                           "build_decode_pos_offsets",
                           "pos_lens_by_sample");
        offsets.push_back(pos_lens_by_sample[sample_id]);
    }
    return offsets;
}

void increment_pos_lens(const std::vector<int>& sample_ids,
                        std::vector<int>& pos_lens_by_sample) {
    for (int sample_id : sample_ids) {
        validate_sample_id(sample_id,
                           static_cast<int>(pos_lens_by_sample.size()),
                           "increment_pos_lens",
                           "pos_lens_by_sample");
        pos_lens_by_sample[sample_id] += 1;
    }
}

EosFilterResult filter_eos_samples(const std::vector<int>& sample_ids,
                                   const std::vector<int>& next_tokens,
                                   int eos_token_id) {
    if (sample_ids.size() != next_tokens.size()) {
        throw std::invalid_argument(
            "filter_eos_samples: sample_ids size must match next_tokens size.");
    }
    EosFilterResult result;
    result.sample_ids.reserve(sample_ids.size());
    result.next_tokens.reserve(next_tokens.size());
    result.cleared_sample_ids.reserve(sample_ids.size());
    for (size_t i = 0; i < sample_ids.size(); ++i) {
        const int sample_id = sample_ids[i];
        const int token = next_tokens[i];
        if (eos_token_id >= 0 && token == eos_token_id) {
            result.cleared_sample_ids.push_back(sample_id);
            continue;
        }
        result.sample_ids.push_back(sample_id);
        result.next_tokens.push_back(token);
    }
    return result;
}

} // namespace easy_llm::generation
