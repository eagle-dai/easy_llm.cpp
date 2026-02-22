#ifndef EASY_LLM_MODELS_CACHE_BATCHING_HPP
#define EASY_LLM_MODELS_CACHE_BATCHING_HPP

#include <vector>

#include "tensor.hpp"

namespace easy_llm {

struct BatchedCacheView {
    Tensor cache;
    std::vector<int> valid_lens;
};

BatchedCacheView build_padded_active_cache(const std::vector<Tensor>& cache_by_sample,
                                           const std::vector<int>& sample_ids);

void apply_valid_length_mask(Tensor& scores, const std::vector<int>& valid_lens);

} // namespace easy_llm

#endif // EASY_LLM_MODELS_CACHE_BATCHING_HPP
