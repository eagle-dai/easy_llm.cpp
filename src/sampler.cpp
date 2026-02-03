#include <sampler.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace easy_llm {

namespace {

int greedy_sample_from_probs(const float* probs, int vocab_size) {
    if (!probs || vocab_size <= 0) {
        throw std::runtime_error("greedy_sample_from_probs - invalid input");
    }
    const auto* end_ptr = probs + vocab_size;
    auto max_it = std::max_element(probs, end_ptr);
    return static_cast<int>(std::distance(probs, max_it));
}

}  // namespace

void Sampler::set_params(float /*temperature*/, float /*top_p*/, int /*top_k*/, bool /*use_greedy*/) {
}

int GreedySampler::sample_from_probs(const float* probs, int vocab_size) {
    return greedy_sample_from_probs(probs, vocab_size);
}

TopKTopPSampler::TopKTopPSampler(float temperature, float top_p, int top_k, bool use_greedy, std::mt19937& rng)
    : temperature_(temperature),
      top_p_(top_p),
      top_k_(top_k),
      use_greedy_(use_greedy),
      rng_(rng) {
}

int TopKTopPSampler::sample_from_probs(const float* probs, int vocab_size) {
    if (!probs || vocab_size <= 0) {
        throw std::runtime_error("TopKTopPSampler::sample_from_probs - invalid input");
    }
    if (use_greedy_) {
        return greedy_sample_from_probs(probs, vocab_size);
    }
    float temperature = temperature_;
    if (temperature <= 0.0f) {
        return greedy_sample_from_probs(probs, vocab_size);
    }
    float inv_temp = 1.0f / std::max(temperature, 1e-6f);
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(static_cast<size_t>(vocab_size));
    bool apply_temp = std::abs(temperature - 1.0f) > 1e-6f;
    for (int i = 0; i < vocab_size; ++i) {
        float p = probs[i];
        float adjusted = apply_temp ? std::pow(std::max(p, 0.0f), inv_temp) : std::max(p, 0.0f);
        candidates.emplace_back(adjusted, i);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (top_k_ > 0 && top_k_ < static_cast<int>(candidates.size())) {
        candidates.resize(static_cast<size_t>(top_k_));
    }

    float total = 0.0f;
    for (const auto& item : candidates) {
        total += item.first;
    }
    if (total <= 0.0f) {
        return greedy_sample_from_probs(probs, vocab_size);
    }
    if (top_p_ > 0.0f && top_p_ < 1.0f) {
        float cumulative = 0.0f;
        size_t cutoff = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            cumulative += candidates[i].first / total;
            cutoff = i + 1;
            if (cumulative >= top_p_) {
                break;
            }
        }
        cutoff = std::max<size_t>(1, cutoff);
        if (cutoff < candidates.size()) {
            candidates.resize(cutoff);
            total = 0.0f;
            for (const auto& item : candidates) {
                total += item.first;
            }
            if (total <= 0.0f) {
                return greedy_sample_from_probs(probs, vocab_size);
            }
        }
    }

    std::uniform_real_distribution<float> dist(0.0f, total);
    float r = dist(rng_);
    float cumulative = 0.0f;
    for (const auto& item : candidates) {
        cumulative += item.first;
        if (r <= cumulative) {
            return item.second;
        }
    }
    return candidates.back().second;
}

void TopKTopPSampler::set_params(float temperature, float top_p, int top_k, bool use_greedy) {
    temperature_ = temperature;
    top_p_ = top_p;
    top_k_ = top_k;
    use_greedy_ = use_greedy;
}

}  // namespace easy_llm
