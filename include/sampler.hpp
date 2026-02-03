#ifndef EASY_LLM_SAMPLER_HPP
#define EASY_LLM_SAMPLER_HPP

#include <random>

namespace easy_llm {

class Sampler {
    public:
    Sampler() = default;
    virtual ~Sampler() = default;

    virtual int sample_from_probs(const float* probs, int vocab_size) = 0;
    virtual void set_params(float temperature, float top_p, int top_k, bool use_greedy);
};

class GreedySampler : public Sampler {
public:
    GreedySampler() = default;
    ~GreedySampler() override = default;

    int sample_from_probs(const float* probs, int vocab_size) override;
};

class TopKTopPSampler : public Sampler {
public:
    TopKTopPSampler(float temperature, float top_p, int top_k, bool use_greedy, std::mt19937& rng);
    ~TopKTopPSampler() override = default;

    int sample_from_probs(const float* probs, int vocab_size) override;
    void set_params(float temperature, float top_p, int top_k, bool use_greedy) override;

private:
    float temperature_;
    float top_p_;
    int top_k_;
    bool use_greedy_;
    std::mt19937& rng_;
};

}  // namespace easy_llm

#endif  // EASY_LLM_SAMPLER_HPP
