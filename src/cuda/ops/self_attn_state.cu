#include "cuda/ops/self_attn.hpp"

#include <memory>
#include <stdexcept>

#include "self_attn_detail.cuh"

namespace easy_llm {
namespace cuda {
namespace ops {

SelfAttnCudaState::SelfAttnCudaState()
    : impl_(std::make_unique<Impl>()) {}

SelfAttnCudaState::~SelfAttnCudaState() = default;

SelfAttnCudaState::SelfAttnCudaState(SelfAttnCudaState&& other) noexcept = default;

SelfAttnCudaState& SelfAttnCudaState::operator=(SelfAttnCudaState&& other) noexcept = default;

void SelfAttnCudaState::init_kv_cache(int batch_size) {
    if (batch_size < 0) {
        throw std::invalid_argument("init_kv_cache: batch_size must be non-negative.");
    }
    impl_->samples.clear();
    impl_->samples.resize(batch_size);
    impl_->decode_graph.reset();
}

void SelfAttnCudaState::clear_kv_cache(int sample_id) {
    if (sample_id < 0 || sample_id >= static_cast<int>(impl_->samples.size())) {
        return;
    }
    SampleCache& sample = impl_->samples[sample_id];
    sample.k = DeviceBuffer();
    sample.v = DeviceBuffer();
    sample.len = 0;
    sample.capacity_len = 0;
    impl_->decode_graph.reset();
}

void SelfAttnCudaState::reset_kv_cache() {
    impl_->samples.clear();
    impl_->decode_graph.reset();
}

int SelfAttnCudaState::cache_len(int sample_id) const {
    if (sample_id < 0 || sample_id >= static_cast<int>(impl_->samples.size())) {
        return 0;
    }
    return impl_->samples[sample_id].len;
}

SelfAttnCudaStats SelfAttnCudaState::stats() const {
    return impl_->stats;
}

void SelfAttnCudaState::reset_stats() {
    impl_->stats = SelfAttnCudaStats{};
}

} // namespace ops
} // namespace cuda
} // namespace easy_llm
