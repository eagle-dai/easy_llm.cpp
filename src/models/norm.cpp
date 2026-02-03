#include "models/norm.hpp"

#include <cmath>
#include <stdexcept>
#include "spdlog/spdlog.h"

#include "utils.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include "models/loader.hpp"

namespace easy_llm {

using std::string;

RMSNorm::RMSNorm() {}

void RMSNorm::load_param(const string& key, ModelParam& model_param) {
    if (model_param.contains(key + ".weight")) {
        weight_ = model_param.get_param(key + ".weight");  
        if (weight_.shape().empty()) {
            throw std::runtime_error("Loaded weight tensor is empty for key: " + key);
        }
        dim_ = weight_.shape()[0];
    } else {
        throw std::runtime_error("Weight parameter not found for RMSNorm key: " + key + ".weight");
    }
}

Tensor RMSNorm::forward(const Tensor& input) const {
    auto input_norm = input;
    auto shape = input_norm.shape();
    auto& data = input_norm.data();
    if (shape.empty()) {
        throw std::invalid_argument("Cannot apply RMSNorm to a tensor with no dimensions.");
    }
    int features = shape.back();
    if (features != dim_) {
        throw std::runtime_error(fmt::format("RMSNorm dimension mismatch: expected {}, got {}", dim_, features));
    }
    auto epsilon = 1e-6f;
    auto num_vectors = data.size() / features;
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (size_t v = 0; v < num_vectors; ++v) {
        auto i = v * features;
        auto sum_sq = 0.0f;
        for (int j = 0; j < features; ++j) {
            auto val = static_cast<float>(data[i + j]);
            sum_sq += val * val;
        }
        auto mean_sq = sum_sq / features;
        auto rms = std::sqrt(mean_sq + epsilon);
        auto inv_rms = 1.0f / rms;

        for (int j = 0; j < features; ++j) {
            data[i + j] = weight_.at(j) * (data[i + j] * data_type(inv_rms));
        }
    }
    return input_norm;
}

}  // namespace easy_llm
