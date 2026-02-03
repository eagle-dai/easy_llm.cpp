#ifndef EASY_LLM_PRECISION_HPP
#define EASY_LLM_PRECISION_HPP

#include <cstddef>

namespace easy_llm {
namespace precision_config {

enum class Kind {
    Fp32,
    Fp16,
    Bf16
};

#if defined(USE_BF16)
constexpr Kind kind = Kind::Bf16;
#elif defined(USE_FP16)
constexpr Kind kind = Kind::Fp16;
#elif defined(USE_FP32)
constexpr Kind kind = Kind::Fp32;
#else
    #error "No valid precision defined!"
#endif

constexpr const char* name() {
    switch (kind) {
        case Kind::Fp32: return "fp32";
        case Kind::Fp16: return "fp16";
        case Kind::Bf16: return "bf16";
        default: return "unknown";
    }
}

constexpr size_t element_bytes() {
    switch (kind) {
        case Kind::Fp32: return 4;
        case Kind::Fp16: return 2;
        case Kind::Bf16: return 2;
        default: return 0;
    }
}

} // namespace precision_config

} // namespace easy_llm

#endif // EASY_LLM_PRECISION_HPP
