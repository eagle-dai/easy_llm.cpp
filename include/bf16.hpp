#ifndef EASY_LLM_BF16_HPP
#define EASY_LLM_BF16_HPP

#include <cstdint>
#include <cstring>

#include <spdlog/fmt/bundled/format.h>

namespace easy_llm {

inline float bf16_to_float(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    // Round the lower 16 bits using round-to-nearest-even
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits = bits + rounding_bias;
    uint16_t bf16 = bits >> 16;
    return bf16;
}

struct Bf16 {
    uint16_t data;
    Bf16() : data(0) {}

    explicit Bf16(float f) : data(float_to_bf16(f)) {}

    operator float() const {
        return bf16_to_float(data); 
    }

    Bf16 operator+(const Bf16 &other) const {
        float result = float(*this) + float(other);
        return Bf16(result);
    }
    
    Bf16 operator*(const Bf16 &other) const {
        float result = float(*this) * float(other);
        return Bf16(result);
    }
    
    Bf16 operator/(const Bf16 &other) const {
        float result = float(*this) / float(other);
        return Bf16(result);
    }
    
    Bf16 operator-() const {
        float result = -float(*this);
        return Bf16(result);
    }

    Bf16 operator-(const Bf16 &other) const {
        float result = float(*this) - float(other);
        return Bf16(result);
    }

    Bf16& operator+=(const Bf16 &other) {
        *this = *this + other;
        return *this;
    }

    Bf16& operator-=(const Bf16 &other) {
        *this = *this - other;
        return *this;
    }

    Bf16& operator*=(const Bf16 &other) {
        *this = *this * other;
        return *this;
    }

    Bf16& operator/=(const Bf16 &other) {
        *this = *this / other;
        return *this;
    }

    bool operator==(const Bf16 &other) const {
        return data == other.data;
    }

    bool operator!=(const Bf16 &other) const {
        return data != other.data;
    }

    bool operator<(const Bf16 &other) const {
        return float(*this) < float(other);
    }

    bool operator>(const Bf16 &other) const {
        return float(*this) > float(other);
    }

    Bf16& operator++() {
        *this += Bf16(1.0f);
        return *this;
    }

    Bf16 operator++(int) {
        Bf16 old = *this;
        *this += Bf16(1.0f);
        return old;
    }

    Bf16& operator--() {
        *this -= Bf16(1.0f);
        return *this;
    }

    Bf16 operator--(int) {
        Bf16 old = *this;
        *this -= Bf16(1.0f);
        return old;
    }
};
}  // namespace easy_llm

namespace fmt {
    template <>
    struct formatter<easy_llm::Bf16> {
      template <typename ParseContext>
      constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }
      template <typename FormatContext>
      auto format(const easy_llm::Bf16& value, FormatContext& ctx) {
        return format_to(ctx.out(), "{}", static_cast<float>(value));
      }
    };
} // namespace fmt

#endif // EASY_LLM_BF16_HPP
