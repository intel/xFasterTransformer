#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "bit_cast.h"

class bfloat16_t {
public:
    bfloat16_t() = default;
    bfloat16_t(float f) { (*this) = f; }
    constexpr bfloat16_t(uint16_t r, bool) : raw_bits_(r) {}

    template <typename IntegerType,
            typename SFINAE = typename std::enable_if<std::is_integral<IntegerType>::value>::type>
    bfloat16_t(const IntegerType i)
        : raw_bits_ {convert_bits_of_normal_or_zero(bit_cast<uint32_t>(static_cast<float>(i)))} {}

    bfloat16_t &operator=(float f) {
        auto iraw = bit_cast<std::array<uint16_t, 2>>(f);
        switch (std::fpclassify(f)) {
            case FP_SUBNORMAL:
            case FP_ZERO:
                raw_bits_ = iraw[1];
                raw_bits_ &= 0x8000;
                break;
            case FP_INFINITE: raw_bits_ = iraw[1]; break;
            case FP_NAN:
                raw_bits_ = iraw[1];
                raw_bits_ |= 1 << 6;
                break;
            case FP_NORMAL:
                const uint32_t rounding_bias = 0x00007FFF + (iraw[1] & 0x1);
                const uint32_t int_raw = bit_cast<uint32_t>(f) + rounding_bias;
                iraw = bit_cast<std::array<uint16_t, 2>>(int_raw);
                raw_bits_ = iraw[1];
                break;
        }

        return *this;
    }

    template <typename IntegerType,
            typename SFINAE = typename std::enable_if<std::is_integral<IntegerType>::value>::type>
    bfloat16_t &operator=(const IntegerType i) {
        return (*this) = bfloat16_t {i};
    }

    operator float() const {
        std::array<uint16_t, 2> iraw = {{0, raw_bits_}};
        return bit_cast<float>(iraw);
    }

    bfloat16_t &operator+=(const float a) {
        (*this) = float {*this} + a;
        return *this;
    }

    static constexpr uint16_t convert_bits_of_normal_or_zero(const uint32_t bits) {
        return uint32_t {bits + uint32_t {0x7FFFU + (uint32_t {bits >> 16} & 1U)}} >> 16;
    }

private:
    uint16_t raw_bits_;
};

static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 2 bytes");
