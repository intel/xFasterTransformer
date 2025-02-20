// Copyright (c) 2025 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once
#include <cstdint>
#include <cmath>

class e4m3_t {
   private:
    uint8_t val;

   public:
    e4m3_t() { val = 0; }

    e4m3_t(uint8_t v) { val = v; }

    e4m3_t &operator=(uint8_t v) {
        val = v;
        return *this;
    }

    e4m3_t &operator=(e4m3_t &v) {
        val = v.val;
        return *this;
    }

    operator float() const {
        // Extract sign, exponent, and mantissa
        uint8_t sign = (val & 0x80) >> 7;
        uint8_t exponent = (val & 0x78) >> 3;
        uint8_t mantissa = val & 0x07;

        float result;
        if (exponent == 0) {
            // Subnormal: use (mantissa/8) * 2^(1-bias) with bias = 7 (i.e. 2^-6)
            result = (mantissa / 8.0f) * powf(2.0f, -6);
        } else {
            // Normalized: implicit 1, so (1 + mantissa/8) * 2^(exponent-bias)
            result = (1.0f + mantissa / 8.0f) * powf(2.0f, exponent - 7);
        }

        return sign ? -result : result;
    }

} __attribute__((packed));
