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
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <immintrin.h>

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

    static void to_bf16(const e4m3_t *fp8_data, uint16_t *bf16_data, size_t count, const float scale = 1.0f) {
        alignas(64) static uint16_t e4m3_to_bf16_table[256] = {
                0x0000,
                0x3B00,
                0x3B80,
                0x3BC0,
                0x3C00,
                0x3C20,
                0x3C40,
                0x3C60,
                0x3C80,
                0x3C90,
                0x3CA0,
                0x3CB0,
                0x3CC0,
                0x3CD0,
                0x3CE0,
                0x3CF0,
                0x3D00,
                0x3D10,
                0x3D20,
                0x3D30,
                0x3D40,
                0x3D50,
                0x3D60,
                0x3D70,
                0x3D80,
                0x3D90,
                0x3DA0,
                0x3DB0,
                0x3DC0,
                0x3DD0,
                0x3DE0,
                0x3DF0,
                0x3E00,
                0x3E10,
                0x3E20,
                0x3E30,
                0x3E40,
                0x3E50,
                0x3E60,
                0x3E70,
                0x3E80,
                0x3E90,
                0x3EA0,
                0x3EB0,
                0x3EC0,
                0x3ED0,
                0x3EE0,
                0x3EF0,
                0x3F00,
                0x3F10,
                0x3F20,
                0x3F30,
                0x3F40,
                0x3F50,
                0x3F60,
                0x3F70,
                0x3F80,
                0x3F90,
                0x3FA0,
                0x3FB0,
                0x3FC0,
                0x3FD0,
                0x3FE0,
                0x3FF0,
                0x4000,
                0x4010,
                0x4020,
                0x4030,
                0x4040,
                0x4050,
                0x4060,
                0x4070,
                0x4080,
                0x4090,
                0x40A0,
                0x40B0,
                0x40C0,
                0x40D0,
                0x40E0,
                0x40F0,
                0x4100,
                0x4110,
                0x4120,
                0x4130,
                0x4140,
                0x4150,
                0x4160,
                0x4170,
                0x4180,
                0x4190,
                0x41A0,
                0x41B0,
                0x41C0,
                0x41D0,
                0x41E0,
                0x41F0,
                0x4200,
                0x4210,
                0x4220,
                0x4230,
                0x4240,
                0x4250,
                0x4260,
                0x4270,
                0x4280,
                0x4290,
                0x42A0,
                0x42B0,
                0x42C0,
                0x42D0,
                0x42E0,
                0x42F0,
                0x4300,
                0x4310,
                0x4320,
                0x4330,
                0x4340,
                0x4350,
                0x4360,
                0x4370,
                0x4380,
                0x4390,
                0x43A0,
                0x43B0,
                0x43C0,
                0x43D0,
                0x43E0,
                0x43F0,
                0x8000,
                0xBB00,
                0xBB80,
                0xBBC0,
                0xBC00,
                0xBC20,
                0xBC40,
                0xBC60,
                0xBC80,
                0xBC90,
                0xBCA0,
                0xBCB0,
                0xBCC0,
                0xBCD0,
                0xBCE0,
                0xBCF0,
                0xBD00,
                0xBD10,
                0xBD20,
                0xBD30,
                0xBD40,
                0xBD50,
                0xBD60,
                0xBD70,
                0xBD80,
                0xBD90,
                0xBDA0,
                0xBDB0,
                0xBDC0,
                0xBDD0,
                0xBDE0,
                0xBDF0,
                0xBE00,
                0xBE10,
                0xBE20,
                0xBE30,
                0xBE40,
                0xBE50,
                0xBE60,
                0xBE70,
                0xBE80,
                0xBE90,
                0xBEA0,
                0xBEB0,
                0xBEC0,
                0xBED0,
                0xBEE0,
                0xBEF0,
                0xBF00,
                0xBF10,
                0xBF20,
                0xBF30,
                0xBF40,
                0xBF50,
                0xBF60,
                0xBF70,
                0xBF80,
                0xBF90,
                0xBFA0,
                0xBFB0,
                0xBFC0,
                0xBFD0,
                0xBFE0,
                0xBFF0,
                0xC000,
                0xC010,
                0xC020,
                0xC030,
                0xC040,
                0xC050,
                0xC060,
                0xC070,
                0xC080,
                0xC090,
                0xC0A0,
                0xC0B0,
                0xC0C0,
                0xC0D0,
                0xC0E0,
                0xC0F0,
                0xC100,
                0xC110,
                0xC120,
                0xC130,
                0xC140,
                0xC150,
                0xC160,
                0xC170,
                0xC180,
                0xC190,
                0xC1A0,
                0xC1B0,
                0xC1C0,
                0xC1D0,
                0xC1E0,
                0xC1F0,
                0xC200,
                0xC210,
                0xC220,
                0xC230,
                0xC240,
                0xC250,
                0xC260,
                0xC270,
                0xC280,
                0xC290,
                0xC2A0,
                0xC2B0,
                0xC2C0,
                0xC2D0,
                0xC2E0,
                0xC2F0,
                0xC300,
                0xC310,
                0xC320,
                0xC330,
                0xC340,
                0xC350,
                0xC360,
                0xC370,
                0xC380,
                0xC390,
                0xC3A0,
                0xC3B0,
                0xC3C0,
                0xC3D0,
                0xC3E0,
                0xC3F0,
        };

        // Check if count is a multiple of 64
        if (__builtin_expect(count % 64, 0)) {
            throw std::runtime_error("count must be a multiple of 64");
        }

        __m512 vscale = _mm512_set1_ps(scale);

        // Process 128 FP8 values at a time
        size_t i = 0;
        for (; i + 127 < count; i += 128) {
            // Load 64 FP8 values (1 byte each = 512 bits)
            __m512i fp8_vec0 = _mm512_loadu_si512((__m512i const *)(fp8_data + i));
            __m512i fp8_vec1 = _mm512_loadu_si512((__m512i const *)(fp8_data + i + 64));

            // Split the 512-bit vector into 4 groups of 16 bytes.
            __m128i group0 = _mm512_castsi512_si128(fp8_vec0);
            __m128i group1 = _mm512_extracti32x4_epi32(fp8_vec0, 1);
            __m128i group2 = _mm512_extracti32x4_epi32(fp8_vec0, 2);
            __m128i group3 = _mm512_extracti32x4_epi32(fp8_vec0, 3);
            __m128i group4 = _mm512_castsi512_si128(fp8_vec1);
            __m128i group5 = _mm512_extracti32x4_epi32(fp8_vec1, 1);
            __m128i group6 = _mm512_extracti32x4_epi32(fp8_vec1, 2);
            __m128i group7 = _mm512_extracti32x4_epi32(fp8_vec1, 3);

            // Convert 16 uint8_t values in each group to 16 32-bit indices.
            __m512i indices0 = _mm512_cvtepu8_epi32(group0);
            __m512i indices1 = _mm512_cvtepu8_epi32(group1);
            __m512i indices2 = _mm512_cvtepu8_epi32(group2);
            __m512i indices3 = _mm512_cvtepu8_epi32(group3);
            __m512i indices4 = _mm512_cvtepu8_epi32(group4);
            __m512i indices5 = _mm512_cvtepu8_epi32(group5);
            __m512i indices6 = _mm512_cvtepu8_epi32(group6);
            __m512i indices7 = _mm512_cvtepu8_epi32(group7);

            // Gather BF16 conversion results from the lookup table.
            __m512i bf16_i32_vec0 = _mm512_i32gather_epi32(indices0, e4m3_to_bf16_table, 2);
            __m512i bf16_i32_vec1 = _mm512_i32gather_epi32(indices1, e4m3_to_bf16_table, 2);
            __m512i bf16_i32_vec2 = _mm512_i32gather_epi32(indices2, e4m3_to_bf16_table, 2);
            __m512i bf16_i32_vec3 = _mm512_i32gather_epi32(indices3, e4m3_to_bf16_table, 2);
            __m512i bf16_i32_vec4 = _mm512_i32gather_epi32(indices4, e4m3_to_bf16_table, 2);
            __m512i bf16_i32_vec5 = _mm512_i32gather_epi32(indices5, e4m3_to_bf16_table, 2);
            __m512i bf16_i32_vec6 = _mm512_i32gather_epi32(indices6, e4m3_to_bf16_table, 2);
            __m512i bf16_i32_vec7 = _mm512_i32gather_epi32(indices7, e4m3_to_bf16_table, 2);

            // Helper lambda: Convert 16 32-bit ints (in a __m512i) to 16 16-bit ints.
            auto convert_32_to_16 = [](__m512i vec) -> __m256i { return _mm512_cvtepi32_epi16(vec); };

            __m256i bf16_i16_vec0 = convert_32_to_16(bf16_i32_vec0);
            __m256i bf16_i16_vec1 = convert_32_to_16(bf16_i32_vec1);
            __m256i bf16_i16_vec2 = convert_32_to_16(bf16_i32_vec2);
            __m256i bf16_i16_vec3 = convert_32_to_16(bf16_i32_vec3);
            __m256i bf16_i16_vec4 = convert_32_to_16(bf16_i32_vec4);
            __m256i bf16_i16_vec5 = convert_32_to_16(bf16_i32_vec5);
            __m256i bf16_i16_vec6 = convert_32_to_16(bf16_i32_vec6);
            __m256i bf16_i16_vec7 = convert_32_to_16(bf16_i32_vec7);

            // Store the 64 BF16 values (16 values per 256-bit vector).
            auto bf16_to_fp32 = [](const __m256i src) -> __m512 {
                __m512i y = _mm512_cvtepu16_epi32(src);
                return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
            };

            _mm256_storeu_si256((__m256i *)(bf16_data + i + 0),
                    (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(bf16_to_fp32(bf16_i16_vec0), vscale)));
            _mm256_storeu_si256((__m256i *)(bf16_data + i + 16),
                    (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(bf16_to_fp32(bf16_i16_vec1), vscale)));
            _mm256_storeu_si256((__m256i *)(bf16_data + i + 32),
                    (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(bf16_to_fp32(bf16_i16_vec2), vscale)));
            _mm256_storeu_si256((__m256i *)(bf16_data + i + 48),
                    (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(bf16_to_fp32(bf16_i16_vec3), vscale)));
            _mm256_storeu_si256((__m256i *)(bf16_data + i + 64),
                    (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(bf16_to_fp32(bf16_i16_vec4), vscale)));
            _mm256_storeu_si256((__m256i *)(bf16_data + i + 80),
                    (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(bf16_to_fp32(bf16_i16_vec5), vscale)));
            _mm256_storeu_si256((__m256i *)(bf16_data + i + 96),
                    (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(bf16_to_fp32(bf16_i16_vec6), vscale)));
            _mm256_storeu_si256((__m256i *)(bf16_data + i + 112),
                    (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(bf16_to_fp32(bf16_i16_vec7), vscale)));
        }

        if (i < count) {
            // Load 64 FP8 values (1 byte each = 512 bits)
            __m512i fp8_vec = _mm512_loadu_si512((__m512i const *)(fp8_data + i));

            // Split the 512-bit vector into 4 groups of 16 bytes.
            __m128i group0 = _mm512_castsi512_si128(fp8_vec);
            __m128i group1 = _mm512_extracti32x4_epi32(fp8_vec, 1);
            __m128i group2 = _mm512_extracti32x4_epi32(fp8_vec, 2);
            __m128i group3 = _mm512_extracti32x4_epi32(fp8_vec, 3);

            // Convert 16 uint8_t values in each group to 16 32-bit indices.
            __m512i indices0 = _mm512_cvtepu8_epi32(group0);
            __m512i indices1 = _mm512_cvtepu8_epi32(group1);
            __m512i indices2 = _mm512_cvtepu8_epi32(group2);
            __m512i indices3 = _mm512_cvtepu8_epi32(group3);

            // Gather BF16 conversion results from the lookup table.
            __m512i bf16_i32_vec0 = _mm512_i32gather_epi32(indices0, e4m3_to_bf16_table, 2);
            __m512i bf16_i32_vec1 = _mm512_i32gather_epi32(indices1, e4m3_to_bf16_table, 2);
            __m512i bf16_i32_vec2 = _mm512_i32gather_epi32(indices2, e4m3_to_bf16_table, 2);
            __m512i bf16_i32_vec3 = _mm512_i32gather_epi32(indices3, e4m3_to_bf16_table, 2);

            // Helper lambda: Convert 16 32-bit ints (in a __m512i) to 16 16-bit ints.
            auto convert_32_to_16 = [](__m512i vec) -> __m256i { return _mm512_cvtepi32_epi16(vec); };

            __m256i bf16_i16_vec0 = convert_32_to_16(bf16_i32_vec0);
            __m256i bf16_i16_vec1 = convert_32_to_16(bf16_i32_vec1);
            __m256i bf16_i16_vec2 = convert_32_to_16(bf16_i32_vec2);
            __m256i bf16_i16_vec3 = convert_32_to_16(bf16_i32_vec3);

            // Store the 64 BF16 values (16 values per 256-bit vector).
            auto bf16_to_fp32 = [](const __m256i src) -> __m512 {
                __m512i y = _mm512_cvtepu16_epi32(src);
                return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
            };

            _mm256_storeu_si256((__m256i *)(bf16_data + i + 0),
                    (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(bf16_to_fp32(bf16_i16_vec0), vscale)));
            _mm256_storeu_si256((__m256i *)(bf16_data + i + 16),
                    (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(bf16_to_fp32(bf16_i16_vec1), vscale)));
            _mm256_storeu_si256((__m256i *)(bf16_data + i + 32),
                    (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(bf16_to_fp32(bf16_i16_vec2), vscale)));
            _mm256_storeu_si256((__m256i *)(bf16_data + i + 48),
                    (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(bf16_to_fp32(bf16_i16_vec3), vscale)));
        }
    }

} __attribute__((packed));
