// Copyright (c) 2024 Intel Corporation
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

#include <immintrin.h>
#include "bfloat16.h"
#include "float16.h"

namespace xft {

inline __m512 set_avx512(float v) {
    return _mm512_set1_ps(v);
}

////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __m512 load_avx512(const T *addr) {
    static_assert(std::is_same_v<T, float>, "Data type in load_avx512 is not supported!");
    __m512 v = _mm512_maskz_loadu_ps(0xffff, addr);
    return v;
}

template <>
inline __m512 load_avx512<bfloat16_t>(const bfloat16_t *addr) {
    __m512 v = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(0xffff, addr));
    return v;
}

template <>
inline __m512 load_avx512<float16_t>(const float16_t *addr) {
    __m512 v = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(0xffff, addr));
    return v;
}

template <>
inline __m512 load_avx512<int8_t>(const int8_t *addr) {
    __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(0xffff, addr)));
    return v;
}

////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __m512 load_avx512(__mmask16 mask, const T *addr) {
    static_assert(std::is_same_v<T, float>, "Data type in load_avx512 is not supported!");
    __m512 v = _mm512_maskz_loadu_ps(mask, addr);
    return v;
}

template <>
inline __m512 load_avx512<bfloat16_t>(__mmask16 mask, const bfloat16_t *addr) {
    __m512 v = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, addr));
    return v;
}

template <>
inline __m512 load_avx512<float16_t>(__mmask16 mask, const float16_t *addr) {
    __m512 v = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, addr));
    return v;
}

template <>
inline __m512 load_avx512<int8_t>(__mmask16 mask, const int8_t *addr) {
    __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(mask, addr)));
    return v;
}

////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void store_avx512(T *addr, __mmask16 mask, __m512 v) {
    static_assert(std::is_same_v<T, float>, "Data type in store_avx512 is not supported!");
    _mm512_mask_storeu_ps(addr, mask, v);
}

template <>
inline void store_avx512<bfloat16_t>(bfloat16_t *addr, __mmask16 mask, __m512 v) {
    _mm256_mask_storeu_epi16(addr, mask, bfloat16_t::cvt_fp32_to_bf16(v));
}

template <>
inline void store_avx512<float16_t>(float16_t *addr, __mmask16 mask, __m512 v) {
    _mm256_mask_storeu_epi16(addr, mask, _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

} // namespace xft