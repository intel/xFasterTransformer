// Copyright (c) 2023 Intel Corporation
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

// Load BF16 and Convert BF16 to FP32
inline __m512 _mm512_loadu_pbh(void const *mem_addr) {
    return _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_epi16(mem_addr));
}

inline __m512 _mm512_maskz_loadu_pbh(__mmask16 k, void const *mem_addr) {
    return _mm512_cvtpbh_ps((__m256bh)_mm256_maskz_loadu_epi16(k, mem_addr));
}

// Convert FP32 to BF16 and Store BF16
inline void _mm512_storeu_pbh(void *mem_addr, __m512 a) {
    _mm256_storeu_epi16(mem_addr, (__m256i)_mm512_cvtneps_pbh(a));
}

inline void _mm512_mask_storeu_pbh(void *mem_addr, __mmask16 k, __m512 a) {
    _mm256_mask_storeu_epi16(mem_addr, k, (__m256i)_mm512_cvtneps_pbh(a));
}