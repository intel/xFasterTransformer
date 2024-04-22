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
#include <cstdint>

#include "intrinsics_util.h"

namespace xft {
template <typename T>
static float absMax(T *x, int size) {
    auto vmax = xft::set_avx512(0);

    for (int i = 0; i < size; i += 16) {
        int remain = size - i;
        __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
        auto vx = _mm512_abs_ps(xft::load_avx512(mask, x + i));
        vmax = _mm512_max_ps(vx, vmax);
    }

    return _mm512_reduce_max_ps(vmax);
}

template <typename T>
static void quantize(int8_t *dst, float *scale, T *src, int size) {
    float max = absMax(src, size);
    float factor = (max == 0 ? 1 : 127.0f / max);
    auto vfactor = xft::set_avx512(factor);

    *scale = max / 127.0f;

    for (int i = 0; i < size; i += 16) {
        int remain = size - i;
        __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
        auto vx = _mm512_mul_ps(xft::load_avx512(mask, src + i), vfactor);
        auto vi = _mm512_cvt_roundps_epi32(vx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        auto vb = _mm512_cvtepi32_epi8(vi);
        _mm_mask_storeu_epi8(dst + i, mask, vb);
    }
}
} // namespace xft