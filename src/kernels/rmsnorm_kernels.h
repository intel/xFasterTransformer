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

#include "bfloat16.h"
#include "float16.h"
#include "my_types.h"

namespace xft {

template <typename T>
void invokeRmsNorm(T *output, const T *input, const T *weight, int rows, int cols, int iStride = -1, int oStride = -1,
        const float epsilon = 1e-6) {
    if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, int8_t>) {
        printf("Type %s not supported!\n", typeid(T).name());
        exit(-1);
    } else {
        printf("Type %s not supported!\n", typeid(T).name());
        exit(-1);
    }
}

template <>
void invokeRmsNorm(float *output, const float *input, const float *weight, int rows, int cols, int iStride, int oStride,
        float epsilon) {
    int size = cols;

    if (iStride == -1) iStride = cols;
    if (oStride == -1) oStride = cols;

#pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        const float *px = input + r * iStride;
        float *py = output + r * oStride;

        float squareSum = 0;

        __m512 vsqare = _mm512_set1_ps(0);

        int col = 0;
        for (; col + 15 < size; col += 16) {
            // SUM(x*x)
            __m512 vx = _mm512_loadu_ps(px + col);
            __m512 tmp = _mm512_mul_ps(vx, vx);
            vsqare = _mm512_add_ps(vsqare, tmp);
        }
        if (col < size) {
            __mmask16 mask = (1 << (size - col)) - 1;
            __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
            __m512 tmp = _mm512_mul_ps(vx, vx);
            vsqare = _mm512_add_ps(vsqare, tmp);
        }

        squareSum = _mm512_reduce_add_ps(vsqare);

        // Variance
        float var = 1 / sqrt(squareSum / size + epsilon);
        __m512 vvar = _mm512_set1_ps(var);

        for (col = 0; col + 15 < size; col += 16) {
            __m512 vx = _mm512_loadu_ps(px + col);
            __m512 vw = _mm512_loadu_ps(weight + col);
            __m512 vy = vx * vvar * vw;
            _mm512_storeu_ps(py + col, vy);
        }
        if (col < size) {
            __mmask16 mask = (1 << (size - col)) - 1;
            __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
            __m512 vw = _mm512_maskz_loadu_ps(mask, weight + col);
            __m512 vy = vx * vvar * vw;
            _mm512_mask_storeu_ps(py + col, mask, vy);
        }
    } // end for rows
}
} // namespace xft