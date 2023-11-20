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
#include <immintrin.h>

#include "bfloat16.h"
#include "dtype.h"
#include "float16.h"
#include "intrinsic_ext.h"
#include "layernorm_kernels.h"
#include "my_types.h"

namespace xft {

void invokeLayerNorm(float *output, const float *input, const float *gamma, const float *beta, int rows, int cols,
        int iStride, int oStride, float epsilon) {

    int size = cols;
    if (iStride == -1) iStride = size;
    if (oStride == -1) oStride = size;

#pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        const float *px = input + r * iStride;
        float *py = output + r * oStride;

        float sum = 0;
        float squareSum = 0;

        __m512 vsum = _mm512_set1_ps(0);
        __m512 vsqare = _mm512_set1_ps(0);

        for (int col = 0; col < size; col += 16) {
            int remain = size - col;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            // SUM(x)
            __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
            vsum = _mm512_add_ps(vsum, vx);

            // SUM(x*x)
            __m512 tmp = _mm512_mul_ps(vx, vx);
            vsqare = _mm512_add_ps(vsqare, tmp);
        }

        sum = _mm512_reduce_add_ps(vsum);
        squareSum = _mm512_reduce_add_ps(vsqare);

        // Mean
        float mean = sum / size;
        __m512 vmean = _mm512_set1_ps(mean);

        // Variance
        float var = 1 / sqrt(squareSum / size - mean * mean + epsilon);
        __m512 vvar = _mm512_set1_ps(var);

        for (int col = 0; col < size; col += 16) {
            int remain = size - col;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
            __m512 vgamma = _mm512_maskz_loadu_ps(mask, gamma + col);
            __m512 vbeta = _mm512_maskz_loadu_ps(mask, beta + col);
            __m512 vy = (vx - vmean) * vgamma * vvar + vbeta;
            _mm512_mask_storeu_ps(py + col, mask, vy);
        }
    }
}

void invokeLayerNorm(bfloat16_t *output, const bfloat16_t *input, const bfloat16_t *gamma, const bfloat16_t *beta,
        int rows, int cols, int iStride, int oStride, float epsilon) {

    int size = cols;
    if (iStride == -1) iStride = size;
    if (oStride == -1) oStride = size;

#pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        const bfloat16_t *px = input + r * iStride;
        bfloat16_t *py = output + r * oStride;

        float sum = 0;
        float squareSum = 0;

        __m512 vsum = _mm512_set1_ps(0);
        __m512 vsqare = _mm512_set1_ps(0);

        for (int col = 0; col < size; col += 16) {
            int remain = size - col;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            // SUM(x)
            __m512 vx = _mm512_maskz_loadu_pbh(mask, px + col);
            vsum = _mm512_add_ps(vsum, vx);

            // SUM(x*x)
            __m512 tmp = _mm512_mul_ps(vx, vx);
            vsqare = _mm512_add_ps(vsqare, tmp);
        }

        sum = _mm512_reduce_add_ps(vsum);
        squareSum = _mm512_reduce_add_ps(vsqare);

        // Mean
        float mean = sum / size;
        __m512 vmean = _mm512_set1_ps(mean);

        // Variance
        float var = 1 / sqrt(squareSum / size - mean * mean + epsilon);
        __m512 vvar = _mm512_set1_ps(var);

        for (int col = 0; col < size; col += 16) {
            int remain = size - col;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = _mm512_maskz_loadu_pbh(mask, px + col);
            __m512 vgamma = _mm512_maskz_loadu_pbh(mask, gamma + col);
            __m512 vbeta = _mm512_maskz_loadu_pbh(mask, beta + col);
            __m512 vy = (vx - vmean) * vgamma * vvar + vbeta;
            _mm512_mask_storeu_pbh(py + col, mask, vy);
        }
    }
}

void invokeLayerNorm(DataType dt, void *output, const void *input, const void *gamma, const void *beta, int rows,
        int cols, int iStride, int oStride, float epsilon) {
    if (dt == DataType::bf16) {
        invokeLayerNorm((bfloat16_t *)output, (const bfloat16_t *)input, (const bfloat16_t *)gamma,
                (const bfloat16_t *)beta, rows, cols, iStride, oStride, epsilon);
    }
}

} // namespace xft