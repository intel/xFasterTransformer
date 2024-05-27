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
#include "intrinsics_util.h"
#include "my_types.h"
#include "rmsnorm_kernels.h"

namespace xft {

template <typename Tout, typename Tin, typename Twei>
void rmsNorm(Tout *output, const Tin *input, const Twei *weight, int rows, int cols, int iStride, int oStride,
        float epsilon) {
    static_assert(std::is_same_v<Tout, float> || std::is_same_v<Tout, bfloat16_t> || std::is_same_v<Tout, float16_t>,
            "Output of rmsNorm must be either float, bfloat16_t or float16.");
    static_assert(std::is_same_v<Tin, float> || std::is_same_v<Tin, bfloat16_t> || std::is_same_v<Tin, float16_t>,
            "Input of rmsNorm must be either float, bfloat16_t or float16.");
    static_assert(std::is_same_v<Twei, float> || std::is_same_v<Twei, bfloat16_t> || std::is_same_v<Twei, float16_t>,
            "Tweight of rmsNorm must be either float, bfloat16_t or float16.");

    int size = cols;
    if (iStride == -1) iStride = cols;
    if (oStride == -1) oStride = cols;

#pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        const Tin *px = input + r * iStride;
        Tout *py = output + r * oStride;

        float squareSum = 0;

        __m512 vsqare = _mm512_set1_ps(0);

        int col = 0;
        for (; col + 15 < size; col += 16) {
            // SUM(x*x)
            __m512 vx = xft::load_avx512(px + col);
            __m512 tmp = _mm512_mul_ps(vx, vx);
            vsqare = _mm512_add_ps(vsqare, tmp);
        }
        if (col < size) {
            __mmask16 mask = (1 << (size - col)) - 1;
            __m512 vx = xft::load_avx512(mask, px + col);
            __m512 tmp = _mm512_mul_ps(vx, vx);
            vsqare = _mm512_add_ps(vsqare, tmp);
        }

        squareSum = _mm512_reduce_add_ps(vsqare);

        // Variance
        float var = 1.0f / sqrtf(squareSum / size + epsilon);
        __m512 vvar = _mm512_set1_ps(var);

        for (col = 0; col + 15 < size; col += 16) {
            __m512 vx = xft::load_avx512(px + col);
            __m512 vw = xft::load_avx512(weight + col);
            __m512 vy = vx * vvar * vw;
            xft::store_avx512(py + col, 0xffff, vy);
        }
        if (col < size) {
            __mmask16 mask = (1 << (size - col)) - 1;
            __m512 vx = xft::load_avx512(mask, px + col);
            __m512 vw = xft::load_avx512(mask, weight + col);
            __m512 vy = vx * vvar * vw;
            xft::store_avx512(py + col, mask, vy);
        }
    } // end for rows
}

void rmsNorm(bfloat16_t *output, const bfloat16_t *input, const bfloat16_t *weight, int rows, int cols, int iStride,
        int oStride, float epsilon) {
    int size = cols;

    if (iStride == -1) iStride = cols;
    if (oStride == -1) oStride = cols;

#pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        const bfloat16_t *px = input + r * iStride;
        bfloat16_t *py = output + r * oStride;

        float squareSum = 0;

        __m512 vsqare = _mm512_set1_ps(0);

        int col = 0;
        for (; col + 15 < size; col += 16) {
            // SUM(x*x)
            __m512 vx = _mm512_loadu_pbh(px + col);
            __m512 tmp = _mm512_mul_ps(vx, vx);
            vsqare = _mm512_add_ps(vsqare, tmp);
        }
        if (col < size) {
            __mmask16 mask = (1 << (size - col)) - 1;
            __m512 vx = _mm512_maskz_loadu_pbh(mask, px + col);
            __m512 tmp = _mm512_mul_ps(vx, vx);
            vsqare = _mm512_add_ps(vsqare, tmp);
        }

        squareSum = _mm512_reduce_add_ps(vsqare);

        // Variance
        float var = 1.0f / sqrtf(squareSum / size + epsilon);
        __m512 vvar = _mm512_set1_ps(var);

        for (col = 0; col + 15 < size; col += 16) {
            __m512 vx = _mm512_loadu_pbh(px + col);
            __m512 vw = _mm512_loadu_pbh(weight + col);
            __m512 vy = vx * vvar * vw;
            _mm512_storeu_pbh(py + col, vy);
        }
        if (col < size) {
            __mmask16 mask = (1 << (size - col)) - 1;
            __m512 vx = _mm512_maskz_loadu_pbh(mask, px + col);
            __m512 vw = _mm512_maskz_loadu_pbh(mask, weight + col);
            __m512 vy = vx * vvar * vw;
            _mm512_mask_storeu_pbh(py + col, mask, vy);
        }
    } // end for rows
}

void rmsNorm(float *output, const float *input, const float *weight, int rows, int cols, int iStride, int oStride,
        float epsilon) {
    rmsNorm<float, float>(output, input, weight, rows, cols, iStride, oStride, epsilon);
}

void rmsNorm(bfloat16_t *output, const float *input, const float *weight, int rows, int cols, int iStride, int oStride,
        float epsilon) {
    rmsNorm<bfloat16_t, float>(output, input, weight, rows, cols, iStride, oStride, epsilon);
}

void rmsNorm(bfloat16_t *output, const bfloat16_t *input, const float *weight, int rows, int cols, int iStride,
        int oStride, float epsilon) {
    rmsNorm<bfloat16_t, bfloat16_t>(output, input, weight, rows, cols, iStride, oStride, epsilon);
}

void rmsNorm(float16_t *output, const float *input, const float *weight, int rows, int cols, int iStride, int oStride,
        float epsilon) {
    rmsNorm<float16_t, float>(output, input, weight, rows, cols, iStride, oStride, epsilon);
}

void rmsNorm(float16_t *output, const float16_t *input, const float *weight, int rows, int cols, int iStride,
        int oStride, float epsilon) {
    rmsNorm<float16_t, float16_t>(output, input, weight, rows, cols, iStride, oStride, epsilon);
}

void rmsNorm(float16_t *output, const float16_t *input, const float16_t *weight, int rows, int cols, int iStride,
        int oStride, float epsilon) {
    rmsNorm<float16_t, float16_t>(output, input, weight, rows, cols, iStride, oStride, epsilon);
}

void invokeRmsNorm(DataType dt, void *output, const void *input, const void *weight, int rows, int cols, int iStride,
        int oStride, float epsilon) {
    if (dt == DataType::fp32) {
        rmsNorm((float *)output, (const float *)input, (const float *)weight, rows, cols, iStride, oStride, epsilon);
    } else if (dt == DataType::bf16) {
        rmsNorm((bfloat16_t *)output, (const bfloat16_t *)input, (const bfloat16_t *)weight, rows, cols, iStride,
                oStride, epsilon);
    } else if (dt == DataType::fp16) {
        rmsNorm((float16_t *)output, (const float16_t *)input, (const float16_t *)weight, rows, cols, iStride, oStride,
                epsilon);
    }
}

} // namespace xft