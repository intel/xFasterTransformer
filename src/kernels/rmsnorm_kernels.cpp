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
#include "my_types.h"
#include "rmsnorm_kernels.h"

namespace xft {

template <typename T>
void rmsNorm(T *output, const float *input, const float *weight, int rows, int cols, int iStride, int oStride,
        float epsilon) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>,
            "Template parameter of rmsNorm must be either float or bfloat16_t");

#if (__GNUC__ > 10) || ((__GNUC__ == 10) && (__GNUC_MINOR__ >= 1))
    auto cvt_fp32_to_bf16 = [&](const __m512 input_vector) { return (__m256i)_mm512_cvtneps_pbh(input_vector); };
#else
    const __m512i nan = _mm512_set1_epi32(0xffff);
    const __m512i ones = _mm512_set1_epi32(0x1);
    const __m512i vec_bias = _mm512_set1_epi32(0x7fff);
    auto cvt_fp32_to_bf16 = [&](const __m512 input_vector) {
        __m512i value = _mm512_castps_si512(input_vector);
        auto mask = _mm512_cmp_ps_mask(input_vector, input_vector, _CMP_ORD_Q);
        auto result = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
        result = _mm512_add_epi32(result, vec_bias);
        result = _mm512_add_epi32(result, value);
        result = _mm512_srli_epi32(result, 16);
        result = _mm512_mask_blend_epi32(mask, nan, result);
        return _mm512_cvtusepi32_epi16(result);
    };
#endif

    int size = cols;
    if (iStride == -1) iStride = cols;
    if (oStride == -1) oStride = cols;

#pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        const float *px = input + r * iStride;
        T *py = output + r * oStride;

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
            if constexpr (std::is_same_v<T, float>) {
                _mm512_storeu_ps(py + col, vy);
            }
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                __m256i vbf16 = cvt_fp32_to_bf16(vy);
                _mm256_mask_storeu_epi16(py + col, 0xffff, vbf16);
            }
        }
        if (col < size) {
            __mmask16 mask = (1 << (size - col)) - 1;
            __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
            __m512 vw = _mm512_maskz_loadu_ps(mask, weight + col);
            __m512 vy = vx * vvar * vw;
            if constexpr (std::is_same_v<T, float>) {
                _mm512_mask_storeu_ps(py + col, mask, vy);
            }
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                __m256i vbf16 = cvt_fp32_to_bf16(vy);
                _mm256_mask_storeu_epi16(py + col, mask, vbf16);
            }
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
        float var = 1 / sqrt(squareSum / size + epsilon);
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
    rmsNorm<float>(output, input, weight, rows, cols, iStride, oStride, epsilon);
}

void rmsNorm(bfloat16_t *output, const float *input, const float *weight, int rows, int cols, int iStride, int oStride,
        float epsilon) {
    rmsNorm<bfloat16_t>(output, input, weight, rows, cols, iStride, oStride, epsilon);
}

void invokeRmsNorm(DataType dt, void *output, const void *input, const void *weight, int rows, int cols, int iStride,
        int oStride, float epsilon) {
    if (dt == DataType::fp32) {
        rmsNorm((float *)output, (const float *)input, (const float *)weight, rows, cols, iStride, oStride, epsilon);
    } else if (dt == DataType::bf16) {
        rmsNorm((bfloat16_t *)output, (const bfloat16_t *)input, (const bfloat16_t *)weight, rows, cols, iStride,
                oStride, epsilon);
    }
}

} // namespace xft