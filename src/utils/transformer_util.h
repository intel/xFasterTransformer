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

#include <cmath>
#include <cstdio>

#include "compile_util.h"

class TransformerUtil {
public:
#if __AVX512F__
    static void layerNorm(float *x, float *y, const float *gamma, const float *beta, int rows, int cols, int stride,
            const float epsilon = 1e-6) {
        int size = cols;

#pragma omp parallel for
        for (int r = 0; r < rows; ++r) {
            float *px = x + r * stride;
            float *py = y + r * stride;

            float sum = 0;
            float squareSum = 0;

            __m512 vsum = _mm512_set1_ps(0);
            __m512 vsqare = _mm512_set1_ps(0);

            for (int c = 0; c < size; c += 16) {
                int remain = size - c;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                // SUM(x)
                __m512 vx = _mm512_maskz_loadu_ps(mask, px + c);
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
#else
    static void layerNorm(float *x, float *y, const float *gamma, const float *beta, int rows, int cols, int stride) {
#pragma omp parallel for
        for (int i = 0; i < rows; ++i) {
            float sum = 0;
            float *px = x + i * stride;
            float *py = y + i * stride;
#pragma omp simd
            for (int j = 0; j < cols; ++j) {
                sum += px[j];
            }

            float mean = sum / cols;

            sum = 0;
#pragma omp simd
            for (int j = 0; j < cols; ++j) {
                float delta = (px[j] - mean);
                sum += delta * delta;
            }
            float tmp = sum / cols + 9.999999960041972e-13;
            float rvariance = 1.0f / sqrt(tmp);

#pragma omp simd
            for (int j = 0; j < cols; ++j) {
                py[j] = (px[j] - mean) * rvariance * gamma[j] + beta[j];
            }
        }
    }
#endif

    static float dotProduct(const float *A, const float *B, int size) {
        __m512 acc = _mm512_setzero_ps();

        for (int i = 0; i < size; i += 16) {
            int remain = size - i;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 va = _mm512_maskz_loadu_ps(mask, A + i);
            __m512 vb = _mm512_maskz_loadu_ps(mask, B + i);
            acc = _mm512_fmadd_ps(va, vb, acc);
        }

        return _mm512_reduce_add_ps(acc);
    }

    // Vector (1 x M) * Matrix (M x N)
    static void gevm(const float *A, const float *B, float *C, int M, int N, int strideB) {
        switch (N) {
            case 64: gevm<64>(A, B, C, M, strideB); return;
            case 128: gevm<128>(A, B, C, M, strideB); return;
        }

        int vectors = (N + 15) / 16;
        __m512 vc[vectors];

        int remain = N % 16;
        __mmask16 tailMask = (remain == 0 ? 0xffff : (1 << remain) - 1); // mask for last vector

        for (int i = 0; i < vectors; ++i) {
            vc[i] = _mm512_setzero_ps();
        }

        for (int i = 0; i < M; ++i) {
            __m512 va = _mm512_set1_ps(A[i]);
            const float *pB = B + i * strideB;

            for (int i = 0; i < vectors - 1; ++i) {
                __m512 vb = _mm512_loadu_ps(pB + i * 16);
                vc[i] = _mm512_fmadd_ps(va, vb, vc[i]);
            }

            __m512 vb = _mm512_maskz_loadu_ps(tailMask, pB + (vectors - 1) * 16);
            vc[vectors - 1] = _mm512_fmadd_ps(va, vb, vc[vectors - 1]);
        }

        for (int i = 0; i < vectors; ++i) {
            __mmask16 mask = ((i == vectors - 1) ? tailMask : 0xffff);
            _mm512_mask_storeu_ps(C + 16 * i, mask, vc[i]);
        }
    }

    // Special version like N = 64
    template <int N>
    static void gevm(const float *A, const float *B, float *C, int M, int strideB) {
        constexpr int vectors = N / 16;
        __m512 vc[vectors];

        for (int i = 0; i < vectors; ++i) {
            vc[i] = _mm512_setzero_ps();
        }

        for (int i = 0; i < M; ++i) {
            __m512 va = _mm512_set1_ps(A[i]);
            const float *pB = B + i * strideB;

            for (int i = 0; i < vectors; ++i) {
                __m512 vb = _mm512_loadu_ps(pB + i * 16);
                vc[i] = _mm512_fmadd_ps(va, vb, vc[i]);
            }
        }

        for (int i = 0; i < vectors; ++i) {
            _mm512_storeu_ps(C + 16 * i, vc[i]);
        }
    }

    // Small GEMM for the transposed B
    // A: not transposed, in shape of M x K, like 32 x 256
    // B: transposed, in shape of N x K, like 32 x 256
    // C: in shape M x N
    template <int M, int N>
    static void small_gemm_transb(const float *A, const float *B, float *C, int K, int lda, int ldb, int ldc) {
        constexpr const int maxCols = 28; // For safe to avoid register spill
        constexpr const int blks = (N + maxCols - 1) / maxCols;
        constexpr const int bn = (N + blks - 1) / blks;

        REQUIRES(N % bn == 0, "N must be multiple of bn.");
        REQUIRES(K % 16 == 0, "K must be multiple of 16.");

        __m512 vres[bn];

        //      bn        bn
        //   ___________________
        //  |   1-->  |         | `
        //  |   2-->  |         | |
        //  |   ...   |         | |
        //  |         |         | M
        //  |         |         | |
        //  |         |         | |
        //  |         |         | /
        //  `````````````````````
        for (int blk = 0; blk < blks; ++blk) {
            const int noff = blk * bn;
            for (int i = 0; i < M; ++i) {
                const float *pa = A + i * lda;
                compile_time_for<bn>::op([&](auto j) { vres[j] = _mm512_set1_ps(0); });
                // Accumulate along k dim
                for (int k = 0; k < K; k += 16) {
                    __m512 va = _mm512_loadu_ps(pa + k);
                    compile_time_for<bn>::op([&](auto j) {
                        const float *pb = B + (noff + j) * ldb;
                        __m512 vb = _mm512_loadu_ps(pb + k);
                        vres[j] = _mm512_fmadd_ps(va, vb, vres[j]);
                    });
                }
                // Reduce and store the result
                compile_time_for<bn>::op([&](auto j) {
                    float v = _mm512_reduce_add_ps(vres[j]);
                    C[i * ldc + noff + j] = v;
                });
            }
        }
    }
};