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
#include "gemm_kernel_ext.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <random>

#include "compile_util.h"
#include "intrinsics_util.h"

template <typename TA, typename TB, int M, int N>
void small_gemm_transb(const TA *A, const TB *B, float *C, int K, int lda, int ldb, int ldc) {
    // vc[0] vc[1]   ... vc[N-1]
    // vc[N] vc[N+1] ...
    // ..
    // vc[(M-1)*N] ...
    __m512 vc[M * N];

    int vecs = (K + 15) / 16; // vector size in AVX512
    __mmask16 mask = (K % 16 == 0 ? 0xffff : (1 << (K % 16)) - 1); // mask for last vector

    compile_time_for<M * N>::op([&vc](auto i) { vc[i] = _mm512_set1_ps(0); });

    // The last vector is not included
    for (int v = 0; v < vecs - 1; ++v) {
        const TA *pA = A + v * 16;
        const TB *pB = B + v * 16;
        __m512 vb[N];
        __m512 va;

        compile_time_for<M * N>::op([&](auto i) {
            constexpr int idx = i;
            // Load from A when reach to first column in vc matrix
            if constexpr (idx % N == 0) {
                va = xft::load_avx512(pA);
                pA += lda;
            }
            // Load from B when reach to first row in vc matrix
            if constexpr (idx < N) {
                vb[idx] = xft::load_avx512(pB);
                pB += ldb;
            }
            constexpr int col = idx % N;
            vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
        });
    }

    // The last vector computing, together with data store
    {
        __m512 vb[N];
        __m512 va;

        const TA *pA = A + (vecs - 1) * 16;
        const TB *pB = B + (vecs - 1) * 16;
        float *pC = C;

        compile_time_for<M * N>::op([&](auto i) {
            constexpr int idx = i;
            if constexpr (idx % N == 0) {
                va = xft::load_avx512(mask, pA);
                pA += lda;
            }
            if constexpr (idx < N) {
                vb[idx] = xft::load_avx512(mask, pB);
                pB += ldb;
            }
            constexpr int col = idx % N;
            vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
            pC[col] = _mm512_reduce_add_ps(vc[idx]);
            // Reach to the row end
            if constexpr (i % N == N - 1) { pC += ldc; }
        });
    }
}

template <typename TA, typename TB>
void small_gemm_transb_6x4(const TA *A, const TB *B, float *C, int K, int lda, int ldb, int ldc) {
    // vc[00] vc[01] vc[02] vc[03]
    // vc[04] vc[05] vc[06] vc[07]
    // vc[08] vc[09] vc[10] vc[11]
    // vc[12] vc[13] vc[14] vc[15]
    // vc[16] vc[17] vc[18] vc[19]
    // vc[20] vc[21] vc[22] vc[23]
    __m512 vc[24];

    int vecs = (K + 15) / 16; // vector size in AVX512
    __mmask16 mask = (K % 16 == 0 ? 0xffff : (1 << (K % 16)) - 1); // mask for last vector

    vc[0] = _mm512_set1_ps(0);
    vc[1] = _mm512_set1_ps(0);
    vc[2] = _mm512_set1_ps(0);
    vc[3] = _mm512_set1_ps(0);
    vc[4] = _mm512_set1_ps(0);
    vc[5] = _mm512_set1_ps(0);
    vc[6] = _mm512_set1_ps(0);
    vc[7] = _mm512_set1_ps(0);
    vc[8] = _mm512_set1_ps(0);
    vc[9] = _mm512_set1_ps(0);
    vc[10] = _mm512_set1_ps(0);
    vc[11] = _mm512_set1_ps(0);
    vc[12] = _mm512_set1_ps(0);
    vc[13] = _mm512_set1_ps(0);
    vc[14] = _mm512_set1_ps(0);
    vc[15] = _mm512_set1_ps(0);
    vc[16] = _mm512_set1_ps(0);
    vc[17] = _mm512_set1_ps(0);
    vc[18] = _mm512_set1_ps(0);
    vc[19] = _mm512_set1_ps(0);
    vc[20] = _mm512_set1_ps(0);
    vc[21] = _mm512_set1_ps(0);
    vc[22] = _mm512_set1_ps(0);
    vc[23] = _mm512_set1_ps(0);

    // The last vector is not included
    for (int v = 0; v < vecs - 1; ++v) {
        const TA *pA = A + v * 16;
        const TB *pB = B + v * 16;
        __m512 vb[4];

        __m512 va = xft::load_avx512(pA);
        pA += lda;
        vb[0] = xft::load_avx512(pB);
        pB += ldb;
        vc[0] = _mm512_fmadd_ps(va, vb[0], vc[0]);
        vb[1] = xft::load_avx512(pB);
        pB += ldb;
        vc[1] = _mm512_fmadd_ps(va, vb[1], vc[1]);
        vb[2] = xft::load_avx512(pB);
        pB += ldb;
        vc[2] = _mm512_fmadd_ps(va, vb[2], vc[2]);
        vb[3] = xft::load_avx512(pB);
        pB += ldb;
        vc[3] = _mm512_fmadd_ps(va, vb[3], vc[3]);

        va = xft::load_avx512(pA);
        pA += lda;
        vc[4] = _mm512_fmadd_ps(va, vb[0], vc[4]);
        vc[5] = _mm512_fmadd_ps(va, vb[1], vc[5]);
        vc[6] = _mm512_fmadd_ps(va, vb[2], vc[6]);
        vc[7] = _mm512_fmadd_ps(va, vb[3], vc[7]);

        va = xft::load_avx512(pA);
        pA += lda;
        vc[8] = _mm512_fmadd_ps(va, vb[0], vc[8]);
        vc[9] = _mm512_fmadd_ps(va, vb[1], vc[9]);
        vc[10] = _mm512_fmadd_ps(va, vb[2], vc[10]);
        vc[11] = _mm512_fmadd_ps(va, vb[3], vc[11]);

        va = xft::load_avx512(pA);
        pA += lda;
        vc[12] = _mm512_fmadd_ps(va, vb[0], vc[12]);
        vc[13] = _mm512_fmadd_ps(va, vb[1], vc[13]);
        vc[14] = _mm512_fmadd_ps(va, vb[2], vc[14]);
        vc[15] = _mm512_fmadd_ps(va, vb[3], vc[15]);

        va = xft::load_avx512(pA);
        pA += lda;
        vc[16] = _mm512_fmadd_ps(va, vb[0], vc[16]);
        vc[17] = _mm512_fmadd_ps(va, vb[1], vc[17]);
        vc[18] = _mm512_fmadd_ps(va, vb[2], vc[18]);
        vc[19] = _mm512_fmadd_ps(va, vb[3], vc[19]);

        va = xft::load_avx512(pA);
        pA += lda;
        vc[20] = _mm512_fmadd_ps(va, vb[0], vc[20]);
        vc[21] = _mm512_fmadd_ps(va, vb[1], vc[21]);
        vc[22] = _mm512_fmadd_ps(va, vb[2], vc[22]);
        vc[23] = _mm512_fmadd_ps(va, vb[3], vc[23]);
    }

    // The last vector computing, together with data store
    const TA *pA = A + (vecs - 1) * 16;
    const TB *pB = B + (vecs - 1) * 16;
    float *pC = C;
    __m512 vb[4];

    __m512 va = xft::load_avx512(mask, pA);
    pA += lda;
    vb[0] = xft::load_avx512(mask, pB);
    pB += ldb;
    vc[0] = _mm512_fmadd_ps(va, vb[0], vc[0]);
    pC[0] = _mm512_reduce_add_ps(vc[0]);
    vb[1] = xft::load_avx512(mask, pB);
    pB += ldb;
    vc[1] = _mm512_fmadd_ps(va, vb[1], vc[1]);
    pC[1] = _mm512_reduce_add_ps(vc[1]);
    vb[2] = xft::load_avx512(mask, pB);
    pB += ldb;
    vc[2] = _mm512_fmadd_ps(va, vb[2], vc[2]);
    pC[2] = _mm512_reduce_add_ps(vc[2]);
    vb[3] = xft::load_avx512(mask, pB);
    pB += ldb;
    vc[3] = _mm512_fmadd_ps(va, vb[3], vc[3]);
    pC[3] = _mm512_reduce_add_ps(vc[3]);
    pC += ldc;

    va = xft::load_avx512(mask, pA);
    pA += lda;
    vc[4] = _mm512_fmadd_ps(va, vb[0], vc[4]);
    pC[0] = _mm512_reduce_add_ps(vc[4]);
    vc[5] = _mm512_fmadd_ps(va, vb[1], vc[5]);
    pC[1] = _mm512_reduce_add_ps(vc[5]);
    vc[6] = _mm512_fmadd_ps(va, vb[2], vc[6]);
    pC[2] = _mm512_reduce_add_ps(vc[6]);
    vc[7] = _mm512_fmadd_ps(va, vb[3], vc[7]);
    pC[3] = _mm512_reduce_add_ps(vc[7]);
    pC += ldc;

    va = xft::load_avx512(mask, pA);
    pA += lda;
    vc[8] = _mm512_fmadd_ps(va, vb[0], vc[8]);
    pC[0] = _mm512_reduce_add_ps(vc[8]);
    vc[9] = _mm512_fmadd_ps(va, vb[1], vc[9]);
    pC[1] = _mm512_reduce_add_ps(vc[9]);
    vc[10] = _mm512_fmadd_ps(va, vb[2], vc[10]);
    pC[2] = _mm512_reduce_add_ps(vc[10]);
    vc[11] = _mm512_fmadd_ps(va, vb[3], vc[11]);
    pC[3] = _mm512_reduce_add_ps(vc[11]);
    pC += ldc;

    va = xft::load_avx512(mask, pA);
    pA += lda;
    vc[12] = _mm512_fmadd_ps(va, vb[0], vc[12]);
    pC[0] = _mm512_reduce_add_ps(vc[12]);
    vc[13] = _mm512_fmadd_ps(va, vb[1], vc[13]);
    pC[1] = _mm512_reduce_add_ps(vc[13]);
    vc[14] = _mm512_fmadd_ps(va, vb[2], vc[14]);
    pC[2] = _mm512_reduce_add_ps(vc[14]);
    vc[15] = _mm512_fmadd_ps(va, vb[3], vc[15]);
    pC[3] = _mm512_reduce_add_ps(vc[15]);
    pC += ldc;

    va = xft::load_avx512(mask, pA);
    pA += lda;
    vc[16] = _mm512_fmadd_ps(va, vb[0], vc[16]);
    pC[0] = _mm512_reduce_add_ps(vc[16]);
    vc[17] = _mm512_fmadd_ps(va, vb[1], vc[17]);
    pC[1] = _mm512_reduce_add_ps(vc[17]);
    vc[18] = _mm512_fmadd_ps(va, vb[2], vc[18]);
    pC[2] = _mm512_reduce_add_ps(vc[18]);
    vc[19] = _mm512_fmadd_ps(va, vb[3], vc[19]);
    pC[3] = _mm512_reduce_add_ps(vc[19]);
    pC += ldc;

    va = xft::load_avx512(mask, pA);
    pA += lda;
    vc[20] = _mm512_fmadd_ps(va, vb[0], vc[20]);
    pC[0] = _mm512_reduce_add_ps(vc[20]);
    vc[21] = _mm512_fmadd_ps(va, vb[1], vc[21]);
    pC[1] = _mm512_reduce_add_ps(vc[21]);
    vc[22] = _mm512_fmadd_ps(va, vb[2], vc[22]);
    pC[2] = _mm512_reduce_add_ps(vc[22]);
    vc[23] = _mm512_fmadd_ps(va, vb[3], vc[23]);
    pC[3] = _mm512_reduce_add_ps(vc[23]);
}

// For the case like M = 1, K = 128/256
template <typename TA, typename TB, int K>
void small_gemm_transb_1xn_fixk(const TA *A, const TB *B, float *C, int N, int lda, int ldb, int ldc) {
    constexpr int vecs = (K + 15) / 16; // vector size in AVX512
    constexpr __mmask16 mask = (K % 16 == 0 ? 0xffff : (1 << (K % 16)) - 1); // mask for last vector
    constexpr int BC = 8; // how many elements computed together

    __m512 va[vecs];

    // Load A
    compile_time_for<vecs>::op([&](auto idx) {
        if constexpr (idx == vecs - 1) {
            va[idx] = xft::load_avx512(mask, A + idx * 16);
        } else {
            va[idx] = xft::load_avx512(0xffff, A + idx * 16);
        }
    });

    // Each loop compute 'BC' elements in C
    int i = 0;
    for (; i + BC - 1 < N; i += BC) {
        const TB *pB = B + i * ldb;

        __m512 vc[BC];
        compile_time_for<BC>::op([&](auto idx) { vc[idx] = _mm512_set1_ps(0); });

        for (int j = 0; j < vecs; ++j) {
            __mmask16 m = (j == vecs - 1 ? mask : 0xffff);
            compile_time_for<BC>::op([&](auto idx) {
                __m512 vb = xft::load_avx512(m, pB + idx * ldb + j * 16);
                vc[idx] = _mm512_fmadd_ps(va[j], vb, vc[idx]);
            });
        }

        // Store to C
        compile_time_for<BC>::op([&](auto idx) { C[i + idx] = _mm512_reduce_add_ps(vc[idx]); });
    }

    // Remain elements
    for (; i < N; ++i) {
        const TB *pB = B + i * ldb;
        __m512 vc = _mm512_set1_ps(0);

        for (int j = 0; j < vecs; ++j) {
            __mmask16 m = (j == vecs - 1 ? mask : 0xffff);
            __m512 vb = xft::load_avx512(m, pB + j * 16);
            vc = _mm512_fmadd_ps(va[j], vb, vc);
        }

        C[i] = _mm512_reduce_add_ps(vc);
    }
}

// For the case like M = 1
template <typename TA, typename TB>
void small_gemm_transb_1xn_dynk(const TA *A, const TB *B, float *C, int N, int K, int lda, int ldb, int ldc) {
    int vecs = (K + 15) / 16; // vector size in AVX512
    __mmask16 mask = (K % 16 == 0 ? 0xffff : (1 << (K % 16)) - 1); // mask for last vector
    constexpr int BC = 16; // how many elements computed together

    // Each loop compute 'BC' elements in C
    int i = 0;
    for (; i + BC - 1 < N; i += BC) {
        const TA *pA = A;
        const TB *pB = B + i * ldb;

        __m512 vc[BC];
        compile_time_for<BC>::op([&](auto idx) { vc[idx] = _mm512_set1_ps(0); });

        for (int j = 0; j < vecs; ++j) {
            __mmask16 m = (j == vecs - 1 ? mask : 0xffff);
            __m512 va = xft::load_avx512(m, pA + j * 16);
            compile_time_for<BC>::op([&](auto idx) {
                __m512 vb = xft::load_avx512(m, pB + idx * ldb + j * 16);
                vc[idx] = _mm512_fmadd_ps(va, vb, vc[idx]);
            });
        }

        // Store to C
        compile_time_for<BC>::op([&](auto idx) { C[i + idx] = _mm512_reduce_add_ps(vc[idx]); });
    }

    // Remain elements
    for (; i < N; ++i) {
        const TA *pA = A;
        const TB *pB = B + i * ldb;
        __m512 vc = _mm512_set1_ps(0);

        for (int j = 0; j < vecs; ++j) {
            __mmask16 m = (j == vecs - 1 ? mask : 0xffff);
            __m512 va = xft::load_avx512(m, pA + j * 16);
            __m512 vb = xft::load_avx512(m, pB + j * 16);
            vc = _mm512_fmadd_ps(va, vb, vc);
        }

        C[i] = _mm512_reduce_add_ps(vc);
    }
}

template <typename TA, typename TB>
void small_gemm_transb_1xn(const TA *A, const TB *B, float *C, int N, int K, int lda, int ldb, int ldc) {
    if (K == 128) {
        small_gemm_transb_1xn_fixk<TA, TB, 128>(A, B, C, N, lda, ldb, ldc);
    } else if (K == 256) {
        small_gemm_transb_1xn_fixk<TA, TB, 256>(A, B, C, N, lda, ldb, ldc);
    } else {
        small_gemm_transb_1xn_dynk(A, B, C, N, K, lda, ldb, ldc);
    }
}

// M is a fixed small number
template <typename TA, typename TB, int M>
void small_gemm_transb(const TA *A, const TB *B, float *C, int N, int K, int lda, int ldb, int ldc) {
    int j = 0;
    const TA *pA = A;
    constexpr int NB = 4;

    for (; j + NB - 1 < N; j += NB) {
        const TB *pB = B + j * ldb;
        if constexpr (M == 6 && NB == 4) {
            small_gemm_transb_6x4(pA, pB, C + j, K, lda, ldb, ldc);
        } else {
            small_gemm_transb<TA, TB, M, NB>(pA, pB, C + j, K, lda, ldb, ldc);
        }
    }

    // Remain part in B
    if (j < N) {
        const TB *pB = B + j * ldb;
        switch (N - j) {
            case 2: small_gemm_transb<TA, TB, M, 2>(pA, pB, C + j, K, lda, ldb, ldc); break;
            case 1: small_gemm_transb<TA, TB, M, 1>(pA, pB, C + j, K, lda, ldb, ldc); break;
            case 3: small_gemm_transb<TA, TB, M, 3>(pA, pB, C + j, K, lda, ldb, ldc); break;
        }
    }
}

// A: M x K
// B: N x K
// C: M x N
template <typename TA, typename TB>
void small_gemm_transb(const TA *A, const TB *B, float *C, int M, int N, int K, int lda, int ldb, int ldc) {
    int i = 0;
    constexpr int MB = 6;

    // Special case for M = 1
    if (M == 1) { return small_gemm_transb_1xn(A, B, C, N, K, lda, ldb, ldc); }

    for (i = 0; i + MB - 1 < M; i += MB) {
        const TA *pA = A + i * lda;
        small_gemm_transb<TA, TB, MB>(pA, B, C + i * ldc, N, K, lda, ldb, ldc);
    }

    // Remain part in A
    if (i < M) {
        const TA *pA = A + i * lda;
        const int remain = M - i;

        switch (remain) {
            case 1: small_gemm_transb<TA, TB, 1>(pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
            case 2: small_gemm_transb<TA, TB, 2>(pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
            case 3: small_gemm_transb<TA, TB, 3>(pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
            case 4: small_gemm_transb<TA, TB, 4>(pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
            case 5: small_gemm_transb<TA, TB, 5>(pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void small_gemm_transb(const float *A, const float *B, float *C, int M, int N, int K, int lda, int ldb, int ldc) {
    small_gemm_transb<float, float>(A, B, C, M, N, K, lda, ldb, ldc);
}

void small_gemm_transb(const float *A, const float16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc) {
    small_gemm_transb<float, float16_t>(A, B, C, M, N, K, lda, ldb, ldc);
}

void small_gemm_transb(
        const bfloat16_t *A, const float16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc) {
    small_gemm_transb<bfloat16_t, float16_t>(A, B, C, M, N, K, lda, ldb, ldc);
}

void small_gemm_transb(
        const bfloat16_t *A, const bfloat16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc) {
    small_gemm_transb<bfloat16_t, bfloat16_t>(A, B, C, M, N, K, lda, ldb, ldc);
}

////////////////////////////////////////////////////////////////////////////////

static void apply_scale(float *C, const float *scale, int M, int N, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; j += 16) {
            int remain = N - j;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
            __m512 v = xft::load_avx512(mask, &C[i * ldc + j]);
            __m512 scaleVec = xft::load_avx512(mask, scale + j);
            v = v * scaleVec;
            xft::store_avx512(&C[i * ldc + j], mask, v);
        }
    }
}

void small_gemm_transb(const float *A, const int8_t *B, const float *bScale, float *C, int M, int N, int K, int lda,
        int ldb, int ldc) {
    small_gemm_transb<float, int8_t>(A, B, C, M, N, K, lda, ldb, ldc);
    if (bScale) { apply_scale(C, bScale, M, N, ldc); }
}

void small_gemm_transb(const bfloat16_t *A, const int8_t *B, const float *bScale, float *C, int M, int N, int K,
        int lda, int ldb, int ldc) {
    small_gemm_transb<bfloat16_t, int8_t>(A, B, C, M, N, K, lda, ldb, ldc);
    if (bScale) { apply_scale(C, bScale, M, N, ldc); }
}