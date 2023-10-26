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

template <int M, int N>
void small_gemm_transb(const float *A, const float *B, float *C, int K, int lda, int ldb, int ldc) {
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
        const float *pA = A + v * 16;
        const float *pB = B + v * 16;
        __m512 vb[N];
        __m512 va;

        compile_time_for<M * N>::op([&](auto i) {
            constexpr int idx = i;
            // Load from A when reach to first column in vc matrix
            if constexpr (idx % N == 0) {
                va = _mm512_loadu_ps(pA);
                pA += lda;
            }
            // Load from B when reach to first row in vc matrix
            if constexpr (idx < N) {
                vb[idx] = _mm512_loadu_ps(pB);
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

        const float *pA = A + (vecs - 1) * 16;
        const float *pB = B + (vecs - 1) * 16;
        float *pC = C;

        compile_time_for<M * N>::op([&](auto i) {
            constexpr int idx = i;
            if constexpr (idx % N == 0) {
                va = _mm512_maskz_loadu_ps(mask, pA);
                pA += lda;
            }
            if constexpr (idx < N) {
                vb[idx] = _mm512_maskz_loadu_ps(mask, pB);
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

void small_gemm_transb_6x4(const float *A, const float *B, float *C, int K, int lda, int ldb, int ldc) {
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
        const float *pA = A + v * 16;
        const float *pB = B + v * 16;
        __m512 vb[4];

        __m512 va = _mm512_loadu_ps(pA);
        pA += lda;
        vb[0] = _mm512_loadu_ps(pB);
        pB += ldb;
        vc[0] = _mm512_fmadd_ps(va, vb[0], vc[0]);
        vb[1] = _mm512_loadu_ps(pB);
        pB += ldb;
        vc[1] = _mm512_fmadd_ps(va, vb[1], vc[1]);
        vb[2] = _mm512_loadu_ps(pB);
        pB += ldb;
        vc[2] = _mm512_fmadd_ps(va, vb[2], vc[2]);
        vb[3] = _mm512_loadu_ps(pB);
        pB += ldb;
        vc[3] = _mm512_fmadd_ps(va, vb[3], vc[3]);

        va = _mm512_loadu_ps(pA);
        pA += lda;
        vc[4] = _mm512_fmadd_ps(va, vb[0], vc[4]);
        vc[5] = _mm512_fmadd_ps(va, vb[1], vc[5]);
        vc[6] = _mm512_fmadd_ps(va, vb[2], vc[6]);
        vc[7] = _mm512_fmadd_ps(va, vb[3], vc[7]);

        va = _mm512_loadu_ps(pA);
        pA += lda;
        vc[8] = _mm512_fmadd_ps(va, vb[0], vc[8]);
        vc[9] = _mm512_fmadd_ps(va, vb[1], vc[9]);
        vc[10] = _mm512_fmadd_ps(va, vb[2], vc[10]);
        vc[11] = _mm512_fmadd_ps(va, vb[3], vc[11]);

        va = _mm512_loadu_ps(pA);
        pA += lda;
        vc[12] = _mm512_fmadd_ps(va, vb[0], vc[12]);
        vc[13] = _mm512_fmadd_ps(va, vb[1], vc[13]);
        vc[14] = _mm512_fmadd_ps(va, vb[2], vc[14]);
        vc[15] = _mm512_fmadd_ps(va, vb[3], vc[15]);

        va = _mm512_loadu_ps(pA);
        pA += lda;
        vc[16] = _mm512_fmadd_ps(va, vb[0], vc[16]);
        vc[17] = _mm512_fmadd_ps(va, vb[1], vc[17]);
        vc[18] = _mm512_fmadd_ps(va, vb[2], vc[18]);
        vc[19] = _mm512_fmadd_ps(va, vb[3], vc[19]);

        va = _mm512_loadu_ps(pA);
        pA += lda;
        vc[20] = _mm512_fmadd_ps(va, vb[0], vc[20]);
        vc[21] = _mm512_fmadd_ps(va, vb[1], vc[21]);
        vc[22] = _mm512_fmadd_ps(va, vb[2], vc[22]);
        vc[23] = _mm512_fmadd_ps(va, vb[3], vc[23]);
    }

    // The last vector computing, together with data store
    const float *pA = A + (vecs - 1) * 16;
    const float *pB = B + (vecs - 1) * 16;
    float *pC = C;
    __m512 vb[4];

    __m512 va = _mm512_maskz_loadu_ps(mask, pA);
    pA += lda;
    vb[0] = _mm512_maskz_loadu_ps(mask, pB);
    pB += ldb;
    vc[0] = _mm512_fmadd_ps(va, vb[0], vc[0]);
    pC[0] = _mm512_reduce_add_ps(vc[0]);
    vb[1] = _mm512_maskz_loadu_ps(mask, pB);
    pB += ldb;
    vc[1] = _mm512_fmadd_ps(va, vb[1], vc[1]);
    pC[1] = _mm512_reduce_add_ps(vc[1]);
    vb[2] = _mm512_maskz_loadu_ps(mask, pB);
    pB += ldb;
    vc[2] = _mm512_fmadd_ps(va, vb[2], vc[2]);
    pC[2] = _mm512_reduce_add_ps(vc[2]);
    vb[3] = _mm512_maskz_loadu_ps(mask, pB);
    pB += ldb;
    vc[3] = _mm512_fmadd_ps(va, vb[3], vc[3]);
    pC[3] = _mm512_reduce_add_ps(vc[3]);
    pC += ldc;

    va = _mm512_maskz_loadu_ps(mask, pA);
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

    va = _mm512_maskz_loadu_ps(mask, pA);
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

    va = _mm512_maskz_loadu_ps(mask, pA);
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

    va = _mm512_maskz_loadu_ps(mask, pA);
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

    va = _mm512_maskz_loadu_ps(mask, pA);
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

// M is a fixed small number
template <int M>
void small_gemm_transb(const float *A, const float *B, float *C, int N, int K, int lda, int ldb, int ldc) {
    int j = 0;
    const float *pA = A;
    constexpr int NB = 4;

    for (; j + NB - 1 < N; j += NB) {
        const float *pB = B + j * ldb;
        if constexpr (M == 6 && NB == 4) {
            small_gemm_transb_6x4(pA, pB, C + j, K, lda, ldb, ldc);
        } else {
            small_gemm_transb<M, NB>(pA, pB, C + j, K, lda, ldb, ldc);
        }
    }

    // Remain part in B
    if (j < N) {
        const float *pB = B + j * ldb;
        switch (N - j) {
            case 2: small_gemm_transb<M, 2>(pA, pB, C + j, K, lda, ldb, ldc); break;
            case 1: small_gemm_transb<M, 1>(pA, pB, C + j, K, lda, ldb, ldc); break;
            case 3: small_gemm_transb<M, 3>(pA, pB, C + j, K, lda, ldb, ldc); break;
        }
    }
}

// A: M x K
// B: N x K
// C: M x N
void small_gemm_transb(const float *A, const float *B, float *C, int M, int N, int K, int lda, int ldb, int ldc) {
    int i = 0;
    constexpr int MB = 6;

    for (i = 0; i + MB - 1 < M; i += MB) {
        const float *pA = A + i * lda;
        small_gemm_transb<MB>(pA, B, C + i * ldc, N, K, lda, ldb, ldc);
    }

    // Remain part in A
    if (i < M) {
        const float *pA = A + i * lda;
        const int remain = M - i;

        switch (remain) {
            case 2: small_gemm_transb<2>(pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
            case 4: small_gemm_transb<4>(pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
            case 1: small_gemm_transb<1>(pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
            case 3: small_gemm_transb<3>(pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
            case 5: small_gemm_transb<5>(pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// To check if the block can be skipped by checking if the attention mask is the lowest value
// lda: leading dimension of the attention mask
template <int COLS>
inline bool isBlockSkippable(const float *attnMask, int lda, int rows) {
    if constexpr (COLS != 4) {
        printf("COLS=%d is not supported in isBlockSkippable\n", COLS);
        exit(-1);
    }

    const float lowest = std::numeric_limits<float>::lowest();

    if (*(attnMask + (rows - 1) * lda) == lowest) { // last row is the lowest, then most likely all lowest
        for (int i = 0; i < rows; ++i) {
            const float *p = attnMask + i * lda;
            if (p[0] != lowest || p[1] != lowest || p[2] != lowest || p[3] != lowest) { return false; }
        }
        return true;
    }

    return false;
}

// M is a fixed small number
template <int M>
void small_gemm_transb(
        const float *attnMask, const float *A, const float *B, float *C, int N, int K, int lda, int ldb, int ldc) {
    int j = 0;
    const float *pA = A;
    constexpr int NB = 4;

    for (; j + NB - 1 < N; j += NB) {
        const float *pB = B + j * ldb;
        if (isBlockSkippable<NB>(attnMask + j, N, M)) { continue; }

        if constexpr (M == 6 && NB == 4) {
            small_gemm_transb_6x4(pA, pB, C + j, K, lda, ldb, ldc);
        } else {
            small_gemm_transb<M, NB>(pA, pB, C + j, K, lda, ldb, ldc);
        }
    }

    // Remain part in B
    if (j < N) {
        const float *pB = B + j * ldb;
        switch (N - j) {
            case 2: small_gemm_transb<M, 2>(pA, pB, C + j, K, lda, ldb, ldc); break;
            case 1: small_gemm_transb<M, 1>(pA, pB, C + j, K, lda, ldb, ldc); break;
            case 3: small_gemm_transb<M, 3>(pA, pB, C + j, K, lda, ldb, ldc); break;
        }
    }
}

// If attention mask is the lowest value in some position, skip the computation
// attnMask: attention mask with the shape of (M, N)
void small_gemm_transb(const float *attnMask, const float *A, const float *B, float *C, int M, int N, int K, int lda,
        int ldb, int ldc) {
    int i = 0;
    constexpr int MB = 6;

    for (i = 0; i + MB - 1 < M; i += MB) {
        const float *pA = A + i * lda;
        small_gemm_transb<MB>(attnMask + i * N, pA, B, C + i * ldc, N, K, lda, ldb, ldc);
    }

    // Remain part in A
    if (i < M) {
        const float *pA = A + i * lda;
        const int remain = M - i;

        switch (remain) {
            case 2: small_gemm_transb<2>(attnMask + i * N, pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
            case 4: small_gemm_transb<4>(attnMask + i * N, pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
            case 1: small_gemm_transb<1>(attnMask + i * N, pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
            case 3: small_gemm_transb<3>(attnMask + i * N, pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
            case 5: small_gemm_transb<5>(attnMask + i * N, pA, B, C + i * ldc, N, K, lda, ldb, ldc); break;
        }
    }
}