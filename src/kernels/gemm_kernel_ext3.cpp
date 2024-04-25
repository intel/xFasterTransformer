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
#include "compile_util.h"
#include "intrinsics_util.h"

#define AVX3_F32_NUM 16
#define INDEX(x, y, ld) ((x) * (ld) + (y))
#define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))

template <int COLS>
constexpr inline int get_max_rows() {
    if constexpr (COLS >= 15) {
        return 1;
    } else if constexpr (COLS >= 10) {
        return 2;
    } else if constexpr (COLS >= 7) {
        return 3;
    } else if constexpr (COLS >= 5) {
        return 4;
    } else if constexpr (COLS >= 4) {
        return 5;
    } else {
        return 6;
    }
}

template <int col, int COLS>
inline __mmask16 get_mask(__mmask16 tailMask) {
    // Not last column, return 0xffff indicating load/store all 16 floats
    if constexpr (col < COLS - 1)
        return (__mmask16)0xffff;
    else
        return tailMask;
}

// This function is for the case of very small M
// ALIGNED_N: the aligned size of N and is a multiple of AVX3_F32_NUM
template <int M, int ALIGNED_N, typename TA, typename TB, typename TC>
void small_sgemm_fixmn(const TA *A, const TB *B, TC *C, int lda, int ldb, int ldc, int actualN, int K, bool acc) {
    constexpr const int ROWS = M;
    constexpr const int COLS = ALIGNED_N / AVX3_F32_NUM;

    __m512 va[ROWS];
    __m512 vb;
    __m512 vc[ROWS * COLS];

    // The mask to load the last vector in B
    const int dwords = (actualN % 16 == 0 ? 16 : actualN % 16);
    const __mmask16 tailMask = (1 << dwords) - 1;

    auto set0 = [&](auto i) { vc[i] = _mm512_setzero_ps(); };
    compile_time_for<ROWS * COLS>::op(set0);

    auto compute = [&](auto i, int k) {
        // Compute in vertical direction
        constexpr const int row = i % ROWS;
        constexpr const int col = i / ROWS;

        if constexpr (col == 0) { va[row] = xft::set_avx512((float)(*ADDRESS(A, row, k, lda))); }

        if constexpr (row == 0) {
            __mmask16 mask = get_mask<col, COLS>(tailMask);
            vb = xft::load_avx512(mask, ADDRESS(B, k, col * AVX3_F32_NUM, ldb));
        }

        constexpr const int idx = INDEX(row, col, COLS);
        vc[idx] = _mm512_fmadd_ps(va[row], vb, vc[idx]);
    };

// Accumulate along k
#pragma unroll(4)
    for (int k = 0; k < K; ++k) {
        compile_time_for<ROWS * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&](auto i) {
        constexpr const int row = i / COLS;
        constexpr const int col = i % COLS;

        __mmask16 mask = get_mask<col, COLS>(tailMask);
        if (acc) { vc[i] = _mm512_add_ps(xft::load_avx512(mask, ADDRESS(C, row, col * AVX3_F32_NUM, ldc)), vc[i]); }
        xft::store_avx512(ADDRESS(C, row, col * AVX3_F32_NUM, ldc), mask, vc[i]);
    };

    compile_time_for<ROWS * COLS>::op(store);
}

// ALIGNED_N: the aligned size of N and is a multiple of AVX3_F32_NUM
template <int ALIGNED_N, typename TA, typename TB, typename TC>
void small_sgemm_fixn(const TA *A, const TB *B, TC *C, int lda, int ldb, int ldc, int M, int N, int K, bool acc) {
    constexpr const int COLS = ALIGNED_N / AVX3_F32_NUM;

    // How many rows of A are computed at the same time
    constexpr const int ROWS = get_max_rows<COLS>();

    int m = 0;
    for (; m + ROWS <= M; m += ROWS) {
        small_sgemm_fixmn<ROWS, ALIGNED_N>(A + m * lda, B, C + m * ldc, lda, ldb, ldc, N, K, acc);
    }

    // Deal with remaining rows
    if (m < M) {
        // Since ROWS is at most 6, the remaining rows are less than 6
        int remain = M - m;
        if (remain == 1) {
            small_sgemm_fixmn<1, ALIGNED_N>(A + m * lda, B, C + m * ldc, lda, ldb, ldc, N, K, acc);
        } else if (remain == 2) {
            small_sgemm_fixmn<2, ALIGNED_N>(A + m * lda, B, C + m * ldc, lda, ldb, ldc, N, K, acc);
        } else if (remain == 3) {
            small_sgemm_fixmn<3, ALIGNED_N>(A + m * lda, B, C + m * ldc, lda, ldb, ldc, N, K, acc);
        } else if (remain == 4) {
            small_sgemm_fixmn<4, ALIGNED_N>(A + m * lda, B, C + m * ldc, lda, ldb, ldc, N, K, acc);
        } else if (remain == 5) {
            small_sgemm_fixmn<5, ALIGNED_N>(A + m * lda, B, C + m * ldc, lda, ldb, ldc, N, K, acc);
        }
    }
}

// N <= 128
template <typename TA, typename TB, typename TC>
void small_sgemm_smalln(const TA *A, const TB *B, TC *C, int lda, int ldb, int ldc, int M, int N, int K, bool acc) {
    constexpr const int maxCols = 8;

    if (N > (maxCols - 1) * AVX3_F32_NUM) {
        small_sgemm_fixn<(maxCols - 0) * AVX3_F32_NUM>(A, B, C, lda, ldb, ldc, M, N, K, acc);
    } else if (N > (maxCols - 2) * AVX3_F32_NUM) {
        small_sgemm_fixn<(maxCols - 1) * AVX3_F32_NUM>(A, B, C, lda, ldb, ldc, M, N, K, acc);
    } else if (N > (maxCols - 3) * AVX3_F32_NUM) {
        small_sgemm_fixn<(maxCols - 2) * AVX3_F32_NUM>(A, B, C, lda, ldb, ldc, M, N, K, acc);
    } else if (N > (maxCols - 4) * AVX3_F32_NUM) {
        small_sgemm_fixn<(maxCols - 3) * AVX3_F32_NUM>(A, B, C, lda, ldb, ldc, M, N, K, acc);
    } else if (N > (maxCols - 5) * AVX3_F32_NUM) {
        small_sgemm_fixn<(maxCols - 4) * AVX3_F32_NUM>(A, B, C, lda, ldb, ldc, M, N, K, acc);
    } else if (N > (maxCols - 6) * AVX3_F32_NUM) {
        small_sgemm_fixn<(maxCols - 5) * AVX3_F32_NUM>(A, B, C, lda, ldb, ldc, M, N, K, acc);
    } else if (N > (maxCols - 7) * AVX3_F32_NUM) {
        small_sgemm_fixn<(maxCols - 6) * AVX3_F32_NUM>(A, B, C, lda, ldb, ldc, M, N, K, acc);
    } else if (N > (maxCols - 8) * AVX3_F32_NUM) {
        small_sgemm_fixn<(maxCols - 7) * AVX3_F32_NUM>(A, B, C, lda, ldb, ldc, M, N, K, acc);
    }
}

template <typename TA, typename TB, typename TC>
void small_sgemm(const TA *A, const TB *B, TC *C, int lda, int ldb, int ldc, int M, int N, int K, bool acc) {
    // Special case
    if (M == 1 && N == 128) { return small_sgemm_fixmn<1, 128>(A, B, C, lda, ldb, ldc, 128, K, acc); }
    if (M == 1 && N == 256) { return small_sgemm_fixmn<1, 256>(A, B, C, lda, ldb, ldc, 256, K, acc); }

    constexpr const int maxCols = 8;
    constexpr const int blockSize = maxCols * AVX3_F32_NUM;

    if (N > blockSize) {
        int blocks = N / blockSize;
        int remainN = N % blockSize;
        for (int i = 0; i < blocks; ++i) {
            small_sgemm_fixn<blockSize>(A, B + i * blockSize, C + i * blockSize, lda, ldb, ldc, M, blockSize, K, acc);
        }
        if (remainN > 0) {
            small_sgemm_smalln(A, B + blocks * blockSize, C + blocks * blockSize, lda, ldb, ldc, M, remainN, K, acc);
        }
    } else {
        small_sgemm_smalln(A, B, C, lda, ldb, ldc, M, N, K, acc);
    }
}

namespace xft {
void small_gemm(const float *A, const float *B, float *C, int M, int N, int K, int lda, int ldb, int ldc, bool acc) {
    ::small_sgemm<float, float, float>(A, B, C, lda, ldb, ldc, M, N, K, acc);
}
void small_gemm(
        const float *A, const float16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc, bool acc) {
    ::small_sgemm<float, float16_t, float>(A, B, C, lda, ldb, ldc, M, N, K, acc);
}
void small_gemm(
        const float *A, const bfloat16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc, bool acc) {
    ::small_sgemm<float, bfloat16_t, float>(A, B, C, lda, ldb, ldc, M, N, K, acc);
}
void small_gemm(
        const float *A, const float16_t *B, bfloat16_t *C, int M, int N, int K, int lda, int ldb, int ldc, bool acc) {
    ::small_sgemm<float, float16_t, bfloat16_t>(A, B, C, lda, ldb, ldc, M, N, K, acc);
}
void small_gemm(
        const float *A, const bfloat16_t *B, bfloat16_t *C, int M, int N, int K, int lda, int ldb, int ldc, bool acc) {
    ::small_sgemm<float, bfloat16_t, bfloat16_t>(A, B, C, lda, ldb, ldc, M, N, K, acc);
}
} // namespace xft