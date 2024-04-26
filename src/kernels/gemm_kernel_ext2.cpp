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
#include "gemm_kernel_ext.h"
#include "intrinsics_util.h"

#define INDEX(x, y, ld) ((x) * (ld) + (y))
#define OFFSET(p, y) ((p) + (y))
#define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))

// Get mask for last column
template <int EXPANDED_N, int col>
constexpr inline __mmask16 get_mask(__mmask16 mask) {
    // Not last column, return 0xffff indicating load/store all 16 floats
    if constexpr (col < EXPANDED_N / 16 - 1)
        return (__mmask16)0xffff;
    else
        return mask;
}

/**
 * @brief Small GEMM kernel for small M (like 1, 2, 3, 4)
 * @tparam TC The type of C
 * @tparam M The number of rows of A and C
 * @tparam N The number of columns of B and C (must be multiple of 16, could be expanded)
 * @tparam N_EXPANDED Whether N is expanded
 * @param A The pointer to A
 * @param B The pointer to B
 * @param bScale The pointer to the scale of B (each row has a scale factor)
 * @param C The pointer to C
 * @param lda The leading dimension of A
 * @param ldb The leading dimension of B
 * @param ldc The leading dimension of C
 * @param actualN The actual number of columns of B and C (When N_EXPANDED = false, actualN = N)
 * @param K The number of columns of A and rows of B
*/
template <typename TC, int M, int N, bool N_EXPANDED = false>
void small_sgemm_smallm(const float *A, const int8_t *B, const float *bScale, TC *C, int lda, int ldb, int ldc,
        int actualN, int K, bool acc) {
    constexpr const int AVX3_F32_NUM = 16;
    constexpr const int COLS = N / AVX3_F32_NUM;

    static_assert(N % AVX3_F32_NUM == 0, "N must be multiple of 16.");

    __m512 va[M];
    __m512 vs; // scale
    __m512 vb;
    __m512 vc[M * COLS];

    // The mask to load the last column
    __mmask16 mask16 = 0xffff;
    if constexpr (N_EXPANDED) {
        const int dwords = (actualN % 16 == 0 ? 16 : actualN % 16);
        mask16 = (1 << dwords) - 1;
    }

    auto set0 = [&](auto i) { vc[i] = xft::set_avx512(0); };
    compile_time_for<M * COLS>::op(set0);

    auto compute = [&](auto i, int k) {
        constexpr const int row = i % M;
        constexpr const int col = i / M;

        if constexpr (col == 0) {
            va[row] = xft::set_avx512(*ADDRESS(A, row, k, lda));
            vs = xft::set_avx512(bScale[k]);
        }

        if constexpr (row == 0) {
            if constexpr (N_EXPANDED) {
                vb = xft::load_avx512(get_mask<N, col>(mask16), ADDRESS(B, k, col * AVX3_F32_NUM, ldb));
            } else {
                vb = xft::load_avx512(ADDRESS(B, k, col * AVX3_F32_NUM, ldb));
            }
            vb = _mm512_mul_ps(vb, vs); // apply scale
        }

        constexpr const int idx = INDEX(row, col, COLS);
        vc[idx] = _mm512_fmadd_ps(va[row], vb, vc[idx]);
    };

// Accumulate along k
#pragma unroll(4)
    for (int k = 0; k < K; ++k) {
        compile_time_for<M * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&](auto i) {
        constexpr const int row = i / COLS;
        constexpr const int col = i % COLS;
        if constexpr (N_EXPANDED) {
            if (acc) {
                vc[i] = xft::load_avx512(get_mask<N, col>(mask16), ADDRESS(C, row, col * AVX3_F32_NUM, ldc)) + vc[i];
            }
            xft::store_avx512(ADDRESS(C, row, col * AVX3_F32_NUM, ldc), get_mask<N, col>(mask16), vc[i]);
        } else {
            if (acc) { vc[i] = xft::load_avx512(ADDRESS(C, row, col * AVX3_F32_NUM, ldc)) + vc[i]; }
            xft::store_avx512(ADDRESS(C, row, col * AVX3_F32_NUM, ldc), 0xffff, vc[i]);
        }
    };
    compile_time_for<M * COLS>::op(store);
}

template <typename TC, int M>
void small_sgemm_smallm(const float *A, const int8_t *B, const float *bScale, TC *C, int lda, int ldb, int ldc, int N,
        int K, bool acc) {
    constexpr const int AVX3_F32_NUM = 16;
    const int COLS = (N + AVX3_F32_NUM - 1) / AVX3_F32_NUM;

    __m512 va;
    __m512 vs; // scale
    __m512 vb[COLS];
    __m512 vc[M * COLS];

    // The mask to load the last column
    __mmask16 mask16 = (1 << (N % 16 == 0 ? 16 : N % 16)) - 1;

    for (int i = 0; i < M * COLS; ++i) {
        vc[i] = xft::set_avx512(0);
    }

// Accumulate along k
#pragma unroll(4)
    for (int k = 0; k < K; ++k) {
        for (int row = 0; row < M; ++row) {
            for (int col = 0; col < COLS; ++col) {
                if (col == 0) {
                    va = xft::set_avx512(*ADDRESS(A, row, k, lda));
                    vs = xft::set_avx512(bScale[k]);
                }

                if (row == 0) {
                    __mmask16 mask = (col == COLS - 1 ? mask16 : 0xffff);
                    vb[col] = xft::load_avx512(mask, ADDRESS(B, k, col * AVX3_F32_NUM, ldb));
                    vb[col] = _mm512_mul_ps(vb[col], vs); // apply scale
                }

                const int idx = INDEX(row, col, COLS);
                vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
            }
        }
    }

    // Store to C
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < COLS; ++col) {
            int i = row * COLS + col;
            __mmask16 mask = (col == COLS - 1 ? mask16 : 0xffff);
            if (acc) { vc[i] = xft::load_avx512(mask, ADDRESS(C, row, col * AVX3_F32_NUM, ldc)) + vc[i]; }
            xft::store_avx512(ADDRESS(C, row, col * AVX3_F32_NUM, ldc), mask, vc[i]);
        }
    }
}

/**
 * @brief Small GEMM kernel designed for LLM for Score * V (Score = Softmax(Q*Kᵀ)), where V is int8_t
*/
template <typename T>
void small_gemm_int8(const float *A, const int8_t *B, const float *bScale, T *C, int M, int N, int K, int lda, int ldb,
        int ldc, bool acc) {
    if (M == 1) {
        if (N == 128) {
            small_sgemm_smallm<T, 1, 128>(A, B, bScale, C, lda, ldb, ldc, N, K, acc);
        } else if (N == 256) {
            small_sgemm_smallm<T, 1, 256>(A, B, bScale, C, lda, ldb, ldc, N, K, acc);
        } else {
            int n = 0;
            for (; n + 255 < N; n += 256) {
                small_sgemm_smallm<T, 1, 256>(A, B + n, bScale, C + n, lda, ldb, ldc, 256, K, acc);
            }
            if (n < N) { small_sgemm_smallm<T, 1>(A, B + n, bScale, C + n, lda, ldb, ldc, N - n, K, acc); }
        }
    } else if (M == 2) {
        int n = 0;
        for (; n + 127 < N; n += 128) {
            small_sgemm_smallm<T, 2, 128>(A, B + n, bScale, C + n, lda, ldb, ldc, 128, K, acc);
        }
        if (n < N) { small_sgemm_smallm<T, 2>(A, B + n, bScale, C + n, lda, ldb, ldc, N - n, K, acc); }
    } else if (M == 3) {
        int n = 0;
        for (; n + 127 < N; n += 128) {
            small_sgemm_smallm<T, 3, 128>(A, B + n, bScale, C + n, lda, ldb, ldc, 128, K, acc);
        }
        if (n < N) { small_sgemm_smallm<T, 3>(A, B + n, bScale, C + n, lda, ldb, ldc, N - n, K, acc); }
    } else if (M == 4) {
        int n = 0;
        for (; n + 95 < N; n += 96) {
            small_sgemm_smallm<T, 4, 96>(A, B + n, bScale, C + n, lda, ldb, ldc, 96, K, acc);
        }
        if (n < N) { small_sgemm_smallm<T, 4>(A, B + n, bScale, C + n, lda, ldb, ldc, N - n, K, acc); }
    } else {
        int m = 0;
        for (; m + 2 < M; m += 3) {
            int n = 0;
            for (; n + 127 < N; n += 128) {
                small_sgemm_smallm<T, 3, 128>(A + m * lda, B + n, bScale, C + m * ldc + n, lda, ldb, ldc, 128, K, acc);
            }
            if (n < N) {
                small_sgemm_smallm<T, 3>(A + m * lda, B + n, bScale, C + m * ldc + n, lda, ldb, ldc, N - n, K, acc);
            }
        }
        if (m < M) {

#define HANDLE_CASE_N128(ROWS) \
    small_sgemm_smallm<T, ROWS, 128>(A + m * lda, B + n, bScale, C + m * ldc + n, lda, ldb, ldc, 128, K, acc);
#define HANDLE_CASE(ROWS) \
    small_sgemm_smallm<T, ROWS>(A + m * lda, B + n, bScale, C + m * ldc + n, lda, ldb, ldc, N - n, K, acc);

            int n = 0;
            for (; n + 127 < N; n += 128) {
                switch (M - m) {
                    case 1: HANDLE_CASE_N128(1); break;
                    case 2: HANDLE_CASE_N128(2); break;
                }
            }
            if (n < N) {
                switch (M - m) {
                    case 1: HANDLE_CASE(1); break;
                    case 2: HANDLE_CASE(2); break;
                }
            }
        }
    }
}

namespace xft {
void small_gemm(const float *A, const int8_t *B, const float *bScale, float *C, int M, int N, int K, int lda, int ldb,
        int ldc, bool acc) {
    small_gemm_int8<float>(A, B, bScale, C, M, N, K, lda, ldb, ldc, acc);
}

void small_gemm(const float *A, const int8_t *B, const float *bScale, bfloat16_t *C, int M, int N, int K, int lda,
        int ldb, int ldc, bool acc) {
    small_gemm_int8<bfloat16_t>(A, B, bScale, C, M, N, K, lda, ldb, ldc, acc);
}

} // namespace xft