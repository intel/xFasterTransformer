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

#include <cmath>
#include <random>

#include "gtest/gtest.h"

template <typename TA, typename TB, typename TC>
static void small_gemm_tranb_ref(const TA *A, const TB *B, TC *C, int M, int N, int K, int lda, int ldb, int ldc) {
    // Loop over the rows of A
    for (int i = 0; i < M; i++) {
        // Loop over the columns of B
        for (int j = 0; j < N; j++) {
            // Compute the dot product of row i of A with column j of B
            float dot_product = 0;
            for (int k = 0; k < K; k++) {
                dot_product += (float)A[i * lda + k] * (float)B[j * ldb + k];
            }
            // Store the result in C[i][j]
            C[i * ldc + j] = dot_product;
        }
    }
}

static void small_gemm_tranb_ref(
        const float *A, const int8_t *B, const float *scale, float *C, int M, int N, int K, int lda, int ldb, int ldc) {
    // Loop over the rows of A
    for (int i = 0; i < M; i++) {
        // Loop over the columns of B
        for (int j = 0; j < N; j++) {
            // Compute the dot product of row i of A with column j of B
            float dot_product = 0;
            for (int k = 0; k < K; k++) {
                dot_product += A[i * lda + k] * B[j * ldb + k] * scale[j];
            }
            // Store the result in C[i][j]
            C[i * ldc + j] = dot_product;
        }
    }
}

// Test function to compare reference and optimized implementations
template <typename TA = float, typename TB = float, typename TC = float>
void test_small_gemm_tranb(int M, int N, int K) {
    TA *A_ref = new TA[M * K];
    TB *B_ref = new TB[K * N];
    TC *C_ref = new TC[M * N];
    TA *A_opt = new TA[M * K];
    TB *B_opt = new TB[K * N];
    TC *C_opt = new TC[M * N];

    // Generate random matrices A and B
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> distr(0.0f, 1.0f);

    for (int i = 0; i < M * K; i++) {
        A_ref[i] = distr(rng);
        A_opt[i] = A_ref[i];
    }
    for (int i = 0; i < K * N; i++) {
        B_ref[i] = distr(rng);
        B_opt[i] = B_ref[i];
    }

    // Compute matrix multiplication with reference implementation
    memset(C_ref, 0, M * N * sizeof(float));
    small_gemm_tranb_ref(A_ref, B_ref, C_ref, M, N, K, K, K, N);

    // Compute matrix multiplication with optimized implementation
    memset(C_opt, 0, M * N * sizeof(float));
    small_gemm_transb(A_opt, B_opt, C_opt, M, N, K, K, K, N);

    // Compare results
    float eps = 1e-4;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            EXPECT_NEAR(C_opt[i * N + j], C_ref[i * N + j], eps);
        }
    }

    delete[] A_ref;
    delete[] B_ref;
    delete[] C_ref;
    delete[] A_opt;
    delete[] B_opt;
    delete[] C_opt;
}

// Test function to compare reference and optimized implementations
void test_small_gemm_tranb_int8(int M, int N, int K) {
    float *A_ref = new float[M * K];
    int8_t *B_ref = new int8_t[K * N];
    float *C_ref = new float[M * N];
    float *A_opt = new float[M * K];
    int8_t *B_opt = new int8_t[K * N];
    float *C_opt = new float[M * N];
    float *scale = new float[N];

    // Generate random matrices A and B
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> distr(0.0f, 1.0f);

    for (int i = 0; i < M * K; i++) {
        A_ref[i] = distr(rng);
        A_opt[i] = A_ref[i];
    }
    for (int i = 0; i < K * N; i++) {
        B_ref[i] = distr(rng) * 200 - 100;
        B_opt[i] = B_ref[i];
    }
    for (int i = 0; i < N; i++) {
        scale[i] = distr(rng) / 100;
    }

    // Compute matrix multiplication with reference implementation
    memset(C_ref, 0, M * N * sizeof(float));
    small_gemm_tranb_ref(A_ref, B_ref, scale, C_ref, M, N, K, K, K, N);

    // Compute matrix multiplication with optimized implementation
    memset(C_opt, 0, M * N * sizeof(float));
    small_gemm_transb(A_opt, B_opt, scale, C_opt, M, N, K, K, K, N);

    // Compare results
    float eps = 1e-4;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            EXPECT_NEAR(C_opt[i * N + j], C_ref[i * N + j], eps);
        }
    }

    delete[] A_ref;
    delete[] B_ref;
    delete[] C_ref;
    delete[] A_opt;
    delete[] B_opt;
    delete[] C_opt;
    delete[] scale;
}

static void test_small_kernel() {
    test_small_gemm_tranb(1, 4, 128);
    test_small_gemm_tranb(2, 4, 128);
    test_small_gemm_tranb(3, 4, 128);
    test_small_gemm_tranb(4, 4, 128);
    test_small_gemm_tranb(5, 4, 128);
    test_small_gemm_tranb(6, 4, 128);

    test_small_gemm_tranb(1, 3, 128);
    test_small_gemm_tranb(2, 3, 128);
    test_small_gemm_tranb(3, 3, 128);
    test_small_gemm_tranb(4, 3, 128);
    test_small_gemm_tranb(5, 3, 128);
    test_small_gemm_tranb(6, 3, 128);

    test_small_gemm_tranb(1, 2, 128);
    test_small_gemm_tranb(2, 2, 128);
    test_small_gemm_tranb(3, 2, 128);
    test_small_gemm_tranb(4, 2, 128);
    test_small_gemm_tranb(5, 2, 128);
    test_small_gemm_tranb(6, 2, 128);

    test_small_gemm_tranb(1, 1, 128);
    test_small_gemm_tranb(2, 1, 128);
    test_small_gemm_tranb(3, 1, 128);
    test_small_gemm_tranb(4, 1, 128);
    test_small_gemm_tranb(5, 1, 128);
    test_small_gemm_tranb(6, 1, 128);

    test_small_gemm_tranb(1, 111, 128);
    test_small_gemm_tranb(1, 111, 256);
}

static void test_bigger_kernel() {
    test_small_gemm_tranb(36, 36, 128);
    test_small_gemm_tranb(35, 35, 128);
    test_small_gemm_tranb(34, 34, 128);
    test_small_gemm_tranb(33, 33, 128);
    test_small_gemm_tranb(32, 32, 128);
    test_small_gemm_tranb(22, 22, 128);
    test_small_gemm_tranb(10, 10, 128);
}

template <typename TA, typename TB, typename TC>
void gemm_ref(const TA *A, const TB *B, TC *C, int M, int N, int K, int lda, int ldb, int ldc, bool acc) {
    // Loop over the rows of A
    for (int i = 0; i < M; i++) {
        // Loop over the columns of B
        for (int j = 0; j < N; j++) {
            // Compute the dot product of row i of A with column j of B
            float dot_product = 0;
            for (int k = 0; k < K; k++) {
                dot_product += (float)A[i * lda + k] * (float)B[k * ldb + j];
            }
            // Store the result in C[i][j]
            if (acc) {
                C[i * ldc + j] = (float)C[i * ldc + j] + dot_product;
            } else {
                C[i * ldc + j] = dot_product;
            }
        }
    }
}

static void gemm_bint8_ref(const float *A, const int8_t *B, const float *bScale, float *C, int M, int N, int K, int lda,
        int ldb, int ldc) {
    // Loop over the rows of A
    for (int i = 0; i < M; i++) {
        // Loop over the columns of B
        for (int j = 0; j < N; j++) {
            // Compute the dot product of row i of A with column j of B
            float dot_product = 0;
            for (int k = 0; k < K; k++) {
                dot_product += A[i * lda + k] * (B[k * ldb + j] * bScale[k]);
            }
            // Store the result in C[i][j]
            C[i * ldc + j] = dot_product;
        }
    }
}

template <typename TA, typename TB, typename TC>
void test_small_gemm(int M, int N, int K, bool acc) {
    const int lda = K;
    const int ldb = N;
    const int ldc = N;
    TA *A = new TA[M * K];
    TB *B = new TB[K * N];
    TC *C = new TC[M * N];
    TC *refC = new TC[M * N];

    // Generate random data for A, B, C
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> distr(-1.0f, 1.0f);
    for (int i = 0; i < M * K; i++) {
        A[i] = static_cast<TA>(distr(rng));
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = static_cast<TB>(distr(rng));
    }
    for (int i = 0; i < M * N; i++) {
        C[i] = static_cast<TC>(distr(rng));
        refC[i] = C[i];
    }

    xft::small_gemm(A, B, C, M, N, K, lda, ldb, ldc, acc);
    gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, acc);

    // Compare results
    float eps = 5 * 1e-2;
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(C[i], refC[i], eps);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] refC;
}

void test_small_gemm_int8(int M, int N, int K) {
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    float *A = new float[M * K];
    int8_t *B = new int8_t[K * N];
    float *bScale = new float[K];
    float *C = new float[M * N];
    float *refC = new float[M * N];

    // Generate random data for A, B, and bScale
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> distr(-1.0f, 1.0f);
    for (int i = 0; i < M * K; i++) {
        A[i] = distr(rng);
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = distr(rng) * 127;
    }
    for (int i = 0; i < K; i++) {
        bScale[i] = distr(rng) / 100;
    }
    memset(C, 0, M * N * sizeof(float));
    memset(refC, 0, M * N * sizeof(float));

    xft::small_gemm(A, B, bScale, C, M, N, K, lda, ldb, ldc);
    gemm_bint8_ref(A, B, bScale, refC, M, N, K, lda, ldb, ldc);

    // Compare results
    float eps = 1e-3;
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(C[i], refC[i], eps);
    }

    delete[] A;
    delete[] B;
    delete[] bScale;
    delete[] C;
    delete[] refC;
}

TEST(small_gemm_tranb, small_gemm_tranb_f32) {
    test_small_kernel();
    test_bigger_kernel();
}

TEST(small_gemm_tranb, small_gemm_tranb_bf16fp16f32) {
    test_small_gemm_tranb<bfloat16_t, float16_t, float>(1, 2, 16);
    test_small_gemm_tranb<bfloat16_t, float16_t, float>(1, 4, 128);
    test_small_gemm_tranb<bfloat16_t, float16_t, float>(1, 4, 256);
}

TEST(small_gemm_tranb, small_gemm_tranb_int8) {
    test_small_gemm_tranb_int8(1, 100, 128);
    test_small_gemm_tranb_int8(2, 101, 256);
}

TEST(small_gemm, small_gemm_int8) {
    for (int m = 0; m < 10; ++m) {
        test_small_gemm_int8(m, 64, 100);
        test_small_gemm_int8(m, 100, 313);
        test_small_gemm_int8(m, 128, 1024);
        test_small_gemm_int8(m, 256, 19);
        test_small_gemm_int8(m, 500, 111);
        test_small_gemm_int8(m, 512, 39);
    }
}

TEST(small_gemm, small_gemm_f32) {
    for (int m = 0; m < 10; ++m) {
        test_small_gemm<float, float, float>(m, 128, 1 + rand() % 100, false);
        test_small_gemm<float, float, float>(m, 128, 1 + rand() % 100, true);

        test_small_gemm<float, float, float>(m, 100, 1 + rand() % 100, false);
        test_small_gemm<float, float, float>(m, 100, 1 + rand() % 100, true);
    }
}

TEST(small_gemm, small_gemm_f32f16bf16) {
    for (int m = 0; m < 10; ++m) {
        test_small_gemm<float, float16_t, bfloat16_t>(m, 128, 1 + rand() % 100, false);
        test_small_gemm<float, float16_t, bfloat16_t>(m, 128, 1 + rand() % 100, true);

        test_small_gemm<float, float16_t, bfloat16_t>(m, 100, 1 + rand() % 100, false);
        test_small_gemm<float, float16_t, bfloat16_t>(m, 100, 1 + rand() % 100, true);
    }
}

TEST(small_gemm, small_gemm_f32bf16bf16) {
    for (int m = 0; m < 10; ++m) {
        test_small_gemm<float, bfloat16_t, bfloat16_t>(m, 128, 1 + rand() % 100, false);
        test_small_gemm<float, bfloat16_t, bfloat16_t>(m, 128, 1 + rand() % 100, true);

        test_small_gemm<float, bfloat16_t, bfloat16_t>(m, 100, 1 + rand() % 100, false);
        test_small_gemm<float, bfloat16_t, bfloat16_t>(m, 100, 1 + rand() % 100, true);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}