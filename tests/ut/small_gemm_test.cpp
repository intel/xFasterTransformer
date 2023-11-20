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
#include <cmath>

#include "xdnn.h"
#include "float16.h"
#include "gtest/gtest.h"

// Reference implementation of matrix multiplication
static void reference_sgemm(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

// Test case for the small_sgemm function
TEST(SmallSgemmTest, MatrixMultiplication) {
    const int M = 3;
    const int N = 4;
    const int K = 2;

    const float A[M * K] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float B[K * N] = {0.5f, 0.5f, 1.0f, 1.0f, 1.5f, 1.5f, 2.0f, 2.0f};
    float C[M * N] = {0.0f};

    const float expected_C[M * N] = {3.5f, 3.5f, 5.0f, 5.0f, 7.5f, 7.5f, 11.0f, 11.0f, 11.5f, 11.5f, 17.0f, 17.0f};

    small_sgemm(M, N, K, A, K, B, N, C, N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            EXPECT_FLOAT_EQ(C[i * N + j], expected_C[i * N + j]);
        }
    }
}

static void testFP32(int M, int N, int K) {
    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];

    // Fill matrices A and B with random values between 0 and 1
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < M * K; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Calculate reference result using reference_sgemm
    float *reference_C = new float[M * N];
    reference_sgemm(M, N, K, A, K, B, N, reference_C, N);

    small_sgemm(M, N, K, A, K, B, N, C, N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            EXPECT_FLOAT_EQ(C[i * N + j], reference_C[i * N + j]);
        }
    }

    // Clean up dynamically allocated memory
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] reference_C;
}

static void testFP16(int M, int N, int K) {
    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];

    // Fill matrices A and B with random values between 0 and 1
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < M * K; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float16_t *f16_B = new float16_t[K * N];
    float16_t::cvt_float_to_float16(B, f16_B, K * N);
    float16_t::cvt_float16_to_float(f16_B, B, K * N);

    // Calculate reference result using reference_sgemm
    float *reference_C = new float[M * N];
    reference_sgemm(M, N, K, A, K, B, N, reference_C, N);

    small_sgemm_f32f16f32(M, N, K, A, K, (const XDNN_FP16 *)f16_B, N, C, N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            EXPECT_FLOAT_EQ(C[i * N + j], reference_C[i * N + j]);
        }
    }

    // Clean up dynamically allocated memory
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] f16_B;
    delete[] reference_C;
}

TEST(SmallSgemmTest, RamdomTestFP32) {
    testFP32(64, 64, 64);
    testFP32(128, 64, 128);
    testFP32(100, 128, 100);
    testFP32(1024, 128, 1024);
}

TEST(SmallSgemmTest, RamdomTestFP16) {
    testFP16(64, 64, 64);
    testFP16(128, 64, 128);
    testFP16(100, 128, 100);
    testFP16(1024, 128, 1024);
}

int main(int argc, char **argv) {
    srand(time(NULL));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
