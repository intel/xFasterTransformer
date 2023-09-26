#include "gemm_kernel_ext.h"

#include <cmath>
#include <random>

#include "gtest/gtest.h"

static void small_gemm_tranb_ref(
        const float *A, const float *B, float *C, int M, int N, int K, int lda, int ldb, int ldc) {
    // Loop over the rows of A
    for (int i = 0; i < M; i++) {
        // Loop over the columns of B
        for (int j = 0; j < N; j++) {
            // Compute the dot product of row i of A with column j of B
            float dot_product = 0;
            for (int k = 0; k < K; k++) {
                dot_product += A[i * lda + k] * B[j * ldb + k];
            }
            // Store the result in C[i][j]
            C[i * ldc + j] += dot_product;
        }
    }
}

// Test function to compare reference and optimized implementations
void test_small_gemm_tranb(int M, int N, int K) {
    float *A_ref = new float[M * K];
    float *B_ref = new float[K * N];
    float *C_ref = new float[M * N];
    float *A_opt = new float[M * K];
    float *B_opt = new float[K * N];
    float *C_opt = new float[M * N];

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

TEST(small_gemm_tranb, small_gemm_tranb) {
    test_small_kernel();
    test_bigger_kernel();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}