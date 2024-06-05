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
#include "matmul_helper.h"

#include <cmath>
#include <memory>
#include <random>

#include "gtest/gtest.h"

// Test function to compare reference and optimized implementations
template <typename TA = float, typename TB = float, typename TC = float, typename Tbias = float>
void test_gemm(MMHelper *mm, int M, int N, int K) {
    std::unique_ptr<TA[]> A = std::make_unique<TA[]>(M * K);
    std::unique_ptr<TB[]> B = std::make_unique<TB[]>(K * N);
    std::unique_ptr<TC[]> C = std::make_unique<TC[]>(M * N);
    std::unique_ptr<Tbias[]> bias = std::make_unique<Tbias[]>(N);

    // Generate random matrices A and B
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> distr(0.0f, 1.0f);

    for (int i = 0; i < M * K; i++) {
        A[i] = static_cast<TA>(distr(rng));
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = static_cast<TB>(distr(rng));
    }
    for (int i = 0; i < N; i++) {
        bias[i] = static_cast<Tbias>(distr(rng));
    }

    std::chrono::system_clock::time_point start, end;
    float during_time;

    memset(C.get(), 0, M * N * sizeof(TC));
    printf("[ RUNTIME  ] MMHelper::compute              M: %d, N: %d, K: %d\t", M, N, K);
    start = std::chrono::high_resolution_clock::now();
    mm->compute(false, M, N, K, 1.0f, A.get(), K, B.get(), nullptr, nullptr, nullptr, 0.0f, C.get(), N);
    end = std::chrono::high_resolution_clock::now();
    during_time = std::chrono::duration<float>(end - start).count();
    printf("%.6f sec\n", during_time);

    memset(C.get(), 0, M * N * sizeof(TC));
    printf("[ RUNTIME  ] MMHelper::compute_bias         M: %d, N: %d, K: %d\t", M, N, K);
    start = std::chrono::high_resolution_clock::now();
    mm->compute_bias(
            false, M, N, K, 1.0f, A.get(), K, B.get(), nullptr, nullptr, nullptr, 0.0f, C.get(), N, bias.get());
    end = std::chrono::high_resolution_clock::now();
    during_time = std::chrono::duration<float>(end - start).count();
    printf("%.6f sec\n", during_time);

    memset(C.get(), 0, M * N * sizeof(TC));
    printf("[ RUNTIME  ] MMHelper::compute_biasadd_relu M: %d, N: %d, K: %d\t", M, N, K);
    start = std::chrono::high_resolution_clock::now();
    mm->compute_biasadd_relu(
            false, M, N, K, 1.0f, A.get(), K, B.get(), nullptr, nullptr, nullptr, 0.0f, C.get(), N, bias.get());
    end = std::chrono::high_resolution_clock::now();
    during_time = std::chrono::duration<float>(end - start).count();
    printf("%.6f sec\n", during_time);

    memset(C.get(), 0, M * N * sizeof(TC));
    printf("[ RUNTIME  ] MMHelper::compute_gelu         M: %d, N: %d, K: %d\t", M, N, K);
    start = std::chrono::high_resolution_clock::now();
    mm->compute_gelu(false, M, N, K, 1.0f, A.get(), K, B.get(), nullptr, nullptr, nullptr, 0.0f, C.get(), N);
    end = std::chrono::high_resolution_clock::now();
    during_time = std::chrono::duration<float>(end - start).count();
    printf("%.6f sec\n", during_time);

    memset(C.get(), 0, M * N * sizeof(TC));
    printf("[ RUNTIME  ] MMHelper::compute_silu         M: %d, N: %d, K: %d\t", M, N, K);
    start = std::chrono::high_resolution_clock::now();
    mm->compute_silu(false, M, N, K, 1.0f, A.get(), K, B.get(), nullptr, nullptr, nullptr, 0.0f, C.get(), N);
    end = std::chrono::high_resolution_clock::now();
    during_time = std::chrono::duration<float>(end - start).count();
    printf("%.6f sec\n", during_time);

    // memset(C.get(), 0, M * N * sizeof(TC));
    // printf("[ RUNTIME  ] MMHelper::compute_resmul       M: %d, N: %d, K: %d\t", M, N, K);
    // start = std::chrono::high_resolution_clock::now();
    // mm->compute_resmul(
    //         false, M, N, K, 1.0f, A.get(), K, B.get(), nullptr, nullptr, nullptr, 0.0f, C.get(), N, A.get(), K);
    // end = std::chrono::high_resolution_clock::now();
    // during_time = std::chrono::duration<float>(end - start).count();
    // printf("%.6f sec\n", during_time);

    // memset(C.get(), 0, M * N * sizeof(TC));
    // printf("[ RUNTIME  ] MMHelper::compute_resext       M: %d, N: %d, K: %d\t", M, N, K);
    // start = std::chrono::high_resolution_clock::now();
    // mm->compute_resext(false, M, N, K, 1.0f, A.get(), K, B.get(), nullptr, nullptr, nullptr, 0.0f, C.get(), N,
    //         bias.get(), 1.0, A.get(), K);
    // end = std::chrono::high_resolution_clock::now();
    // during_time = std::chrono::duration<float>(end - start).count();
    // printf("%.6f sec\n", during_time);

    // memset(C.get(), 0, M * N * sizeof(TC));
    // printf("[ RUNTIME  ] MMHelper::compute_residential  M: %d, N: %d, K: %d\t", M, N, K);
    // start = std::chrono::high_resolution_clock::now();
    // mm->compute_residential(false, M, N, K, 1.0f, A.get(), K, B.get(), nullptr, nullptr, nullptr, 0.0f, C.get(), N,
    //         bias.get(), A.get(), K);
    // end = std::chrono::high_resolution_clock::now();
    // during_time = std::chrono::duration<float>(end - start).count();
    // printf("%.6f sec\n", during_time);
}

// TEST(MMHelper, gemm_f32f16f32) {
//     std::vector<int> M(8192);
//     std::generate(M.begin(), M.end(), [n = 1]() mutable { return n += 1; });
//     std::vector<int> N = {4096, 5120, 7168, 8192};
//     std::vector<int> K = {4096, 5120, 7168, 8192, 11008, 13696, 13824, 28672};

//     for (int j = 0; j < N.size(); ++j) {
//         std::unique_ptr<MMHelper> mm = std::make_unique<MMHelper>(xft::DeviceKind::iCPU, 0);
//         for (int t = j; t < K.size(); ++t) {
//             for (int i = 0; i < M.size(); ++i) {
//                 test_gemm<float, float16_t, float>(mm.get(), M[i], N[j], K[t]);
//             }
//         }
//         std::string name;
//         std::cout << "Enter your name: ";
//         std::getline(std::cin, name);
//     }
// }

TEST(MMHelper, gemm_f32bf16f32) {
    std::vector<int> M(8192);
    std::generate(M.begin(), M.end(), [n = 1]() mutable { return n += 1; });
    std::vector<int> N = {4096, 5120, 7168, 8192};
    std::vector<int> K = {4096, 5120, 7168, 8192, 11008, 13696, 13824, 28672};

    for (int j = 0; j < N.size(); ++j) {
        std::unique_ptr<MMHelper> mm = std::make_unique<MMHelper>(xft::DeviceKind::iCPU, 0);
        for (int t = j; t < K.size(); ++t) {
            for (int i = 0; i < M.size(); ++i) {
                test_gemm<float, bfloat16_t, float>(mm.get(), M[i], N[j], K[t]);
            }
        }
        std::string name;
        std::cout << "Enter your name: ";
        std::getline(std::cin, name);
    }
}

// TEST(MMHelper, gemm_bf16bf16bf16) {
//     std::vector<int> M(8192);
//     std::generate(M.begin(), M.end(), [n = 1]() mutable { return n += 1; });
//     std::vector<int> N = {4096, 5120, 7168, 8192};
//     std::vector<int> K = {4096, 5120, 7168, 8192, 11008, 13696, 13824, 28672};

//     for (int j = 0; j < N.size(); ++j) {
//         std::unique_ptr<MMHelper> mm = std::make_unique<MMHelper>(xft::DeviceKind::iCPU, 0);
//         for (int t = j; t < K.size(); ++t) {
//             for (int i = 0; i < M.size(); ++i) {
//                 test_gemm<bfloat16_t, bfloat16_t, bfloat16_t>(mm.get(), M[i], N[j], K[t]);
//             }
//         }
//         std::string name;
//         std::cout << "Enter your name: ";
//         std::getline(std::cin, name);
//     }
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}