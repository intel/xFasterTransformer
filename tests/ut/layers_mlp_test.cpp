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
#include <chrono>
#include <cmath>
#include <type_traits>

#include "bfloat16.h"
#include "float16.h"
#include "layers_mlp.h"
#include "gtest/gtest.h"

template <typename T>
static void matmul(int m, int n, int k, const float *A, const float *B, float *C) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * n + j] = 0.0f;
            for (int q = 0; q < k; ++q) {
                C[i * n + j] += static_cast<float>(static_cast<T>(A[i * k + q]))
                        * static_cast<float>(static_cast<T>(B[q * n + j]));
            }
        }
    }
}

template <typename T>
static void refMLPLLaMA(int numTokens, int hiddenSize, int intermediateSize, float *output, int outputStride,
        const float *input, int inputStride, const float *gateWeight, const float *upWeight, const float *downWeight) {
    // self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    float *gate_proj = (float *)aligned_alloc(64, numTokens * intermediateSize * sizeof(float));
    float *up_proj = (float *)aligned_alloc(64, numTokens * intermediateSize * sizeof(float));
    memset(gate_proj, 0, numTokens * intermediateSize * sizeof(float));
    memset(up_proj, 0, numTokens * intermediateSize * sizeof(float));

    matmul<T>(numTokens, intermediateSize, hiddenSize, input, gateWeight, gate_proj);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < numTokens; ++i) {
        for (int j = 0; j < intermediateSize; ++j) {
            gate_proj[i * intermediateSize + j] = (1.0f / (1.0f + std::exp(-gate_proj[i * intermediateSize + j])))
                    * gate_proj[i * intermediateSize + j];
        }
    }

    matmul<T>(numTokens, intermediateSize, hiddenSize, input, upWeight, up_proj);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < numTokens; ++i) {
        for (int j = 0; j < intermediateSize; ++j) {
            gate_proj[i * intermediateSize + j] *= up_proj[i * intermediateSize + j];
        }
    }

    matmul<T>(numTokens, hiddenSize, intermediateSize, gate_proj, downWeight, output);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < numTokens; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            output[i * hiddenSize + j] += input[i * hiddenSize + j];
        }
    }

    free(gate_proj);
    free(up_proj);
}

template <typename T>
static void compareMLPLLaMA(
        int numTokens, int hiddenSize, int intermediateSize, float *gateW, float *upW, float *downW) {
    float *input = (float *)aligned_alloc(64, numTokens * hiddenSize * sizeof(float));
    float *ourOutput = (float *)aligned_alloc(64, numTokens * hiddenSize * sizeof(float));
    float *refOutput = (float *)aligned_alloc(64, numTokens * hiddenSize * sizeof(float));
    memset(ourOutput, 0, numTokens * hiddenSize * sizeof(float));
    memset(refOutput, 0, numTokens * hiddenSize * sizeof(float));

    for (int i = 0; i < numTokens * hiddenSize; ++i) {
        input[i] = static_cast<float>(1.0f * rand() / RAND_MAX);
    }

    xft::DataType dt = xft::DataType::unknown;
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        dt = xft::DataType::bf16;
    } else if constexpr (std::is_same<T, float16_t>::value) {
        dt = xft::DataType::fp16;
    } else {
        printf("Unsupported data type\n");
        GTEST_FAIL();
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    invokeMLPLLaMA(dt, xft::ActivationType::SILU, numTokens, hiddenSize, intermediateSize, (void *)ourOutput, hiddenSize,
            (const void *)input, hiddenSize, (const void *)gateW, (const void *)upW, (const void *)downW);
    auto end = std::chrono::high_resolution_clock::now();
    float during_time = std::chrono::duration<float>(end - start).count();
    printf("[ RUNTIME  ] XFT::invokeMLPLLaMA %.6f sec\n", during_time);

    refMLPLLaMA<T>(numTokens, hiddenSize, intermediateSize, (float *)refOutput, hiddenSize,
            (const float *)input, hiddenSize, (const float *)gateW, (const float *)upW, (const float *)downW);

    for (int i = 0; i < numTokens * hiddenSize; ++i) {
        EXPECT_EQ(std::abs(refOutput[i] - ourOutput[i]) > 0.01
                        && std::abs((refOutput[i] - ourOutput[i]) / refOutput[i]) > 0.01,
                false);
    }

    free(input);
    free(ourOutput);
    free(refOutput);
}

template <typename T>
void test_MLPLLaMA(void) {
    int hiddenSize = 4096;
    int intermediateSize = 11008;

    float *gateW = (float *)aligned_alloc(64, hiddenSize * intermediateSize * sizeof(float));
    float *upW = (float *)aligned_alloc(64, hiddenSize * intermediateSize * sizeof(float));
    float *downW = (float *)aligned_alloc(64, intermediateSize * hiddenSize * sizeof(float));

    for (int i = 0; i < hiddenSize * intermediateSize; ++i) {
        gateW[i] = static_cast<float>(0.5f * rand() / RAND_MAX);
        upW[i] = static_cast<float>(0.5f * rand() / RAND_MAX);
        downW[i] = static_cast<float>(0.5f * rand() / RAND_MAX);
    }

    compareMLPLLaMA<T>(18, hiddenSize, intermediateSize, gateW, upW, downW);
    compareMLPLLaMA<T>(16, hiddenSize, intermediateSize, gateW, upW, downW);
    compareMLPLLaMA<T>(16, hiddenSize, intermediateSize, gateW, upW, downW);
    compareMLPLLaMA<T>(16, hiddenSize, intermediateSize, gateW, upW, downW);
    compareMLPLLaMA<T>(16, hiddenSize, intermediateSize, gateW, upW, downW);
    compareMLPLLaMA<T>(10, hiddenSize, intermediateSize, gateW, upW, downW);
    compareMLPLLaMA<T>(4, hiddenSize, intermediateSize, gateW, upW, downW);
    compareMLPLLaMA<T>(2, hiddenSize, intermediateSize, gateW, upW, downW);
    compareMLPLLaMA<T>(1, hiddenSize, intermediateSize, gateW, upW, downW);
    compareMLPLLaMA<T>(2, hiddenSize, intermediateSize, gateW, upW, downW);
    compareMLPLLaMA<T>(4, hiddenSize, intermediateSize, gateW, upW, downW);
    compareMLPLLaMA<T>(6, hiddenSize, intermediateSize, gateW, upW, downW);
    compareMLPLLaMA<T>(16, hiddenSize, intermediateSize, gateW, upW, downW);
    compareMLPLLaMA<T>(16, hiddenSize, intermediateSize, gateW, upW, downW);

    free(gateW);
    free(upW);
    free(downW);
}

TEST(MLPLLaMA, bfloat16_t) {
    test_MLPLLaMA<bfloat16_t>();
}

TEST(MLPLLaMA, float16_t) {
    test_MLPLLaMA<float16_t>();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}