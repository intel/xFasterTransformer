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
#include <type_traits>

#include "bfloat16.h"
#include "float16.h"
#include "layers_mlp.h"
#include "gtest/gtest.h"

template <typename T>
static void refMLPLLaMA(int numTokens, int hiddenSize, int intermediateSize, T *output,
        int outputStride, const T *input, int inputStride, const T *gateWeight, const T *upWeight,
        const T *downWeight) {
    memset(output, 0, numTokens * hiddenSize * sizeof(T));
}

template <typename T>
static void compareMLPLLaMA(int numTokens, int hiddenSize, int intermediateSize) {

    float *input = (float *)aligned_alloc(64, numTokens * hiddenSize * sizeof(float));
    float *gateW = (float *)aligned_alloc(64, hiddenSize * intermediateSize * sizeof(float));
    float *upW = (float *)aligned_alloc(64, hiddenSize * intermediateSize * sizeof(float));
    float *downW = (float *)aligned_alloc(64, intermediateSize * hiddenSize * sizeof(float));
    float *ourOutput = (float *)aligned_alloc(64, numTokens * hiddenSize * sizeof(float));
    float *refOutput = (float *)aligned_alloc(64, numTokens * hiddenSize * sizeof(float));

    for (int i = 0; i < numTokens * hiddenSize; ++i) {
        input[i] = static_cast<float>(1.0f * rand() / RAND_MAX);
    }

    for (int i = 0; i < hiddenSize * intermediateSize; ++i) {
        gateW[i] = static_cast<float>(1.0f * rand() / RAND_MAX);
        upW[i] = static_cast<float>(1.0f * rand() / RAND_MAX);
        downW[i] = static_cast<float>(1.0f * rand() / RAND_MAX);
    }

    if constexpr (std::is_same<T, bfloat16_t>::value) {
        invokeMLPLLaMA(xft::DataType::bf16, numTokens, hiddenSize, intermediateSize, (void *)ourOutput,
                hiddenSize, (const void *)input, hiddenSize, (const void *)gateW, (const void *)upW,
                (const void *)downW);
        refMLPLLaMA<bfloat16_t>(numTokens, hiddenSize, intermediateSize, (bfloat16_t *)refOutput,
                hiddenSize, (const bfloat16_t *)input, hiddenSize, (const bfloat16_t *)gateW, (const bfloat16_t *)upW,
                (const bfloat16_t *)downW);
    }

    for (int i = 0; i < numTokens * hiddenSize; ++i) {
        EXPECT_LT(std::abs(refOutput[i] - (float)ourOutput[i]), 0.01);
    }

    free(input);
    free(gateW);
    free(upW);
    free(downW);
    free(ourOutput);
    free(refOutput);
}

TEST(MLPLLaMA, bfloat16_t) {
    compareMLPLLaMA<bfloat16_t>(128, 4096, 11008);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}