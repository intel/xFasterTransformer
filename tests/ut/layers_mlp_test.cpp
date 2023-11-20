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
static void refMLPLLaMA(int batchSize, int inputSeqLen, int hiddenSize, int intermediateSize, T *output,
        int outputStride, const T *input, int inputStride, const T *gateWeight, const T *upWeight,
        const T *downWeight) {
    memset(output, 0, batchSize * inputSeqLen * hiddenSize * sizeof(T));
}

template <typename T>
static void compareMLPLLaMA(int batch_size, int seq_length, int hidden_size, int intermediate_size) {

    float *input = (float *)aligned_alloc(64, batch_size * seq_length * hidden_size * sizeof(float));
    float *gateW = (float *)aligned_alloc(64, hidden_size * intermediate_size * sizeof(float));
    float *upW = (float *)aligned_alloc(64, hidden_size * intermediate_size * sizeof(float));
    float *downW = (float *)aligned_alloc(64, intermediate_size * hidden_size * sizeof(float));
    float *ourOutput = (float *)aligned_alloc(64, batch_size * seq_length * hidden_size * sizeof(float));
    float *refOutput = (float *)aligned_alloc(64, batch_size * seq_length * hidden_size * sizeof(float));

    for (int i = 0; i < batch_size * seq_length * hidden_size; ++i) {
        input[i] = static_cast<float>(1.0f * rand() / RAND_MAX);
    }

    for (int i = 0; i < hidden_size * intermediate_size; ++i) {
        gateW[i] = static_cast<float>(1.0f * rand() / RAND_MAX);
        upW[i] = static_cast<float>(1.0f * rand() / RAND_MAX);
        downW[i] = static_cast<float>(1.0f * rand() / RAND_MAX);
    }

    if constexpr (std::is_same<T, bfloat16_t>::value) {
        invokeMLPLLaMA(xft::DataType::bf16, batch_size, seq_length, hidden_size, intermediate_size, (void *)ourOutput,
                hidden_size, (const void *)input, hidden_size, (const void *)gateW, (const void *)upW,
                (const void *)downW);
        refMLPLLaMA<bfloat16_t>(batch_size, seq_length, hidden_size, intermediate_size, (bfloat16_t *)refOutput,
                hidden_size, (const bfloat16_t *)input, hidden_size, (const bfloat16_t *)gateW, (const bfloat16_t *)upW,
                (const bfloat16_t *)downW);
    }

    for (int i = 0; i < batch_size * seq_length * hidden_size; ++i) {
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
    // compareMLPLLaMA<bfloat16_t>(1, 6, 32, 128);
    compareMLPLLaMA<bfloat16_t>(1, 128, 4096, 11008);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}