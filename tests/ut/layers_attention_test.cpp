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
#include "layers_attention.h"
#include "gtest/gtest.h"

template <typename T>
static void compareAttentionLLaMA(int attHeadDim, int attHeadNum, int kvHeadNum, int maxPositions, int maxPosEmbed,
        int maxSeqLength, int hiddenSize, const float *ln1Gamma, const float *ln1Beta, const void *queryWeight,
        const void *keyWeight, const void *valueWeight, const void *attnOutWeight) {
    int batchSize = 1;
    int inputSeqLen = 18;
    int pastSeqLen = 0;
    int currentSeqLen = inputSeqLen;
    int step = 0;

    float *input = (float *)aligned_alloc(64, batchSize * inputSeqLen * hiddenSize * sizeof(float));
    float *ourOutput = (float *)aligned_alloc(64, batchSize * inputSeqLen * hiddenSize * sizeof(float));
    memset(ourOutput, 0, batchSize * inputSeqLen * hiddenSize * sizeof(float));

    for (int i = 0; i < batchSize * inputSeqLen * hiddenSize; ++i) {
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
    invokeAttentionLLaMA(dt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum, maxPositions, maxPosEmbed,
            maxSeqLength, pastSeqLen, currentSeqLen, step, hiddenSize, (void *)ourOutput, hiddenSize,
            (const void *)input, hiddenSize, ln1Gamma, ln1Beta, (const void *)queryWeight, (const void *)keyWeight,
            (const void *)valueWeight, (const void *)attnOutWeight);
    auto end = std::chrono::high_resolution_clock::now();
    float during_time = std::chrono::duration<float>(end - start).count();
    printf("[ RUNTIME  ] XFT::invokeAttentionLLaMA %.6f sec\n", during_time);

    free(input);
    free(ourOutput);
}

template <typename T>
void test_AttentionLLaMA(void) {
    int maxPosEmbed = 4096;
    int maxPositions = maxPosEmbed;
    int maxSeqLength = maxPosEmbed;
    int hiddenSize = 4096;
    int attHeadNum = 32;
    int attHeadDim = hiddenSize / attHeadNum;
    int kvHeadNum = 32;
    int qSize = attHeadDim * attHeadNum;
    int kvSize = attHeadDim * kvHeadNum;

    float *ln1Gamma = (float *)aligned_alloc(64, hiddenSize * sizeof(float));
    float *ln1Beta = (float *)aligned_alloc(64, hiddenSize * sizeof(float));
    float *qkvProj = (float *)aligned_alloc(64, hiddenSize * (qSize + 2 * kvSize) * sizeof(float));
    float *oProj = (float *)aligned_alloc(64, hiddenSize * hiddenSize * sizeof(float));

    for (int i = 0; i < hiddenSize; ++i) {
        ln1Gamma[i] = static_cast<float>(0.5f * rand() / RAND_MAX);
        ln1Beta[i] = static_cast<float>(0.5f * rand() / RAND_MAX);
    }

    for (int i = 0; i < hiddenSize * (qSize + 2 * kvSize); ++i) {
        qkvProj[i] = static_cast<float>(0.5f * rand() / RAND_MAX);
    }

    for (int i = 0; i < hiddenSize * hiddenSize; ++i) {
        oProj[i] = static_cast<float>(0.5f * rand() / RAND_MAX);
    }

    compareAttentionLLaMA<T>(attHeadDim, attHeadNum, kvHeadNum, maxPositions, maxPosEmbed, maxSeqLength, hiddenSize,
            ln1Gamma, ln1Beta, qkvProj, qkvProj + qSize, qkvProj + kvSize, oProj);

    free(ln1Gamma);
    free(ln1Beta);
    free(qkvProj);
    free(oProj);
}

TEST(AttentionLLaMA, bfloat16_t) {
    test_AttentionLLaMA<bfloat16_t>();
}

TEST(AttentionLLaMA, float16_t) {
    test_AttentionLLaMA<float16_t>();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}