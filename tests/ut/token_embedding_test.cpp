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
#include <cmath>
#include <type_traits>

#include "bfloat16.h"
#include "float16.h"
#include "layers_norm.h"
#include "gtest/gtest.h"

#include "token_embedding_kernels.h"

template <typename OutT, typename WeiT>
static void TestTokenEmbeddingKernel(const int vocabSize, const int hiddenSize, const int tokenSize) {
    WeiT *embTable = (WeiT *)aligned_alloc(64, vocabSize * hiddenSize * sizeof(WeiT));
    int *tokenId = (int *)aligned_alloc(64, tokenSize * sizeof(int));
    OutT *output = (OutT *)aligned_alloc(64, tokenSize * hiddenSize * sizeof(OutT));

    for (int i = 0; i < vocabSize; i++) {
        for (int j = 0; j < hiddenSize; j++) {
            embTable[i * hiddenSize + j] = static_cast<WeiT>(rand()) / RAND_MAX;
        }
    }

    for (int i = 0; i < tokenSize; i++) {
        tokenId[i] = rand() % vocabSize;
    }

    xft::tokenEmbedding<OutT, WeiT>(output, tokenId, embTable, tokenSize, hiddenSize);

    for (int i = 0; i < tokenSize; i++) {
        int id = tokenId[i];
        for (int j = 0; j < hiddenSize; j++) {
            EXPECT_FLOAT_EQ(float(output[i * hiddenSize + j]), float(embTable[id * hiddenSize + j]));
        }
    }

    free(embTable);
    free(tokenId);
    free(output);
}

#define UT_EMBEDDING(MN, OutT, WeiT, VS, HS, BS, SL)                               \
    TEST(TokenEmbeddingKernel, MN##_BS##BS##_Lens##SL##_OutT##OutT##_WeiT##WeiT) { \
        TestTokenEmbeddingKernel<OutT, WeiT>((VS), (HS), (BS) * (SL));             \
    }

UT_EMBEDDING(Llama2_7B, float_t, float16_t, 32000, 4096, 1, 64);
UT_EMBEDDING(Llama2_7B, float_t, float16_t, 32000, 4096, 1, 256);
UT_EMBEDDING(Llama2_7B, float_t, float16_t, 32000, 4096, 1, 512);
UT_EMBEDDING(Llama2_7B, float_t, float16_t, 32000, 4096, 8, 512);

UT_EMBEDDING(Llama2_7B, float16_t, float16_t, 32000, 4096, 1, 64);
UT_EMBEDDING(Llama2_7B, float16_t, float16_t, 32000, 4096, 1, 256);
UT_EMBEDDING(Llama2_7B, float16_t, float16_t, 32000, 4096, 1, 512);
UT_EMBEDDING(Llama2_7B, float16_t, float16_t, 32000, 4096, 8, 512);

// UT_EMBEDDING(Llama2_7B, bfloat16_t, bfloat16_t, 32000, 4096, 1, 64);
// UT_EMBEDDING(Llama2_7B, bfloat16_t, bfloat16_t, 32000, 4096, 1, 256);
// UT_EMBEDDING(Llama2_7B, bfloat16_t, bfloat16_t, 32000, 4096, 1, 512);
// UT_EMBEDDING(Llama2_7B, bfloat16_t, bfloat16_t, 32000, 4096, 512, 512);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}