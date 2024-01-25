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
#include <ctime>
#include <iostream>

#include "rotary_embedding.h"
#include "gtest/gtest.h"

static bool compare(const float *result, const float *ground_truth, const int size, const float diff = 0.001) {
    for (int i = 0; i < size; ++i) {
        if (abs(ground_truth[i] - result[i]) > diff) { return false; }
    }
    return true;
}

TEST(RotrayEmbedding, RotrayEmbeddingTest) {
    int bs = 2, seq = 2, headnum = 2, dim = 2;
    int max_len = 10;
    int qkshape[5] = {bs, seq, headnum, dim, headnum};
    int pos_ids[2] = {1, 0};
    int stride = bs * seq, size = bs * seq * headnum * dim;
    LlamaRotaryEmbedding RotrayEmbeddingTest(dim, max_len);
    float q[16] = {4, 4, 4, 4, 3, 2, 1, 1, 4, 4, 2, 1, 4, 1, 3, 0};
    float k[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float q_groundtruth[16]
            = {-1.20467, 5.52709, -1.20467, 5.52709, 3, 2, 1, 1, -1.20467, 5.52709, 0.239134, 2.22324, 4, 1, 3, 0};
    float k_groundtruth[16]
            = {-0.301169, 1.38177, -0.301169, 1.38177, 1, 1, 1, 1, -0.301169, 1.38177, -0.301169, 1.38177, 1, 1, 1, 1};
    RotrayEmbeddingTest.forward(q, k, stride, stride, qkshape, pos_ids);
    EXPECT_TRUE(compare(q, q_groundtruth, size));
    EXPECT_TRUE(compare(k, k_groundtruth, size));
}

TEST(RotrayEmbedding, BF16Test) {
    int bs = 2, seq = 2, headnum = 2, dim = 2;
    int max_len = 10;
    int qkshape[5] = {bs, seq, headnum, dim, headnum};
    int pos_ids[2] = {1, 0};
    int stride = bs * seq, size = bs * seq * headnum * dim;

    float q_fp32[16] = {4, 4, 4, 4, 3, 2, 1, 1, 4, 4, 2, 1, 4, 1, 3, 0};
    float k_fp32[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float q_groundtruth[16]
            = {-1.20467, 5.52709, -1.20467, 5.52709, 3, 2, 1, 1, -1.20467, 5.52709, 0.239134, 2.22324, 4, 1, 3, 0};
    float k_groundtruth[16]
            = {-0.301169, 1.38177, -0.301169, 1.38177, 1, 1, 1, 1, -0.301169, 1.38177, -0.301169, 1.38177, 1, 1, 1, 1};

    bfloat16_t q[16];
    bfloat16_t k[16];
    bfloat16_t::cvt_float_to_bfloat16(q_fp32, q, 16);
    bfloat16_t::cvt_float_to_bfloat16(k_fp32, k, 16);

    LlamaRotaryEmbedding RotrayEmbeddingTest(dim, max_len);
    RotrayEmbeddingTest.forward(q, k, stride, stride, qkshape, pos_ids);

    bfloat16_t::cvt_bfloat16_to_float(q, q_fp32, 16);
    bfloat16_t::cvt_bfloat16_to_float(k, k_fp32, 16);

    EXPECT_TRUE(compare(q_fp32, q_groundtruth, size, 0.01));
    EXPECT_TRUE(compare(k_fp32, k_groundtruth, size, 0.01));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}