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

#include "alibi_embedding.h"
#include "gtest/gtest.h"

static bool compare(const float *result, const float *groundTruth, const int size) {
    const float diff = 0.001;
    for (int i = 0; i < size; ++i) {
        if (abs(groundTruth[i] - result[i]) > diff) { return false; }
    }
    return true;
}

TEST(AlibiEmbedding, AlibiEmbeddingTest) {
    int seqLen = 6, headNum = 6, headIdx = 4;
    AlibiEmbedding alibi(headNum, seqLen);
    float *biasMatrx = (float *)malloc(seqLen * seqLen * sizeof(float));
    alibi.alibiGetBias(headIdx, seqLen, biasMatrx);

    float groundTruth[36] = {0.0000, 0.0098, 0.0197, 0.0295, 0.0394, 0.0492, -0.0098, 0.0000, 0.0098, 0.0197, 0.0295,
            0.0394, -0.0197, -0.0098, 0.0000, 0.0098, 0.0197, 0.0295, -0.0295, -0.0197, -0.0098, 0.0000, 0.0098, 0.0197,
            -0.0394, -0.0295, -0.0197, -0.0098, 0.0000, 0.0098, -0.0492, -0.0394, -0.0295, -0.0197, -0.0098, 0.0000};
    int size = 36;
    EXPECT_TRUE(compare(biasMatrx, groundTruth, size));

    free(biasMatrx);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}