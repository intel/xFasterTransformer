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

#include "opt_decoder.h"
#include "gtest/gtest.h"

template <typename T = float>
static void reorder(int *idx, int bs) {
    int headNum = 32;
    int headSize = 128;
    int seqLen = 10;

    KVCacheTensor<T> tensor;
    tensor.resize(seqLen, bs, headNum, headSize);

#pragma omp parallel for collapse(3)
    for (int seqIdx = 0; seqIdx < seqLen; ++seqIdx) {
        for (int batchIdx = 0; batchIdx < bs; ++batchIdx) {
            for (int headIdx = 0; headIdx < headNum; ++headIdx) {
                // Set each position to a unique value
                T val = (T)(1.0f * (seqIdx * bs + batchIdx) + 0.01f * headIdx);
                T *data = tensor.getSequence(seqIdx, batchIdx, headIdx).first;
                std::fill(data, data + headSize, val);
            }
        }
    }

    tensor.reorder(idx, bs, 0, seqLen);

#pragma omp parallel for collapse(3)
    for (int seqIdx = 0; seqIdx < seqLen; ++seqIdx) {
        for (int batchIdx = 0; batchIdx < bs; ++batchIdx) {
            for (int headIdx = 0; headIdx < headNum; ++headIdx) {
                T val = (T)(1.0f * (seqIdx * bs + idx[batchIdx]) + 0.01f * headIdx);
                T *data = tensor.getSequence(seqIdx, batchIdx, headIdx).first;
                for (int i = 0; i < headSize; ++i) {
                    EXPECT_NEAR(data[i], val, 0.00001f);
                }
            }
        }
    }
}

TEST(reorder4, reorder4) {
    int idx1[] = {0, 2, 1, 3};
    reorder(idx1, sizeof(idx1) / sizeof(int));

    int idx2[] = {0, 0, 0, 0};
    reorder(idx2, sizeof(idx2) / sizeof(int));

    int idx3[] = {1, 1, 0, 0};
    reorder(idx3, sizeof(idx3) / sizeof(int));

    int idx4[] = {1, 0, 2, 1};
    reorder(idx4, sizeof(idx4) / sizeof(int));

    int idx5[] = {0, 1, 0, 1};
    reorder(idx5, sizeof(idx5) / sizeof(int));
}

TEST(reorder6, reorder6) {
    int idx1[] = {0, 2, 1, 4, 3, 3};
    reorder(idx1, sizeof(idx1) / sizeof(int));

    int idx2[] = {1, 2, 3, 4, 5, 0};
    reorder(idx2, sizeof(idx2) / sizeof(int));

    int idx3[] = {1, 3, 2, 4, 3, 5};
    reorder(idx3, sizeof(idx3) / sizeof(int));

    int idx4[] = {2, 1, 0, 4, 5, 0};
    reorder(idx4, sizeof(idx4) / sizeof(int));

    int idx5[] = {2, 0, 1, 5, 5, 4};
    reorder(idx5, sizeof(idx5) / sizeof(int));
}

// Test for beam size 4, user side batch size 2
TEST(reorder8, reorder8) {
    for (int i = 0; i < 256; ++i) {
        int idx[8];
        for (int j = 0; j < 4; ++j) {
            idx[j] = rand() % 4;
            idx[j + 4] = rand() % 4 + 4;
        }
        reorder(idx, sizeof(idx) / sizeof(int));
    }
}

TEST(reorder6More, reorder6More) {
    for (int i = 0; i < 256; ++i) {
        int id4 = i % 4;
        int id123 = i / 4;
        int id3 = id123 % 4;
        int id12 = id123 / 4;
        int id2 = id12 % 4;
        int id1 = id12 / 4;
        int idx[] = {id1, id2, id3, id4, 5, 4};
        reorder(idx, sizeof(idx) / sizeof(int));
    }
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    srand(time(NULL));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
