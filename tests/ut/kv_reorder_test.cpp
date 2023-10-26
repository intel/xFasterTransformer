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

static void swapValues(float *p1, float *p2, int size) {
    int i = 0;
    for (; i + 15 < size; i += 16) {
        __m512 v1 = _mm512_loadu_ps(p1 + i);
        __m512 v2 = _mm512_loadu_ps(p2 + i);
        _mm512_storeu_ps(p1 + i, v2);
        _mm512_storeu_ps(p2 + i, v1);
    }

    if (i < size) {
        int remain = size - i;
        __mmask16 mask = (1 << remain) - 1;

        __m512 v1 = _mm512_maskz_loadu_ps(mask, p1 + i);
        __m512 v2 = _mm512_maskz_loadu_ps(mask, p2 + i);
        _mm512_mask_storeu_ps(p1 + i, mask, v2);
        _mm512_mask_storeu_ps(p2 + i, mask, v1);
    }
}

static void skippableCopy(float *dst, float *src, int size) {
    // Copy only when different
    // TODO: check if there are any risks
    if (*(uint64_t *)dst != *(uint64_t *)src) { memcpy(dst, src, size * sizeof(float)); }
}

template <typename T>
static bool valueExist(T *arr, int size, T val) {
    for (int i = 0; i < size; ++i) {
        if (arr[i] == val) { return true; }
    }
    return false;
}

static void reorder(float *keys, float *values, int accSeqLen, int cols, int *idx, int size) {
    // Temporary buffer used for reorder
    float *extraKeyBuf = (float *)aligned_alloc(64, 2 * (size - 1) * cols * sizeof(float));
    float *extraValBuf = extraKeyBuf + (size - 1) * cols;

    // This for loop is copied from the source code
    for (int s = 0; s < accSeqLen; ++s) {
        int extraBufIdx = 0;
        int remapped[size];
        memcpy(remapped, idx, size * sizeof(int));

        for (int i = 0; i < size; ++i) {
            int from = remapped[i];
            if (from < i) { // The source line already reordered
                // Current line will be used in future, thus save to extra buffer
                if (valueExist(remapped + i + 1, size - i - 1, i)) {
                    memcpy(extraKeyBuf + extraBufIdx * cols, keys + i * cols, cols * sizeof(float));
                    memcpy(extraValBuf + extraBufIdx * cols, values + i * cols, cols * sizeof(float));

                    // When need line i, should look into temporary buffer, (extraBufIdx - size) < 0, always
                    std::replace(remapped + i + 1, remapped + size, i, extraBufIdx - size);
                    extraBufIdx += 1;
                }

                if (from < 0) { // copy from extraBuf
                    skippableCopy(keys + i * cols, extraKeyBuf + (from + size) * cols, cols);
                    skippableCopy(values + i * cols, extraValBuf + (from + size) * cols, cols);
                } else {
                    skippableCopy(keys + i * cols, keys + from * cols, cols);
                    skippableCopy(values + i * cols, values + from * cols, cols);
                }
            } else if (from > i) {
                // Just need to swap
                if (remapped[from] == i) {
                    swapValues(keys + i * cols, keys + from * cols, cols);
                    swapValues(values + i * cols, values + from * cols, cols);

                    // Update the map information
                    std::transform(remapped + i + 1, remapped + size, remapped + i + 1, [&](int num) {
                        if (num == i) {
                            return from;
                        } else if (num == from) {
                            return i;
                        }
                        return num;
                    });
                }
                // Current line will be used in future, thus save to extra buffer
                else if (valueExist(remapped + i + 1, size - i - 1, i)) {
                    memcpy(extraKeyBuf + extraBufIdx * cols, keys + i * cols, cols * sizeof(float));
                    memcpy(extraValBuf + extraBufIdx * cols, values + i * cols, cols * sizeof(float));

                    // When need line i, should look into temporary buffer, (extraBufIdx - size) < 0, always
                    std::replace(remapped + i + 1, remapped + size, i, extraBufIdx - size);
                    extraBufIdx += 1;

                    skippableCopy(keys + i * cols, keys + from * cols, cols);
                    skippableCopy(values + i * cols, values + from * cols, cols);

                    // When need line 'from', should look into line i
                    std::replace(remapped + i + 1, remapped + size, from, i);
                }
                // Current line will never be used in futre, just overwrite it
                else {
                    skippableCopy(keys + i * cols, keys + from * cols, cols);
                    skippableCopy(values + i * cols, values + from * cols, cols);

                    // When need line 'from', should look into line i
                    std::replace(remapped + i + 1, remapped + size, from, i);
                }
            }
        }

        keys += size * cols;
        values += size * cols;
    }

    // Clean up
    free(extraKeyBuf);
}

static void reorder(int *idx) {
    int bs = 6;
    int cols = 128;
    int accSeqLen = 3;

    float *keys = (float *)aligned_alloc(64, accSeqLen * bs * cols * sizeof(float));
    float *values = (float *)aligned_alloc(64, accSeqLen * bs * cols * sizeof(float));

    for (int i = 0; i < accSeqLen * bs; ++i) {
        std::fill(keys + i * cols, keys + (i + 1) * cols, 1.0f * i);
        std::fill(values + i * cols, values + (i + 1) * cols, 1.0f * i);
    }

    reorder(keys, values, accSeqLen, cols, idx, bs);

    for (int i = 0; i < accSeqLen; ++i) {
        for (int b = 0; b < bs; ++b) {
            EXPECT_NEAR(keys[(i * bs + b) * cols], i * bs + 1.0f * idx[b], 0.00001f);
            EXPECT_NEAR(values[(i * bs + b) * cols], i * bs + 1.0f * idx[b], 0.00001f);
        }
    }

    free(keys);
    free(values);
}

TEST(reorder1, reorder1) {
    int idx[] = {0, 2, 1, 4, 3, 3};
    reorder(idx);
}

TEST(reorder2, reorder2) {
    int idx[] = {1, 2, 3, 4, 5, 0};
    reorder(idx);
}

TEST(reorder3, reorder3) {
    int idx[] = {1, 3, 2, 4, 3, 5};
    reorder(idx);
}

TEST(reorder4, reorder4) {
    int idx[] = {2, 1, 0, 4, 5, 0};
    reorder(idx);
}

TEST(reorder5, reorder5) {
    int idx[] = {2, 0, 1, 5, 5, 4};
    reorder(idx);
}

TEST(reorderAll, reorderAll) {
    for (int i = 0; i < 256; ++i) {
        int id4 = i % 4;
        int id123 = i / 4;
        int id3 = id123 % 4;
        int id12 = id123 / 4;
        int id2 = id12 % 4;
        int id1 = id12 / 4;
        int idx[] = {id1, id2, id3, id4, 5, 4};
        reorder(idx);
    }
}

////////////////////////////////////////////////////////////////////////////////

void computeLogSoftmax(float *input, float *output, int size) {
    float max = input[0];
#pragma omp parallel for reduction(max : max)
    for (int i = 1; i < size; i++) {
        if (input[i] > max) { max = input[i]; }
    }

    float sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < size; i++) {
        sum += std::exp(input[i] - max);
    }

    float logsum = std::log(sum);
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        output[i] = input[i] - max - logsum;
    }
}

TEST(reorderForward, reorderForward) {
    const char *modelPath = "/data/1-gpu";
    OptDecoder<float16_t> decoder(modelPath);

    const int batchSize = 2;
    const int beamSize = 3;

    // Initial input
    const int seqLen = 22;
    int samples[batchSize * beamSize][seqLen] = {
            {2, 100, 17, 27, 119, 10080, 10288, 6, 38, 17, 27, 119, 2602, 14, 38, 17, 27, 548, 56, 5, 945, 7},
            {2, 100, 17, 27, 119, 10080, 10288, 6, 38, 17, 27, 119, 2602, 14, 38, 17, 27, 548, 56, 5, 945, 7},
            {2, 100, 17, 27, 119, 10080, 10288, 6, 38, 17, 27, 119, 2602, 14, 38, 17, 27, 548, 56, 5, 945, 7},
            {2, 4993, 41, 1946, 9, 1431, 6, 9469, 6, 5, 5385, 9, 5, 446, 9, 7395, 174, 1865, 5, 80, 2380, 2442},
            {2, 4993, 41, 1946, 9, 1431, 6, 9469, 6, 5, 5385, 9, 5, 446, 9, 7395, 174, 1865, 5, 80, 2380, 2442},
            {2, 4993, 41, 1946, 9, 1431, 6, 9469, 6, 5, 5385, 9, 5, 446, 9, 7395, 174, 1865, 5, 80, 2380, 2442},
    };

    // Subsequent input IDs, dumped from transformers solution
    int inputIds[][6] = {
            {28, 1807, 3594, 444, 44, 22},
            {10, 5, 5, 4102, 48, 5525},
            {233, 82, 82, 15, 4, 8},
            {9, 9, 9, 50118, 5, 10},
            {42, 5, 402, 50118, 346, 696},
    };

    // Beam indices dumped from transformers solution
    int beamIndices[][6] = {
            {0, 0, 0, 3, 3, 3},
            {0, 2, 1, 3, 4, 5},
            {0, 2, 1, 3, 3, 3},
            {0, 1, 2, 4, 3, 3},
    };

    // Logits output dumped from transformers solution
    float expectedLogits[][6] = {
            {4.7295, 3.9903, 4.5394, 0.5524, -1.2528, -0.8553},
            {3.2884, 3.0396, 2.7616, 2.1482, -0.7856, -0.6693},
            {7.2427, 4.7973, 4.7772, 0.2183, 4.0118, 1.9940},
            {6.1418, 2.7603, 2.8243, 4.1816, 0.3531, -0.6773},
    };

    // To get first token
    int64_t dims[3] = {batchSize, beamSize, seqLen};
    std::tuple<float *, int, int> result = decoder.forward((int *)samples, dims, 0);

    int vocSize = std::get<2>(result);
    float *p0 = std::get<0>(result);
    float *p1 = p0 + beamSize * vocSize;

    computeLogSoftmax(p0, p0, vocSize);
    computeLogSoftmax(p1, p1, vocSize);

    float maxVal0 = p0[0], maxVal1 = p1[0];
    int maxIndex0 = 0, maxIndex1 = 0;
    for (int i = 1; i < vocSize; i++) {
        if (p0[i] > maxVal0) {
            maxVal0 = p0[i];
            maxIndex0 = i;
        }
        if (p1[i] > maxVal1) {
            maxVal1 = p1[i];
            maxIndex1 = i;
        }
    }

    // Equals 28, 444
    EXPECT_EQ(maxIndex0, inputIds[0][0]);
    EXPECT_EQ(maxIndex1, inputIds[0][beamSize]);

    float preLogSoftmax[6] = {};

    // Subsequent tokens generation
    for (int i = 0; i < 4; ++i) {
        decoder.reorderCache(beamIndices[i], batchSize * beamSize);

        int64_t dims[3] = {batchSize, beamSize, 1};
        std::tuple<float *, int, int> result = decoder.forward(inputIds[i], dims, i + 1);

        float *p = std::get<0>(result);
        int vocSize = std::get<2>(result);

        for (int b = 0; b < 6; ++b) {
            EXPECT_NEAR(p[b * vocSize], expectedLogits[i][b], 0.001f);
        }
    }
}

int main(int argc, char **argv) {
    srand(time(NULL));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
