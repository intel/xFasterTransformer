#include <ctime>
#include <iostream>

#include "rotary_embedding.h"
#include "gtest/gtest.h"

static bool compare(const float *result, const float *ground_truth, const int size) {
    const float diff = 0.001;
    for (int i = 0; i < size; ++i) {
        if (abs(ground_truth[i] - result[i]) > diff) { return false; }
    }
    return true;
}

TEST(RotrayEmbedding, RotrayEmbeddingTest) {
    int bs = 2, seq = 2, headnum = 2, dim = 2;
    int max_len = 10;
    int qkshape[4] = {bs, seq, headnum, dim};
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}