#include <ctime>
#include <iostream>

#include "alibi_embedding.h"
#include "gtest/gtest.h"

static bool compare(const float *result, const float *ground_truth, const int size) {
    const float diff = 0.001;
    for (int i = 0; i < size; ++i) {
        if (abs(ground_truth[i] - result[i]) > diff) { return false; }
    }
    return true;
}

TEST(AlibiEmbedding, AlibiEmbeddingTest) {
    int seq_len = 6, head_num = 6, headIdx = 4;
    AlibiEmbedding alibi(head_num, seq_len);
    float *bias_matrx = (float *)malloc(seq_len * seq_len * sizeof(float));
    alibi.alibi_get_bias(headIdx, seq_len, bias_matrx);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            std::cout << bias_matrx[i * seq_len + j] << " ";
        }
        std::cout << std::endl;
    }

    float groundtruth[36] = {0.0000, 0.0098, 0.0197, 0.0295, 0.0394, 0.0492, -0.0098, 0.0000, 0.0098, 0.0197, 0.0295,
            0.0394, -0.0197, -0.0098, 0.0000, 0.0098, 0.0197, 0.0295, -0.0295, -0.0197, -0.0098, 0.0000, 0.0098, 0.0197,
            -0.0394, -0.0295, -0.0197, -0.0098, 0.0000, 0.0098, -0.0492, -0.0394, -0.0295, -0.0197, -0.0098, 0.0000};
    int size = 36;
    EXPECT_TRUE(compare(bias_matrx, groundtruth, size));

    free(bias_matrx);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}