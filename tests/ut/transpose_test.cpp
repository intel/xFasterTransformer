#include <cmath>

#include "transpose_util.h"
#include "gtest/gtest.h"

template <typename T>
static void transpose_ref(T *src, T *dst, int rows, int cols, int lda, int ldb) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[j * ldb + i] = src[i * lda + j];
        }
    }
}

static void compareTranspose(int rows, int cols) {
    int lda = cols;
    int ldb = rows;

    float *weight = (float *)aligned_alloc(64, rows * cols * sizeof(float));
    float *ourRet = (float *)aligned_alloc(64, rows * cols * sizeof(float));
    float *refRet = (float *)aligned_alloc(64, rows * cols * sizeof(float));

    for (int i = 0; i < rows * cols; ++i) {
        weight[i] = 1.0f * rand() / RAND_MAX;
    }

    transpose_ref(weight, refRet, rows, cols, lda, ldb);

    TransposeUtil::transpose(weight, ourRet, rows, cols);

    for (int i = 0; i < rows * cols; ++i) {
        EXPECT_EQ(ourRet[i], refRet[i]);
    }

    free(weight);
    free(ourRet);
    free(refRet);
}

TEST(transpose, transpose) {
    compareTranspose(128, 128);
    compareTranspose(5120, 5120);
    compareTranspose(5120, 5120 * 3);
    compareTranspose(rand() % 100 + 100, rand() % 100 + 100);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}