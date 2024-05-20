#include <algorithm>
#include "attention_kernels.h"
#include "matmul_helper.h"
#include "gtest/gtest.h"

// Reference implementation of matrix multiplication
static void mmRef(
        bfloat16_t *A, bfloat16_t *B, bfloat16_t *C, int M, int N, int K, int lda, int ldb, int ldc, bool transB) {
    if (transB) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += (float)A[m * lda + k] * (float)B[n * ldb + k];
                }
                C[m * ldc + n] = sum;
            }
        }
    } else {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += (float)A[m * lda + k] * (float)B[k * ldb + n];
                }
                C[m * ldc + n] = sum;
            }
        }
    }
}

static void selfAttentionRef(bfloat16_t *output, bfloat16_t *query, bfloat16_t *key, bfloat16_t *value, int qHeadNum,
        int kvHeadNum, int headSize, int oStride, int qStride, int kvStride, int tokenSize, const float scale) {
    const int groupSize = qHeadNum / kvHeadNum;

#pragma omp parallel for
    for (int h = 0; h < qHeadNum; h++) {
        bfloat16_t *q = query + h * headSize;
        bfloat16_t *k = key + (h / groupSize) * headSize;
        bfloat16_t *v = value + (h / groupSize) * headSize;

        bfloat16_t *score = (bfloat16_t *)malloc(tokenSize * tokenSize * sizeof(bfloat16_t));
        float *fscore = (float *)malloc(tokenSize * tokenSize * sizeof(float));

        // Compute matmul between 'query' and 'key'
        mmRef(q, k, score, tokenSize, tokenSize, headSize, qStride, kvStride, tokenSize, true);

        // Scale
        for (int i = 0; i < tokenSize; i++) {
            for (int j = 0; j < tokenSize; j++) {
                fscore[i * tokenSize + j] = (float)score[i * tokenSize + j] * scale;
            }
        }

        // Compute softmax of the result
        for (int i = 0; i < tokenSize; i++) {
            float sum = 0.0f;
            float maxVal = std::numeric_limits<float>::lowest();
            int size = i + 1; // causual attention
            for (int j = 0; j < size; j++) {
                maxVal = std::max(maxVal, fscore[i * tokenSize + j]);
            }
            for (int j = 0; j < size; j++) {
                sum += std::exp(fscore[i * tokenSize + j] - maxVal);
            }
            float rsum = 1.0f / sum;
            for (int j = 0; j < size; j++) {
                score[i * tokenSize + j] = (bfloat16_t)(std::exp(fscore[i * tokenSize + j] - maxVal) * rsum);
            }
            for (int j = size; j < tokenSize; ++j) {
                score[i * tokenSize + j] = 0.0f;
            }
        }

        // Compute matmul between result and 'value'
        auto out = output + h * headSize;
        mmRef(score, v, out, tokenSize, headSize, tokenSize, tokenSize, kvStride, oStride, false);

        free(score);
        free(fscore);
    }
}

// Reference implementation of self-attention
static void selfAttentionRef(bfloat16_t *output, bfloat16_t *query, bfloat16_t *key, bfloat16_t *value, int qHeadNum,
        int kvHeadNum, int headSize, int oStride, int qStride, int kvStride, int batchSize, const int *tokenSizes,
        const float scale) {

    int rowOffsets[batchSize] = {0};
    for (int i = 1; i < batchSize; i++) {
        rowOffsets[i] = rowOffsets[i - 1] + tokenSizes[i - 1];
    }

    for (int b = 0; b < batchSize; b++) {
        auto q = query + rowOffsets[b] * qStride;
        auto k = key + rowOffsets[b] * kvStride;
        auto v = value + rowOffsets[b] * kvStride;
        selfAttentionRef(output + rowOffsets[b] * oStride, q, k, v, qHeadNum, kvHeadNum, headSize, oStride, qStride,
                kvStride, tokenSizes[b], scale);
    }
}

void testSelfAttention(
        int headSize, int qHeadNum, int kvHeadNum, int *tokenSizes, int batchSize, bool bSeparateCopy = true) {
    const int qkvStride = headSize * (qHeadNum + 2 * kvHeadNum);
    const int hiddenSize = headSize * qHeadNum;
    const float scale = 1.0f / std::sqrt(headSize);
    int threadNum = -1;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid == 0) { threadNum = omp_get_num_threads(); }
    }

    int totalTokens = 0;
    int maxTokens = 0;
    for (int i = 0; i < batchSize; i++) {
        totalTokens += tokenSizes[i];
        maxTokens = std::max(maxTokens, tokenSizes[i]);
    }

    // Buffers
    bfloat16_t *ourOutput = (bfloat16_t *)xft::alloc(totalTokens * hiddenSize * sizeof(bfloat16_t));
    bfloat16_t *refOutput = (bfloat16_t *)xft::alloc(totalTokens * hiddenSize * sizeof(bfloat16_t));
    bfloat16_t *kvCache
            = (bfloat16_t *)xft::alloc(2 * batchSize * maxTokens * (kvHeadNum * headSize) * sizeof(bfloat16_t));
    bfloat16_t *qkv = (bfloat16_t *)xft::alloc(totalTokens * qkvStride * sizeof(bfloat16_t));

    bfloat16_t *kCache = kvCache;
    bfloat16_t *vCache = kvCache + batchSize * maxTokens * (kvHeadNum * headSize);

    bfloat16_t *query = qkv;
    bfloat16_t *key = qkv + headSize * qHeadNum;
    bfloat16_t *value = qkv + headSize * (qHeadNum + kvHeadNum);

    float factor = 1.0f / RAND_MAX;
    for (int i = 0; i < totalTokens * qkvStride; i++) {
        qkv[i] = bfloat16_t(rand() * factor);
    }

    // Call the function
    if (bSeparateCopy) {
        xft::selfAttention_SeparateCopy<true>(
                ourOutput, query, key, value, qHeadNum, kvHeadNum, headSize, hiddenSize, qkvStride, qkvStride,
                batchSize, tokenSizes, scale, nullptr, threadNum,
                [&](int b, int h, int s) {
                    return kCache + b * maxTokens * kvHeadNum * headSize + s * kvHeadNum * headSize + h * headSize;
                },
                [&](int b, int h, int s) {
                    return vCache + b * maxTokens * kvHeadNum * headSize + s * kvHeadNum * headSize + h * headSize;
                });
    } else {
        xft::selfAttention_FusedCopy(
                ourOutput, query, key, value, qHeadNum, kvHeadNum, headSize, hiddenSize, qkvStride, qkvStride,
                batchSize, tokenSizes, scale, nullptr, threadNum,
                [&](int b, int h, int s) {
                    return kCache + b * maxTokens * kvHeadNum * headSize + s * kvHeadNum * headSize + h * headSize;
                },
                [&](int b, int h, int s) {
                    return vCache + b * maxTokens * kvHeadNum * headSize + s * kvHeadNum * headSize + h * headSize;
                });
    }

    selfAttentionRef(refOutput, query, key, value, qHeadNum, kvHeadNum, headSize, hiddenSize, qkvStride, qkvStride,
            batchSize, tokenSizes, scale);

    // Verify the correctness of the function
    for (int i = 0; i < totalTokens * hiddenSize; i++) {
        ASSERT_NEAR((float)refOutput[i], (float)ourOutput[i], 1e-2);
    }

    // Clean up
    free(ourOutput);
    free(refOutput);
    free(kvCache);
    free(qkv);
}

TEST(AttentionKernelsTest, SeparateCopyTest1) {
    int batchSize = 1;
    int tokenSizes[batchSize] = {80};
    testSelfAttention(128, 2, 2, tokenSizes, batchSize);
}

TEST(AttentionKernelsTest, SeparateCopyTest2) {
    int batchSize = 1;
    int tokenSizes[batchSize] = {100};
    testSelfAttention(128, 6, 2, tokenSizes, batchSize);
}

TEST(AttentionKernelsTest, SeparateCopyTest3) {
    int batchSize = 2;
    int tokenSizes[batchSize] = {100, 200};
    testSelfAttention(128, 8, 2, tokenSizes, batchSize);
}

TEST(AttentionKernelsTest, SeparateCopyTest4) {
    int batchSize = 3;
    int tokenSizes[batchSize] = {100, 101, 102};
    testSelfAttention(128, 8, 2, tokenSizes, batchSize);
}

TEST(AttentionKernelsTest, SeparateCopyTest5) {
    int batchSize = 4;
    int tokenSizes[batchSize] = {100, 55, 111, 203};
    testSelfAttention(128, 8, 2, tokenSizes, batchSize);
}

TEST(AttentionKernelsTest, FusedCopyTest1) {
    int batchSize = 1;
    int tokenSizes[batchSize] = {100};
    testSelfAttention(128, 2, 2, tokenSizes, batchSize, false);
}

TEST(AttentionKernelsTest, FusedCopyTest2) {
    int batchSize = 2;
    int tokenSizes[batchSize] = {100, 101};
    testSelfAttention(128, 4, 4, tokenSizes, batchSize, false);
}

TEST(AttentionKernelsTest, FusedCopyTest3) {
    int batchSize = 4;
    int tokenSizes[batchSize] = {100, 101, 102, 103};
    testSelfAttention(128, 4, 4, tokenSizes, batchSize, false);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
