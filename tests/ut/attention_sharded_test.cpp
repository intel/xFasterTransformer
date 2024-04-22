#include "attention_kernels.h"
#include "gemm_kernel_ext.h"
#include "gtest/gtest.h"

#include <cstdlib>

// Define a fixture for the unit tests
class AttentionKernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any necessary data or resources for the tests
    }

    void TearDown() override {
        // Clean up any resources allocated in SetUp()
    }
};

template <typename T>
static void softmaxRef(T *input, int size) {
    float max = (float)input[0];
    for (int i = 1; i < size; ++i) {
        if ((float)input[i] > max) { max = (float)input[i]; }
    }

    float sum = 0;
    for (int i = 0; i < size; ++i) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }

    float recip = 1.0f / sum;
    for (int i = 0; i < size; ++i) {
        input[i] *= recip;
    }
}

template <typename T1, typename T2, typename T3>
static void gemmRef(T1 *A, T2 *B, T3 *C, int M, int N, int K, int lda, int ldb, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += (float)A[m * lda + k] * (float)B[k * ldb + n];
            }
            C[m * ldc + n] = sum;
        }
    }
}

// Reference impl.
// Define the missing function crossAttentionRef
static void crossAttentionRef(bfloat16_t *output, const bfloat16_t *query, const float16_t *key, const float16_t *value,
        const float *attnMask, int inputSeqLen, int presentSeqLen, int qHeadNum, int headSize, float scale) {
    assert(inputSeqLen == 1);

    int qkvStride = qHeadNum * headSize;
    float *scores = (float *)malloc(qHeadNum * inputSeqLen * presentSeqLen * sizeof(float));

    for (int h = 0; h < qHeadNum; ++h) {
        auto pquery = query + h * headSize;
        auto pkey = key + h * headSize;
        auto pvalue = value + h * headSize;
        auto pscore = scores + h * inputSeqLen * presentSeqLen;

        // Q * Kᵀ
        for (int seq = 0; seq < presentSeqLen; ++seq) {
            float sum = 0;
            for (int i = 0; i < headSize; ++i) {
                sum += (float)pquery[i] * (float)pkey[i];
            }
            pscore[seq] = sum * scale;
            pkey += qkvStride;
        }

        // Score = Softmax(Q * Kᵀ)
        softmaxRef(pscore, presentSeqLen);

        // Score * V
        gemmRef(pscore, pvalue, output + h * headSize, 1, headSize, presentSeqLen, presentSeqLen, qkvStride, qkvStride);
    }
}

static void test(int inputSeqLen, int pastSeqLen, int headNum, int threadNum, int headSize = 128) {
    // Define the input parameters for the function
    const int presentSeqLen = inputSeqLen + pastSeqLen;
    const int qHeadNum = headNum;
    const int kvHeadNum = headNum;
    const int oStride = qHeadNum * headSize;
    const int qStride = qHeadNum * headSize;
    const int kvStride = kvHeadNum * headSize;
    const int batchSize = 1;
    const float scale = 1 / sqrtf(headSize);

    // Set OpenMP threads
    omp_set_num_threads(threadNum);

    // Define the input and output arrays
    bfloat16_t *query = (bfloat16_t *)malloc(batchSize * inputSeqLen * qHeadNum * headSize * sizeof(bfloat16_t));
    bfloat16_t *output = (bfloat16_t *)malloc(batchSize * inputSeqLen * qHeadNum * headSize * sizeof(bfloat16_t));
    bfloat16_t *refOut = (bfloat16_t *)malloc(batchSize * inputSeqLen * qHeadNum * headSize * sizeof(bfloat16_t));
    float16_t *key = (float16_t *)malloc(batchSize * presentSeqLen * kvHeadNum * headSize * sizeof(float16_t));
    float16_t *value = (float16_t *)malloc(batchSize * presentSeqLen * kvHeadNum * headSize * sizeof(float16_t));
    float *attnMask = (float *)malloc(batchSize * inputSeqLen * presentSeqLen * sizeof(float));

    // Initialize the input arrays
    for (int i = 0; i < batchSize * inputSeqLen * qHeadNum * headSize; i++) {
        query[i] = (bfloat16_t)((rand() % 1000 - 500) / 500.0f);
    }
    for (int i = 0; i < batchSize * presentSeqLen * kvHeadNum * headSize; i++) {
        key[i] = (float16_t)((rand() % 1000 - 500) / 500.0f);
        value[i] = (float16_t)((rand() % 1000 - 500) / 500.0f);
    }

    memset(attnMask, 0, batchSize * inputSeqLen * presentSeqLen * sizeof(float));

    // Call the function under test
    xft::crossAttnShardedHead<bfloat16_t, float16_t>(
            output, query, attnMask, inputSeqLen, presentSeqLen, qHeadNum, headSize, oStride, qStride, batchSize, scale,
            threadNum, [&](int b, int qHead) { return std::make_tuple(key + qHead * headSize, kvStride, nullptr); },
            [&](int b, int qHead) { return std::make_tuple(value + qHead * headSize, kvStride, nullptr); });

    for (int i = 0; i < batchSize; ++i) {
        auto pout = refOut + i * inputSeqLen * oStride;
        auto pquery = query + i * inputSeqLen * qStride;
        auto pkey = key + i * presentSeqLen * kvStride;
        auto pvalue = value + i * presentSeqLen * kvStride;
        crossAttentionRef(pout, pquery, pkey, pvalue, attnMask, inputSeqLen, presentSeqLen, qHeadNum, headSize, scale);
    }

    // Compare values
    for (int i = 0; i < batchSize * inputSeqLen * qHeadNum * headSize; i++) {
        ASSERT_NEAR((float)output[i], (float)refOut[i], 1e-3);
    }

    // Clean up
    free(query);
    free(output);
    free(refOut);
    free(key);
    free(value);
    free(attnMask);
}

// Define a test case
TEST_F(AttentionKernelsTest, CrossAttnShardedHeadTest1) {
    test(1, 2, 1, 2, 16);
    test(1, 3, 8, 32, 16);
    test(1, 4, 8, 32, 16);
    test(1, 71, 4, 8);
}

TEST_F(AttentionKernelsTest, CrossAttnShardedHeadTest2) {
    test(1, 38, 4, 12);
    test(1, 100, 12, 36);
    test(1, 1024, 12, 36);
}

// Run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}