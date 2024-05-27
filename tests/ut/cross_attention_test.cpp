#include "attention_kernels.h"
#include "gemm_kernel_ext.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

// Define a fixture for the unit tests
class AttentionKernelsTest : public ::testing::Test {
protected:
    void SetUp() override { srand(time(NULL)); }

    void TearDown() override {
        // Clean up any resources allocated in SetUp()
    }
};

class Timer {
private:
    std::chrono::steady_clock::time_point startTime;
    std::string name;

public:
    Timer(const std::string &n) : name(n) { startTime = std::chrono::steady_clock::now(); }

    void stop() {
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << name << ": " << duration << " milliseconds" << std::endl;
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

// Reference impl. for one sample/batch
// query, key, value is [SeqLen, headNum, headSize]
// presentSeqLen = pastSeqLen + inputSeqLen
static void crossAttentionRef(bfloat16_t *output, const bfloat16_t *query, const float16_t *key, const float16_t *value,
        const float *attnMask, int inputSeqLen, int presentSeqLen, int qHeadNum, int kvHeadNum, int headSize,
        float scale) {
    assert(inputSeqLen == 1);
    assert(qHeadNum % kvHeadNum == 0);

    int hiddenSize = qHeadNum * headSize;
    int kvStride = kvHeadNum * headSize;
    float *scores = (float *)malloc(qHeadNum * inputSeqLen * presentSeqLen * sizeof(float));

    for (int qHdx = 0; qHdx < qHeadNum; ++qHdx) {
        int kvHdx = qHdx / (qHeadNum / kvHeadNum);
        auto pquery = query + qHdx * headSize;
        auto pkey = key + kvHdx * headSize;
        auto pvalue = value + kvHdx * headSize;
        auto pscore = scores + qHdx * inputSeqLen * presentSeqLen;

        // Q * Kᵀ
        for (int seq = 0; seq < presentSeqLen; ++seq) {
            float sum = 0;
            for (int i = 0; i < headSize; ++i) {
                sum += (float)pquery[i] * (float)pkey[i];
            }
            pscore[seq] = sum * scale;
            pkey += kvStride;
        }

        // Score = Softmax(Q * Kᵀ)
        softmaxRef(pscore, presentSeqLen);
#ifdef DEBUG
        printf("pscore: ");
        for (int i = 0; i < presentSeqLen; ++i) {
            printf("%.6f ", pscore[i]);
        }
        printf("\n");
#endif

        // Score * V
        gemmRef(pscore, pvalue, output + qHdx * headSize, 1, headSize, presentSeqLen, presentSeqLen, kvStride,
                hiddenSize);
    }

    free(scores);
}

static void testShardedHead(int inputSeqLen, int pastSeqLen, int qHeadNum, int kvHeadNum, int headSize, int threadNum) {
    // Define the input parameters for the function
    const int presentSeqLen = inputSeqLen + pastSeqLen;
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
            output, query, inputSeqLen, presentSeqLen, qHeadNum, headSize, oStride, qStride, batchSize, scale,
            threadNum,
            [&](int b, int qHead) {
                auto kvHead = qHead / (qHeadNum / kvHeadNum);
                return std::make_tuple(key + kvHead * headSize, kvStride, nullptr);
            },
            [&](int b, int qHead) {
                auto kvHead = qHead / (qHeadNum / kvHeadNum);
                return std::make_tuple(value + kvHead * headSize, kvStride, nullptr);
            },
            [&](int b, int qHead, int srcLen, int tgtLen) { return attnMask + b * srcLen * tgtLen; });

    for (int i = 0; i < batchSize; ++i) {
        auto pout = refOut + i * inputSeqLen * oStride;
        auto pquery = query + i * inputSeqLen * qStride;
        auto pkey = key + i * presentSeqLen * kvStride;
        auto pvalue = value + i * presentSeqLen * kvStride;
        crossAttentionRef(
                pout, pquery, pkey, pvalue, attnMask, inputSeqLen, presentSeqLen, qHeadNum, kvHeadNum, headSize, scale);
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

static void testCrossAttnByHead(int qHeadNum, int kvHeadNum, int headSize, int threadNum) {
    // Define the input parameters for the function
    const int hiddenSize = qHeadNum * headSize;
    const int oStride = qHeadNum * headSize;
    const int qkvStride = (qHeadNum + 2 * kvHeadNum) * headSize;
    const int batchSize = 3;

    int inputSeqLen[batchSize];
    int inputOffsets[batchSize];
    int pastSeqLens[batchSize];
    int totalSeqLen = 0;
    for (int i = 0; i < batchSize; i++) {
        inputSeqLen[i] = 1;
        pastSeqLens[i] = rand() % 100 + 20;
        inputOffsets[i] = totalSeqLen;
        totalSeqLen += inputSeqLen[i];
    }

    const float scale = 1 / sqrtf(headSize);

    // Set OpenMP threads
    omp_set_num_threads(threadNum);

    // Define the input and output arrays
    bfloat16_t *qkv = (bfloat16_t *)malloc(totalSeqLen * qkvStride * sizeof(bfloat16_t));
    bfloat16_t *output = (bfloat16_t *)malloc(totalSeqLen * hiddenSize * sizeof(bfloat16_t));
    bfloat16_t *refOut = (bfloat16_t *)malloc(totalSeqLen * hiddenSize * sizeof(bfloat16_t));
    float16_t *keyCaches[batchSize];
    float16_t *valueCaches[batchSize];

    // Initialize the input arrays
    for (int i = 0; i < totalSeqLen * qkvStride; i++) {
        qkv[i] = (bfloat16_t)((rand() % 1000 - 500) / 500.0f);
    }

// Initialize the key and value caches
#pragma omp parallel for
    for (int i = 0; i < batchSize; i++) {
        auto capacity = pastSeqLens[i] + inputSeqLen[i];
        keyCaches[i] = (float16_t *)malloc(capacity * kvHeadNum * headSize * sizeof(float16_t));
        valueCaches[i] = (float16_t *)malloc(capacity * kvHeadNum * headSize * sizeof(float16_t));
        for (int j = 0; j < capacity * kvHeadNum * headSize; j++) {
            keyCaches[i][j] = (float16_t)((rand() % 1000 - 500) / 500.0f);
            valueCaches[i][j] = (float16_t)((rand() % 1000 - 500) / 500.0f);
        }
    }

    // To call the function under test
    auto query = qkv;
    auto key = qkv + qHeadNum * headSize;
    auto value = qkv + (qHeadNum + kvHeadNum) * headSize;
    auto qStride = qkvStride;
    auto kvStride = qkvStride;

    Timer timer("xft::crossAttnByHead");
    xft::crossAttnByHead<bfloat16_t, float16_t>(
            output, query, key, value, qHeadNum, kvHeadNum, headSize, oStride, qStride, kvStride, batchSize,
            inputSeqLen, pastSeqLens, true, scale, nullptr, threadNum,
            [&](int b, int kvHdx) {
                return std::make_tuple(keyCaches[b] + kvHdx * headSize, kvHeadNum * headSize, nullptr);
            },
            [&](int b, int kvHdx) {
                return std::make_tuple(valueCaches[b] + kvHdx * headSize, kvHeadNum * headSize, nullptr);
            });
    timer.stop();

#pragma omp parallel for num_threads(1)
    for (int i = 0; i < batchSize; ++i) {
        auto pout = refOut + inputOffsets[i] * oStride;
        auto pquery = query + inputOffsets[i] * qkvStride;
        auto pkey = keyCaches[i];
        auto pvalue = valueCaches[i];
        crossAttentionRef(pout, pquery, pkey, pvalue, nullptr, inputSeqLen[i], pastSeqLens[i] + inputSeqLen[i],
                qHeadNum, kvHeadNum, headSize, scale);
    }

    // Compare values
    for (int i = 0; i < totalSeqLen * hiddenSize; i++) {
        ASSERT_NEAR((float)output[i], (float)refOut[i], 3 * 1e-3);
    }

    // Clean up
    free(qkv);
    free(output);
    free(refOut);
    for (int i = 0; i < batchSize; i++) {
        free(keyCaches[i]);
        free(valueCaches[i]);
    }
}

// Define a test case
TEST_F(AttentionKernelsTest, CrossAttnShardedHeadTest) {
    // testShardedHead(inputSeqLen, pastSeqLen, qHeadNum, kvHeadNum, headSize, threadNum)
    testShardedHead(1, 2, 1, 1, 16, 2);
    testShardedHead(1, 3, 8, 8, 128, 16);
    testShardedHead(1, 4, 8, 8, 128, 16);
    testShardedHead(1, 71, 8, 8, 128, 32);
    testShardedHead(1, 999, 8, 2, 128, 24);
    testShardedHead(1, 111, 12, 2, 128, 36);
    testShardedHead(1, 1000, 32, 8, 128, 64);
}

TEST_F(AttentionKernelsTest, CrossAttnByHeadTest) {
    // testCrossAttnByHead(qHeadNum, kvHeadNum, headSize, threadNum)
    testCrossAttnByHead(2, 1, 128, 2);
    testCrossAttnByHead(32, 8, 128, 16);
    testCrossAttnByHead(32, 32, 128, 16);

    testCrossAttnByHead(2, 1, 256, 2);
    testCrossAttnByHead(32, 8, 256, 32);
    testCrossAttnByHead(32, 32, 256, 32);
}

// Run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
