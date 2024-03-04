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
#pragma once
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "my_types.h"
#include "split_util.h"

struct RopeParams {
    float base;
    std::string type;
    float scale;
    int orgMaxPosEmbed;
    float extraPolFactor;
    float attnFactor;
    float betaFast;
    float betaSlow;

public:
    RopeParams(float theta = 10000.0, std::string vtype = "", float vscale = 1.0, int vorgMaxPosEmbed = 2048,
            float vextraPolFactor = 1, float vattnFactor = 1, float vbetaFast = 32, float vbetaSlow = 1)
        : base(theta)
        , type(vtype)
        , scale(vscale)
        , orgMaxPosEmbed(vorgMaxPosEmbed)
        , extraPolFactor(vextraPolFactor)
        , attnFactor(vattnFactor)
        , betaFast(vbetaFast)
        , betaSlow(vbetaSlow) {}
};

class MMHelper;

struct DecoderContext {
    // # of mini-batch
    int batchSize;
    // # of tokens
    int inputSeqLen;
    // For custom usage
    int reserved1;

    // Other configuration
    int vocabSize;
    int embeddingSize;
    int maxPositions;
    int maxPosEmbed;
    int maxSeqLength; // From Qwen model's seq_length
    int layers;

    // For BERT-base, hidden_size=768
    const int hiddenSize;
    // For BERT-base, intermediate_size=3072
    const int intermediateSize;
    // For BERT-base, attHeadNum=12
    const int attHeadNum;
    const int kvHeadNum;
    // attHeadSize = hiddenSize / attHeadNum
    int attHeadSize;
    // attFactor = 1 / sqrtf(attHeadSize)
    float attFactor;

    // norm epsilon
    float epsilon;

    // rope scaling parameters
    RopeParams *ropeParamsPtr;

    // Which split this context is for
    const int splitIdx;
    // # of splits (the same as NUMA node number in the system)
    const int numSplit;

    // For pipeline parallel and tensor parallel config
    int ppSize = 1; // pipeline parallel stage size
    int ppRank = 0; // pipeline parallel stage rank
    int tpSize = 1; // tensor parallel size
    int tpRank = 0; // tensor parallel rank

    enum ActivationType { RELU, GELU, SWIGLU, SILU };
    ActivationType actType;

    // # of thread
    int numThreads;

    float *qkScores; // attention score

    // Please look into the comments in resize function to see how buffers are arranged
    hpj::Matrix<float> normBuf; // buf for the first layer norm
    hpj::Matrix<float> tmpBuf; // tmp buffer, same size as output
    hpj::Matrix<float> qkvMatMul; // query, key, value
    hpj::Matrix<float> imOut; // intermediate output

    MMHelper *mmHelper;

private:
    float *rawBuffer;
    uint64_t rawBufSize; // how many floats

    // Detail buffer capacity
    uint64_t size1;
    uint64_t size2;
    uint64_t size3;

public:
    DecoderContext(int _layers, int _hiddenSize, int _attHeadNum, int _kvHeadNum, int _imSize, const std::string &act,
            float epsilon, int _vocabSize, int _embeddingSize, int _maxPositions, int _maxPosEmbed, int _maxSeqLength,
            int _splitIdx, int _splits, int _ppSize = 1, int _ppRank = 0, RopeParams *_ropeParamsPtr = nullptr,
            int numThreads = 0)
        : layers(_layers)
        , hiddenSize(_hiddenSize)
        , intermediateSize(_imSize)
        , attHeadNum(_attHeadNum)
        , kvHeadNum(_kvHeadNum)
        , vocabSize(_vocabSize)
        , embeddingSize(_embeddingSize)
        , maxPositions(_maxPositions)
        , maxPosEmbed(_maxPosEmbed)
        , maxSeqLength(_maxSeqLength)
        , ropeParamsPtr(_ropeParamsPtr)
        , splitIdx(_splitIdx)
        , numSplit(_splits)
        , ppSize(_ppSize)
        , ppRank(_ppRank)
        , tpSize(_splits)
        , tpRank(_splitIdx)
        , epsilon(epsilon) {
        if (attHeadNum != 0) {
            this->attHeadSize = hiddenSize / attHeadNum;
            this->attFactor = 1 / sqrtf(attHeadSize);
        }

        // Set the default value (don't worry, it can be changed later)
        this->batchSize = 1;
        this->inputSeqLen = 1;
        this->numThreads = numThreads;

        if (numThreads == 0) {
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                if (tid == 0) { this->numThreads = omp_get_num_threads(); }
            }
        }

        this->rawBufSize = 4 * 32 * intermediateSize + 4 * attHeadNum * 32 * 32; // assume bs=4, seq=32
        this->rawBuffer = (float *)aligned_alloc(64, sizeof(float) * rawBufSize);
        memset(this->rawBuffer, 0, sizeof(float) * rawBufSize);

        if (act == "relu") {
            this->actType = RELU;
        } else if (act == "gelu") {
            this->actType = GELU;
        } else if (act == "silu") {
            this->actType = SILU;
        } else if (act == "swiglu") {
            this->actType = SWIGLU;
        } else {
            printf("unsupported activation: %s\n", act.c_str());
            exit(-1);
        }
    }

    void dump() {
        printf("batch_size=%d\n", batchSize);
        printf("inputSeqLen=%d\n", inputSeqLen);

        printf("hiddenSize=%d\n", hiddenSize);
        printf("intermediateSize=%d\n", intermediateSize);
        printf("attHeadNum=%d\n", attHeadNum);
        printf("kvHeadNum=%d\n", kvHeadNum);
        printf("attHeadSize=%d\n", attHeadSize);
        printf("attFactor=%f\n", attFactor);

        printf("numThreads=%d\n", numThreads);
    }

    // Resize to make sure the buffer is big enough
    // |---------|---------|--------|
    // | normBuf |qkvMatMul|qkScores|
    // |         |  imOut  | tmpBuf |
    void resize(int batchSize, int inputSeqLen, bool preSeqLen) {
        this->batchSize = batchSize;
        this->inputSeqLen = inputSeqLen;

        // Check total required size
        const int pad = 0; // 4;
        int hiddenStride = (hiddenSize % 512 == 0 ? hiddenSize + pad
                                                  : hiddenSize); // stride for matrix with columns of hiddenSize
        int responsibleHead
                = splitIdx < (attHeadNum % numSplit) ? (attHeadNum / numSplit + 1) : (attHeadNum / numSplit);
        int qCols = responsibleHead * attHeadSize;
        int kCols = qCols / (attHeadNum / kvHeadNum);
        int vCols = kCols;
        int qkvCols = qCols + kCols + vCols;
        int qkvStride = (qkvCols % 512 == 0 ? qkvCols + pad : qkvCols); // stride for the concated QKV
        int mlpFactor = (this->actType == SILU || this->actType == SWIGLU) ? 2 : 1;
        auto range = SplitUtil::getTaskRange(intermediateSize, numSplit, splitIdx);
        int imCols = range.second - range.first;
        int imStride = (imCols % 512 == 0 ? imCols + pad : imCols); // stride for intermediate output

        uint64_t normSize = (uint64_t)batchSize * inputSeqLen * hiddenStride;
        uint64_t qkvSize = (uint64_t)batchSize * inputSeqLen * qkvStride;
        uint64_t imOutSize = (uint64_t)batchSize * inputSeqLen * imStride * mlpFactor;

        int presentSeqLen = preSeqLen + 1;
        int paddedSize = (presentSeqLen + 15) / 16 * 16;

        // Note: the score buffer for first token generation is not padded
        uint64_t scoreBufSize = preSeqLen > 0 ? (uint64_t)batchSize * responsibleHead * inputSeqLen * paddedSize
                                              : (uint64_t)batchSize * responsibleHead * inputSeqLen * inputSeqLen;
        uint64_t tmpBufSize = (uint64_t)batchSize * inputSeqLen * hiddenStride;

        size1 = normSize;
        size2 = qkvSize < imOutSize ? imOutSize : qkvSize;
        size3 = tmpBufSize < scoreBufSize ? scoreBufSize : tmpBufSize;

        uint64_t total = size1 + size2 + size3;
        if (total > this->rawBufSize) {
            this->rawBufSize = total;
            free(this->rawBuffer);

            this->rawBuffer = (float *)aligned_alloc(64, sizeof(float) * rawBufSize);
            memset(this->rawBuffer, 0, sizeof(float) * rawBufSize);
        }

        // Assign the buffer
        this->qkScores = this->rawBuffer + size1 + size2;
        normBuf.Assign(this->rawBuffer, batchSize * inputSeqLen, hiddenSize, hiddenStride);
        tmpBuf.Assign(this->qkScores, batchSize * inputSeqLen, hiddenSize, hiddenStride);
        imOut.Assign(this->rawBuffer + size1, batchSize * inputSeqLen, imCols, imStride);
        qkvMatMul.Assign(this->rawBuffer + size1, batchSize * inputSeqLen, qkvCols, qkvStride);
    }

    uint64_t getScoreCapacity() {
        return size3;
    }

    ~DecoderContext() { free(this->rawBuffer); }
};
