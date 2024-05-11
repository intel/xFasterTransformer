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

#include "allocator.h"
#include <filesystem>

#include "INIReader.h"
#include "my_types.h"
#include "simple_mem_pool.h"
#include "split_util.h"

namespace fs = std::filesystem;

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

    // Runtime configuration
    // # of mini-batch
    int batchSize;
    // # of tokens
    int inputSeqLen;
    // For custom usage
    int reserved1;

#ifdef PIPELINE_PARALLEL
    int sequenceID;
#endif

    // Model structure configuration
    int vocabSize;
    int embeddingSize;
    int maxPositions;
    int maxPosEmbed;
    int maxSeqLength; // From Qwen model's seq_length
    bool useLogN; // From Qwen model
    bool useNTK; // From Qwen model
    int layers;
    int hiddenSize;
    int intermediateSize;
    int attHeadNum;
    int kvHeadNum;
    int attHeadSize;
    float attFactor;
    float epsilon;

    // rope scaling parameters
    RopeParams *ropeParamsPtr;

    // Which split this context is for
    int splitIdx;
    // # of splits (the same as NUMA node number in the system)
    int numSplit;

    // For pipeline parallel and tensor parallel config
    int ppSize = 1; // pipeline parallel stage size
    int ppRank = 0; // pipeline parallel stage rank
    int tpSize = 1; // tensor parallel size
    int tpRank = 0; // tensor parallel rank

    enum ActivationType { RELU, GELU, SWIGLU, SILU };
    ActivationType actType;

    // # of thread
    int numThreads;

    // Please look into the comments in resize function to see how buffers are arranged
    xft::Matrix<float> normBuf; // buf for the first layer norm
    xft::Matrix<float> tmpBuf; // tmp buffer, same size as output
    xft::Matrix<float> qkvMatMul; // query, key, value
    xft::Matrix<float> imOut; // intermediate output

    MMHelper *mmHelper;

    std::string configPath;
    INIReader configReader;
    std::string sectionName;

private:
    float *rawBuffer;
    uint64_t rawBufSize; // how many floats

    // Detail buffer capacity
    uint64_t size1;
    uint64_t size2;
    uint64_t size3;

public:
    DecoderContext(int _layers, int _hiddenSize, int _headSize, int _attHeadNum, int _kvHeadNum, int _imSize, const std::string &act,
            float epsilon, int _vocabSize, int _embeddingSize, int _maxPositions, int _maxPosEmbed, int _maxSeqLength,
            int _splitIdx, int _splits, int _ppSize = 1, int _ppRank = 0, RopeParams *_ropeParamsPtr = nullptr,
            bool _useLogN = true, bool _useNTK = true, int numThreads = 0)
        : layers(_layers)
        , hiddenSize(_hiddenSize)
        , attHeadSize(_headSize)
        , intermediateSize(_imSize)
        , attHeadNum(_attHeadNum)
        , kvHeadNum(_kvHeadNum)
        , vocabSize(_vocabSize)
        , embeddingSize(_embeddingSize)
        , maxPositions(_maxPositions)
        , maxPosEmbed(_maxPosEmbed)
        , maxSeqLength(_maxSeqLength)
        , useLogN(_useLogN)
        , useNTK(_useNTK)
        , ropeParamsPtr(_ropeParamsPtr)
        , splitIdx(_splitIdx)
        , numSplit(_splits)
        , ppSize(_ppSize)
        , ppRank(_ppRank)
        , tpSize(_splits)
        , tpRank(_splitIdx)
        , epsilon(epsilon) {
        if (attHeadNum != 0) {
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
        this->rawBuffer = (float *)xft::alloc(sizeof(float) * rawBufSize);
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

    DecoderContext(std::string _configPath, std::string _sectionName = "") {
        this->ResetConfigReader(_configPath, _sectionName);
    }

    void ResetConfigReader(std::string _configPath, std::string _sectionName = "") {
        this->configPath = _configPath;
        this->configReader = INIReader(_configPath);
        if (this->configReader.ParseError() < 0) {
            printf("Config File %s Can't be loaded!", configPath.c_str());
            exit(-1);
        }

        if (_sectionName == "") {
            if (!this->configReader.Sections().empty()) {
                this->sectionName = *(this->configReader.Sections().begin());
            } else {
                printf("Config File %s Can't be loaded!", configPath.c_str());
                exit(-1);
            }
        }
    }

    template <typename T>
    void GetAttr(const std::string &attrName, T *attrValue) {
        if constexpr (std::is_integral<T>::value) { // int
            *attrValue = this->configReader.GetInteger(this->sectionName, attrName);
        } else if constexpr (std::is_floating_point<T>::value) { // float
            *attrValue = this->configReader.GetFloat(this->sectionName, attrName);
        } else if constexpr (std::is_same<T, bool>::value) { // bool
            printf("the func GetBoolean(%s) should have default value!", attrName);
            exit(-1);
        } else { // string
            *attrValue = this->configReader.Get(this->sectionName, attrName);
        }
    }

    template <typename T>
    void GetAttr(const std::string &attrName, T *attrValue, const T defValue) {
        if constexpr (std::is_integral<T>::value) { // int
            *attrValue = this->configReader.GetInteger(this->sectionName, attrName, defValue);
        } else if constexpr (std::is_floating_point<T>::value) { // float
            *attrValue = this->configReader.GetFloat(this->sectionName, attrName, defValue);
        } else if constexpr (std::is_same<T, bool>::value) { // bool
            *attrValue = this->configReader.GetBoolean(this->sectionName, attrName, defValue);
        } else { // string
            *attrValue = this->configReader.Get(this->sectionName, attrName, defValue);
        }
    }

    bool cached(const std::string &name) { return SimpleMemPool::instance().cached(name); }

    template <typename T>
    T *getBuffer(const std::string &name, size_t size, size_t alignment = 64) {
        return (T *)SimpleMemPool::instance().getBuffer(name, sizeof(T) * size, alignment);
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
    // | normBuf |qkvMatMul|        |
    // |         |  imOut  | tmpBuf |
    void resize(int totalInSeqLen) {
        // Check total required size
        int responsibleHead
                = splitIdx < (attHeadNum % numSplit) ? (attHeadNum / numSplit + 1) : (attHeadNum / numSplit);
        int qCols = responsibleHead * attHeadSize;
        int kCols = qCols / (attHeadNum / kvHeadNum);
        int vCols = kCols;
        int qkvCols = qCols + kCols + vCols;
        int mlpFactor = (this->actType == GELU || this->actType == SILU || this->actType == SWIGLU) ? 2 : 1;
        auto range = SplitUtil::getTaskRange(intermediateSize, numSplit, splitIdx);
        int imCols = range.second - range.first;

        uint64_t normSize = (uint64_t)totalInSeqLen * hiddenSize;
        uint64_t qkvSize = (uint64_t)totalInSeqLen * qkvCols;
        uint64_t imOutSize = (uint64_t)totalInSeqLen * imCols * mlpFactor;
        uint64_t tmpBufSize = (uint64_t)totalInSeqLen * hiddenSize;

        size1 = normSize;
        size2 = qkvSize < imOutSize ? imOutSize : qkvSize;
        size3 = tmpBufSize;

        uint64_t total = size1 + size2 + size3;
        if (total > this->rawBufSize) {
            this->rawBufSize = total;
            free(this->rawBuffer);

            this->rawBuffer = (float *)xft::alloc(sizeof(float) * rawBufSize);
            memset(this->rawBuffer, 0, sizeof(float) * rawBufSize);
        }

        // Assign the buffer
        normBuf.Assign(this->rawBuffer, totalInSeqLen, hiddenSize, hiddenSize);
        tmpBuf.Assign(this->rawBuffer + size1 + size2, totalInSeqLen, hiddenSize, hiddenSize);
        imOut.Assign(this->rawBuffer + size1, totalInSeqLen, imCols, imCols);
        qkvMatMul.Assign(this->rawBuffer + size1, totalInSeqLen, qkvCols, qkvCols);
    }

    // TODO: deprecate it
    void resize(int batchSize, int inputSeqLen, bool preSeqLen) {
        this->batchSize = batchSize;
        this->inputSeqLen = inputSeqLen;

        this->resize(inputSeqLen * batchSize);
    }

    uint64_t getScoreCapacity() {
        // Return real size instead of size3
        return rawBufSize - size1 - size2;
    }

    ~DecoderContext() { free(this->rawBuffer); }
};