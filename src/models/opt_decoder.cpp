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
#include "opt_decoder.h"

#include <immintrin.h>

#include <cstdio>
#include <iostream>

#include "INIReader.h"
#include "compile_util.h"
#include "opt_decoder.h"
#include "transpose_util.h"

template <typename WeiT>
OptDecoder<WeiT>::OptDecoder(const std::string &modelPath)
    : CommonDecoder<Attention<WeiT, QKPO_Dummy, LayerNorm>, MLP<WeiT>>(modelPath, "gpt") {
    // Context
    DecoderContext *ctx = this->getContext();

    // Embedding
    embedding = new OptEmbedding<float16_t>(ctx);
    setEmbeddingWeights(modelPath);

    // Final LN
    setFinalLnWeight(modelPath);
}

template <typename WeiT>
OptDecoder<WeiT>::~OptDecoder() {
    delete embedding;
}

template <typename WeiT>
void OptDecoder<WeiT>::setEmbeddingWeights(const std::string &modelPath) {
    int vocabSize = embedding->getVocabSize();
    int embeddingSize = embedding->getEmbeddingSize();
    int maxPos = embedding->getMaxPositions();
    int hiddenSize = embedding->getHiddenSize();

    float *tokenEmb = (float *)malloc(vocabSize * embeddingSize * sizeof(float));
    float *posEmb = (float *)malloc(maxPos * hiddenSize * sizeof(float));

    loadWeight(modelPath + "/model.wte.bin", tokenEmb, vocabSize * embeddingSize, this->getDataType());
    loadWeight(modelPath + "/model.wpe.bin", posEmb, maxPos * hiddenSize, this->getDataType());

    embedding->setWeights(tokenEmb, posEmb);

    free(tokenEmb);
    free(posEmb);
}

template <typename WeiT>
void OptDecoder<WeiT>::setFinalLnWeight(const std::string &modelPath) {
    int hiddenSize = embedding->getHiddenSize();

    float *gamma = (float *)malloc(hiddenSize * sizeof(float));
    float *beta = (float *)malloc(hiddenSize * sizeof(float));

    loadWeight(modelPath + "/model.final_layernorm.weight.bin", gamma, hiddenSize, this->getDataType());
    loadWeight(modelPath + "/model.final_layernorm.bias.bin", beta, hiddenSize, this->getDataType());

    finalLN.setWeight(gamma, beta, hiddenSize);
    free(gamma);
    free(beta);
}

template <typename WeiT>
void OptDecoder<WeiT>::prepareAttnMask(int *ids, int step) {
    DecoderContext *ctx = this->getContext();
    int seqLen = ctx->inputSeqLen;

    if (step == 0) {
        int sizeRequired = ctx->batchSize * seqLen * seqLen;
        float *mask = this->getAttnMask(sizeRequired);
        for (int b = 0; b < ctx->batchSize; ++b) {
            auto pmask = mask + b * seqLen * seqLen;
            for (int i = 0; i < seqLen; ++i) {
                memset(pmask + i * seqLen, 0, (i + 1) * sizeof(float)); // bottom left are 0
                std::fill_n(pmask + i * seqLen + i + 1, seqLen - i - 1, std::numeric_limits<float>::lowest());
            }
        }
    } else if (seqLen > 1) {
        int sizeRequired = ctx->batchSize * this->accSeqLen * seqLen;
        float *mask = this->getAttnMask(sizeRequired);
        for (int b = 0; b < ctx->batchSize; ++b) {
            auto pmask = mask + b * this->accSeqLen * seqLen;
            int pastLen = this->accSeqLen - seqLen;
            for (int i = 0; i < seqLen; ++i) {
                memset(pmask + i * this->accSeqLen, 0, (pastLen + i + 1) * sizeof(float));
                std::fill_n(pmask + i * this->accSeqLen + pastLen + i + 1, seqLen - i - 1,
                        std::numeric_limits<float>::lowest());
            }
        }
    } else {
        int sizeRequired = ctx->batchSize * this->accSeqLen;
        float *mask = this->getAttnMask(sizeRequired);
        memset(mask, 0, sizeRequired * sizeof(float)); // all elements are 0
    }
}

template <typename WeiT>
void OptDecoder<WeiT>::embeddingForward(int *ids, float *buf, int batchSize, int seqLen) {
    int pastSeqLen = this->accSeqLen;
    if (pastSeqLen == 0 && this->prefixSharing) { pastSeqLen += this->prefixSeqLen; }
    // Prepare position data for positional embedding
    int positions[batchSize * seqLen];
    for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < seqLen; ++i) {
            positions[b * seqLen + i] = i + pastSeqLen;
        }
    }

    // Embedding
    embedding->forward(ids, positions, buf, batchSize, seqLen);
}

template <typename WeiT>
void OptDecoder<WeiT>::lastLayerNormForward(float *input, float *output, int rows) {
    finalLN.forward(input, output, rows);
}

template class OptDecoder<float>;
template class OptDecoder<float16_t>;
template class OptDecoder<bfloat16_t>;
template class OptDecoder<int8_t>;
template class OptDecoder<w8a8_t>;
template class OptDecoder<uint4x2_t>;
template class OptDecoder<nf4x2_t>;
