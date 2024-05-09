// Copyright (c) 2023-2024 Intel Corporation
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

template <typename WeiT, typename KVCacheT>
OptDecoder<WeiT, KVCacheT>::OptDecoder(const std::string &modelPath)
    : CommonDecoder<Attention<WeiT, QKPO_Dummy, LayerNorm>, MLP<WeiT>, KVCacheT>(modelPath, "gpt") {
    // Context
    DecoderContext *ctx = this->getContext();

    // Embedding
    embedding = new OptEmbedding<float16_t>(ctx);
    setEmbeddingWeights(modelPath);

    // Final LN
    setFinalLnWeight(modelPath);
}

template <typename WeiT, typename KVCacheT>
OptDecoder<WeiT, KVCacheT>::~OptDecoder() {
    delete embedding;
}

template <typename WeiT, typename KVCacheT>
void OptDecoder<WeiT, KVCacheT>::setEmbeddingWeights(const std::string &modelPath) {
    int vocabSize = embedding->getVocabSize();
    int embeddingSize = embedding->getEmbeddingSize();
    int maxPos = embedding->getMaxPositions();
    int hiddenSize = embedding->getHiddenSize();

    float *tokenEmb = (float *)malloc(vocabSize * embeddingSize * sizeof(float));
    float *posEmb = (float *)malloc(maxPos * hiddenSize * sizeof(float));

    loadWeight(modelPath + "/model.wte.bin", tokenEmb, vocabSize * embeddingSize);
    loadWeight(modelPath + "/model.wpe.bin", posEmb, maxPos * hiddenSize);

    embedding->setWeights(tokenEmb, posEmb);

    free(tokenEmb);
    free(posEmb);
}

template <typename WeiT, typename KVCacheT>
void OptDecoder<WeiT, KVCacheT>::setFinalLnWeight(const std::string &modelPath) {
    finalLN.setWeight(modelPath + "/model.final_layernorm.weight.bin", modelPath + "/model.final_layernorm.bias.bin",
            embedding->getHiddenSize());
}

template <typename WeiT, typename KVCacheT>
void OptDecoder<WeiT, KVCacheT>::prepareAttnMask(int *ids, int step) {
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

template <typename WeiT, typename KVCacheT>
void OptDecoder<WeiT, KVCacheT>::embeddingForward(int *ids, float *buf, int tokenSize) {
    int pastSeqLen = this->accSeqLen;
    if (pastSeqLen == 0 && this->prefixSharing) { pastSeqLen += this->prefixSeqLen; }

    // Prepare position data for positional embedding
    int batchSize = 1;
    int seqLen = tokenSize;
    int positions[batchSize * seqLen];
    for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < seqLen; ++i) {
            positions[b * seqLen + i] = i + pastSeqLen;
        }
    }

    // Embedding
    embedding->forward(ids, positions, buf, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void OptDecoder<WeiT, KVCacheT>::embeddingForward(float *output, const std::vector<SequenceMeta *> &sequences) {
    // Calculate the total number of input tokens
    int inputTokens = 0;
    for (int i = 0; i < sequences.size(); ++i) {
        inputTokens += sequences[i]->getInputSeqLen();
    }

    // Prepare position data for positional embedding
    int idBuf[256];
    int posBuf[256];

    int *ids = inputTokens <= 256 ? idBuf : (int *)malloc(inputTokens * sizeof(int));
    int *positions = inputTokens <= 256 ? posBuf : (int *)malloc(inputTokens * sizeof(int));

    int idx = 0;
    for (int i = 0; i < sequences.size(); ++i) {
        auto pastSeqLen = sequences[i]->getPastSeqLen();
        auto inputTokens = sequences[i]->getInputTokens();
        for (int j = 0; j < sequences[i]->getInputSeqLen(); ++j) {
            ids[idx] = inputTokens[pastSeqLen + j];
            positions[idx] = pastSeqLen + j;
            idx += 1;
        }
    }

    // Embedding
    embedding->forward(ids, positions, output, inputTokens);

    if (inputTokens > 256) {
        free(ids);
        free(positions);
    }
}

template <typename WeiT, typename KVCacheT>
void OptDecoder<WeiT, KVCacheT>::lastLayerNormForward(float *input, float *output, int rows) {
    finalLN.forward(input, output, rows);
}

IMPLEMENT_MODEL(OptDecoder, gpt)
