// Copyright (c) 2024-2025 Intel Corporation
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
#include "deepseek.h"

template <typename WeiT, typename KVCacheT>
DeepSeekLLM<WeiT, KVCacheT>::DeepSeekLLM(const std::string &modelPath)
    : CommonDecoder<DeepSeekAttention<WeiT, DeekSeekV2RotaryEmbedding, typename TypeSelector<WeiT>::InType, typename TypeSelector<WeiT>::ImType,
                            typename TypeSelector<WeiT>::OutType>,
              DeepSeekMoE<WeiT, typename TypeSelector<WeiT>::InType, typename TypeSelector<WeiT>::ImType,
                      typename TypeSelector<WeiT>::OutType>,
              KVCacheT>(modelPath, "deepseek_v2") {
    // Context
    DecoderContext *ctx = this->getContext();

    // Embedding
    embedding = new TokenEmbedding<float16_t>(ctx);
    setEmbeddingWeights(modelPath);

    // Final LN
    setFinalLnWeight(modelPath);
}

template <typename WeiT, typename KVCacheT>
DeepSeekLLM<WeiT, KVCacheT>::~DeepSeekLLM() {
    delete embedding;
}

template <typename WeiT, typename KVCacheT>
void DeepSeekLLM<WeiT, KVCacheT>::setEmbeddingWeights(const std::string &modelPath) {
    embedding->setWeights(modelPath + "/model.wte.bin");
}

template <typename WeiT, typename KVCacheT>
void DeepSeekLLM<WeiT, KVCacheT>::setFinalLnWeight(const std::string &modelPath) {
    finalLN.setWeight(modelPath + "/model.final_layernorm.weight.bin", "", embedding->getHiddenSize());
}

template <typename WeiT, typename KVCacheT>
void DeepSeekLLM<WeiT, KVCacheT>::prepareAttnMask(int *ids, int step) {
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
        memset(mask, 0, ctx->batchSize * this->accSeqLen * sizeof(float)); // all elements are 0
    }
}

template <typename WeiT, typename KVCacheT>
void DeepSeekLLM<WeiT, KVCacheT>::embeddingForward(int *ids, float *output, int tokenSize) {
    embedding->forward(ids, output, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void DeepSeekLLM<WeiT, KVCacheT>::embeddingForward(int *ids, bfloat16_t *output, int tokenSize) {
    embedding->forward(ids, output, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void DeepSeekLLM<WeiT, KVCacheT>::embeddingForward(int *ids, float16_t *output, int tokenSize) {
    embedding->forward(ids, output, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void DeepSeekLLM<WeiT, KVCacheT>::lastLayerNormForward(float *input, float *output, int rows) {
    finalLN.forward(input, output, rows);
}

template <typename WeiT, typename KVCacheT>
void DeepSeekLLM<WeiT, KVCacheT>::lastLayerNormForward(bfloat16_t *input, bfloat16_t *output, int rows) {
    finalLN.forward(input, output, rows);
}

template <typename WeiT, typename KVCacheT>
void DeepSeekLLM<WeiT, KVCacheT>::lastLayerNormForward(float16_t *input, float16_t *output, int rows) {
    finalLN.forward(input, output, rows);
}

IMPLEMENT_DS_MODEL(DeepSeekLLM, deepseek_v2)
