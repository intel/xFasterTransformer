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
#include <limits>

#include "yarn_llama.h"

template <typename WeiT, typename KVCacheT>
YaRNLlama<WeiT, KVCacheT>::YaRNLlama(const std::string &modelPath)
    : CommonDecoder<
            RopeScalingAttention<WeiT, LlamaYaRNScaledRotaryEmbedding, RmsNorm, typename TypeSelector<WeiT>::InType,
                    typename TypeSelector<WeiT>::ImType, typename TypeSelector<WeiT>::OutType, true>,
            LlamaMLP<WeiT, typename TypeSelector<WeiT>::InType, typename TypeSelector<WeiT>::ImType,
                    typename TypeSelector<WeiT>::OutType>,
            KVCacheT>(modelPath, "yarn_llama") {
    // Context
    DecoderContext *ctx = this->getContext();

    // Embedding
    embedding = new TokenEmbedding<float16_t>(ctx);
    setEmbeddingWeights(modelPath);

    // Final LN
    setFinalLnWeight(modelPath);
}

template <typename WeiT, typename KVCacheT>
YaRNLlama<WeiT, KVCacheT>::~YaRNLlama() {
    delete embedding;
}

template <typename WeiT, typename KVCacheT>
void YaRNLlama<WeiT, KVCacheT>::setEmbeddingWeights(const std::string &modelPath) {
    embedding->setWeights(modelPath + "/model.wte.bin");
}

template <typename WeiT, typename KVCacheT>
void YaRNLlama<WeiT, KVCacheT>::setFinalLnWeight(const std::string &modelPath) {
    finalLN.setWeight(modelPath + "/model.final_layernorm.weight.bin", "", embedding->getHiddenSize());
}

// Prepare attention_mask which is like:
// def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
//     # create causal mask
//     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
template <typename WeiT, typename KVCacheT>
void YaRNLlama<WeiT, KVCacheT>::prepareAttnMask(int *ids, int step) {
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
void YaRNLlama<WeiT, KVCacheT>::embeddingForward(int *ids, float *output, int tokenSize) {
    embedding->forward(ids, output, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void YaRNLlama<WeiT, KVCacheT>::embeddingForward(int *ids, bfloat16_t *output, int tokenSize) {
    embedding->forward(ids, output, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void YaRNLlama<WeiT, KVCacheT>::embeddingForward(int *ids, float16_t *output, int tokenSize) {
    embedding->forward(ids, output, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void YaRNLlama<WeiT, KVCacheT>::lastLayerNormForward(float *input, float *output, int rows) {
    finalLN.forward(input, output, rows);
}

template <typename WeiT, typename KVCacheT>
void YaRNLlama<WeiT, KVCacheT>::lastLayerNormForward(bfloat16_t *input, bfloat16_t *output, int rows) {
    finalLN.forward(input, output, rows);
}

template <typename WeiT, typename KVCacheT>
void YaRNLlama<WeiT, KVCacheT>::lastLayerNormForward(float16_t *input, float16_t *output, int rows) {
    finalLN.forward(input, output, rows);
}

IMPLEMENT_MODEL(YaRNLlama, yarn_llama)
