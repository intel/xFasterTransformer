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

#include "qwen2.h"

template <typename WeiT, typename KVCacheT>
Qwen2LLM<WeiT, KVCacheT>::Qwen2LLM(const std::string &modelPath)
    : CommonDecoder<Attention<WeiT, LlamaRotaryEmbedding, RmsNorm, typename TypeSelector<WeiT>::InType,
                            typename TypeSelector<WeiT>::ImType, typename TypeSelector<WeiT>::OutType, true>,
            LlamaMLP<WeiT, typename TypeSelector<WeiT>::InType, typename TypeSelector<WeiT>::ImType,
                    typename TypeSelector<WeiT>::OutType>,
            KVCacheT>(modelPath, "qwen2") {
    // Context
    DecoderContext *ctx = this->getContext();

    // Embedding
    embedding = new TokenEmbedding<float16_t>(ctx);
    setEmbeddingWeights(modelPath);

    // Final LN
    setFinalLnWeight(modelPath);
}

template <typename WeiT, typename KVCacheT>
Qwen2LLM<WeiT, KVCacheT>::~Qwen2LLM() {
    delete embedding;
}

template <typename WeiT, typename KVCacheT>
void Qwen2LLM<WeiT, KVCacheT>::setEmbeddingWeights(const std::string &modelPath) {
    embedding->setWeights(modelPath + "/model.wte.bin");
}

template <typename WeiT, typename KVCacheT>
void Qwen2LLM<WeiT, KVCacheT>::setFinalLnWeight(const std::string &modelPath) {
    finalLN.setWeight(modelPath + "/model.final_layernorm.weight.bin", "", embedding->getHiddenSize());
}

// Prepare attention_mask which is like:
// def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
//     # create causal mask
//     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
//     combined_attention_mask = None
//     if input_shape[-1] > 1:
//         combined_attention_mask = _make_causal_mask(
//             input_shape,
//             inputs_embeds.dtype,
//             device=inputs_embeds.device,
//             past_key_values_length=past_key_values_length,
//         )
//     if attention_mask is not None:
//         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
//         expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
//             inputs_embeds.device
//         )
//         combined_attention_mask = (
//             expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
//         )
//     return combined_attention_mask
template <typename WeiT, typename KVCacheT>
void Qwen2LLM<WeiT, KVCacheT>::prepareAttnMask(int *ids, int step) {
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
void Qwen2LLM<WeiT, KVCacheT>::embeddingForward(int *ids, float *output, int tokenSize) {
    embedding->forward(ids, output, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void Qwen2LLM<WeiT, KVCacheT>::embeddingForward(int *ids, bfloat16_t *output, int tokenSize) {
    embedding->forward(ids, output, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void Qwen2LLM<WeiT, KVCacheT>::embeddingForward(int *ids, float16_t *output, int tokenSize) {
    embedding->forward(ids, output, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void Qwen2LLM<WeiT, KVCacheT>::lastLayerNormForward(float *input, float *output, int rows) {
    finalLN.forward(input, output, rows);
}

template <typename WeiT, typename KVCacheT>
void Qwen2LLM<WeiT, KVCacheT>::lastLayerNormForward(bfloat16_t *input, bfloat16_t *output, int rows) {
    finalLN.forward(input, output, rows);
}

template <typename WeiT, typename KVCacheT>
void Qwen2LLM<WeiT, KVCacheT>::lastLayerNormForward(float16_t *input, float16_t *output, int rows) {
    finalLN.forward(input, output, rows);
}

IMPLEMENT_MODEL(Qwen2LLM, qwen2)
