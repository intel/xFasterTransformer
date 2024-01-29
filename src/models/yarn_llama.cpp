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
#include <limits>

#include "yarn_llama.h"

template <typename WeiT>
YaRNLlama<WeiT>::YaRNLlama(const std::string &modelPath)
    : CommonDecoder<RopeScalingAttention<WeiT, LlamaYaRNScaledRotaryEmbedding, RmsNorm>, LlamaMLP<WeiT>, float>(
            modelPath, "yarn_llama") {
    // Context
    DecoderContext *ctx = this->getContext();

    // Embedding (no need position embed)
    embedding = new TokenEmbedding<float16_t>(ctx);
    setEmbeddingWeights(modelPath);

    // Final LN
    setFinalLnWeight(modelPath);
}

template <typename WeiT>
YaRNLlama<WeiT>::~YaRNLlama() {
    delete embedding;
}

template <typename WeiT>
void YaRNLlama<WeiT>::setEmbeddingWeights(const std::string &modelPath) {
    int vocabSize = embedding->getVocabSize();
    int hiddenSize = embedding->getHiddenSize();

    float *tokenEmb = (float *)malloc(vocabSize * hiddenSize * sizeof(float));

    loadWeight(modelPath + "/model.wte.bin", tokenEmb, vocabSize * hiddenSize, this->getDataType());

    embedding->setWeights(tokenEmb);

    free(tokenEmb);
}

template <typename WeiT>
void YaRNLlama<WeiT>::setFinalLnWeight(const std::string &modelPath) {
    int hiddenSize = embedding->getHiddenSize();

    float *gamma = (float *)malloc(hiddenSize * sizeof(float));

    loadWeight(modelPath + "/model.final_layernorm.weight.bin", gamma, hiddenSize, this->getDataType());

    finalLN.setWeight(gamma, nullptr, hiddenSize);

    free(gamma);
}

// Prepare attention_mask which is like:
// def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
//     # create causal mask
//     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
template <typename WeiT>
void YaRNLlama<WeiT>::prepareAttnMask(int *ids, int step) {
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

template <typename WeiT>
void YaRNLlama<WeiT>::embeddingForward(int *ids, float *output, int batchSize, int seqLen) {
    embedding->forward(ids, output, batchSize, seqLen);
}

template <typename WeiT>
void YaRNLlama<WeiT>::lastLayerNormForward(float *input, float *output, int rows) {
    finalLN.forward(input, output, rows);
}

template class YaRNLlama<float>;
template class YaRNLlama<float16_t>;
template class YaRNLlama<bfloat16_t>;
template class YaRNLlama<int8_t>;
template class YaRNLlama<w8a8_t>;
template class YaRNLlama<uint4x2_t>;
template class YaRNLlama<nf4x2_t>;
