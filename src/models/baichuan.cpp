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

#include "baichuan.h"

template <typename WeiT>
Baichuan<WeiT>::Baichuan(const std::string &modelPath)
    : CommonDecoder<BaichuanAttention<WeiT, LlamaRotaryEmbedding, RmsNorm>, LlamaMLP<WeiT>, float>(
            modelPath, "baichuan") {
    // Context
    DecoderContext *ctx = this->getContext();

    // Embedding (no need position embed)
    embedding = new TokenEmbedding<float16_t>(ctx);
    setEmbeddingWeights(modelPath);

    // Final LN
    setFinalLnWeight(modelPath);
}

template <typename WeiT>
Baichuan<WeiT>::~Baichuan() {
    delete embedding;
}

template <typename WeiT>
void Baichuan<WeiT>::setEmbeddingWeights(const std::string &modelPath) {
    int vocabSize = embedding->getVocabSize();
    int hiddenSize = embedding->getHiddenSize();

    float *tokenEmb = (float *)malloc(vocabSize * hiddenSize * sizeof(float));

    loadWeight(modelPath + "/model.wte.bin", tokenEmb, vocabSize * hiddenSize, this->getDataType());

    embedding->setWeights(tokenEmb);

    free(tokenEmb);
}

template <typename WeiT>
void Baichuan<WeiT>::setFinalLnWeight(const std::string &modelPath) {
    int hiddenSize = embedding->getHiddenSize();

    float *gamma = (float *)malloc(hiddenSize * sizeof(float));

    loadWeight(modelPath + "/model.final_layernorm.weight.bin", gamma, hiddenSize, this->getDataType());

    finalLN.setWeight(gamma, nullptr, hiddenSize);

    free(gamma);
}

// Prepare attention_mask which is like:
//def _get_interleave(n):
//    def _get_interleave_power_of_2(n):
//        start = (2 ** (-2 ** -(math.log2(n) - 3)))
//        ratio = start
//        return [start * ratio ** i for i in range(n)]
//
//    if math.log2(n).is_integer():
//        return _get_interleave_power_of_2(n)
//    else:
//        closest_power_of_2 = 2 ** math.floor(math.log2(n))
//        return _get_interleave_power_of_2(closest_power_of_2) + \
//               _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
//def _gen_alibi_mask(n_head, max_pos):
//    """used in inference only"""
//    slopes = torch.Tensor(_get_interleave(n_head))
//    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_pos).unsqueeze(0).unsqueeze(0).expand(
//        n_head, -1, -1)
//    alibi = alibi.view(n_head, 1, max_pos)
//    alibi_mask = torch.triu(
//        _fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1
//    )
//    alibi_mask = alibi_mask.unsqueeze(0) + alibi
//    return alibi_mask

template <typename WeiT>
void Baichuan<WeiT>::prepareAttnMaskBase(int *ids, int step) {
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
void Baichuan<WeiT>::prepareAttnMask(int *ids, int step) {
    DecoderContext *ctx = this->getContext();
    if (ctx->maxPosEmbed > 0) {
        // Base Mask for CausalLM
        prepareAttnMaskBase(ids, step);
        return;
    }

    // Alibi Mask
    int seqLen = ctx->inputSeqLen;

    int responsibleHeads = BaichuanAttention<WeiT>::getResponsibleHeads();
    // alibi mask slope for each head
    const float *slopes = BaichuanAttention<WeiT>::getAlibiSlopes();

    if (step == 0) {
        int sizeRequired = responsibleHeads * seqLen * seqLen;
        float *mask = this->getAttnMask(sizeRequired);
        for (int h = 0; h < responsibleHeads; ++h) {
            auto pmask = mask + h * seqLen * seqLen;
            for (int i = 0; i < seqLen; ++i) {
                memset(pmask + i * seqLen, 0, (i + 1) * sizeof(float)); // bottom left are 0
                // attention mask added with alibi mask
                for (int j = 0; j < i + 1; ++j) {
                    pmask[i * seqLen + j] += j * slopes[h];
                }
                std::fill_n(pmask + i * seqLen + i + 1, seqLen - i - 1, std::numeric_limits<float>::lowest());
            }
        }
    } else if (seqLen > 1) {
        int sizeRequired = responsibleHeads * this->accSeqLen * seqLen;
        float *mask = this->getAttnMask(sizeRequired);
        for (int h = 0; h < responsibleHeads; ++h) {
            auto pmask = mask + h * this->accSeqLen * seqLen;
            int pastLen = this->accSeqLen - seqLen;
            for (int i = 0; i < seqLen; ++i) {
                memset(pmask + i * this->accSeqLen, 0, (pastLen + i + 1) * sizeof(float));
                // attention mask added with alibi mask
                for (int j = 0; j < pastLen + i + 1; ++j) {
                    pmask[i * this->accSeqLen + j] += j * slopes[h];
                }
                std::fill_n(pmask + i * this->accSeqLen + pastLen + i + 1, seqLen - i - 1,
                        std::numeric_limits<float>::lowest());
            }
        }
    } else {
        int sizeRequired = responsibleHeads * this->accSeqLen;
        float *mask = this->getAttnMask(sizeRequired);
        for (int h = 0; h < responsibleHeads; ++h) {
            auto pmask = mask + h * this->accSeqLen;
            // attention mask added with alibi mask
            for (int j = 0; j < this->accSeqLen; ++j) {
                pmask[j] = j * slopes[h];
            }
        }
    }
}

template <typename WeiT>
void Baichuan<WeiT>::embeddingForward(int *ids, float *output, int batchSize, int seqLen) {
    embedding->forward(ids, output, batchSize, seqLen);
}

template <typename WeiT>
void Baichuan<WeiT>::lastLayerNormForward(float *input, float *output, int rows) {
    finalLN.forward(input, output, rows);
}

template class Baichuan<float>;
template class Baichuan<float16_t>;
template class Baichuan<bfloat16_t>;
template class Baichuan<int8_t>;
template class Baichuan<w8a8_t>;
template class Baichuan<uint4x2_t>;
template class Baichuan<nf4x2_t>;
