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
#include <algorithm>
#include <limits>

#include "INIReader.h"
#include "chatglm2.h"

template <typename WeiT, typename NormT>
ChatGLM2<WeiT, NormT>::ChatGLM2(const std::string &modelPath, const std::string &modelType)
    : CommonDecoder<ChatGLM2Attention<WeiT, ChatGLM2RotaryEmbedding, NormT, true>, ChatGLM2MLP<WeiT, NormT, true>>(
            modelPath, modelType) {
    this->positionIds = nullptr;
    this->posBufSize = 0;

    // Context
    DecoderContext *ctx = this->getContext();

    // Embedding
    embedding = new TokenEmbedding<float16_t>(ctx);
    setEmbeddingWeights(modelPath);

    // Final LN
    setFinalLnWeight(modelPath);
}

template <typename WeiT, typename NormT>
ChatGLM2<WeiT, NormT>::~ChatGLM2() {
    delete embedding;

    if (positionIds) { free(positionIds); }
}

template <typename WeiT, typename NormT>
void ChatGLM2<WeiT, NormT>::setEmbeddingWeights(const std::string &modelPath) {
    int vocabSize = embedding->getVocabSize();
    int hiddenSize = embedding->getHiddenSize();

    float *tokenEmb = (float *)malloc(vocabSize * hiddenSize * sizeof(float));

    loadWeight(modelPath + "/model.wte.bin", tokenEmb, vocabSize * hiddenSize, this->getDataType());

    embedding->setWeights(tokenEmb);

    free(tokenEmb);
}

template <typename WeiT, typename NormT>
void ChatGLM2<WeiT, NormT>::setFinalLnWeight(const std::string &modelPath) {
    int hiddenSize = embedding->getHiddenSize();

    float *gamma = (float *)malloc(hiddenSize * sizeof(float));
    float *beta = (float *)malloc(hiddenSize * sizeof(float));

    loadWeight(modelPath + "/model.final_layernorm.weight.bin", gamma, hiddenSize, this->getDataType());

    finalLN.setWeight(gamma, beta, hiddenSize);

    free(gamma);
    free(beta);
}

// Prepare attention_mask
// Python code:
// def get_masks(self, input_ids, device):
//     batch_size, seq_length = input_ids.shape
//     context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
//     attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
//     attention_mask.tril_()
//     for i, context_length in enumerate(context_lengths):
//         attention_mask[i, :, :context_length] = 1
//     attention_mask.unsqueeze_(1)
//     attention_mask = (attention_mask < 0.5).bool()
//
//     return attention_mask
template <typename WeiT, typename NormT>
void ChatGLM2<WeiT, NormT>::prepareAttnMask(int *ids, int step) {
    DecoderContext *ctx = this->getContext();
    int seqLen = ctx->inputSeqLen;
    int sizeRequired = ctx->batchSize * seqLen * seqLen;

    if (step == 0) {
        float *mask = this->getAttnMask(sizeRequired);
        int startId = this->getStartId();

        for (int b = 0; b < ctx->batchSize; ++b) {
            int contextLen = -1;
            auto it = std::find(ids + b * seqLen, ids + (b + 1) * seqLen, startId);
            if (it != ids + (b + 1) * seqLen) { contextLen = std::distance(ids + b * seqLen, it); }

            auto pmask = mask + b * seqLen * seqLen;
            for (int i = 0; i < seqLen; ++i) {
                int zeroLen = contextLen > (i + 1) ? contextLen : (i + 1);
                memset(pmask + i * seqLen, 0, zeroLen * sizeof(float)); // bottom left or 0:contextLen are 0
                std::fill_n(pmask + i * seqLen + zeroLen, seqLen - zeroLen, std::numeric_limits<float>::lowest());
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

template <typename WeiT, typename NormT>
void ChatGLM2<WeiT, NormT>::embeddingForward(int *ids, float *output, int batchSize, int seqLen) {
    embedding->forward(ids, output, batchSize, seqLen);
}

template <typename WeiT, typename NormT>
void ChatGLM2<WeiT, NormT>::lastLayerNormForward(float *input, float *output, int rows) {
    finalLN.forward(input, output, rows);
}

// Return the position_ids + block_position_ids
// if position_ids is None:
//     position_ids = self.get_position_ids(input_ids, device=input_ids.device)
// if not is_first_forward:
//     position_ids = position_ids[..., -1:]
//     input_ids = input_ids[:, -1:]
//     def get_position_ids(self, input_ids, device):
// batch_size, seq_length = input_ids.shape
// position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
// return position_ids
template <typename WeiT, typename NormT>
int *ChatGLM2<WeiT, NormT>::getPositionIds(int *ids, int batchSize, int seqLen, int step) {
    // Prepare buffer
    int sizeNeeded = (batchSize * seqLen + 63) / 64 * 64; // position_ids + block_position_ids
    if (posBufSize < sizeNeeded) {
        if (positionIds) { free(positionIds); }
        posBufSize = sizeNeeded + 8; // whatever, a little bigger
        positionIds = (int *)aligned_alloc(64, posBufSize * sizeof(int));
    }
    if (step == 0) {
        lastBlockPositions.clear();
        for (int i = 0; i < batchSize; ++i) {
            int *pos = positionIds + i * seqLen;
            for (int j = 0; j < seqLen; ++j) {
                pos[j] = j;
            }
            lastBlockPositions.emplace_back(seqLen);
        }
    } else {
        if (batchSize > lastBlockPositions.size()) {
            int userSideBS = lastBlockPositions.size();
            int beamSize = batchSize / userSideBS;
            std::vector<int> tmp(lastBlockPositions);
            lastBlockPositions.clear();

            lastBlockPositions.reserve(batchSize);
            for (int i = 0; i < userSideBS; ++i) {
                lastBlockPositions.insert(lastBlockPositions.begin() + i * beamSize, beamSize, tmp[i]);
            }
        }
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < seqLen; j++) {
                positionIds[i * seqLen + j] = lastBlockPositions[i] + j;
            }
            lastBlockPositions[i] += seqLen;
        }
    }
    return positionIds;
}

template class ChatGLM2<float>;
template class ChatGLM2<float16_t>;
template class ChatGLM2<bfloat16_t>;
template class ChatGLM2<int8_t>;
template class ChatGLM2<uint4x2_t>;
template class ChatGLM2<nf4x2_t>;