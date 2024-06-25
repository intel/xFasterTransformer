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
#include <algorithm>
#include <limits>

#include "INIReader.h"
#include "allocator.h"
#include "chatglm2.h"

template <typename WeiT, typename KVCacheT>
ChatGLM2<WeiT, KVCacheT>::ChatGLM2(const std::string &modelPath, const std::string &modelType)
    : CommonDecoder<Attention<WeiT, ChatGLM2RotaryEmbedding, RmsNorm, typename TypeSelector<WeiT>::InType,
                            typename TypeSelector<WeiT>::ImType, typename TypeSelector<WeiT>::OutType, true>,
            ChatGLM2MLP<WeiT, typename TypeSelector<WeiT>::InType, typename TypeSelector<WeiT>::ImType,
                    typename TypeSelector<WeiT>::OutType, RmsNorm, true>,
            KVCacheT>(modelPath, modelType) {
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

template <typename WeiT, typename KVCacheT>
ChatGLM2<WeiT, KVCacheT>::~ChatGLM2() {
    delete embedding;

    if (positionIds) { free(positionIds); }
}

template <typename WeiT, typename KVCacheT>
void ChatGLM2<WeiT, KVCacheT>::setEmbeddingWeights(const std::string &modelPath) {
    embedding->setWeights(modelPath + "/model.wte.bin");
}

template <typename WeiT, typename KVCacheT>
void ChatGLM2<WeiT, KVCacheT>::setFinalLnWeight(const std::string &modelPath) {
    finalLN.setWeight(modelPath + "/model.final_layernorm.weight.bin", "", embedding->getHiddenSize());
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
template <typename WeiT, typename KVCacheT>
void ChatGLM2<WeiT, KVCacheT>::prepareAttnMask(int *ids, int step) {
    DecoderContext *ctx = this->getContext();
    int seqLen = ctx->inputSeqLen;

    if (step == 0) {
        int sizeRequired = ctx->batchSize * seqLen * seqLen;
        float *mask = this->getAttnMask(sizeRequired);
        for (int b = 0; b < ctx->batchSize; ++b) {
            auto pmask = mask + b * seqLen * seqLen;
            for (int i = 0; i < seqLen; ++i) {
                int zeroLen = i + 1;
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

template <typename WeiT, typename KVCacheT>
void ChatGLM2<WeiT, KVCacheT>::embeddingForward(int *ids, float *output, int tokenSize) {
    embedding->forward(ids, output, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void ChatGLM2<WeiT, KVCacheT>::embeddingForward(int *ids, bfloat16_t *output, int tokenSize) {
    embedding->forward(ids, output, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void ChatGLM2<WeiT, KVCacheT>::embeddingForward(int *ids, float16_t *output, int tokenSize) {
    embedding->forward(ids, output, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void ChatGLM2<WeiT, KVCacheT>::lastLayerNormForward(float *input, float *output, int rows) {
    finalLN.forward(input, output, rows);
}

template <typename WeiT, typename KVCacheT>
void ChatGLM2<WeiT, KVCacheT>::lastLayerNormForward(bfloat16_t *input, bfloat16_t *output, int rows) {
    finalLN.forward(input, output, rows);
}

template <typename WeiT, typename KVCacheT>
void ChatGLM2<WeiT, KVCacheT>::lastLayerNormForward(float16_t *input, float16_t *output, int rows) {
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
template <typename WeiT, typename KVCacheT>
int *ChatGLM2<WeiT, KVCacheT>::getPositionIds(int *ids, int batchSize, int seqLen, int step) {
    // Prepare buffer
    int sizeNeeded = (batchSize * seqLen + 63) / 64 * 64; // position_ids + block_position_ids
    if (posBufSize < sizeNeeded) {
        if (positionIds) { free(positionIds); }
        posBufSize = sizeNeeded + 8; // whatever, a little bigger
        positionIds = (int *)xft::alloc(posBufSize * sizeof(int));
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

IMPLEMENT_MODEL(ChatGLM2, chatglm2)
