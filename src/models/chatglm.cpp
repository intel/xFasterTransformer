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
#include "chatglm.h"

template <typename WeiT, typename KVCacheT>
ChatGLM<WeiT, KVCacheT>::ChatGLM(const std::string &modelPath)
    : CommonDecoder<ChatGlmAttention<WeiT, RotaryEmbedding2D, LayerNorm>, ChatGlmMLP<WeiT>, KVCacheT>(
            modelPath, "chatglm") {
    std::string configPath = modelPath + "/config.ini";
    INIReader reader = INIReader(configPath);

    this->maskTokenId = reader.GetInteger("chatglm", "mask_token_id", 130000);
    this->gmaskTokenId = reader.GetInteger("chatglm", "gmask_token_id", 130001);

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
ChatGLM<WeiT, KVCacheT>::~ChatGLM() {
    delete embedding;

    if (positionIds) { free(positionIds); }
}

template <typename WeiT, typename KVCacheT>
void ChatGLM<WeiT, KVCacheT>::setEmbeddingWeights(const std::string &modelPath) {
    embedding->setWeights(modelPath + "/model.wte.bin");
}

template <typename WeiT, typename KVCacheT>
void ChatGLM<WeiT, KVCacheT>::setFinalLnWeight(const std::string &modelPath) {
    finalLN.setWeight(modelPath + "/model.final_layernorm.weight.bin", modelPath + "/model.final_layernorm.bias.bin",
            embedding->getHiddenSize());
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
void ChatGLM<WeiT, KVCacheT>::prepareAttnMask(int *ids, int step) {
    DecoderContext *ctx = this->getContext();
    int seqLen = ctx->inputSeqLen;

    if (step == 0) {
        int sizeRequired = ctx->batchSize * seqLen * seqLen;
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
    } else {
        int sizeRequired = ctx->batchSize * this->accSeqLen;
        float *mask = this->getAttnMask(sizeRequired);
        memset(mask, 0, ctx->batchSize * this->accSeqLen * sizeof(float)); // all elements are 0
    }
}

template <typename WeiT, typename KVCacheT>
void ChatGLM<WeiT, KVCacheT>::embeddingForward(int *ids, float *output, int tokenSize) {
    embedding->forward(ids, output, tokenSize);
}

template <typename WeiT, typename KVCacheT>
void ChatGLM<WeiT, KVCacheT>::lastLayerNormForward(float *input, float *output, int rows) {
    finalLN.forward(input, output, rows);
}

// Return the position_ids + block_position_ids
template <typename WeiT, typename KVCacheT>
int *ChatGLM<WeiT, KVCacheT>::getPositionIds(int *ids, int batchSize, int seqLen, int step) {
    if (step == 0) {
        maskPositions.clear();
        lastBlockPositions.clear();

        for (int i = 0; i < batchSize; ++i) {
            int *p = ids + i * seqLen;

            // Python code:
            // mask_token = gMASK if gMASK in seq else MASK
            // mask_positions.append(seq.index(mask_token))
            int maskPos = -1;
            bool gMaskDetected = false;
            for (int s = 0; s < seqLen; ++s) {
                // Always use gMASK if gMASK detected
                if (p[s] == gmaskTokenId) {
                    gMaskDetected = true;
                    maskPositions.emplace_back(s);
                    break;
                }
                // Detect MASK (not gMASK)
                if (p[s] == maskTokenId && maskPos == -1) { maskPos == s; }
            }

            if (!gMaskDetected) { maskPositions.emplace_back(maskPos); }
        }

        // Prepare buffer
        int sizeNeeded = 2 * batchSize * seqLen; // position_ids + block_position_ids
        if (posBufSize < sizeNeeded) {
            if (positionIds) { free(positionIds); }
            posBufSize = sizeNeeded + 8; // whatever, a little bigger
            positionIds = (int *)xft::alloc(posBufSize * sizeof(int));
        }

        // position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        // context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        // position_ids[i, context_length:] = mask_positions[i]
        // block_position_ids = [torch.cat((
        //     torch.zeros(context_length, dtype=torch.long, device=device),
        //     torch.arange(seq_length - context_length, dtype=torch.long, device=device) + 1
        // ))
        int bosId = this->getStartId();
        for (int i = 0; i < batchSize; ++i) {
            int *pos = positionIds + i * seqLen * 2;
            int *blockPos = pos + seqLen;

            int contextLength = -1;
            int *pIds = ids + i * seqLen;
            auto it = std::find(pIds, pIds + seqLen, bosId);

            if (unlikely(it == pIds + seqLen)) {
                printf("WARNING: cannot find bos_token_id, unexpected!\n");
                continue;
            }

            contextLength = std::distance(pIds, it);

            for (int j = 0; j < contextLength; ++j) {
                pos[j] = j;
                blockPos[j] = 0;
            }

            for (int j = contextLength; j < seqLen; ++j) {
                pos[j] = maskPositions[i];
                blockPos[j] = j - contextLength + 1;
            }

            lastBlockPositions.emplace_back(seqLen - contextLength);
        }
    } else {
        if (batchSize > maskPositions.size()) {
            int userSideBS = maskPositions.size();
            int beamSize = batchSize / userSideBS;
            std::vector<int> tmpMaskP(maskPositions);
            std::vector<int> tmpLastBlockP(lastBlockPositions);
            maskPositions.clear();
            lastBlockPositions.clear();

            maskPositions.reserve(batchSize);
            lastBlockPositions.reserve(batchSize);
            for (int i = 0; i < userSideBS; ++i) {
                maskPositions.insert(maskPositions.begin() + i * beamSize, beamSize, tmpMaskP[i]);
                lastBlockPositions.insert(lastBlockPositions.begin() + i * beamSize, beamSize, tmpLastBlockP[i]);
            }
        }
        for (int i = 0; i < batchSize; ++i) {
            positionIds[i * 2] = maskPositions[i];
            positionIds[i * 2 + 1] = lastBlockPositions[i] + 1;
            lastBlockPositions[i] += 1;
        }
    }

    return positionIds;
}

template <typename WeiT, typename KVCacheT>
void ChatGLM<WeiT, KVCacheT>::setPrefix(int *ids, int seqLen) {
    printf("[ERROR] ChatGLM doesn't support prefix sharing.\n");
    exit(-1);
}

IMPLEMENT_MODEL(ChatGLM, chatglm)