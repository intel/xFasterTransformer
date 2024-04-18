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
#pragma once

#include <vector>
#include "attn_chatglm.h"
#include "common_decoder.h"
#include "layer_norm.h"
#include "mlp_chatglm.h"
#include "rope_2d.h"
#include "token_embedding.h"

template <typename WeiT>
class ChatGLM : public CommonDecoder<ChatGlmAttention<WeiT, RotaryEmbedding2D, LayerNorm>, ChatGlmMLP<WeiT>> {
public:
    ChatGLM(const std::string &modelPath);
    ~ChatGLM();

    void prepareAttnMask(int *ids, int step);
    void embeddingForward(int *ids, float *output, int batchSize, int seqLen);
    void lastLayerNormForward(float *input, float *output, int rows);
    int *getPositionIds(int *ids, int batchSize, int seqLen, int step) override;
    void setPrefix(int *ids, int seqLen) override;

private:
    void setEmbeddingWeights(const std::string &modelPath);
    void setFinalLnWeight(const std::string &modelPath);

private:
    TokenEmbedding<float16_t> *embedding;
    LayerNorm finalLN;

    // Mask token ID from configuration
    int maskTokenId;
    int gmaskTokenId;

    // Mask position in current sequence, used to generate position_ids
    // mask_positions, use_gmasks = [], []
    // for seq in seqs:
    //     mask_token = gMASK if gMASK in seq else MASK
    //     use_gmask = mask_token == gMASK
    //     mask_positions.append(seq.index(mask_token))
    //     use_gmasks.append(use_gmask)
    std::vector<int> maskPositions;

    // Record last block positions
    std::vector<int> lastBlockPositions;

    // position_ids + block_position_ids
    // For input_ids [ 74747,  83400,  66846, 130001, 130004], position_ids is:
    // [0, 1, 2, 3, 3] + [0, 0, 0, 0, 1], as gmask_token_id = 130001
    // and next is [3] + [2], ...
    int *positionIds;
    int posBufSize;
};

REGISTER_DECODER(ChatGLM, chatglm, float)
REGISTER_DECODER(ChatGLM, chatglm, float16_t)
REGISTER_DECODER(ChatGLM, chatglm, bfloat16_t)
REGISTER_DECODER(ChatGLM, chatglm, int8_t)
REGISTER_DECODER(ChatGLM, chatglm, w8a8_t)
REGISTER_DECODER(ChatGLM, chatglm, uint4x2_t)
REGISTER_DECODER(ChatGLM, chatglm, nf4x2_t)
REGISTER_HYBRID_MODEL(ChatGLM, chatglm, bfloat16_t, float16_t)
REGISTER_HYBRID_MODEL(ChatGLM, chatglm, bfloat16_t, int8_t)
REGISTER_HYBRID_MODEL(ChatGLM, chatglm, bfloat16_t, w8a8_t)
REGISTER_HYBRID_MODEL(ChatGLM, chatglm, bfloat16_t, uint4x2_t)
REGISTER_HYBRID_MODEL(ChatGLM, chatglm, bfloat16_t, nf4x2_t)
REGISTER_HYBRID_MODEL(ChatGLM, chatglm, w8a8_t, int8_t)
REGISTER_HYBRID_MODEL(ChatGLM, chatglm, w8a8_t, uint4x2_t)
REGISTER_HYBRID_MODEL(ChatGLM, chatglm, w8a8_t, nf4x2_t)