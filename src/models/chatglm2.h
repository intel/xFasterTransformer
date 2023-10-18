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
#pragma once

#include <vector>
#include "attn_chatglm2.h"
#include "common_decoder.h"
#include "layer_norm.h"
#include "mlp_chatglm2.h"
#include "rms_norm.h"
#include "rotary_embedding_chatglm2.h"
#include "token_embedding.h"

template <typename WeiT, typename NormT = RmsNorm>
class ChatGLM2 : public CommonDecoder<ChatGLM2Attention<WeiT, ChatGLM2RotaryEmbedding, NormT, true>,
                         ChatGLM2MLP<WeiT, NormT, true>> {
public:
    ChatGLM2(const std::string &modelPath);
    ~ChatGLM2();

    virtual void prepareAttnMask(int *ids, int step);
    virtual void embeddingForward(int *ids, float *output, int batchSize, int seqLen);
    virtual void lastLayerNormForward(float *input, float *output, int rows);
    virtual int *getPositionIds(int *ids, int batchSize, int seqLen, int step) override;

private:
    virtual void setEmbeddingWeights(const std::string &modelPath);
    virtual void setFinalLnWeight(const std::string &modelPath);

private:
    TokenEmbedding<float16_t> *embedding;
    NormT finalLN;

    // Record last block positions
    std::vector<int> lastBlockPositions;

    // position_ids
    // For input_ids [ 74747,  83400,  66846, 130001, 130004], position_ids is:
    // [0, 1, 2, 3, 4]
    // and next is [5], ...
    int *positionIds;
    int posBufSize;
};
