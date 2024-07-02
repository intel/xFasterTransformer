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
#include "attention.h"
#include "common_decoder.h"
#include "mlp_chatglm2.h"
#include "rms_norm.h"
#include "rotary_embedding_chatglm2.h"
#include "token_embedding.h"

template <typename WeiT, typename KVCacheT>
class ChatGLM2
    : public CommonDecoder<Attention<WeiT, ChatGLM2RotaryEmbedding, RmsNorm, typename TypeSelector<WeiT>::InType,
                                   typename TypeSelector<WeiT>::ImType, typename TypeSelector<WeiT>::OutType, true>,
              ChatGLM2MLP<WeiT, typename TypeSelector<WeiT>::InType, typename TypeSelector<WeiT>::ImType,
                      typename TypeSelector<WeiT>::OutType, RmsNorm, true>,
              KVCacheT> {
public:
    ChatGLM2(const std::string &modelPath, const std::string &modelType = "chatglm2");
    ~ChatGLM2();

    virtual void prepareAttnMask(int *ids, int step);
    virtual void embeddingForward(int *ids, float *output, int tokenSize);
    virtual void embeddingForward(int *ids, bfloat16_t *output, int tokenSize);
    virtual void embeddingForward(int *ids, float16_t *output, int tokenSize);
    virtual void lastLayerNormForward(float *input, float *output, int rows);
    virtual void lastLayerNormForward(bfloat16_t *input, bfloat16_t *output, int rows);
    virtual void lastLayerNormForward(float16_t *input, float16_t *output, int rows);
    virtual int *getPositionIds(int *ids, int batchSize, int seqLen, int step) override;

private:
    virtual void setEmbeddingWeights(const std::string &modelPath);
    virtual void setFinalLnWeight(const std::string &modelPath);

private:
    TokenEmbedding<float16_t> *embedding;
    RmsNorm finalLN;

    // Record last block positions
    std::vector<int> lastBlockPositions;

    // position_ids
    // For input_ids [ 74747,  83400,  66846, 130001, 130004], position_ids is:
    // [0, 1, 2, 3, 4]
    // and next is [5], ...
    int *positionIds;
    int posBufSize;
};

REGISTER_MODEL(ChatGLM2, chatglm2)
