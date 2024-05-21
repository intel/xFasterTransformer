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

#include "attn_baichuan.h"
#include "common_decoder.h"
#include "layers_norm.h"
#include "mlp_llama.h"
#include "rotary_embedding.h"
#include "token_embedding.h"

template <typename WeiT, typename KVCacheT>
class Baichuan
    : public CommonDecoder<BaichuanAttention<WeiT, LlamaRotaryEmbedding, RmsNorm, typename TypeSelector<WeiT>::InType,
                                   typename TypeSelector<WeiT>::ImType, typename TypeSelector<WeiT>::OutType, true>,
              LlamaMLP<WeiT, typename TypeSelector<WeiT>::InType, typename TypeSelector<WeiT>::ImType,
                      typename TypeSelector<WeiT>::OutType>,
              KVCacheT> {
public:
    Baichuan(const std::string &modelPath);
    ~Baichuan();

    void prepareAttnMaskBase(int *ids, int step);
    void prepareAttnMask(int *ids, int step);
    void embeddingForward(int *ids, float *output, int tokenSize);
    void embeddingForward(int *ids, bfloat16_t *output, int tokenSize);
    void lastLayerNormForward(float *input, float *output, int rows);
    void lastLayerNormForward(bfloat16_t *input, bfloat16_t *output, int rows);

private:
    void setEmbeddingWeights(const std::string &modelPath);
    void setFinalLnWeight(const std::string &modelPath);

private:
    TokenEmbedding<float16_t> *embedding;
    RmsNorm finalLN;
};

REGISTER_MODEL(Baichuan, baichuan)
