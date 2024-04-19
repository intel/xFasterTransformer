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

#include "common_decoder.h"
#include "mlp_llama.h"
#include "rms_norm.h"
#include "rotary_embedding.h"
#include "token_embedding_gemma.h"
#include "type_selector.h"

template <typename WeiT>
class GemmaLLM
    : public CommonDecoder<Attention<WeiT, LlamaRotaryEmbedding, RmsNorm, typename TypeSelector<WeiT>::InType,
                                   typename TypeSelector<WeiT>::ImType, typename TypeSelector<WeiT>::OutType, true>,
              LlamaMLP<WeiT, typename TypeSelector<WeiT>::InType, typename TypeSelector<WeiT>::ImType,
                      typename TypeSelector<WeiT>::OutType>,
              typename TypeSelector<WeiT>::KVCacheType> {
public:
    GemmaLLM(const std::string &modelPath);
    ~GemmaLLM();

    void prepareAttnMask(int *ids, int step);

    void embeddingForward(int *ids, float *output, int batchSize, int seqLen);
    void embeddingForward(int *ids, bfloat16_t *output, int batchSize, int seqLen);

    void lastLayerNormForward(float *input, float *output, int rows);
    void lastLayerNormForward(bfloat16_t *input, bfloat16_t *output, int rows);

private:
    void setEmbeddingWeights(const std::string &modelPath);
    void setFinalLnWeight(const std::string &modelPath);

private:
    GemmaTokenEmbedding<float16_t> *embedding;
    RmsNorm finalLN;
};

REGISTER_DECODER(GemmaLLM, gemma, float)
REGISTER_DECODER(GemmaLLM, gemma, float16_t)
REGISTER_DECODER(GemmaLLM, gemma, bfloat16_t)
REGISTER_DECODER(GemmaLLM, gemma, int8_t)
REGISTER_DECODER(GemmaLLM, gemma, w8a8_t)
REGISTER_DECODER(GemmaLLM, gemma, uint4x2_t)
REGISTER_DECODER(GemmaLLM, gemma, nf4x2_t)
REGISTER_HYBRID_MODEL(GemmaLLM, gemma, bfloat16_t, float16_t)
REGISTER_HYBRID_MODEL(GemmaLLM, gemma, bfloat16_t, int8_t)
REGISTER_HYBRID_MODEL(GemmaLLM, gemma, bfloat16_t, w8a8_t)
REGISTER_HYBRID_MODEL(GemmaLLM, gemma, bfloat16_t, uint4x2_t)
REGISTER_HYBRID_MODEL(GemmaLLM, gemma, bfloat16_t, nf4x2_t)
REGISTER_HYBRID_MODEL(GemmaLLM, gemma, w8a8_t, int8_t)
REGISTER_HYBRID_MODEL(GemmaLLM, gemma, w8a8_t, uint4x2_t)
REGISTER_HYBRID_MODEL(GemmaLLM, gemma, w8a8_t, nf4x2_t)