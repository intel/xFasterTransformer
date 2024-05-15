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
#include <map>
#include <string>
#include <vector>

#include "abstract_decoder.h"
#include "attention.h"
#include "common_decoder.h"
#include "dist_linear.h"
#include "float16.h"
#include "layer_norm.h"
#include "messenger.h"
#include "mlp_standard.h"
#include "opt_embedding.h"
#include "transformer_ctx.h"

template <typename WeiT, typename KVCacheT>
class OptDecoder : public CommonDecoder<Attention<WeiT, QKPO_Dummy, LayerNorm>, MLP<WeiT>, KVCacheT> {
public:
    OptDecoder(const std::string &modelPath);
    ~OptDecoder();

    void prepareAttnMask(int *ids, int step);
    void embeddingForward(int *ids, float *output, int tokenSize);
    void embeddingForward(float *output, const std::vector<SequenceMeta *> &sequences);
    void lastLayerNormForward(float *input, float *output, int rows);

private:
    void setEmbeddingWeights(const std::string &modelPath);
    void setFinalLnWeight(const std::string &modelPath);

private:
    OptEmbedding<float16_t> *embedding;
    LayerNorm finalLN;
};

REGISTER_MODEL(OptDecoder, gpt)
