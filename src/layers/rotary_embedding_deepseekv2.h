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
#include <cmath>
#include <cstring>
#include <iostream>

#include "bfloat16.h"
#include "float16.h"
#include "transformer_ctx.h"

class DeekSeekV2RotaryEmbedding {
public:
    DeekSeekV2RotaryEmbedding(DecoderContext *ctx);

    ~DeekSeekV2RotaryEmbedding() {}

    // For continuous batching
    void forward(float *query, float *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads,
            int *positionIds);
    void forward(bfloat16_t *query, bfloat16_t *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads,
            int *positionIds);
    void forward(float16_t *query, float16_t *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads,
            int *positionIds);

private:
    void initYarn(const RopeParams *ropeParamsPtr);
    float yarnGetMscale(float scale, float mscale);
    void yarnFindRange(int &low, int &high, int betaFast, int betaSlow, int dim, float base, int orgMaxPosEmbed);
    void yarnLinearRampMask(float *invFreqMask, int low, int high, int dim, float extraFactor);
    void yarnLlamaCalEmb(float scale, float attnFactor);
#ifdef _DS_ROTARY_DEBUG_
    void test();
#endif
private:
    int dim = -1;
    int inv_freq_size = -1;
    int max_position_embeddings = -1;
    float *emb_cos = nullptr;
    float *emb_sin = nullptr;
};
