// Copyright (c) 2024 Intel Corporation
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

#include "bfloat16.h"
#include "float16.h"

namespace xft {

void llamaSetCosSinCache(const float *invFreq, float *embCos, float *embSin, int invFreqSize,
        int max_position_embeddings = 2048, float scale = 1.0);

void llamaApplyRotaryPosEmbeding(float *query, float *key, int qStride, int kStride, float *emb_cos, float *emb_sin,
        int inv_freq_size, const int *qkShape, const int *positionIds);

void llamaApplyRotaryPosEmbeding(bfloat16_t *query, bfloat16_t *key, int qStride, int kStride, float *emb_cos,
        float *emb_sin, int inv_freq_size, const int *qkShape, const int *positionIds);

void llamaApplyRotaryPosEmbeding(float16_t *query, float16_t *key, int qStride, int kStride, float *emb_cos,
        float *emb_sin, int inv_freq_size, const int *qkShape, const int *positionIds);

// For continous batching
void llamaApplyRotaryPosEmbed(float *query, float *key, float *embCos, float *embSin, int qStride, int kStride, int dim,
        int totSeqLen, int qHeads, int kHeads, const int *positionIds);

void llamaApplyRotaryPosEmbed(bfloat16_t *query, bfloat16_t *key, float *emb_cos, float *emb_sin, int qStride,
        int kStride, int dim, int totSeqLen, int qHeads, int kHeads, const int *positionIds);

void llamaApplyRotaryPosEmbed(float16_t *query, float16_t *key, float *emb_cos, float *emb_sin, int qStride,
        int kStride, int dim, int totSeqLen, int qHeads, int kHeads, const int *positionIds);


void chatglm2ApplyRotaryPosEmbeding(float *query, float *key, int qStride, int kStride, float *emb_cos, float *emb_sin,
        int inv_freq_size, const int *qkShape, const int *positionIds);

void chatglm2ApplyRotaryPosEmbeding(bfloat16_t *query, bfloat16_t *key, int qStride, int kStride, float *emb_cos,
        float *emb_sin, int inv_freq_size, const int *qkShape, const int *positionIds);

void chatglm2ApplyRotaryPosEmbeding(float16_t *query, float16_t *key, int qStride, int kStride, float *emb_cos,
        float *emb_sin, int inv_freq_size, const int *qkShape, const int *positionIds);

} // namespace xft
