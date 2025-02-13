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
#include "rotary_embedding_deepseekv2.h"
#include <algorithm>
#include <cmath>
#include "allocator.h"
#include "compile_util.h"
#include "rotary_embedding_kernels.h"

DeekSeekV2RotaryEmbedding::DeekSeekV2RotaryEmbedding(DecoderContext *ctx) {
    const std::string embCosPrefix = "deepseek_emb_cos";
    const std::string embSinPrefix = "deepseek_emb_sin";

    bool initialized = false;
    float base = -1;
    int rope_ratio = 1;

    this->ropeDim = ctx->ropeDim;
    this->nopeDim = ctx->nopeDim;
    this->maxPositionEmbeddings = ctx->maxPosEmbed;
    this->invFreqSize = (ropeDim + 1) / 2;
    const RopeParams *ropeParamsPtr = ctx->ropeParamsPtr;
    if (ropeParamsPtr == nullptr) return;
    // assert ropeParam in Context
    assert(ropeParamsPtr->type == "yarn");

    std::string embCosStr = embCosPrefix + "_" + ropeParamsPtr->type;
    std::string embSinStr = embSinPrefix + "_" + ropeParamsPtr->type;

    if (ctx->cached(embCosStr)) { initialized = true; }

    this->embCos = ctx->getBuffer<float>(embCosStr, maxPositionEmbeddings * invFreqSize * 2);
    this->embSin = ctx->getBuffer<float>(embSinStr, maxPositionEmbeddings * invFreqSize * 2);
    // printf("maxPositionEmbeddings=%d, dim=%d, invFreqSize=%d\n", maxPositionEmbeddings, dim, invFreqSize);

    if (!initialized) {
        initialized = true;
        if (ropeParamsPtr->type == "yarn") {
            initYarn(ropeParamsPtr);
        } else {
            printf("Unsupported type=%s\n", ropeParamsPtr->type.c_str());
            exit(-2);
        }
    } else if (ropeDim != this->invFreqSize * 2) {
        printf("Incorrect dim=%d, invFreqSize=%d\n", ropeDim, this->invFreqSize);
        exit(-1);
    }
}

void DeekSeekV2RotaryEmbedding::initYarn(const RopeParams *ropeParamsPtr) {
    int low, high;
    yarnFindRange(low, high, ropeParamsPtr->betaFast, ropeParamsPtr->betaSlow, ropeDim, ropeParamsPtr->base,
            ropeParamsPtr->orgMaxPosEmbed);

    float *inv_freq_mask = (float *)malloc(invFreqSize * sizeof(float));
    yarnLinearRampMask(inv_freq_mask, low, high, invFreqSize, ropeParamsPtr->extraPolFactor);
    float *inv_freq = (float *)malloc(invFreqSize * sizeof(float));
    for (size_t i = 0; i < invFreqSize; i++) {
        float freq_extra = 1.0 / pow(ropeParamsPtr->base, float(i * 2) / ropeDim);
        inv_freq[i]
                = freq_extra / ropeParamsPtr->scale * (1 - inv_freq_mask[i]) + freq_extra * inv_freq_mask[i];
    }
    float scale = yarnGetMscale(ropeParamsPtr->scale, ropeParamsPtr->mscale);
    scale /= yarnGetMscale(ropeParamsPtr->scale, ropeParamsPtr->mscaleAllDim);
    xft::llamaSetCosSinCache(inv_freq, embCos, embSin, invFreqSize, maxPositionEmbeddings, scale);
    free(inv_freq_mask);
    free(inv_freq);
}

float DeekSeekV2RotaryEmbedding::yarnGetMscale(float scale, float mscale) {
    if (scale <= 1) return 1.0;
    return 0.1 * mscale * std::log(scale) + 1.0;
}

void DeekSeekV2RotaryEmbedding::yarnFindRange(
        int &low, int &high, int betaFast, int betaSlow, int dim, float base, int orgMaxPosEmbed) {
    float flow = dim * std::log(orgMaxPosEmbed / (betaFast * 2 * M_PI)) / (2 * std::log(base));
    float fhigh = dim * std::log(orgMaxPosEmbed / (betaSlow * 2 * M_PI)) / (2 * std::log(base));
    low = std::max(int(floor(flow)), 0);
    high = std::min(int(ceil(fhigh)), dim - 1);
}

void DeekSeekV2RotaryEmbedding::yarnLinearRampMask(float *invFreqMask, int low, int high, int dim, float extraFactor) {
    float min = low, max = high;
    if (min == max) max += 0.001;

    for (int i = 0; i < dim; ++i) {
        invFreqMask[i] = ((float)i - min) / (max - min);
    }

    for (int i = 0; i < dim; ++i) {
        invFreqMask[i] = (1.0 - std::clamp(invFreqMask[i], 0.0f, 1.0f)) * extraFactor;
    }
}

// For continuous batching
void DeekSeekV2RotaryEmbedding::forward(
        float *query, float *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads, int *positionIds) {
    xft::deepseekv2ApplyRotaryPosEmbed(
            query, key, embCos, embSin, qStride, kStride, invFreqSize,
            totSeqLen, qHeads, kHeads, positionIds, nopeDim);
}

void DeekSeekV2RotaryEmbedding::forward(bfloat16_t *query, bfloat16_t *key, int totSeqLen, int qStride, int kStride,
        int qHeads, int kHeads, int *positionIds) {
    xft::deepseekv2ApplyRotaryPosEmbed(
            query, key, embCos, embSin, qStride, kStride, invFreqSize,
            totSeqLen, qHeads, kHeads, positionIds, nopeDim);
}

void DeekSeekV2RotaryEmbedding::forward(float16_t *query, float16_t *key, int totSeqLen, int qStride, int kStride,
        int qHeads, int kHeads, int *positionIds) {
    xft::deepseekv2ApplyRotaryPosEmbed(
            query, key, embCos, embSin, qStride, kStride, invFreqSize,
            totSeqLen, qHeads, kHeads, positionIds, nopeDim);
}