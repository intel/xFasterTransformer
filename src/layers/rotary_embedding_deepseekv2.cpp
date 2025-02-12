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
    const std::string emb_cos_str_prefix = "deepseek_emb_cos";
    const std::string emb_sin_str_prefix = "deepseek_emb_sin";

    bool initialized = false;
    float base = -1;
    int rope_ratio = 1;

    this->dim = ctx->ropeDim;
    this->max_position_embeddings = ctx->maxPosEmbed;
    this->inv_freq_size = (dim + 1) / 2;
    const RopeParams *ropeParamsPtr = ctx->ropeParamsPtr;
    if (ropeParamsPtr == nullptr) return;
    // assert ropeParam in Context
    assert(ropeParamsPtr->type == "yarn");

    std::string emb_cos_str = emb_cos_str_prefix + "_" + ropeParamsPtr->type;
    std::string emb_sin_str = emb_sin_str_prefix + "_" + ropeParamsPtr->type;

    if (ctx->cached(emb_cos_str)) { initialized = true; }

    this->emb_cos = ctx->getBuffer<float>(emb_cos_str, max_position_embeddings * inv_freq_size * 2);
    this->emb_sin = ctx->getBuffer<float>(emb_sin_str, max_position_embeddings * inv_freq_size * 2);
    // printf("max_position_embeddings=%d, dim=%d, inv_freq_size=%d\n", max_position_embeddings, dim, inv_freq_size);

    if (!initialized) {
        initialized = true;
        if (ropeParamsPtr->type == "yarn") {
            initYarn(ropeParamsPtr);
#ifdef _DS_ROTARY_DEBUG_
            test();
#endif // _DS_ROTARY_DEBUG_
        } else {
            printf("Unsupported type=%s\n", ropeParamsPtr->type.c_str());
            exit(-2);
        }
    } else if (dim != this->inv_freq_size * 2) {
        printf("Incorrect dim=%d, inv_freq_size=%d\n", dim, this->inv_freq_size);
        exit(-1);
    }
}

#ifdef _DS_ROTARY_DEBUG_
void DeekSeekV2RotaryEmbedding::test() {
    int qHeads = 128;
    int kHeads = 1;
    int qStride = 64;
    int kStride = 64;
    int totInSeqLen = 4;
    int position_ids[4] = {0, 1, 2, 3};
    int size = totInSeqLen * kStride;
    float query[1 * qHeads * size] = {0};
    float key[1 * kHeads * size] = {-0.1643, 0.3330, 0.7862, 1.3314, 0.9643, 1.0741, 0.9989, -0.3220, -1.1456, 0.1899, -0.5767,
            1.1961, 0.5975, 0.6341, 0.8131, -0.2354, -0.1308, 0.7986, 0.9103, 1.2525, -2.1596, 0.5284, -0.0414, 0.8128,
            -0.2848, 0.2458, -1.5502, 0.8092, -1.7414, 0.3593, 0.3483, -0.2657, 0.0383, -1.1891, -2.7815, 0.1983,
            0.4790, -0.6628, 0.3901, 1.0671, 0.7974, 1.5437, 1.0416, 0.5651, 0.0830, 0.6451, 2.8310, -1.2749, 0.6692,
            -0.4510, -2.5409, 0.4628, 1.6221, 0.0329, 0.5014, 1.9395, -2.5793, -6.9318, -1.2709, -4.2719, 2.4604,
            -2.3658, 6.2353, -3.8382, 2.5141, 0.5135, -0.9972, 1.9395, -0.4488, 0.1512, -2.3239, 0.2964, 1.2265,
            -1.9953, 1.9795, -1.8108, 1.1441, -1.7430, 0.3978, 1.2690, 1.3218, -0.2237, -0.4944, 0.7673, -0.5980,
            0.0729, 0.2531, -0.4006, -1.6054, -0.0733, -0.0621, -0.7856, -1.4979, -0.4555, 0.2672, 0.5128, 0.9324,
            -0.5812, -1.0707, 1.1672, -0.6723, -1.1052, 0.4626, -0.0213, -0.2103, 2.0959, 0.8914, 0.1753, 0.5183,
            -0.0707, 0.4267, 1.4086, -0.2222, -0.4101, 0.0736, 1.5434, -0.2661, 0.1780, 0.8560, -1.8014, 0.9748,
            -1.1361, -0.6257, -1.2926, 0.3565, 2.9209, -2.5979, 0.9305, 1.9866, -0.2056, -1.0562, 1.6671, -0.6728,
            1.0173, -1.6868, 0.6089, 0.5797, -1.0960, 0.5725, -0.4312, 0.5870, -1.7321, 0.5408, 0.4553, 0.7293, 0.3024,
            -0.2002, 1.2927, 0.8355, 0.5003, 0.8769, 0.1872, -0.9717, 1.1202, 0.0872, 0.1613, -0.3936, 1.0500, 0.2107,
            0.0614, -0.6115, -0.2237, -1.4662, 0.7227, 0.3783, -0.4672, -0.0518, -0.0093, 0.2649, -0.6475, 0.1529,
            0.0984, 0.7707, 0.5207, 0.1501, 0.1223, -0.3678, -0.3164, -1.1864, 0.5413, 1.0936, -0.7345, -0.6268, 0.0995,
            -1.2786, -0.3070, 2.2235, 1.3397, 0.2993, 0.1789, -0.8245, 1.1018, 2.3845, 0.2943, -0.7233, 1.9675, -0.9256,
            -0.3503, -2.4593, 0.2645, 1.3441, -1.8189, 1.1882, -2.0555, 1.0467, -1.3211, 0.2701, 0.8963, 1.4127,
            -0.6718, 0.0529, 1.1852, -0.3097, 0.1192, 0.9529, -0.2667, -1.8352, 0.8040, 0.4957, -0.6748, -0.5061,
            -0.6760, -0.3526, 1.1586, -1.0376, 0.2851, -1.2863, 1.2476, -0.9885, 0.3816, 0.0856, -0.3509, 0.4306,
            0.9960, 0.3778, 0.0582, -0.2763, 0.3970, -1.1822, 1.0903, -0.7010, -0.3043, -0.0295, 1.2954, -0.0391,
            -0.7410, 2.1768, -3.2658, 1.6495, -1.0283, 0.0767, -1.1388, 1.9509, 4.8640, -3.9358, -0.0995};
    for (int i=0;i<qHeads;i++) {
        memcpy(query+i*size, key, size);
    }
    // rope.forward(
    //     query, key, totInSeqLen, qBuffer.Stride(), loraBuffer.Stride(), ctx->attHeadNum, 1, posIds.data())
    printf("start DeekSeekV2RotaryEmbedding::forward\n");
    forward(query, key, totInSeqLen, qStride, kStride, qHeads, kHeads, position_ids);
    printf("finished DeekSeekV2RotaryEmbedding::forward\n");
    printf("key_res=[");
    for(int i=0;i<totInSeqLen;i++) {
        printf("[");
        for(int j=0;j<kStride;j++) {
            printf("%f, ", key[i*kStride+j]);
        }
        printf("]\n");
    }
    printf("]\n");
}
#endif // _DS_ROTARY_DEBUG_

void DeekSeekV2RotaryEmbedding::initYarn(const RopeParams *ropeParamsPtr) {
    int low, high;
    yarnFindRange(low, high, ropeParamsPtr->betaFast, ropeParamsPtr->betaSlow, dim, ropeParamsPtr->base,
            ropeParamsPtr->orgMaxPosEmbed);

    float *inv_freq_mask = (float *)malloc(inv_freq_size * sizeof(float));
    yarnLinearRampMask(inv_freq_mask, low, high, inv_freq_size, ropeParamsPtr->extraPolFactor);
    float *inv_freq = (float *)malloc(inv_freq_size * sizeof(float));
    for (size_t i = 0; i < inv_freq_size; i++) {
        float freq_extra = 1.0 / pow(ropeParamsPtr->base, float(i * 2) / dim);
        inv_freq[i]
                = freq_extra / ropeParamsPtr->scale * (1 - inv_freq_mask[i]) + freq_extra * inv_freq_mask[i];
    }
    float scale = yarnGetMscale(ropeParamsPtr->scale, ropeParamsPtr->mscale);
    scale /= yarnGetMscale(ropeParamsPtr->scale, ropeParamsPtr->mscaleAllDim);
    xft::llamaSetCosSinCache(inv_freq, emb_cos, emb_sin, inv_freq_size, max_position_embeddings, scale);
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
            query, key, emb_cos, emb_sin, qStride, kStride, inv_freq_size, totSeqLen, qHeads, kHeads, positionIds);
}

void DeekSeekV2RotaryEmbedding::forward(bfloat16_t *query, bfloat16_t *key, int totSeqLen, int qStride, int kStride,
        int qHeads, int kHeads, int *positionIds) {
    xft::deepseekv2ApplyRotaryPosEmbed(
            query, key, emb_cos, emb_sin, qStride, kStride, inv_freq_size, totSeqLen, qHeads, kHeads, positionIds);
}

void DeekSeekV2RotaryEmbedding::forward(float16_t *query, float16_t *key, int totSeqLen, int qStride, int kStride,
        int qHeads, int kHeads, int *positionIds) {
    xft::deepseekv2ApplyRotaryPosEmbed(
            query, key, emb_cos, emb_sin, qStride, kStride, inv_freq_size, totSeqLen, qHeads, kHeads, positionIds);
}