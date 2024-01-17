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
#include "yarn_scaled_rotary_embedding.h"

#include "compile_util.h"

static int maxSeqLenCached = -1;
static int invFreqSize = -1;
static float *invFreq;
static float *embCos = nullptr;
static float *embSin = nullptr;

bool LlamaYaRNScaledRotaryEmbedding::initialized = false;

// dim: equals to head size
LlamaYaRNScaledRotaryEmbedding::LlamaYaRNScaledRotaryEmbedding(
        const int dim, const int maxPosEmbed, const RopeParams *ropeParamsPtr) {
    // skip the init of parent class
    if (ropeParamsPtr == nullptr) return;
    if (!initialized) {
        initialized = true;

        maxSeqLenCached = maxPosEmbed;
        invFreqSize = (dim + 1) / 2;
        // assert ropeParam in Context
        assert(ropeParamsPtr->type == "yarn");

        int low, high;
        yarnFindRange(low, high, ropeParamsPtr->betaFast, ropeParamsPtr->betaSlow, dim, ropeParamsPtr->base,
                ropeParamsPtr->orgMaxPosEmbed);

        float *invFreqMask = (float *)malloc(invFreqSize * sizeof(float));
        yarnLinearRampMask(invFreqMask, low, high, invFreqSize, ropeParamsPtr->extraPolFactor);

        invFreq = (float *)malloc(invFreqSize * sizeof(float));
        for (size_t i = 0; i < invFreqSize; i++) {
            invFreq[i] = 1.0 / pow(ropeParamsPtr->base, float(i * 2) / dim);
            invFreq[i] = invFreq[i] / ropeParamsPtr->scale * (1 - invFreqMask[i]) + invFreq[i] * invFreqMask[i];
        }
        free(invFreqMask);

        yarnLlamaCalEmb(ropeParamsPtr->scale, ropeParamsPtr->attnFactor);
    } else if (dim != invFreqSize * 2) {
        printf("Incorrect dim=%d, inv_freq_size=%d\n", dim, invFreqSize);
        exit(-1);
    }
};

void LlamaYaRNScaledRotaryEmbedding::yarnFindRange(
        int &low, int &high, int betaFast, int betaSlow, int dim, float base, int orgMaxPosEmbed) {
    float flow = dim * std::log(orgMaxPosEmbed / (betaFast * 2 * M_PI)) / (2 * std::log(base));
    float fhigh = dim * std::log(orgMaxPosEmbed / (betaSlow * 2 * M_PI)) / (2 * std::log(base));
    low = std::max(int(floor(flow)), 0);
    high = std::min(int(ceil(fhigh)), dim - 1);
}

void LlamaYaRNScaledRotaryEmbedding::yarnLinearRampMask(
        float *invFreqMask, int low, int high, int dim, float extraFactor) {
    float min = low, max = high;
    if (min == max) max += 0.001;

    for (int i = 0; i < dim; ++i) {
        invFreqMask[i] = ((float)i - min) / (max - min);
    }

    for (int i = 0; i < dim; ++i) {
        invFreqMask[i] = (1.0 - std::clamp(invFreqMask[i], 0.0f, 1.0f)) * extraFactor;
    }
}

void LlamaYaRNScaledRotaryEmbedding::yarnLlamaCalEmb(float scale, float attnFactor) {
    float mscale;
    if (scale <= 1)
        mscale = 1.0;
    else
        mscale = 0.1 * std::log(scale) + 1.0;
    mscale *= attnFactor;

    embCos = (float *)aligned_alloc(64, maxSeqLenCached * (invFreqSize * 2) * sizeof(float));
    embSin = (float *)aligned_alloc(64, maxSeqLenCached * (invFreqSize * 2) * sizeof(float));

#pragma omp parallel for
    for (size_t i = 0; i < maxSeqLenCached; i++) {
        float *pcos = embCos + i * invFreqSize * 2;
        float *psin = embSin + i * invFreqSize * 2;

        for (size_t j = 0; j < invFreqSize; j++) {
            float tmp = i * invFreq[j];
            float cosTmp = std::cos(tmp) * mscale;
            float sinTmp = std::sin(tmp) * mscale;

            pcos[j] = cosTmp;
            pcos[j + invFreqSize] = cosTmp;
            psin[j] = sinTmp;
            psin[j + invFreqSize] = sinTmp;
        }
    }
}

// def rotate_half(x):
//     """Rotates half the hidden dims of the input."""
//     x1 = x[..., : x.shape[-1] // 2]
//     x2 = x[..., x.shape[-1] // 2 :]
//     return torch.cat((-x2, x1), dim=-1)
// def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
//     # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
//     cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
//     sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
//     cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
//     sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
//     q_embed = (q * cos) + (rotate_half(q) * sin)
//     k_embed = (k * cos) + (rotate_half(k) * sin)
//     return q_embed, k_embed
//
// qk_shape: 4 values of [batch_size, seq_len, head_num, head_size]
// position_ids: an array in the size of seq_len
// query and key is the matrix like below:
//
// |<------------------------------ head_size * head_num --------------------------------->|
// |_head_size|___________________________________________________________________________________
// |          |          |          |          |          |          |          |          |    ^
// |          |          |          |          |          |          |          |          |    |
// |          |          |          |          |          |          |          |          | bs*seq_len
// |          |          |          |          |          |          |          |          |    |
// |          |          |          |          |          |          |          |          |    |
// |__________|__________|__________|__________|__________|__________|__________|__________|____v__
void LlamaYaRNScaledRotaryEmbedding::forward(
        float *query, float *key, int qStride, int kStride, const int *qkShape, const int *positionIds) {
    int dim = invFreqSize * 2;
    REQUIRES(dim == qkShape[3], "Incorrect shape, this dimention is not the head size.");

    const int batchSize = qkShape[0];
    const int seqLen = qkShape[1];
    const int qHeads = qkShape[2];
    const int kHeads = qkShape[4];
    const int heads = std::max(qHeads, kHeads);
    const int half = invFreqSize;

    // for (size_t i = 0; i < emb_size; i++) {
    //     emb[i] = x[i] * emb_cos[position_ids[i % cached_size / dim]][i % dim];
    //     int offset = (i % dim + inv_freq_size) % dim;
    //     float sign = ((offset < inv_freq_size) * 1) + ((offset >= inv_freq_size) * -1);
    //     emb[i] += x[(i - i % dim) + offset] * sign * emb_sin[position_ids[i % cached_size / dim]][i % dim];
    // }
#pragma omp parallel for collapse(3)
    for (int head = 0; head < heads; ++head) {
        for (int bs = 0; bs < batchSize; ++bs) {
            for (int seq = 0; seq < seqLen; ++seq) {
                int pos = positionIds[seq];
                float *pcos = embCos + pos * dim;
                float *psin = embSin + pos * dim;

                float *q = query + bs * seqLen * qStride + seq * qStride + head * dim;
                float *k = key + bs * seqLen * kStride + seq * kStride + head * dim;
#pragma omp simd
                for (int i = 0; i < half; ++i) {
                    if (head < qHeads) {
                        auto q1 = q[i];
                        q[i] = q[i] * pcos[i] - q[i + half] * psin[i];
                        q[i + half] = q[i + half] * pcos[i + half] + q1 * psin[i + half];
                    }
                    if (head < kHeads) {
                        auto k1 = k[i];
                        k[i] = k[i] * pcos[i] - k[i + half] * psin[i];
                        k[i + half] = k[i + half] * pcos[i + half] + k1 * psin[i + half];
                    }
                }
            }
        }
    }
}
