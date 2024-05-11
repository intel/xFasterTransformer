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
#include "rotary_embedding_kernels.h"
#include "intrinsics_util.h"

namespace xft {

void llamaSetCosSinCache(
        const float *invFreq, float *embCos, float *embSin, int invFreqSize, int maxPositionEmbeddings, float scale) {

#pragma omp parallel for
    for (size_t i = 0; i < maxPositionEmbeddings; i++) {
        float *pcos = embCos + i * invFreqSize;
        float *psin = embSin + i * invFreqSize;

        for (size_t j = 0; j < invFreqSize; j++) {
            float tmp = i * invFreq[j];
            float cosTmp = std::cos(tmp) * scale;
            float sinTmp = std::sin(tmp) * scale;

            pcos[j] = cosTmp;
            psin[j] = sinTmp;
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

void llamaApplyRotaryPosEmbed(float *query, float *key, float *emb_cos, float *emb_sin, int qStride, int kStride,
        int dim, int totSeqLen, int qHeads, int kHeads, const int *positionIds) {
    const int half = (dim + 1) / 2;
    const int heads = std::max(qHeads, kHeads);

#pragma omp parallel for collapse(2)
    for (int head = 0; head < heads; ++head) {
        for (int seq = 0; seq < totSeqLen; ++seq) {
            int pos = positionIds[seq];
            float *pcos = emb_cos + pos * half;
            float *psin = emb_sin + pos * half;

            float *q = query + seq * qStride + head * dim;
            float *k = key + seq * kStride + head * dim;

#pragma omp simd
            for (int i = 0; i < half; ++i) {
                if (head < qHeads) {
                    auto q1 = q[i];
                    q[i] = q1 * pcos[i] - q[i + half] * psin[i];
                    q[i + half] = q[i + half] * pcos[i] + q1 * psin[i];
                }
                if (head < kHeads) {
                    auto k1 = k[i];
                    k[i] = k1 * pcos[i] - k[i + half] * psin[i];
                    k[i + half] = k[i + half] * pcos[i] + k1 * psin[i];
                }
            }
        }
    }
}

void llamaApplyRotaryPosEmbed(bfloat16_t *query, bfloat16_t *key, float *emb_cos, float *emb_sin, int qStride,
        int kStride, int dim, int totSeqLen, int qHeads, int kHeads, const int *positionIds) {
    const int half = (dim + 1) / 2;
    const int heads = std::max(qHeads, kHeads);

#pragma omp parallel for collapse(2)
    for (int head = 0; head < heads; ++head) {
        for (int seq = 0; seq < totSeqLen; ++seq) {
            int pos = positionIds[seq];
            float *pcos = emb_cos + pos * half;
            float *psin = emb_sin + pos * half;

            bfloat16_t *q = query + seq * qStride + head * dim;
            bfloat16_t *k = key + seq * kStride + head * dim;

            // Process chunks of 16 elements at a time
            for (int i = 0; i < half; i += 16) {
                int remain = half - i;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                __m512 pCosVec = _mm512_maskz_loadu_ps(mask, &pcos[i]);
                __m512 pSinVec = _mm512_maskz_loadu_ps(mask, &psin[i]);

                // Compute something like:
                // q[i] = q[i] * pcos[i] - q[i + half] * psin[i];
                // q[i + half] = q[i + half] * pcos[i] + q[i] * psin[i];
                if (head < qHeads) {
                    __m512 qVec = xft::load_avx512(mask, &q[i]);
                    __m512 qHalfVec = xft::load_avx512(mask, &q[i + half]);
                    __m512 qNew = _mm512_fmsub_ps(qVec, pCosVec, _mm512_mul_ps(qHalfVec, pSinVec));
                    __m512 qHalfNew = _mm512_fmadd_ps(qHalfVec, pCosVec, _mm512_mul_ps(qVec, pSinVec));
                    xft::store_avx512(&q[i], mask, qNew);
                    xft::store_avx512(&q[i + half], mask, qHalfNew);
                }

                if (head < kHeads) {
                    __m512 kVec = xft::load_avx512(mask, &k[i]);
                    __m512 kHalfVec = xft::load_avx512(mask, &k[i + half]);
                    __m512 kNew = _mm512_fmsub_ps(kVec, pCosVec, _mm512_mul_ps(kHalfVec, pSinVec));
                    __m512 kHalfNew = _mm512_fmadd_ps(kHalfVec, pCosVec, _mm512_mul_ps(kVec, pSinVec));
                    xft::store_avx512(&k[i], mask, kNew);
                    xft::store_avx512(&k[i + half], mask, kHalfNew);
                }
            }
        }
    }
}

} // namespace xft
