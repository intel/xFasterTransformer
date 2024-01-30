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
#include "rotary_embedding.h"

#include "compile_util.h"

static int max_seq_len_cached = -1;
static int inv_freq_size = -1;
static float *inv_freq;
static float *emb_cos = nullptr;
static float *emb_sin = nullptr;

bool LlamaRotaryEmbedding::initialized = false;

// dim: equals to head size
LlamaRotaryEmbedding::LlamaRotaryEmbedding(const int dim, const int max_position_embeddings, const float base) {
    if (!initialized) {
        initialized = true;

        max_seq_len_cached = max_position_embeddings;
        inv_freq_size = (dim + 1) / 2;
        inv_freq = (float *)malloc(inv_freq_size * sizeof(float));
        for (size_t i = 0; i < inv_freq_size; i++) {
            inv_freq[i] = 1.0 / pow(base, float(i * 2) / dim);
        }

        llamaCalEmb();
    } else if (dim != inv_freq_size * 2) {
        printf("Incorrect dim=%d, inv_freq_size=%d\n", dim, inv_freq_size);
        exit(-1);
    }
};

void LlamaRotaryEmbedding::llamaCalEmb() {
    emb_cos = (float *)aligned_alloc(64, max_seq_len_cached * (inv_freq_size * 2) * sizeof(float));
    emb_sin = (float *)aligned_alloc(64, max_seq_len_cached * (inv_freq_size * 2) * sizeof(float));

#pragma omp parallel for
    for (size_t i = 0; i < max_seq_len_cached; i++) {
        float *pcos = emb_cos + i * inv_freq_size * 2;
        float *psin = emb_sin + i * inv_freq_size * 2;

        for (size_t j = 0; j < inv_freq_size; j++) {
            float tmp = i * inv_freq[j];
            float cos_tmp = std::cos(tmp);
            float sin_tmp = std::sin(tmp);

            pcos[j] = cos_tmp;
            pcos[j + inv_freq_size] = cos_tmp;
            psin[j] = sin_tmp;
            psin[j + inv_freq_size] = sin_tmp;
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
void LlamaRotaryEmbedding::forward(
        float *query, float *key, int qStride, int kStride, const int *qkShape, const int *positionIds) {
    int dim = inv_freq_size * 2;
    REQUIRES(dim == qkShape[3], "Incorrect shape, this dimention is not the head size.");

    const int batchSize = qkShape[0];
    const int seqLen = qkShape[1];
    const int qHeads = qkShape[2];
    const int kHeads = qkShape[4];
    const int heads = std::max(qHeads, kHeads);
    const int half = inv_freq_size;

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
                float *pcos = emb_cos + pos * dim;
                float *psin = emb_sin + pos * dim;

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

void LlamaRotaryEmbedding::forward(
        bfloat16_t *query, bfloat16_t *key, int qStride, int kStride, const int *qkShape, const int *positionIds) {
    int dim = inv_freq_size * 2;
    REQUIRES(dim == qkShape[3], "Incorrect shape, this dimention is not the head size.");

    const int batchSize = qkShape[0];
    const int seqLen = qkShape[1];
    const int qHeads = qkShape[2];
    const int kHeads = qkShape[4];
    const int heads = std::max(qHeads, kHeads);
    const int half = inv_freq_size;

#pragma omp parallel for collapse(3)
    for (int head = 0; head < heads; ++head) {
        for (int bs = 0; bs < batchSize; ++bs) {
            for (int seq = 0; seq < seqLen; ++seq) {
                int pos = positionIds[seq];
                float *pcos = emb_cos + pos * dim;
                float *psin = emb_sin + pos * dim;

                bfloat16_t *q = query + bs * seqLen * qStride + seq * qStride + head * dim;
                bfloat16_t *k = key + bs * seqLen * kStride + seq * kStride + head * dim;

                // Process chunks of 16 elements at a time
                for (int i = 0; i < half; i += 16) {
                    int remain = half - i;
                    __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                    __m512 pCosVec = _mm512_maskz_loadu_ps(mask, &pcos[i]);
                    __m512 pCosHalfVec = _mm512_maskz_loadu_ps(mask, &pcos[i + half]);
                    __m512 pSinVec = _mm512_maskz_loadu_ps(mask, &psin[i]);
                    __m512 pSinHalfVec = _mm512_maskz_loadu_ps(mask, &psin[i + half]);

                    // Compute something like:
                    // q[i] = q[i] * pcos[i] - q[i + half] * psin[i];
                    // q[i + half] = q[i + half] * pcos[i + half] + q[i] * psin[i + half];
                    if (head < qHeads) {
                        __m512 qVec = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &q[i]));
                        __m512 qHalfVec = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &q[i + half]));
                        __m512 qNew = _mm512_fmsub_ps(qVec, pCosVec, _mm512_mul_ps(qHalfVec, pSinVec));
                        __m512 qHalfNew = _mm512_fmadd_ps(qHalfVec, pCosHalfVec, _mm512_mul_ps(qVec, pSinHalfVec));
                        _mm256_mask_storeu_epi16(&q[i], mask, bfloat16_t::cvt_fp32_to_bf16(qNew));
                        _mm256_mask_storeu_epi16(&q[i + half], mask, bfloat16_t::cvt_fp32_to_bf16(qHalfNew));
                    }

                    if (head < kHeads) {
                        __m512 kVec = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &k[i]));
                        __m512 kHalfVec = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &k[i + half]));
                        __m512 kNew = _mm512_fmsub_ps(kVec, pCosVec, _mm512_mul_ps(kHalfVec, pSinVec));
                        __m512 kHalfNew = _mm512_fmadd_ps(kHalfVec, pCosHalfVec, _mm512_mul_ps(kVec, pSinHalfVec));
                        _mm256_mask_storeu_epi16(&k[i], mask, bfloat16_t::cvt_fp32_to_bf16(kNew));
                        _mm256_mask_storeu_epi16(&k[i + half], mask, bfloat16_t::cvt_fp32_to_bf16(kHalfNew));
                    }
                }
            }
        }
    }
}