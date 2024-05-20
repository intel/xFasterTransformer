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
#include "compile_util.h"
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

template <typename T>
static inline void llamaApplyRotaryPosEmbeding(T *query, T *key, int qStride, int kStride, float *emb_cos,
        float *emb_sin, int inv_freq_size, const int *qkShape, const int *positionIds) {
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
                float *pcos = emb_cos + pos * half;
                float *psin = emb_sin + pos * half;

                T *q = query + bs * seqLen * qStride + seq * qStride + head * dim;
                T *k = key + bs * seqLen * kStride + seq * kStride + head * dim;

                // Process chunks of 16 elements at a time
                for (int i = 0; i < half; i += 16) {
                    int remain = half - i;
                    __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                    __m512 pCosVec = _mm512_maskz_loadu_ps(mask, &pcos[i]);
                    __m512 pSinVec = _mm512_maskz_loadu_ps(mask, &psin[i]);

                    // Compute something like:
                    // q[i] = q[i] * pcos[i] - q[i + half] * psin[i];
                    // q[i + half] = q[i + half] * pcos[i + half] + q[i] * psin[i + half];
                    if constexpr (std::is_same_v<T, float>) {
                        if (head < qHeads) {
                            __m512 qVec = _mm512_maskz_loadu_ps(mask, &q[i]);
                            __m512 qHalfVec = _mm512_maskz_loadu_ps(mask, &q[i + half]);
                            __m512 qNew = _mm512_fmsub_ps(qVec, pCosVec, _mm512_mul_ps(qHalfVec, pSinVec));
                            __m512 qHalfNew = _mm512_fmadd_ps(qHalfVec, pCosVec, _mm512_mul_ps(qVec, pSinVec));
                            _mm512_mask_storeu_ps(&q[i], mask, qNew);
                            _mm512_mask_storeu_ps(&q[i + half], mask, qHalfNew);
                        }

                        if (head < kHeads) {
                            __m512 kVec = _mm512_maskz_loadu_ps(mask, &k[i]);
                            __m512 kHalfVec = _mm512_maskz_loadu_ps(mask, &k[i + half]);
                            __m512 kNew = _mm512_fmsub_ps(kVec, pCosVec, _mm512_mul_ps(kHalfVec, pSinVec));
                            __m512 kHalfNew = _mm512_fmadd_ps(kHalfVec, pCosVec, _mm512_mul_ps(kVec, pSinVec));
                            _mm512_mask_storeu_ps(&k[i], mask, kNew);
                            _mm512_mask_storeu_ps(&k[i + half], mask, kHalfNew);
                        }
                    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                        if (head < qHeads) {
                            __m512 qVec = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &q[i]));
                            __m512 qHalfVec
                                    = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &q[i + half]));
                            __m512 qNew = _mm512_fmsub_ps(qVec, pCosVec, _mm512_mul_ps(qHalfVec, pSinVec));
                            __m512 qHalfNew = _mm512_fmadd_ps(qHalfVec, pCosVec, _mm512_mul_ps(qVec, pSinVec));
                            _mm256_mask_storeu_epi16(&q[i], mask, bfloat16_t::cvt_fp32_to_bf16(qNew));
                            _mm256_mask_storeu_epi16(&q[i + half], mask, bfloat16_t::cvt_fp32_to_bf16(qHalfNew));
                        }

                        if (head < kHeads) {
                            __m512 kVec = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &k[i]));
                            __m512 kHalfVec
                                    = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &k[i + half]));
                            __m512 kNew = _mm512_fmsub_ps(kVec, pCosVec, _mm512_mul_ps(kHalfVec, pSinVec));
                            __m512 kHalfNew = _mm512_fmadd_ps(kHalfVec, pCosVec, _mm512_mul_ps(kVec, pSinVec));
                            _mm256_mask_storeu_epi16(&k[i], mask, bfloat16_t::cvt_fp32_to_bf16(kNew));
                            _mm256_mask_storeu_epi16(&k[i + half], mask, bfloat16_t::cvt_fp32_to_bf16(kHalfNew));
                        }
                    } else if constexpr (std::is_same_v<T, float16_t>) {
                        if (head < qHeads) {
                            __m512 qVec = float16_t::cvt_fp16_to_fp32(_mm256_maskz_loadu_epi16(mask, &q[i]));
                            __m512 qHalfVec = float16_t::cvt_fp16_to_fp32(_mm256_maskz_loadu_epi16(mask, &q[i + half]));
                            __m512 qNew = _mm512_fmsub_ps(qVec, pCosVec, _mm512_mul_ps(qHalfVec, pSinVec));
                            __m512 qHalfNew = _mm512_fmadd_ps(qHalfVec, pCosVec, _mm512_mul_ps(qVec, pSinVec));
                            _mm256_mask_storeu_epi16(&q[i], mask, float16_t::cvt_fp32_to_fp16(qNew));
                            _mm256_mask_storeu_epi16(&q[i + half], mask, float16_t::cvt_fp32_to_fp16(qHalfNew));
                        }

                        if (head < kHeads) {
                            __m512 kVec = float16_t::cvt_fp16_to_fp32(_mm256_maskz_loadu_epi16(mask, &k[i]));
                            __m512 kHalfVec = float16_t::cvt_fp16_to_fp32(_mm256_maskz_loadu_epi16(mask, &k[i + half]));
                            __m512 kNew = _mm512_fmsub_ps(kVec, pCosVec, _mm512_mul_ps(kHalfVec, pSinVec));
                            __m512 kHalfNew = _mm512_fmadd_ps(kHalfVec, pCosVec, _mm512_mul_ps(kVec, pSinVec));
                            _mm256_mask_storeu_epi16(&k[i], mask, float16_t::cvt_fp32_to_fp16(kNew));
                            _mm256_mask_storeu_epi16(&k[i + half], mask, float16_t::cvt_fp32_to_fp16(kHalfNew));
                        }
                    } else {
                        printf("%s:%d: Need to define llamaApplyRotaryPosEmbeding kernel data type.\n", __FILE__,
                                __LINE__);
                        exit(-1);
                    }
                }
            }
        }
    }
}

void llamaApplyRotaryPosEmbeding(float *query, float *key, int qStride, int kStride, float *emb_cos, float *emb_sin,
        int inv_freq_size, const int *qkShape, const int *positionIds) {
    llamaApplyRotaryPosEmbeding<float>(
            query, key, qStride, kStride, emb_cos, emb_sin, inv_freq_size, qkShape, positionIds);
}

void llamaApplyRotaryPosEmbeding(bfloat16_t *query, bfloat16_t *key, int qStride, int kStride, float *emb_cos,
        float *emb_sin, int inv_freq_size, const int *qkShape, const int *positionIds) {
    llamaApplyRotaryPosEmbeding<bfloat16_t>(
            query, key, qStride, kStride, emb_cos, emb_sin, inv_freq_size, qkShape, positionIds);
}

void llamaApplyRotaryPosEmbeding(float16_t *query, float16_t *key, int qStride, int kStride, float *emb_cos,
        float *emb_sin, int inv_freq_size, const int *qkShape, const int *positionIds) {
    llamaApplyRotaryPosEmbeding<float16_t>(
            query, key, qStride, kStride, emb_cos, emb_sin, inv_freq_size, qkShape, positionIds);
}

template <typename T>
static inline void llamaApplyRotaryPosEmbed(T *query, T *key, float *emb_cos, float *emb_sin, int qStride, int kStride,
        int dim, int totSeqLen, int qHeads, int kHeads, const int *positionIds) {
    const int half = (dim + 1) / 2;
    const int heads = std::max(qHeads, kHeads);

#pragma omp parallel for collapse(2)
    for (int head = 0; head < heads; ++head) {
        for (int seq = 0; seq < totSeqLen; ++seq) {
            int pos = positionIds[seq];
            float *pcos = emb_cos + pos * half;
            float *psin = emb_sin + pos * half;

            T *q = query + seq * qStride + head * dim;
            T *k = key + seq * kStride + head * dim;

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

void llamaApplyRotaryPosEmbed(float *query, float *key, float *emb_cos, float *emb_sin, int qStride, int kStride,
        int dim, int totSeqLen, int qHeads, int kHeads, const int *positionIds) {
    llamaApplyRotaryPosEmbed<float>(
            query, key, emb_cos, emb_sin, qStride, kStride, dim, totSeqLen, qHeads, kHeads, positionIds);
}

void llamaApplyRotaryPosEmbed(bfloat16_t *query, bfloat16_t *key, float *emb_cos, float *emb_sin, int qStride,
        int kStride, int dim, int totSeqLen, int qHeads, int kHeads, const int *positionIds) {
    llamaApplyRotaryPosEmbed<bfloat16_t>(
            query, key, emb_cos, emb_sin, qStride, kStride, dim, totSeqLen, qHeads, kHeads, positionIds);
}

void llamaApplyRotaryPosEmbed(float16_t *query, float16_t *key, float *emb_cos, float *emb_sin, int qStride,
        int kStride, int dim, int totSeqLen, int qHeads, int kHeads, const int *positionIds) {
    llamaApplyRotaryPosEmbed<float16_t>(
            query, key, emb_cos, emb_sin, qStride, kStride, dim, totSeqLen, qHeads, kHeads, positionIds);
}


static inline void chatglm2PrepareSinCos(__m512 a, __m512 b, __m512 *result) {
    const __m512i mask = _mm512_set_epi32(
            0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00);

    *result = _mm512_permutex2var_ps(a, mask, b);
}

static inline void chatglm2InterleaveQK(__m512 a, __m512 b, __m512 *result0, __m512 *result1) {
    const __m512i mask0 = _mm512_set_epi32(
            0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00);

    const __m512i mask1 = _mm512_set_epi32(
            0x1f, 0x1d, 0x1b, 0x19, 0x17, 0x15, 0x13, 0x11, 0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01);

    *result0 = _mm512_permutex2var_ps(a, mask0, b);
    *result1 = _mm512_permutex2var_ps(a, mask1, b);
}

static inline void chatglm2DeinterleaveQK(__m512 a, __m512 b, __m512 *result0, __m512 *result1) {
    const __m512i mask0 = _mm512_set_epi32(
            0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14, 0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01, 0x10, 0x00);

    const __m512i mask1 = _mm512_set_epi32(
            0x1f, 0x0f, 0x1e, 0x0e, 0x1d, 0x0d, 0x1c, 0x0c, 0x1b, 0x0b, 0x1a, 0x0a, 0x19, 0x09, 0x18, 0x08);

    *result0 = _mm512_permutex2var_ps(a, mask0, b);
    *result1 = _mm512_permutex2var_ps(a, mask1, b);
}

template <typename T>
static inline void chatglm2ApplyRotaryPosEmbeding(T *query, T *key, int qStride, int kStride, float *emb_cos,
        float *emb_sin, int inv_freq_size, const int *qk_shape, const int *position_ids) {
    int dim = inv_freq_size * 2;
    REQUIRES(dim == qk_shape[3], "Incorrect shape, last dimention is not the head size.");
    const int batch_size = qk_shape[0];
    const int seq_len = qk_shape[1];
    const int head_num = qk_shape[2] + qk_shape[4];
    const int half = inv_freq_size;

#pragma omp parallel for
    for (int head = 0; head < head_num; ++head) {
        int off = head * dim;
        for (int bs = 0; bs < batch_size; ++bs) {
            for (int seq = 0; seq < seq_len; ++seq) {
                T *pF = query + off;

                int pos = position_ids[seq];
                float *pcos = emb_cos + pos * dim;
                float *psin = emb_sin + pos * dim;

                for (int i = 0; i < half; i += 32) {
                    __mmask16 mask = 0xffff;
                    __m512 tmp0, tmp1, pCosVec, pSinVec, qVec0, qVec1;
                    //TODO:  can directly load/save with shuffle??
                    tmp0 = _mm512_maskz_loadu_ps(mask, &pcos[i]);
                    tmp1 = _mm512_maskz_loadu_ps(mask, &pcos[i + 16]);
                    chatglm2PrepareSinCos(tmp0, tmp1, &pCosVec);

                    tmp0 = _mm512_maskz_loadu_ps(mask, &psin[i]);
                    tmp1 = _mm512_maskz_loadu_ps(mask, &psin[i + 16]);
                    chatglm2PrepareSinCos(tmp0, tmp1, &pSinVec);

                    if constexpr (std::is_same_v<T, float>) {
                        tmp0 = _mm512_maskz_loadu_ps(mask, &pF[i]);
                        tmp1 = _mm512_maskz_loadu_ps(mask, &pF[i + 16]);
                    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                        tmp0 = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &pF[i]));
                        tmp1 = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &pF[i + 16]));
                    } else if constexpr (std::is_same_v<T, float16_t>) {
                        tmp0 = float16_t::cvt_fp16_to_fp32(_mm256_maskz_loadu_epi16(mask, &pF[i]));
                        tmp1 = float16_t::cvt_fp16_to_fp32(_mm256_maskz_loadu_epi16(mask, &pF[i + 16]));
                    } else {
                        printf("%s:%d: Need to define chatglm2ApplyRotaryPosEmbeding kernel data type.\n", __FILE__,
                                __LINE__);
                        exit(-1);
                    }

                    chatglm2InterleaveQK(tmp0, tmp1, &qVec0, &qVec1);

                    __m512 qNew0 = _mm512_fmsub_ps(qVec0, pCosVec, _mm512_mul_ps(qVec1, pSinVec));
                    __m512 qNew1 = _mm512_fmadd_ps(qVec0, pSinVec, _mm512_mul_ps(qVec1, pCosVec));

                    chatglm2DeinterleaveQK(qNew0, qNew1, &tmp0, &tmp1);

                    if constexpr (std::is_same_v<T, float>) {
                        _mm512_mask_storeu_ps(&pF[i], mask, tmp0);
                        _mm512_mask_storeu_ps(&pF[i + 16], mask, tmp1);
                    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                        _mm256_mask_storeu_epi16(&pF[i], mask, bfloat16_t::cvt_fp32_to_bf16(tmp0));
                        _mm256_mask_storeu_epi16(&pF[i + 16], mask, bfloat16_t::cvt_fp32_to_bf16(tmp1));
                    } else if constexpr (std::is_same_v<T, float16_t>) {
                        _mm256_mask_storeu_epi16(&pF[i], mask, float16_t::cvt_fp32_to_fp16(tmp0));
                        _mm256_mask_storeu_epi16(&pF[i + 16], mask, float16_t::cvt_fp32_to_fp16(tmp1));
                    } else {
                        printf("%s:%d: Need to define chatglm2ApplyRotaryPosEmbeding kernel data type.\n", __FILE__,
                                __LINE__);
                        exit(-1);
                    }
                }
                off += qStride;
            }
        }
    }
}

void chatglm2ApplyRotaryPosEmbeding(float *query, float *key, int qStride, int kStride, float *emb_cos, float *emb_sin,
        int inv_freq_size, const int *qkShape, const int *positionIds) {
    chatglm2ApplyRotaryPosEmbeding<float>(
            query, key, qStride, kStride, emb_cos, emb_sin, inv_freq_size, qkShape, positionIds);
}

void chatglm2ApplyRotaryPosEmbeding(bfloat16_t *query, bfloat16_t *key, int qStride, int kStride, float *emb_cos,
        float *emb_sin, int inv_freq_size, const int *qkShape, const int *positionIds) {
    chatglm2ApplyRotaryPosEmbeding<bfloat16_t>(
            query, key, qStride, kStride, emb_cos, emb_sin, inv_freq_size, qkShape, positionIds);
}

void chatglm2ApplyRotaryPosEmbeding(float16_t *query, float16_t *key, int qStride, int kStride, float *emb_cos,
        float *emb_sin, int inv_freq_size, const int *qkShape, const int *positionIds) {
    chatglm2ApplyRotaryPosEmbeding<float16_t>(
            query, key, qStride, kStride, emb_cos, emb_sin, inv_freq_size, qkShape, positionIds);
}

} // namespace xft
