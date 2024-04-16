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
#include "rotary_embedding_chatglm2.h"

#include "allocator.h"
#include "compile_util.h"

static int max_seq_len_cached = -1;
static int inv_freq_size = -1;
static float *inv_freq;
static float *emb_cos = nullptr;
static float *emb_sin = nullptr;

bool ChatGLM2RotaryEmbedding::initialized = false;

// dim: equals to head size
ChatGLM2RotaryEmbedding::ChatGLM2RotaryEmbedding(const int dim, const int max_position_embeddings, const float base) {
    if (!initialized) {
        initialized = true;

        max_seq_len_cached = max_position_embeddings;
        inv_freq_size = (dim + 1) / 2;
        inv_freq = (float *)malloc(inv_freq_size * sizeof(float));
#pragma omp parallel for
        for (size_t i = 0; i < inv_freq_size; i++) {
            inv_freq[i] = 1.0 / pow(base, float(i * 2) / dim);
        }

        glm2CalEmb();
    } else if (dim != inv_freq_size * 2) {
        printf("Incorrect dim=%d, inv_freq_size=%d\n", dim, inv_freq_size);
        exit(-1);
    }
};

void ChatGLM2RotaryEmbedding::glm2CalEmb() {
    emb_cos = (float *)xft::alloc(max_seq_len_cached * (inv_freq_size * 2) * sizeof(float));
    emb_sin = (float *)xft::alloc(max_seq_len_cached * (inv_freq_size * 2) * sizeof(float));

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

// def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
// #x : [sq, b, np, hn]
//     sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
//     rot_dim = rope_cache.shape[-2] * 2
//     x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
// #truncate to support variable sizes
//     rope_cache = rope_cache[:sq]
//     xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
//     rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
//     print('### rope_cache={}, x={}, sq={}, b={}, np={}, hn={}'.format(rope_cache.shape, xshaped.shape, sq, b, np, hn))
//     x_out2 = torch.stack(
//         [
//             xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
//             xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
//         ],
//         -1,
//     )
//     x_out2 = x_out2.flatten(3)
//     return torch.cat((x_out2, x_pass), dim=-1)

void ChatGLM2RotaryEmbedding::forward(
        float *query, float *key, int qStride, int kStride, const int *qk_shape, const int *position_ids) {
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
                float *p1 = query + off;

                int pos = position_ids[seq];
                float *pcos = emb_cos + pos * dim;
                float *psin = emb_sin + pos * dim;

#pragma omp simd
                for (int i = 0; i < half; i += 2) {
                    auto t1 = p1[i];
                    p1[i] = p1[i] * pcos[i] - p1[i + 1] * psin[i];
                    p1[i + 1] = p1[i + 1] * pcos[i] + t1 * psin[i];
                }
                off += qStride;
            }
        }
    }
}

inline void ChatGLM2RotaryEmbedding::prepare_sincos(__m512 a, __m512 b, __m512 *result) {
    const __m512i mask = _mm512_set_epi32(
            0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00);

    *result = _mm512_permutex2var_ps(a, mask, b);
}

inline void ChatGLM2RotaryEmbedding::interleave_qk(__m512 a, __m512 b, __m512 *result0, __m512 *result1) {
    const __m512i mask0 = _mm512_set_epi32(
            0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00);

    const __m512i mask1 = _mm512_set_epi32(
            0x1f, 0x1d, 0x1b, 0x19, 0x17, 0x15, 0x13, 0x11, 0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01);

    *result0 = _mm512_permutex2var_ps(a, mask0, b);
    *result1 = _mm512_permutex2var_ps(a, mask1, b);
}

inline void ChatGLM2RotaryEmbedding::deinterleave_qk(__m512 a, __m512 b, __m512 *result0, __m512 *result1) {
    const __m512i mask0 = _mm512_set_epi32(
            0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14, 0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01, 0x10, 0x00);

    const __m512i mask1 = _mm512_set_epi32(
            0x1f, 0x0f, 0x1e, 0x0e, 0x1d, 0x0d, 0x1c, 0x0c, 0x1b, 0x0b, 0x1a, 0x0a, 0x19, 0x09, 0x18, 0x08);

    *result0 = _mm512_permutex2var_ps(a, mask0, b);
    *result1 = _mm512_permutex2var_ps(a, mask1, b);
}

void ChatGLM2RotaryEmbedding::forward(
        bfloat16_t *query, bfloat16_t *key, int qStride, int kStride, const int *qk_shape, const int *position_ids) {
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
                bfloat16_t *pBF = query + off;

                int pos = position_ids[seq];
                float *pcos = emb_cos + pos * dim;
                float *psin = emb_sin + pos * dim;

                for (int i = 0; i < half; i += 32) {
                    __mmask16 mask = 0xffff;
                    __m512 tmp0, tmp1, pCosVec, pSinVec, qVec0, qVec1;
                    //TODO:  can directly load/save with shuffle??
                    tmp0 = _mm512_maskz_loadu_ps(mask, &pcos[i]);
                    tmp1 = _mm512_maskz_loadu_ps(mask, &pcos[i + 16]);
                    prepare_sincos(tmp0, tmp1, &pCosVec);

                    tmp0 = _mm512_maskz_loadu_ps(mask, &psin[i]);
                    tmp1 = _mm512_maskz_loadu_ps(mask, &psin[i + 16]);
                    prepare_sincos(tmp0, tmp1, &pSinVec);

                    tmp0 = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &pBF[i]));
                    tmp1 = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &pBF[i + 16]));

                    interleave_qk(tmp0, tmp1, &qVec0, &qVec1);

                    __m512 qNew0 = _mm512_fmsub_ps(qVec0, pCosVec, _mm512_mul_ps(qVec1, pSinVec));
                    __m512 qNew1 = _mm512_fmadd_ps(qVec0, pSinVec, _mm512_mul_ps(qVec1, pCosVec));

                    deinterleave_qk(qNew0, qNew1, &tmp0, &tmp1);

                    _mm256_mask_storeu_epi16(&pBF[i], mask, bfloat16_t::cvt_fp32_to_bf16(tmp0));
                    _mm256_mask_storeu_epi16(&pBF[i + 16], mask, bfloat16_t::cvt_fp32_to_bf16(tmp1));
                }
                off += qStride;
            }
        }
    }
}
