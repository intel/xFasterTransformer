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
#include "rope_2d.h"

#include "allocator.h"
#include "compile_util.h"
#include "float16.h"

static int max_seq_len_cached = -1;
static int inv_freq_size = -1;
static float *inv_freq;
static float *emb_cos = nullptr;
static float *emb_sin = nullptr;

bool RotaryEmbedding2D::initialized = false;

// dim: equals to head size, but in rotary_2d, we need to be a half of head size
RotaryEmbedding2D::RotaryEmbedding2D(const int _dim, const int max_position_embeddings, const float base) {
    int dim = _dim / 2;

    if (!initialized) {
        initialized = true;

        max_seq_len_cached = max_position_embeddings;
        inv_freq_size = (dim + 1) / 2;
        inv_freq = (float *)malloc(inv_freq_size * sizeof(float));
        for (size_t i = 0; i < inv_freq_size; i++) {
            inv_freq[i] = 1.0 / pow(base, float(i * 2) / dim);
        }

        // Convert to FP16 (align with ChatGLM)
        float16_t *buf = (float16_t *)malloc(inv_freq_size * sizeof(float16_t));
        float16_t::cvt_float_to_float16(inv_freq, buf, inv_freq_size);
        float16_t::cvt_float16_to_float(buf, inv_freq, inv_freq_size);
        free(buf);

        prepareEmbedding();
    } else if (dim != inv_freq_size * 2) {
        printf("Incorrect dim=%d, inv_freq_size=%d\n", dim, inv_freq_size);
        exit(-1);
    }
};

void RotaryEmbedding2D::prepareEmbedding() {
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

// FOR PYTHON CODE LIKE BELOW:
//
// q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
// k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
// cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
// position_ids, block_position_ids = position_ids[:, 0, :].transpose(0, 1).contiguous(), \
//     position_ids[:, 1, :].transpose(0, 1).contiguous()
// q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
// q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
// query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
// key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))
//
// def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
//     # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
//     cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
//         F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
//     q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
//     return q, k
//
// qk_shape: 4 values of [batch_size, seq_len, head_num, head_size]
// positions: position_ids + block_position_ids, with the size of 2*seq_len
// query and key is the matrix like below:
//
// |<------------------------------ head_size * head_num --------------------------------->|
// |_head_size|_____________________________________________________________________________  ____
// |          |          |          |          |          |          |          |          |    ^
// |          |          |          |          |          |          |          |          |    |
// |          |          |          |          |          |          |          |          | bs*seq_len
// |          |          |          |          |          |          |          |          |    |
// |          |          |          |          |          |          |          |          |    |
// |__________|__________|__________|__________|__________|__________|__________|__________|  __v__
void RotaryEmbedding2D::forward(
        float *query, float *key, int qStride, int kStride, const int *qk_shape, const int *positions) {
    int dim = inv_freq_size * 2;
    REQUIRES(dim * 2 == qk_shape[3], "Incorrect shape, last dimention is not the head size.");

    const int batch_size = qk_shape[0];
    const int seq_len = qk_shape[1];
    const int head_num = qk_shape[2];
    const int head_size = qk_shape[3];
    const int half = inv_freq_size;

#pragma omp parallel for
    for (int head = 0; head < head_num; ++head) {
        int off = head * head_size;
        int row = 0;

        const int *position_ids = positions;
        const int *block_position_ids = positions + seq_len;

        for (int bs = 0; bs < batch_size; ++bs) {
            for (int seq = 0; seq < seq_len; ++seq) {
                float *p1 = query + row * qStride + off;
                float *p2 = key + row * kStride + off;

                int pos = position_ids[seq];
                if (unlikely(pos >= max_seq_len_cached)) {
                    printf("Unexpected position (%d), please expand the rotary table.\n", pos);
                    exit(-1);
                }

                float *pcos = emb_cos + pos * dim;
                float *psin = emb_sin + pos * dim;

#pragma omp simd
                for (int i = 0; i < half; ++i) {
                    auto t1 = p1[i];
                    auto t2 = p2[i];

                    p1[i] = p1[i] * pcos[i] - p1[i + half] * psin[i];
                    p2[i] = p2[i] * pcos[i] - p2[i + half] * psin[i];

                    p1[i + half] = p1[i + half] * pcos[i + half] + t1 * psin[i + half];
                    p2[i + half] = p2[i + half] * pcos[i + half] + t2 * psin[i + half];
                }

                // Update params to compute for the second half
                p1 += dim;
                p2 += dim;

                pos = block_position_ids[seq];
                if (unlikely(pos >= max_seq_len_cached)) {
                    printf("Unexpected block position (%d), please expand the rotary table.\n", pos);
                    exit(-1);
                }

                pcos = emb_cos + pos * dim;
                psin = emb_sin + pos * dim;

#pragma omp simd
                for (int i = 0; i < half; ++i) {
                    auto t1 = p1[i];
                    auto t2 = p2[i];

                    p1[i] = p1[i] * pcos[i] - p1[i + half] * psin[i];
                    p2[i] = p2[i] * pcos[i] - p2[i + half] * psin[i];

                    p1[i + half] = p1[i + half] * pcos[i + half] + t1 * psin[i + half];
                    p2[i + half] = p2[i + half] * pcos[i + half] + t2 * psin[i + half];
                }

                row += 1;
            } // end seq

            // Update position_ids and block_position_ids
            position_ids += seq_len * 2;
            block_position_ids += seq_len * 2;
        } // end bs
    } // end head
}

void RotaryEmbedding2D::forward(
        float *query, float *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads, int *positionIds) {
    printf("Unsupported RotaryEmbedding2D in cb mode!\n");
    exit(1);
}
