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

void ChatGLM2RotaryEmbedding::forward(
        bfloat16_t *query, bfloat16_t *key, int qStride, int kStride, const int *qk_shape, const int *position_ids) {
    xft::chatglm2ApplyRotaryPosEmbeding(
            query, key, qStride, kStride, emb_cos, emb_sin, inv_freq_size, qk_shape, position_ids);
}

void ChatGLM2RotaryEmbedding::forward(
        float16_t *query, float16_t *key, int qStride, int kStride, const int *qk_shape, const int *position_ids) {
    xft::chatglm2ApplyRotaryPosEmbeding(
            query, key, qStride, kStride, emb_cos, emb_sin, inv_freq_size, qk_shape, position_ids);
}

// For continuous batching
void ChatGLM2RotaryEmbedding::forward(
        float *query, float *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads, int *positionIds) {
    printf("Unsupported ChatGLM2RotaryEmbedding in cb mode !\n");
    exit(1);
}

void ChatGLM2RotaryEmbedding::forward(bfloat16_t *query, bfloat16_t *key, int totSeqLen, int qStride, int kStride,
        int qHeads, int kHeads, int *positionIds) {
    printf("Unsupported ChatGLM2RotaryEmbedding in cb mode !\n");
    exit(1);
}

void ChatGLM2RotaryEmbedding::forward(float16_t *query, float16_t *key, int totSeqLen, int qStride, int kStride,
        int qHeads, int kHeads, int *positionIds) {
    printf("Unsupported ChatGLM2RotaryEmbedding in cb mode !\n");
    exit(1);
}