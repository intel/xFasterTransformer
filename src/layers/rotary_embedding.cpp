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

#include "allocator.h"
#include "compile_util.h"

LlamaRotaryEmbedding::LlamaRotaryEmbedding(DecoderContext *ctx) {
    const std::string inv_freq_str = "inv_freq";
    const std::string emb_cos_str = "emb_cos";
    const std::string emb_sin_str = "emb_sin";

    this->dim = ctx->attHeadSize;
    this->max_position_embeddings = ctx->maxPosEmbed;
    ctx->GetAttr("rope_theta", &this->base, 10000);
    ctx->GetAttr("rope_type", &this->rope_type, std::to_string(-1));

    if (this->rope_type == "linear") ctx->GetAttr("scaling_factor", &this->scaling_factor, 1.0f);

    inv_freq_size = (dim + 1) / 2;

    emb_cos = ctx->getBuffer<float>(emb_cos_str, max_position_embeddings * inv_freq_size);
    emb_sin = ctx->getBuffer<float>(emb_sin_str, max_position_embeddings * inv_freq_size);

    if (!ctx->cached(inv_freq_str)) {
        inv_freq = ctx->getBuffer<float>(inv_freq_str, inv_freq_size);

        for (size_t i = 0; i < inv_freq_size; i++) {
            inv_freq[i] = 1.0 / pow(base, float(i * 2) / dim);
            inv_freq[i] /= this->scaling_factor;
        }
        xft::llamaSetCosSinCache(inv_freq, emb_cos, emb_sin, inv_freq_size, max_position_embeddings);
    } else if (dim != inv_freq_size * 2) {
        printf("Incorrect dim=%d, inv_freq_size=%d\n", dim, inv_freq_size);
        exit(-1);
    }
}

// This API is deprecated, will delete after all rotary embed code refactor.
LlamaRotaryEmbedding::LlamaRotaryEmbedding(const int dim, const int max_position_embeddings, const float base) {
    this->dim = dim;
    inv_freq_size = (dim + 1) / 2;

    inv_freq = (float *)malloc(inv_freq_size * sizeof(float));
    emb_cos = (float *)xft::alloc(max_position_embeddings * inv_freq_size * sizeof(float));
    emb_sin = (float *)xft::alloc(max_position_embeddings * inv_freq_size * sizeof(float));
    for (size_t i = 0; i < inv_freq_size; i++) {
        inv_freq[i] = 1.0 / pow(base, float(i * 2) / dim);
    }

    xft::llamaSetCosSinCache(inv_freq, emb_cos, emb_sin, inv_freq_size, max_position_embeddings);
}

// query and key is the matrix like below:
//
//   |<------------------------------ head_num * head_size --------------------------------->|
//   |_head_size|_____________________________________________________________________________ _ _ _ _
//   |          |          |          |          |          |          |          |          |     ^
//   |          |          |          |          |          |          |          |          |     |
//   |          |          |          |          |          |          |          |          |  bs*seq_len
//   |          |          |          |          |          |          |          |          |     |
//   |          |          |          |          |          |          |          |          |     |
//   |__________|__________|__________|__________|__________|__________|__________|__________|_ _ _v_
//
// inv_freq:
//    _____
//   |_____| 1
//  head_size/2
//
// emb_cos:        emb_sin:
//    _____          _____
//   |     |        |     |
//   |     |        |     |
//   |     |        |     |
//   |     |        |     | max_position_embeddings
//   |     |        |     |
//   |     |        |     |
//   |_____|        |_____|
//  head_size/2    head_size/2

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
                float *pcos = emb_cos + pos * half;
                float *psin = emb_sin + pos * half;

                float *q = query + bs * seqLen * qStride + seq * qStride + head * dim;
                float *k = key + bs * seqLen * kStride + seq * kStride + head * dim;
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
}

void LlamaRotaryEmbedding::forward(
        bfloat16_t *query, bfloat16_t *key, int qStride, int kStride, const int *qkShape, const int *positionIds) {
    xft::llamaApplyRotaryPosEmbeding(query, key, qStride, kStride, emb_cos, emb_sin, inv_freq_size, qkShape, positionIds);
}

void LlamaRotaryEmbedding::forward(
        float16_t *query, float16_t *key, int qStride, int kStride, const int *qkShape, const int *positionIds) {
    xft::llamaApplyRotaryPosEmbeding(query, key, qStride, kStride, emb_cos, emb_sin, inv_freq_size, qkShape, positionIds);
}

// For continuous batching
void LlamaRotaryEmbedding::forward(
        float *query, float *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads, int *positionIds) {
    xft::llamaApplyRotaryPosEmbed(
            query, key, emb_cos, emb_sin, qStride, kStride, this->dim, totSeqLen, qHeads, kHeads, positionIds);
}

void LlamaRotaryEmbedding::forward(bfloat16_t *query, bfloat16_t *key, int totSeqLen, int qStride, int kStride,
        int qHeads, int kHeads, int *positionIds) {
    xft::llamaApplyRotaryPosEmbed(
            query, key, emb_cos, emb_sin, qStride, kStride, this->dim, totSeqLen, qHeads, kHeads, positionIds);
}

void LlamaRotaryEmbedding::forward(float16_t *query, float16_t *key, int totSeqLen, int qStride, int kStride,
        int qHeads, int kHeads, int *positionIds) {
    xft::llamaApplyRotaryPosEmbed(
            query, key, emb_cos, emb_sin, qStride, kStride, this->dim, totSeqLen, qHeads, kHeads, positionIds);
}