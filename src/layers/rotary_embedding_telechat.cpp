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
#include "rotary_embedding_telechat.h"

#include "allocator.h"
#include "compile_util.h"

// unordered_map<base, tuple<emb_cos, emb_sin>>
static std::unordered_map<float, std::tuple<float *, float *>> embCosSin;
static float *cur_emb_cos = nullptr;
static float *cur_emb_sin = nullptr;

bool TeleChatRotaryEmbedding::initialized = false;
int TeleChatRotaryEmbedding::max_seq_len_cached = -1;
int TeleChatRotaryEmbedding::inv_freq_size = -1;

inline float calBaseValue(const int &true_seq_len, const float &base_initial, const int &dim) {
    float context_value = log((float)true_seq_len / (float)4096) / log(2.0) + 1;
    float ntk_alpha = pow((float)2.0, ceil(context_value)) - 1;
    ntk_alpha = std::max(ntk_alpha, (float)1.0);
    float new_base = base_initial * pow(ntk_alpha, (float)dim / (dim - 2));
    return new_base;
}

// dim: equals to head size
TeleChatRotaryEmbedding::TeleChatRotaryEmbedding(const int dim, const int max_position_embeddings, const float base)
    : max_training_seqlen(max_position_embeddings) {
    this->dim = dim;
    this->base_initial = base;
    this->base = base;
    this->default_base = calBaseValue(max_training_seqlen, base, dim);
    if (!initialized) {
        this->max_seq_len_cached = max_training_seqlen;
        this->inv_freq_size = (dim + 1) / 2;
        float *inv_freq = (float *)malloc(this->inv_freq_size * sizeof(float));
#pragma omp parallel for
        for (size_t i = 0; i < this->inv_freq_size; i++) {
            inv_freq[i] = 1.0 / pow(base, float(i * 2) / dim);
        }

        CalEmb(inv_freq, base, embCosSin, max_training_seqlen);
        free(inv_freq);

        auto &value = embCosSin[base];
        cur_emb_cos = std::get<0>(value);
        cur_emb_sin = std::get<1>(value);
        initialized = true;
    } else if (dim != this->inv_freq_size * 2) {
        printf("Incorrect dim=%d, inv_freq_size=%d\n", dim, this->inv_freq_size);
        exit(-1);
    }
};

TeleChatRotaryEmbedding::~TeleChatRotaryEmbedding() {}

float TeleChatRotaryEmbedding::getNewBaseValue(const int true_seq_len) {
    if (true_seq_len <= max_training_seqlen) { return this->default_base; }
    return calBaseValue(true_seq_len, this->base_initial, this->dim);
}

void TeleChatRotaryEmbedding::CalEmb(
        float *inv_freq, float base, std::unordered_map<float, std::tuple<float *, float *>> &embCosSin, int seqLen) {
    float *emb_cos = (float *)xft::alloc(this->max_seq_len_cached * (this->inv_freq_size * 2) * sizeof(float));
    float *emb_sin = (float *)xft::alloc(this->max_seq_len_cached * (this->inv_freq_size * 2) * sizeof(float));

    embCosSin[base] = std::make_tuple(emb_cos, emb_sin);

    // if seqLen > max_training_seqlen, we will scale the value by m_scale
    if (seqLen <= max_training_seqlen) {
#pragma omp parallel for
        for (size_t i = 0; i < this->max_seq_len_cached; i++) {
            float *pcos = emb_cos + i * this->inv_freq_size * 2;
            float *psin = emb_sin + i * this->inv_freq_size * 2;

            for (size_t j = 0; j < this->inv_freq_size; j++) {
                float tmp = i * inv_freq[j];
                float cos_tmp = std::cos(tmp);
                float sin_tmp = std::sin(tmp);

                pcos[j] = cos_tmp;
                pcos[j + this->inv_freq_size] = cos_tmp;
                psin[j] = sin_tmp;
                psin[j + this->inv_freq_size] = sin_tmp;
            }
        }
    } else {
        float m_scale = 0.1 * log((float)seqLen / (float)max_training_seqlen) + 1.0;
#pragma omp parallel for
        for (size_t i = 0; i < this->max_seq_len_cached; i++) {
            float *pcos = emb_cos + i * this->inv_freq_size * 2;
            float *psin = emb_sin + i * this->inv_freq_size * 2;

            for (size_t j = 0; j < this->inv_freq_size; j++) {
                float tmp = i * inv_freq[j];
                float cos_tmp = std::cos(tmp) * m_scale;
                float sin_tmp = std::sin(tmp) * m_scale;

                pcos[j] = cos_tmp;
                pcos[j + this->inv_freq_size] = cos_tmp;
                psin[j] = sin_tmp;
                psin[j + this->inv_freq_size] = sin_tmp;
            }
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
void TeleChatRotaryEmbedding::forward(
        float *query, float *key, int qStride, int kStride, const int *qkShape, const int *positionIds) {
    int dim = this->inv_freq_size * 2;
    REQUIRES(dim == qkShape[3], "Incorrect shape, this dimention is not the head size.");

    const int batchSize = qkShape[0];
    const int seqLen = qkShape[1];
    const int qHeads = qkShape[2];
    const int kHeads = qkShape[4];
    const int pastKeyLength = qkShape[6];
    const int heads = std::max(qHeads, kHeads);
    const int half = this->inv_freq_size;
    int kv_len = seqLen + pastKeyLength;

    /*** In QWEN torch version, they acturally used seq_len+past_key_len as kv_len to calculate new base
        kv_seq_len = hidden_states.size()[1]
        if past_key_values[0] is not None:
            # past key values[0][0] shape: bs * seq_len * head_num * dim
            if self.use_cache_quantization:
                kv_seq_len += past_key_values[0][0][0].shape[2]
            else:
                kv_seq_len += past_key_values[0][0].shape[1]

    ***/
    float new_base = getNewBaseValue(kv_len);
    if (std::abs(new_base - this->base) > 1e-5) {
        this->base = new_base;
        auto it = embCosSin.find(new_base);
        if (it == embCosSin.end()) {
            float *inv_freq = (float *)malloc(this->inv_freq_size * sizeof(float));
#pragma omp parallel for
            for (size_t i = 0; i < this->inv_freq_size; i++) {
                inv_freq[i] = 1.0 / pow(new_base, float(i * 2) / dim);
            }
            CalEmb(inv_freq, new_base, embCosSin, seqLen);
            free(inv_freq);
        }

        auto &value = embCosSin[new_base];
        cur_emb_cos = std::get<0>(value);
        cur_emb_sin = std::get<1>(value);
    }

#pragma omp parallel for collapse(3)
    for (int head = 0; head < heads; ++head) {
        for (int bs = 0; bs < batchSize; ++bs) {
            for (int seq = 0; seq < seqLen; ++seq) {
                int pos = positionIds[seq];
                float *pcos = cur_emb_cos + pos * dim;
                float *psin = cur_emb_sin + pos * dim;

                float *q = query + bs * seqLen * qStride + seq * qStride + head * dim;
                float *k = key + bs * seqLen * kStride + seq * kStride + head * dim;
#pragma omp simd
                for (int i = 0; i < half; ++i) {
                    if (head < qHeads) {
                        auto q1 = q[i];
                        q[i] = (q[i] * pcos[i] - q[i + half] * psin[i]);
                        q[i + half] = (q[i + half] * pcos[i + half] + q1 * psin[i + half]);
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

void TeleChatRotaryEmbedding::forward(
        bfloat16_t *query, bfloat16_t *key, int qStride, int kStride, const int *qkShape, const int *positionIds) {
    int dim = this->inv_freq_size * 2;
    REQUIRES(dim == qkShape[3], "Incorrect shape, this dimention is not the head size.");

    const int batchSize = qkShape[0];
    const int seqLen = qkShape[1];
    const int qHeads = qkShape[2];
    const int kHeads = qkShape[4];
    const int pastKeyLength = qkShape[6];
    const int heads = std::max(qHeads, kHeads);
    const int half = this->inv_freq_size;
    int kv_len = seqLen + pastKeyLength;

    float new_base = getNewBaseValue(kv_len);
    if (std::abs(new_base - this->base) > 1e-5) {
        this->base = new_base;

        auto it = embCosSin.find(new_base);
        if (it == embCosSin.end()) {
            float *inv_freq = (float *)malloc(this->inv_freq_size * sizeof(float));
#pragma omp parallel for
            for (size_t i = 0; i < this->inv_freq_size; i++) {
                inv_freq[i] = 1.0 / pow(base, float(i * 2) / dim);
            }
            CalEmb(inv_freq, new_base, embCosSin, seqLen);
            free(inv_freq);
        }

        auto &value = embCosSin[new_base];
        cur_emb_cos = std::get<0>(value);
        cur_emb_sin = std::get<1>(value);
    }

    xft::teleChatApplyRotaryPosEmbeding(
            query, key, qStride, kStride, cur_emb_cos, cur_emb_sin, inv_freq_size, qkShape, positionIds);
}

void TeleChatRotaryEmbedding::forward(
        float16_t *query, float16_t *key, int qStride, int kStride, const int *qkShape, const int *positionIds) {
    int dim = this->inv_freq_size * 2;
    REQUIRES(dim == qkShape[3], "Incorrect shape, this dimention is not the head size.");

    const int seqLen = qkShape[1];
    const int maxSeqLength = qkShape[5];
    const int pastKeyLength = qkShape[6];
    int kv_len = seqLen + pastKeyLength;

    float new_base = getNewBaseValue(kv_len);
    if (std::abs(new_base - this->base) > 1e-5) {
        this->base = new_base;

        auto it = embCosSin.find(new_base);
        if (it == embCosSin.end()) {
            float *inv_freq = (float *)malloc(this->inv_freq_size * sizeof(float));
#pragma omp parallel for
            for (size_t i = 0; i < this->inv_freq_size; i++) {
                inv_freq[i] = 1.0 / pow(base, float(i * 2) / dim);
            }
            CalEmb(inv_freq, new_base, embCosSin, seqLen);
            free(inv_freq);
        }

        auto &value = embCosSin[new_base];
        cur_emb_cos = std::get<0>(value);
        cur_emb_sin = std::get<1>(value);
    }

    xft::teleChatApplyRotaryPosEmbeding(
            query, key, qStride, kStride, cur_emb_cos, cur_emb_sin, inv_freq_size, qkShape, positionIds);
}

// For continuous batching
void TeleChatRotaryEmbedding::forward(
        float *query, float *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads, int *positionIds) {
    xft::teleChatApplyRotaryPosEmbed(
            query, key, cur_emb_cos, cur_emb_sin, qStride, kStride, this->dim, totSeqLen, qHeads, kHeads, positionIds);
}

void TeleChatRotaryEmbedding::forward(bfloat16_t *query, bfloat16_t *key, int totSeqLen, int qStride, int kStride,
        int qHeads, int kHeads, int *positionIds) {
    xft::teleChatApplyRotaryPosEmbed(
            query, key, cur_emb_cos, cur_emb_sin, qStride, kStride, this->dim, totSeqLen, qHeads, kHeads, positionIds);
}

void TeleChatRotaryEmbedding::forward(float16_t *query, float16_t *key, int totSeqLen, int qStride, int kStride,
        int qHeads, int kHeads, int *positionIds) {
    xft::teleChatApplyRotaryPosEmbed(
            query, key, cur_emb_cos, cur_emb_sin, qStride, kStride, this->dim, totSeqLen, qHeads, kHeads, positionIds);
}
