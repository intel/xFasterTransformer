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

    // dim: equals to head size
    ctx->GetAttr("size_per_head", &this->dim);
    ctx->GetAttr("max_pos_seq_len", &this->max_position_embeddings, 2048);
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
        }
        llamaCalEmb(inv_freq, max_position_embeddings);
#ifdef GPU
        device = ctx->device;
        if (device != nullptr) {
            sycl::queue *gpu_queue = static_cast<sycl::queue *>(device);
            float *emb_cos_bak = emb_cos;
            float *emb_sin_bak = emb_sin;
            emb_cos = ctx->getBuffer<float>(emb_cos_str + "_gpu", max_position_embeddings * inv_freq_size, gpu_queue);
            emb_sin = ctx->getBuffer<float>(emb_sin_str + "_gpu", max_position_embeddings * inv_freq_size, gpu_queue);
            gpu_queue->memcpy(emb_cos, emb_cos_bak, max_position_embeddings * inv_freq_size * sizeof(float)).wait();
            gpu_queue->memcpy(emb_sin, emb_sin_bak, max_position_embeddings * inv_freq_size * sizeof(float)).wait();
            ctx->freeBuffer(emb_cos_str);
            ctx->freeBuffer(emb_sin_str);
        }
#endif
    } else if (dim != inv_freq_size * 2) {
        printf("Incorrect dim=%d, inv_freq_size=%d\n", dim, inv_freq_size);
        exit(-1);
    }
}

// This API is deprecated, will delete after all rotary embed code refactor.
LlamaRotaryEmbedding::LlamaRotaryEmbedding(const int dim, const int max_position_embeddings, const float base) {}

void LlamaRotaryEmbedding::llamaCalEmb(const float *inv_freq, const int max_position_embeddings) {
#pragma omp parallel for
    for (size_t i = 0; i < max_position_embeddings; i++) {
        float *pcos = emb_cos + i * inv_freq_size;
        float *psin = emb_sin + i * inv_freq_size;

        for (size_t j = 0; j < inv_freq_size; j++) {
            float tmp = i * inv_freq[j] / this->scaling_factor;
            float cos_tmp = std::cos(tmp);
            float sin_tmp = std::sin(tmp);

            pcos[j] = cos_tmp;
            psin[j] = sin_tmp;
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

#ifdef GPU

void LlamaRotaryEmbedding::forward(
        float *query, float *key, int qStride, int kStride, const int *qkShape, const int *positionIds) {
    const int batchSize = qkShape[0];
    const int seqLen = qkShape[1];
    const int qHeads = qkShape[2];
    const int kHeads = qkShape[4];
    const int head_num = std::max(qHeads, kHeads);
    const int head_size = qkShape[3];
    const int half_head_size = (head_size + 1) / 2;
    using namespace sycl;

    auto rope_kernel
            = [](sycl::nd_item<3> &item, const float *embCos, const float *embSin, const int qHeads, const int kHeads,
                      const int seq_size, const int head_size, const int half, float *query, float *key, int qStride,
                      int kStride, const sycl::accessor<int, 1, sycl::access::mode::read> &positionIds) {
                  size_t idx_bs_seq = item.get_global_id(0);
                  size_t idx_head_num = item.get_global_id(1);
                  size_t idx_half_head_dim = item.get_global_id(2);

                  size_t pos = positionIds[idx_bs_seq % seq_size];
                  float cos = embCos[pos * half + idx_half_head_dim];
                  float sin = embSin[pos * half + idx_half_head_dim];

                  float *q = query + idx_bs_seq * qStride + idx_head_num * head_size + idx_half_head_dim;
                  float *k = key + idx_bs_seq * kStride + idx_head_num * head_size + idx_half_head_dim;

                  if (idx_head_num < qHeads) {
                      auto q1 = q[0];
                      q[0] = q1 * cos - q[half] * sin;
                      q[half] = q[half] * cos + q1 * sin;
                  }
                  if (idx_head_num < kHeads) {
                      auto k1 = k[0];
                      k[0] = k1 * cos - k[half] * sin;
                      k[half] = k[half] * cos + k1 * sin;
                  }
              };

    // Reorder input
    sycl::queue *gpu_queue = static_cast<sycl::queue *>(device);
    float *embCos = emb_cos;
    float *embSin = emb_sin;

    sycl::buffer<int, 1> positionIdsBuf(positionIds, sycl::range<1>(seqLen));
    gpu_queue->submit([&](sycl::handler &cgh) {
        sycl::accessor position(positionIdsBuf, cgh, sycl::read_only);
        sycl::range<3> globalSize(batchSize * seqLen, head_num, half_head_size);
        sycl::range<3> workGroupSize(1, 1, 1);

        cgh.parallel_for<class kernel_rope>(
                sycl::nd_range(globalSize, workGroupSize), [=, this](sycl::nd_item<3> item) {
                    rope_kernel(item, embCos, embSin, qHeads, kHeads, seqLen, head_size, half_head_size, query, key,
                            qStride, kStride, position);
                });
    });
    gpu_queue->wait();
}

#else

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

                bfloat16_t *q = query + bs * seqLen * qStride + seq * qStride + head * dim;
                bfloat16_t *k = key + bs * seqLen * kStride + seq * kStride + head * dim;

                // Process chunks of 16 elements at a time
                for (int i = 0; i < half; i += 16) {
                    int remain = half - i;
                    __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                    __m512 pCosVec = _mm512_maskz_loadu_ps(mask, &pcos[i]);
                    __m512 pSinVec = _mm512_maskz_loadu_ps(mask, &psin[i]);

                    // Compute something like:
                    // q[i] = q[i] * pcos[i] - q[i + half] * psin[i];
                    // q[i + half] = q[i + half] * pcos[i + half] + q[i] * psin[i + half];
                    if (head < qHeads) {
                        __m512 qVec = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &q[i]));
                        __m512 qHalfVec = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &q[i + half]));
                        __m512 qNew = _mm512_fmsub_ps(qVec, pCosVec, _mm512_mul_ps(qHalfVec, pSinVec));
                        __m512 qHalfNew = _mm512_fmadd_ps(qHalfVec, pCosVec, _mm512_mul_ps(qVec, pSinVec));
                        _mm256_mask_storeu_epi16(&q[i], mask, bfloat16_t::cvt_fp32_to_bf16(qNew));
                        _mm256_mask_storeu_epi16(&q[i + half], mask, bfloat16_t::cvt_fp32_to_bf16(qHalfNew));
                    }

                    if (head < kHeads) {
                        __m512 kVec = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &k[i]));
                        __m512 kHalfVec = bfloat16_t::cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, &k[i + half]));
                        __m512 kNew = _mm512_fmsub_ps(kVec, pCosVec, _mm512_mul_ps(kHalfVec, pSinVec));
                        __m512 kHalfNew = _mm512_fmadd_ps(kHalfVec, pCosVec, _mm512_mul_ps(kVec, pSinVec));
                        _mm256_mask_storeu_epi16(&k[i], mask, bfloat16_t::cvt_fp32_to_bf16(kNew));
                        _mm256_mask_storeu_epi16(&k[i + half], mask, bfloat16_t::cvt_fp32_to_bf16(kHalfNew));
                    }
                }
            }
        }
    }
}

#endif // GPU