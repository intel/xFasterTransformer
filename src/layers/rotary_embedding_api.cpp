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
#include "layers_rotary_embedding.h"
#include "compile_util.h"
#include "bfloat16.h"
namespace xft {

void rotaryEmbeddingKernel(const int64_t *positionIds, float *query, float *key, const float *embCos,
        const float *embSin, const int dim, const int qStride, const int kStride, const int numTokens,
        const int headNum, const int headSize, const int numKvHeads = 0) {
    REQUIRES(dim == headSize, "Incorrect shape, rot_dim is not the head size.");
    const int half = (dim + 1) / 2;

#pragma omp parallel for
    for (int head = 0; head < headNum; ++head) {
        int off = head * dim;

        for (int row = 0; row < numTokens; ++row) {
            float *p1 = query + row * qStride + off;
            float *p2 = key + row * kStride + off;

            int pos = positionIds[row];
            const float *pcos = embCos + pos * dim;
            const float *psin = embSin + pos * dim;

#pragma omp simd
            for (int i = 0; i < half; ++i) {
                auto t1 = p1[i];
                auto t2 = p2[i];

                p1[i] = p1[i] * pcos[i] - p1[i + half] * psin[i];
                p2[i] = p2[i] * pcos[i] - p2[i + half] * psin[i];

                p1[i + half] = p1[i + half] * pcos[i + half] + t1 * psin[i + half];
                p2[i + half] = p2[i + half] * pcos[i + half] + t2 * psin[i + half];
            }
        }
    }
}

void rotaryEmbeddingKernel(const int64_t *position_ids, bfloat16_t *query, bfloat16_t *key, const bfloat16_t *embCos,
        const bfloat16_t *embSin, const int dim, const int qStride, const int kStride, const int numTokens,
        const int headNum, const int headSize, const int numKvHeads = 0) {
    REQUIRES(dim == headSize, "Incorrect shape, rot_dim is not the head size.");
    const int half = (dim + 1) / 2;

#pragma omp parallel for
    for (int head = 0; head < headNum; ++head) {
        int off = head * dim;

        for (int row = 0; row < numTokens; ++row) {
            bfloat16_t *p1 = query + row * qStride + off;
            bfloat16_t *p2 = key + row * kStride + off;

            int pos = position_ids[row];
            const bfloat16_t *pcos = embCos + pos * dim;
            const bfloat16_t *psin = embSin + pos * dim;

#pragma omp simd
            for (int i = 0; i < half; ++i) {
                auto t1 = p1[i];
                auto t2 = p2[i];

                p1[i] = p1[i] * pcos[i] - p1[i + half] * psin[i];
                p2[i] = p2[i] * pcos[i] - p2[i + half] * psin[i];

                p1[i + half] = p1[i + half] * pcos[i + half] + t1 * psin[i + half];
                p2[i + half] = p2[i + half] * pcos[i + half] + t2 * psin[i + half];
            }
        }
    }
}

void invokeRotaryEmbedding(DataType dt, const int64_t *positionIds, void *query, void *key, const void *embCos,
        const void *embSin, const int dim, const int qStride, const int kStride, const int numTokens, const int headNum,
        const int headSize, const int numKvHeads) {
    if (dt == DataType::bf16) {
        rotaryEmbeddingKernel(positionIds, (bfloat16_t *)query, (bfloat16_t *)key, (bfloat16_t *)embCos,
                (bfloat16_t *)embSin, dim, qStride, kStride, numTokens, headNum, headSize);
    } else if (dt == DataType::fp16) {
        rotaryEmbeddingKernel(positionIds, (float *)query, (float *)key, (float *)embCos, (float *)embSin, dim, qStride,
                kStride, numTokens, headNum, headSize);
    }
}
} // namespace xft
