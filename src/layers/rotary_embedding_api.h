#pragma once
#include <cmath>
#include <cstring>
#include <iostream>
#include "compile_util.h"

template <typename scalar_t>
void xft_rotary_embedding_kernel(
    const int64_t *__restrict__ position_ids, // [num_tokens]
    scalar_t *__restrict__ query,             // [num_tokens, head_num, head_size]
    scalar_t *__restrict__ key,               // [num_tokens, head_num, head_size] or [[num_tokens, num_kv_heads, head_size]]

    const scalar_t *__restrict__ emb_cos, // [max_position, dim]
    const scalar_t *__restrict__ emb_sin, // [max_position, dim]
    const int dim,                        // rot_dim
    const int qStride,
    const int kStride,
    const int num_tokens,
    const int head_num,
    const int head_size,
    const int num_kv_heads = 0)
{
    REQUIRES(dim == head_size, "Incorrect shape, rot_dim is not the head size.");
    return;
    const int half = (dim + 1) / 2; // inv_freq_size

#pragma omp parallel for
    for (int head = 0; head < head_num; ++head)
    {
        int off = head * dim;

        for (int row = 0; row < num_tokens; ++row)
        {
            float *p1 = query + row * qStride + off;
            float *p2 = key + row * kStride + off;

            int pos = position_ids[row];
            const float *pcos = emb_cos + pos * dim;
            const float *psin = emb_sin + pos * dim;

#pragma omp simd
            for (int i = 0; i < half; ++i)
            {
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
