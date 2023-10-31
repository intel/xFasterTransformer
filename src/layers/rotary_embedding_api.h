#pragma once
#include "compile_util.h"
#include <cmath>
#include <cstring>
#include <iostream>

void xft_rotary_embedding_kernel(
    const int64_t *__restrict__ position_ids, // [num_tokens]
    float *__restrict__ query,         // [num_tokens, head_num, head_size]
    float *__restrict__ key,           // [num_tokens, head_num, head_size]
    const float *__restrict__ emb_cos, // [max_position, dim]
    const float *__restrict__ emb_sin, // [max_position, dim]
    const int dim,                     // rot_dim
    const int qStride, const int kStride, const int num_tokens,
    const int head_num, const int head_size, const int num_kv_heads);