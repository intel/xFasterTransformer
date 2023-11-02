#pragma once
#include "bfloat16.h"
#include "compile_util.h"
#include <cmath>
#include <cstring>
#include <iostream>

void xft_rotary_embedding_kernel(
    const int64_t *__restrict__ position_ids, // [num_tokens]
    bfloat16_t *__restrict__ query,         // [num_tokens, head_num, head_size]
    bfloat16_t *__restrict__ key,           // [num_tokens, head_num, head_size]
    const bfloat16_t *__restrict__ emb_cos, // [max_position, dim]
    const bfloat16_t *__restrict__ emb_sin, // [max_position, dim]
    const int dim,                          // rot_dim
    const int qstride, const int kstride, const int num_tokens,
    const int head_num, const int head_size, const int num_kv_heads);