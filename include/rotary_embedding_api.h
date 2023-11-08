#pragma once
#include <cmath>
#include <cstring>
#include <iostream>
#include "compile_util.h"
#include "dtype.h"

namespace xft {
void rotaryEmbeddingKernel(DataType dt,
        const int64_t *positionIds, // [num_tokens]
        void *query, // [num_tokens, head_num, head_size]
        void *key, // [num_tokens, head_num, head_size]
        const void *embCos, // [max_position, dim]
        const void *embSin, // [max_position, dim]
        const int dim, // rot_dim
        const int qStride, const int kStride, const int numTokens, const int headNum, const int headSize,
        const int numKvHeads);
}