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
#include <immintrin.h>

#include "bfloat16.h"
#include "copy_util.h"
#include "float16.h"
#include "my_types.h"

namespace xft {

template <typename OutT, typename WeiT>
void tokenEmbedding(OutT *output, const int *tokenId, const WeiT *embTable, const int tokenSize, const int hiddenSize) {
    for (int i = 0; i < tokenSize; ++i) {
        int id = tokenId[i];
        xft::copy(output + i * hiddenSize, embTable + id * hiddenSize, hiddenSize);
    }
}

template void tokenEmbedding<float, float>(
        float *output, const int *tokenId, const float *weight, const int tokenSize, const int hiddenSize);
template void tokenEmbedding<float16_t, float16_t>(
        float16_t *output, const int *tokenId, const float16_t *weight, const int tokenSize, const int hiddenSize);
template void tokenEmbedding<bfloat16_t, bfloat16_t>(
        bfloat16_t *output, const int *tokenId, const bfloat16_t *weight, const int tokenSize, const int hiddenSize);

template void tokenEmbedding<float, float16_t>(
        float *output, const int *tokenId, const float16_t *weight, const int tokenSize, const int hiddenSize);
template void tokenEmbedding<float, bfloat16_t>(
        float *output, const int *tokenId, const bfloat16_t *weight, const int tokenSize, const int hiddenSize);
template void tokenEmbedding<bfloat16_t, float16_t>(
        bfloat16_t *output, const int *tokenId, const float16_t *weight, const int tokenSize, const int hiddenSize);
template void tokenEmbedding<float16_t, bfloat16_t>(
        float16_t *output, const int *tokenId, const bfloat16_t *weight, const int tokenSize, const int hiddenSize);
} // namespace xft