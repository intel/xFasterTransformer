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
#pragma once

#include <immintrin.h>

#include "bfloat16.h"
#include "float16.h"
#include "my_types.h"

#include "token_embedding_kernels.h"

namespace xft {
/**
 * @brief Token embedding function that computes embeddings for tokens in a sequence.
 * 
 * @tparam OutT Output data type for the embeddings, only support (fp32/fp16/bf16).
 * @tparam weiT Data type for the token weights, only support (fp32/fp16/bf16).
 * @param output Pointer to the output array where embeddings will be stored.
 * @param tokenId Pointer to the array containing token IDs.
 * @param weight Pointer to the array containing token weights for embedding lookup.
 * @param tokenSize Total number of tokens in the input array.
 * @param seqLen Length of each sequence (number of tokens).
 * @param hiddenSize Size of the hidden dimension for each token embedding.
 */
template <typename OutT, typename weiT>
void tokenEmbedding(OutT *output, const int *tokenId, const weiT *weight, const int tokenSize, const int hiddenSize);

} // namespace xft