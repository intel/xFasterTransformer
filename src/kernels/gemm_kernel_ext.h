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
#include <cstdio>
#include <typeinfo>

#include "bfloat16.h"
#include "float16.h"
#include "sgemm.h"
#include "sgemm_f32f16f32.h"
#include "sgemm_f32f16bf16.h"

// Single thread small gemm
void small_gemm_transb(const float *A, const float *B, float *C, int M, int N, int K, int lda, int ldb, int ldc);
void small_gemm_transb(const float *A, const float16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc);
void small_gemm_transb(const bfloat16_t *A, const float16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc);

// Single thread small gemm with attention mask (skip skippable computation according to attnMask)
void small_gemm_transb(const float *attnMask, const float *A, const float *B, float *C, int M, int N, int K, int lda,
        int ldb, int ldc);
void small_gemm_transb(const float *attnMask, const float *A, const float16_t *B, float *C, int M, int N, int K,
        int lda, int ldb, int ldc);
void small_gemm_transb(const float *attnMask, const float *A, const bfloat16_t *B, float *C, int M, int N, int K,
        int lda, int ldb, int ldc);
void small_gemm_transb(const float *attnMask, const bfloat16_t *A, const bfloat16_t *B, float *C, int M, int N, int K,
        int lda, int ldb, int ldc);
void small_gemm_transb(const float *attnMask, const bfloat16_t *A, const float16_t *B, float *C, int M, int N, int K,
        int lda, int ldb, int ldc);

////////////////////////////////////////////////////////////////////////////////

namespace xft {
// Single thread small gemm
template <typename TA, typename TB, typename TC>
void small_gemm(const TA *A, const TB *B, TC *C, int M, int N, int K, int lda, int ldb, int ldc) {
    printf("Error: small_gemm(%s, %s, %s) is not supported yet!\n", typeid(TA).name(), typeid(TB).name(),
            typeid(TC).name());
    exit(-1);
}

template <>
inline void small_gemm(const float *A, const float *B, float *C, int M, int N, int K, int lda, int ldb, int ldc) {
    xdnn_sgemm_single_thread(false, false, M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
}

template <>
inline void small_gemm(const float *A, const float16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc) {
    xdnn_sgemm_f32f16f32_single_thread(false, false, M, N, K, 1.0f, A, lda, (const XDNN_FP16 *)B, ldb, 0.0f, C, ldc);
}

template <>
inline void small_gemm(const float *A, const float16_t *B, bfloat16_t *C, int M, int N, int K, int lda, int ldb, int ldc) {
    small_sgemm_f32f16bf16(false, M, N, K, A, lda, (const XDNN_FP16 *)B, ldb, (XDNN_BF16 *)C, ldc);
}
} // namespace xft