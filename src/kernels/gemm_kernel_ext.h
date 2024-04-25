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
#include <cstdint>
#include <cstdio>
#include <typeinfo>

#include "bfloat16.h"
#include "float16.h"
#include "sgemm.h"
#include "sgemm_f32f16bf16.h"
#include "sgemm_f32f16f32.h"

// Single thread small gemm
void small_gemm_transb(const float *A, const float *B, float *C, int M, int N, int K, int lda, int ldb, int ldc);
void small_gemm_transb(const float *A, const float16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc);
void small_gemm_transb(
        const bfloat16_t *A, const float16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc);
void small_gemm_transb(
        const bfloat16_t *A, const bfloat16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc);
void small_gemm_transb(
        const float *A, const int8_t *B, const float *bScale, float *C, int M, int N, int K, int lda, int ldb, int ldc);
void small_gemm_transb(const bfloat16_t *A, const int8_t *B, const float *bScale, float *C, int M, int N, int K,
        int lda, int ldb, int ldc);

////////////////////////////////////////////////////////////////////////////////

namespace xft {
// Single thread small gemm
void small_gemm(
        const float *A, const float *B, float *C, int M, int N, int K, int lda, int ldb, int ldc, bool acc = false);
void small_gemm(
        const float *A, const float16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc, bool acc = false);
void small_gemm(const float *A, const bfloat16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc,
        bool acc = false);
void small_gemm(const float *A, const float16_t *B, bfloat16_t *C, int M, int N, int K, int lda, int ldb, int ldc,
        bool acc = false);
void small_gemm(const float *A, const bfloat16_t *B, bfloat16_t *C, int M, int N, int K, int lda, int ldb, int ldc,
        bool acc = false);

// INT8 versions
void small_gemm(const float *A, const int8_t *B, const float *bScale, float *C, int M, int N, int K, int lda, int ldb,
        int ldc, bool acc = false);
void small_gemm(const float *A, const int8_t *B, const float *bScale, bfloat16_t *C, int M, int N, int K, int lda,
        int ldb, int ldc, bool acc = false);
} // namespace xft