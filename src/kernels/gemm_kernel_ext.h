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
#include "float16.h"

// Single thread small gemm
void small_gemm_transb(const float *A, const float *B, float *C, int M, int N, int K, int lda, int ldb, int ldc);
void small_gemm_transb(const float *A, const float16_t *B, float *C, int M, int N, int K, int lda, int ldb, int ldc);

// Single thread small gemm with attention mask (skip skippable computation according to attnMask)
void small_gemm_transb(const float *attnMask, const float *A, const float *B, float *C, int M, int N, int K, int lda,
        int ldb, int ldc);
void small_gemm_transb(const float *attnMask, const float *A, const float16_t *B, float *C, int M, int N, int K,
        int lda, int ldb, int ldc);
