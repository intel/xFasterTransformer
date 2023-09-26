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
