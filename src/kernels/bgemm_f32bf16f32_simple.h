#ifndef __BGEMM_F32BF16F32_SIMPLE_H__
#define __BGEMM_F32BF16F32_SIMPLE_H__

#include "../common/bfloat16.h"

extern "C" {
// clang-format off
// To compute bgemm: C = alpha * A * (bfloat16_t *)B + beta * C
void ig_bgemm_f32bf16f32(bool transA, bool transB, int M, int N, int K,
        float alpha, const float *A, int lda, const bfloat16_t *B, int ldb,
        float beta, float *C, int ldc);

// To pack matrix B
// transB: if the input 'b' is transposed or not
// B is in K x N if transB = false
// B is in N x K if transB = true
void ig_bgemm_f32bf16f32_packb(bool transB, int N, int K, const bfloat16_t *B, int ldb, bfloat16_t *packedB);

// To compute bgemm: C = alpha * A * (bfloat16_t *)packedB + beta * C
// Note: there is no ldb, as B is packed in compact format
void ig_bgemm_f32bf16f32_compute(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const bfloat16_t *packedB,
        float beta, float *C, int ldc);

// To compute bgemm w/ bias_add: C = SILU(alpha * A * packedB + beta * C)
void ig_bgemm_f32bf16f32_compute_silu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const bfloat16_t *packedB,
        float beta, float *C, int ldc);

// Extended residential
// C = alpha * A * packedB + beta * C + bias + gamma * res
// ldres, residential matrix stride
void ig_bgemm_f32bf16f32_compute_resext(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const bfloat16_t *packedB,
        float beta, float *C, int ldc, const float *bias,
        float gamma, const float *res, int ldres);

// C = [alpha * A * packedB + beta * C] * res
// ldres, residential matrix stride
void ig_bgemm_f32bf16f32_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const bfloat16_t *packedB,
        float beta, float *C, int ldc, const float *res, int ldres);

// To compute bgemm w/ bias_add: C = alpha * A * (bfloat16_t *)packedB + beta * C + bias
void ig_bgemm_f32bf16f32_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const bfloat16_t *packedB,
        float beta, float *C, int ldc, const float *bias);

// To compute bgemm w/ bias_add: C = RELU(alpha * A * (bfloat16_t *)packedB + beta * C + bias)
void ig_bgemm_f32bf16f32_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const bfloat16_t *packedB,
        float beta, float *C, int ldc, const float *bias);

// C = alpha * A * (bfloat16_t *)packedB + beta * C + bias + res
// ldres, redidential matrix stride
void ig_bgemm_f32bf16f32_compute_residential(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const bfloat16_t *packedB,
        float beta, float *C, int ldc, const float *bias, const float *res, int ldres);

// ================================================================================
// Below is single thread small bgemm
// ================================================================================
void small_bgemm_f32bf16f32(int M, int N, int K, const float *A, int lda, const bfloat16_t *B, int ldb, float *C, int ldc);
// clang-format on
}

#endif
