#ifndef __SGEMM_SIMPLE_H__
#define __SGEMM_SIMPLE_H__

extern "C" {
// clang-format off
// To compute sgemm: C = alpha * A * B + beta * C
void ig_sgemm(bool transA, bool transB, int M, int N, int K,
        float alpha, const float *A, int lda, const float *B, int ldb,
        float beta, float *C, int ldc);

// To pack matrix B
// transB: if the input 'b' is transposed or not
// B is in K x N if transB = false
// B is in N x K if transB = true
void ig_sgemm_packb(bool transB, int N, int K, const float *B, int ldb, float *packedB);

// To compute sgemm: C = alpha * A * packedB + beta * C
// Note: there is no ldb, as B is packed in compact format
void ig_sgemm_compute(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const float *packedB,
        float beta, float *C, int ldc);

// To compute sgemm w/ bias_add: C = SILU(alpha * A * packedB + beta * C)
void ig_sgemm_compute_silu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const float *packedB,
        float beta, float *C, int ldc);

// To compute sgemm w/ bias_add: C = alpha * A * packedB + beta * C + bias
void ig_sgemm_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const float *packedB,
        float beta, float *C, int ldc, const float *bias);

// To compute sgemm w/ bias_add: C = RELU(alpha * A * packedB + beta * C + bias)
void ig_sgemm_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const float *packedB,
        float beta, float *C, int ldc, const float *bias);

// C = alpha * A * packedB + beta * C + bias + res
// ldres, residential matrix stride
void ig_sgemm_compute_residential(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const float *packedB,
        float beta, float *C, int ldc, const float *bias, const float *res, int ldres);

// Extended residential
// C = alpha * A * packedB + beta * C + bias + gamma * res
// ldres, residential matrix stride
void ig_sgemm_compute_resext(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const float *packedB,
        float beta, float *C, int ldc, const float *bias, 
        float gamma, const float *res, int ldres);

// C = [alpha * A * packedB + beta * C] * res
// ldres, residential matrix stride
void ig_sgemm_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const float *packedB,
        float beta, float *C, int ldc, const float *res, int ldres);

// ================================================================================
// Below is single thread small sgemm
// ================================================================================
void small_sgemm(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc);
// clang-format on
}

#endif
