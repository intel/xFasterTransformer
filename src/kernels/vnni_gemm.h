#ifndef __VNNI_GEMM_H
#define __VNNI_GEMM_H

#include <cstdint>

extern "C" {
// clang-format off
// To pack matrix B to packedB
// Note:
// (1) K must be multiple of 4, otherwise will fail
// (2) If transB = false, B is in K x N, ldb >= N
// (3) If transB = true,  B is in N X K, ldb >= K
void igemm_pack_b(int8_t *B, int8_t *packedB, int K, int N, int ldb, bool transB);

// General int8 GEMM
// Note: there is no ldb, as B is packed in compact format
void igemm(uint8_t *A, int8_t *packedB, int32_t *C, int M, int N, int K, int lda, int ldc);

// Int8 GEMM considered compensation
// zpA: zero point of A (A is quantized per tensor, thus only one zero point)
// compensationB: sum of each column of B
// C = (A - zp) * B
void igemm_compensation(const uint8_t *A, const int8_t *packedB, int32_t *C,
                        const int32_t zpA, const int32_t *compensationB,
                        int M, int N, int K, int lda, int ldc);

// Int8 GEMM (A is uint8, B is int8)
// scalesA: scales of A (per row)
// zpsA: zero points of A
// scalesB: scales of B (per column)
// compensationB: sum of each column of B
// bias: C = A * packedB + bias, it has N elements
//
// Note:
// (1) For quantization of A, formula is: Q(x) = round (x / scale + zp); For B, it is symmetric, thus zp = 0
// (2) "_rc" means: A is quantized per channel (row), B is quantized per_channel_symmetric
void igemm_dense_rc(const uint8_t *A, const int8_t *packedB, float *C,
                    float *scalesA, int32_t *zpsA,
                    float *scalesB, const int32_t *compensationB,
                    const float *bias,
                    int M, int N, int K, int lda, int ldc);

// "_tc" version: A is quantized per_tensor_affine, B is quantized per_channel_symmetric
void igemm_dense_tc(const uint8_t *A, const int8_t *packedB, float *C,
                    float scaleA, int32_t zpA,
                    float *scalesB, const int32_t *compensationB,
                    const float *bias,
                    int M, int N, int K, int lda, int ldc);

// "_rt" means: A is quantized per channel (row), B is quantized per_tensor_symmetric
void igemm_dense_rt(const uint8_t *A, const int8_t *packedB, float *C,
                    float *scalesA, int32_t *zpsA,
                    float scaleB, const int32_t *compensationB,
                    const float *bias,
                    int M, int N, int K, int lda, int ldc);

// "_tt" version: A is quantized per_tensor_affine, B is quantized per_tensor_symmetric
void igemm_dense_tt(const uint8_t *A, const int8_t *packedB, float *C,
                    float scaleA, int32_t zpA,
                    float scaleB, const int32_t *compensationB,
                    const float *bias,
                    int M, int N, int K, int lda, int ldc);
// clang-format on
}

#endif