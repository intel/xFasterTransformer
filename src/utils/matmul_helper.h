#pragma once
#include "bfloat16.h"
#include "bgemm_f32bf16f32_simple.h"
#include "float16.h"
#include "hgemm_f32f16f32_simple.h"
#include "hgemm_f32i8f32_simple.h"
#include "my_types.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_version.h"
#include "sgemm_f32f16f32_simple.h"
#include "sgemm_f32i8f32_simple.h"
#include "sgemm_simple.h"
#include "timeline.h"
#include "transformer_ctx.h"

#define USE_AMX_M 2

class MMHelper {
public:
    // Pack the MatMul weight from 'src(rows, cols)' to 'weight'
    // Note: if ctx->numSplits > 1, 'w' will get splited
    // trans: 'src' is transposed or not
    // verticalSplit: vertical split or horizontal split, vertical vs. horizontal:
    //  _________________________            _________________________
    // |            |            |          |                         |
    // |            |            |          |_________________________|
    // |            |            |          |                         |
    // |____________|____________|          |_________________________|
    //           vertical                            horizontal
    //
    // ****************************************************************************
    //
    // Vertical split like the left one, but if transposed, like the right one
    //      |<-------- cols ----------|           |<-------- rows ----------|
    //  _    _________________________        _    _________________________
    //  ↑   |            |            |       ↑   |                         |
    //  |   |            |            |       |   |_________________________|
    // rows |            |            |      cols |                         |
    //  ↓   |____________|____________|       ↓   |_________________________|
    //             not_transposed                          transposed
    //
    // ****************************************************************************
    //
    // Horizontal split like the right one, but if transposed, like the left one
    //      |<-------- rows ----------|           |<-------- cols ----------|
    //  _    _________________________        _    _________________________
    //  ↑   |            |            |       ↑   |                         |
    //  |   |            |            |       |   |_________________________|
    // cols |            |            |      rows |                         |
    //  ↓   |____________|____________|       ↓   |_________________________|
    //               transposed                          not_transposed
    //
    template <typename WeiT>
    static void convertWeight(bool trans, int rows, int cols, const float *src, int numSplit, int splitIdx,
            bool verticalSplit, hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight) {
        // FP32 transpose
        if constexpr (std::is_same_v<WeiT, float>) {
            if (verticalSplit) {
                int colsPerSplit = cols / numSplit;
                if (trans) {
                    quantizedWeight.Resize(colsPerSplit, rows);
                    const float *base = src + quantizedWeight.Rows() * quantizedWeight.Stride() * splitIdx;
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        memcpy(quantizedWeight.Data() + i * quantizedWeight.Stride(), base + i * rows,
                                quantizedWeight.Cols());
                    }
                } else {
                    quantizedWeight.Resize(rows, colsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        memcpy(quantizedWeight.Data() + i * quantizedWeight.Stride(),
                                src + i * cols + quantizedWeight.Cols() * splitIdx, quantizedWeight.Cols());
                    }
                }
            } else {
                int rowsPerSplit = rows / numSplit;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        memcpy(quantizedWeight.Data() + i * quantizedWeight.Stride(),
                                src + i * rows + quantizedWeight.Cols() * splitIdx, quantizedWeight.Cols());
                    }
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    const float *base = src + quantizedWeight.Rows() * quantizedWeight.Stride() * splitIdx;
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        memcpy(quantizedWeight.Data() + i * quantizedWeight.Stride(), base + i * cols,
                                quantizedWeight.Cols());
                    }
                }
            }
        }

        // FP32 -> FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
            if (verticalSplit) {
                int colsPerSplit = cols / numSplit;
                if (trans) { // right
                    quantizedWeight.Resize(colsPerSplit, rows);
                    const float *base = src + quantizedWeight.Rows() * quantizedWeight.Stride() * splitIdx;
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        float16_t::cvt_float_to_float16(base + i * rows,
                                quantizedWeight.Data() + i * quantizedWeight.Stride(), quantizedWeight.Cols());
                    }
                } else {
                    quantizedWeight.Resize(rows, colsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        float16_t::cvt_float_to_float16(src + i * cols + quantizedWeight.Cols() * splitIdx,
                                quantizedWeight.Data() + i * quantizedWeight.Stride(), quantizedWeight.Cols());
                    }
                }
            } else {
                int rowsPerSplit = rows / numSplit;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        float16_t::cvt_float_to_float16(src + i * rows + quantizedWeight.Cols() * splitIdx,
                                quantizedWeight.Data() + i * quantizedWeight.Stride(), quantizedWeight.Cols());
                    }
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    const float *base = src + quantizedWeight.Rows() * quantizedWeight.Stride() * splitIdx;
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        float16_t::cvt_float_to_float16(base + i * cols,
                                quantizedWeight.Data() + i * quantizedWeight.Stride(), quantizedWeight.Cols());
                    }
                }
            }
        }

        // FP32 -> BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
            if (verticalSplit) {
                int colsPerSplit = cols / numSplit;
                if (trans) { // right
                    quantizedWeight.Resize(colsPerSplit, rows);
                    const float *base = src + quantizedWeight.Rows() * quantizedWeight.Stride() * splitIdx;
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        for (int j = 0; j < quantizedWeight.Cols(); ++j) {
                            quantizedWeight.Data()[i * quantizedWeight.Stride() + j] = bfloat16_t(base[i * rows + j]);
                        }
                    }
                } else {
                    quantizedWeight.Resize(rows, colsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        for (int j = 0; j < quantizedWeight.Cols(); ++j) {
                            quantizedWeight.Data()[i * quantizedWeight.Stride() + j]
                                    = bfloat16_t(src[i * cols + quantizedWeight.Cols() * splitIdx + j]);
                        }
                    }
                }
            } else {
                int rowsPerSplit = rows / numSplit;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        for (int j = 0; j < quantizedWeight.Cols(); ++j) {
                            quantizedWeight.Data()[i * quantizedWeight.Stride() + j]
                                    = bfloat16_t(src[i * rows + quantizedWeight.Cols() * splitIdx + j]);
                        }
                    }
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    const float *base = src + quantizedWeight.Rows() * quantizedWeight.Stride() * splitIdx;
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        for (int j = 0; j < quantizedWeight.Cols(); ++j) {
                            quantizedWeight.Data()[i * quantizedWeight.Stride() + j] = bfloat16_t(base[i * cols + j]);
                        }
                    }
                }
            }
        }

        // FP32 -> INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
            if (verticalSplit) {
                int colsPerSplit = cols / numSplit;
                if (trans) {
                    quantizedWeight.Resize(colsPerSplit, rows);
                    scaleWeight.Resize(colsPerSplit);
                    zeroWeight.Resize(colsPerSplit);
                } else {
                    quantizedWeight.Resize(rows, colsPerSplit);
                    scaleWeight.Resize(colsPerSplit);
                    zeroWeight.Resize(colsPerSplit);
                }
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
                ig_sgemm_f32i8f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * colsPerSplit * splitIdx) : (src + colsPerSplit * splitIdx),
                        trans ? rows : cols, 0.9999f, quantizedWeight.Data(), trans ? rows : colsPerSplit,
                        scaleWeight.Data(), zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
                ig_hgemm_f32i8f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * colsPerSplit * splitIdx) : (src + colsPerSplit * splitIdx),
                        trans ? rows : cols, 0.9999f, quantizedWeight.Data(), trans ? rows : colsPerSplit,
                        scaleWeight.Data(), zeroWeight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
            } else {
                int rowsPerSplit = rows / numSplit;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
                    scaleWeight.Resize(cols);
                    zeroWeight.Resize(cols);
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    scaleWeight.Resize(cols);
                    zeroWeight.Resize(cols);
                }
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
                ig_sgemm_f32i8f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + rowsPerSplit * splitIdx) : (src + rowsPerSplit * cols * splitIdx),
                        trans ? rows : cols, 0.9999f, quantizedWeight.Data(), trans ? rowsPerSplit : cols,
                        scaleWeight.Data(), zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
                ig_hgemm_f32i8f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + rowsPerSplit * splitIdx) : (src + rowsPerSplit * cols * splitIdx),
                        trans ? rows : cols, 0.9999f, quantizedWeight.Data(), trans ? rowsPerSplit : cols,
                        scaleWeight.Data(), zeroWeight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
            }
        }
    }

    template <typename WeiT>
    static void convertWeight(bool trans, int rows, int cols, const float *src, hpj::Matrix<WeiT> &quantizedWeight,
            hpj::Vector<float> &scaleWeight, hpj::Vector<float> &zeroWeight) {
        convertWeight(trans, rows, cols, src, 1, 0, true, quantizedWeight, scaleWeight, zeroWeight);
    }

    template <typename WeiT>
    static void convertWeight(DecoderContext *ctx, bool trans, int rows, int cols, const float *src, bool verticalSplit,
            hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float> &scaleWeight, hpj::Vector<float> &zeroWeight) {
        convertWeight(trans, rows, cols, src, ctx->numSplit, ctx->splitIdx, verticalSplit, quantizedWeight, scaleWeight,
                zeroWeight);
    }

    template <typename WeiT>
    static void packWeight(bool trans, hpj::Matrix<WeiT> &src, hpj::Matrix<WeiT> &weight) {
        weight.Resize(trans ? src.Cols() : src.Rows(), trans ? src.Rows() : src.Cols());
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            ig_sgemm_packb(trans, trans ? src.Rows() : src.Cols(), trans ? src.Cols() : src.Rows(), src.Data(),
                    src.Stride(), weight.Data());
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            ig_sgemm_f32f16f32_packb(trans, trans ? src.Rows() : src.Cols(), trans ? src.Cols() : src.Rows(),
                    src.Data(), src.Stride(), weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            ig_hgemm_f32f16f32_packb(trans, trans ? src.Rows() : src.Cols(), trans ? src.Cols() : src.Rows(),
                    src.Data(), src.Stride(), weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            ig_sgemm_f32bf16f32_packb(trans, trans ? src.Rows() : src.Cols(), trans ? src.Cols() : src.Rows(),
                    src.Data(), src.Stride(), weight.Data());
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            ig_bgemm_f32bf16f32_packb(trans, trans ? src.Rows() : src.Cols(), trans ? src.Cols() : src.Rows(),
                    src.Data(), src.Stride(), weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            ig_sgemm_f32i8f32_packb(trans, trans ? src.Rows() : src.Cols(), trans ? src.Cols() : src.Rows(), src.Data(),
                    src.Stride(), weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            ig_hgemm_f32i8f32_packb(trans, trans ? src.Rows() : src.Cols(), trans ? src.Cols() : src.Rows(), src.Data(),
                    src.Stride(), weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename WeiT>
    static void compute(bool transA, int M, int N, int K, float alpha, const float *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, float beta, float *C, int ldc) {
        TimeLine t("MMHelper.compute");
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            ig_sgemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            ig_sgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            ig_hgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            ig_sgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                ig_amx_sgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
            } else {
                ig_bgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            ig_sgemm_f32i8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            ig_hgemm_f32i8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename WeiT>
    static void compute_bias(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, float *C, int ldc,
            const float *bias) {
        TimeLine t("MMHelper.compute_bias");
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            ig_sgemm_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            ig_sgemm_f32f16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            ig_hgemm_f32f16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            ig_sgemm_f32bf16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                ig_amx_sgemm_f32bf16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
            } else {
                ig_bgemm_f32bf16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            ig_sgemm_f32i8f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            ig_hgemm_f32i8f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename WeiT>
    static void compute_biasadd_relu(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, float *C, int ldc,
            const float *bias) {
        TimeLine t("MMHelper.compute_biasadd_relu");
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            ig_sgemm_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            ig_sgemm_f32f16f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            ig_hgemm_f32f16f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            ig_sgemm_f32bf16f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                ig_amx_sgemm_f32bf16f32_compute_biasadd_relu(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
            } else {
                ig_bgemm_f32bf16f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            ig_sgemm_f32i8f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            ig_hgemm_f32i8f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename WeiT>
    static void compute_silu(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, float *C, int ldc) {
        TimeLine t("MMHelper.compute_silu");
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            ig_sgemm_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            ig_sgemm_f32f16f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            ig_hgemm_f32f16f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            ig_sgemm_f32bf16f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                ig_amx_sgemm_f32bf16f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
            } else {
                ig_bgemm_f32bf16f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            ig_sgemm_f32i8f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            ig_hgemm_f32i8f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename WeiT>
    static void compute_resmul(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, float *C, int ldc,
            const float *res, int ldres) {
        TimeLine t("MMHelper.compute_resmul");
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            ig_sgemm_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            ig_sgemm_f32f16f32_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            ig_hgemm_f32f16f32_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            ig_sgemm_f32bf16f32_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                ig_amx_sgemm_f32bf16f32_compute_resmul(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
            } else {
                ig_bgemm_f32bf16f32_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            ig_sgemm_f32i8f32_compute_resmul(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            ig_hgemm_f32i8f32_compute_resmul(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename WeiT>
    static void compute_residential(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, float *C, int ldc,
            const float *bias, const float *res, int ldres) {
        TimeLine t("MMHelper.compute_residential");
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            ig_sgemm_compute_residential(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            ig_sgemm_f32f16f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            ig_hgemm_f32f16f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            ig_sgemm_f32bf16f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                ig_amx_sgemm_f32bf16f32_compute_residential(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
            } else {
                ig_bgemm_f32bf16f32_compute_residential(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            ig_sgemm_f32i8f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            ig_hgemm_f32i8f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename WeiT>
    static void compute_resext(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, float *C, int ldc,
            const float *bias, float gamma, float *res, int ldres) {
        TimeLine t("MMHelper.compute_resext");
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            ig_sgemm_compute_resext(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            ig_sgemm_f32f16f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            ig_hgemm_f32f16f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            ig_sgemm_f32bf16f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < M; ++i) {
                    for (int j = 0; j < N; ++j) {
                        res[i * ldres + j] = res[i * ldres + j] * gamma;
                    }
                }
                ig_amx_sgemm_f32bf16f32_compute_residential(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
            } else {
                ig_bgemm_f32bf16f32_compute_resext(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            ig_sgemm_f32i8f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            ig_hgemm_f32i8f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

private:
    static dnnl::engine &get_dnnl_engine() {
        static dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        return engine;
    }

    static dnnl::stream &get_dnnl_stream() {
        static dnnl::stream engine_stream(get_dnnl_engine());
        return engine_stream;
    }

    static void ig_amx_sgemm_f32bf16f32_compute(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const bfloat16_t *packedB, float beta, float *C, int ldc) {
        TimeLine t("MMHelper.ig_amx_sgemm_f32bf16f32_compute");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        // Source (A), weights (B), and destination (C) matrix dimensions.
        memory::dims input_dims = {M, K};
        memory::dims weight_dims = {K, N};
        memory::dims output_dims = {M, N};

        // Create user memory descriptors and memory objects for src, weights, bias, and dst.
        auto user_input_md = memory::desc(input_dims, dt::f32, tag::ab);
        auto user_input_mem = memory(user_input_md, get_dnnl_engine(), const_cast<float *>(A));

        // Create memory descriptors and memory objects for src, weights, bias, and dst.
        auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
        auto weight_md = memory::desc(weight_dims, dt::bf16, tag::BA16a64b2a);
        auto output_md = memory::desc(output_dims, dt::f32, tag::ab);

        // Create primitive descriptor.
        auto matmul_pd = matmul::primitive_desc(get_dnnl_engine(), input_md, weight_md, output_md);

        // Repack and convert input data.
        auto input_mem = memory(matmul_pd.src_desc(), get_dnnl_engine());
        auto weight_mem = memory(matmul_pd.weights_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(packedB));
        auto output_mem = memory(matmul_pd.dst_desc(), get_dnnl_engine(), C);

        // Create the primitive.
        auto matmul_prim = matmul(matmul_pd);
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});

        // Executions.
        reorder(user_input_mem, input_mem).execute(get_dnnl_stream(), user_input_mem, input_mem);
        matmul_prim.execute(get_dnnl_stream(), matmul_args);
        get_dnnl_stream().wait();
    }

    static void ig_amx_sgemm_f32bf16f32_compute_biasadd(bool transA, int M, int N, int K, float alpha, const float *A,
            int lda, const bfloat16_t *packedB, float beta, float *C, int ldc, const float *bias) {
        TimeLine t("MMHelper.ig_amx_sgemm_f32bf16f32_compute_biasadd");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        // Source (A), weights (B), and destination (C) matrix dimensions.
        memory::dims input_dims = {M, K};
        memory::dims weight_dims = {K, N};
        memory::dims bias_dims = {1, N};
        memory::dims output_dims = {M, N};

        // Create user memory descriptors and memory objects for src, weights, bias, and dst.
        auto user_input_md = memory::desc(input_dims, dt::f32, tag::ab);
        auto user_input_mem = memory(user_input_md, get_dnnl_engine(), const_cast<float *>(A));

        // Create memory descriptors and memory objects for src, weights, bias, and dst.
        auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
        auto weight_md = memory::desc(weight_dims, dt::bf16, tag::BA16a64b2a);
        auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
        auto output_md = memory::desc(output_dims, dt::f32, tag::ab);

        // Create primitive descriptor.
        auto matmul_pd = matmul::primitive_desc(get_dnnl_engine(), input_md, weight_md, bias_md, output_md);

        // Repack and convert input data.
        auto input_mem = memory(matmul_pd.src_desc(), get_dnnl_engine());
        auto weight_mem = memory(matmul_pd.weights_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(packedB));
        auto bias_mem = memory(matmul_pd.bias_desc(), get_dnnl_engine(), const_cast<float *>(bias));
        auto output_mem = memory(matmul_pd.dst_desc(), get_dnnl_engine(), C);

        // Create the primitive.
        auto matmul_prim = matmul(matmul_pd);
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});

        // Executions.
        reorder(user_input_mem, input_mem).execute(get_dnnl_stream(), user_input_mem, input_mem);
        matmul_prim.execute(get_dnnl_stream(), matmul_args);
        get_dnnl_stream().wait();
    }

    static void ig_amx_sgemm_f32bf16f32_compute_biasadd_relu(bool transA, int M, int N, int K, float alpha,
            const float *A, int lda, const bfloat16_t *packedB, float beta, float *C, int ldc, const float *bias) {
        TimeLine t("MMHelper.ig_amx_sgemm_f32bf16f32_compute_biasadd_relu");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        // Source (A), weights (B), and destination (C) matrix dimensions.
        memory::dims input_dims = {M, K};
        memory::dims weight_dims = {K, N};
        memory::dims bias_dims = {1, N};
        memory::dims output_dims = {M, N};

        // Create memory descriptors and memory objects for src, weights, bias, and dst.
        auto user_input_md = memory::desc(input_dims, dt::f32, tag::ab);
        auto user_input_mem = memory(user_input_md, get_dnnl_engine(), const_cast<float *>(A));

        // Create primitive descriptor.
        auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
        auto weight_md = memory::desc(weight_dims, dt::bf16, tag::BA16a64b2a);
        auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
        auto output_md = memory::desc(output_dims, dt::f32, tag::ab);

        // Create primitive post-ops (ReLU).
        const float post_alpha = 0.0f;
        const float post_beta = 0.0f;
        post_ops matmul_ops;
        matmul_ops.append_eltwise(algorithm::eltwise_relu, post_alpha, post_beta);
        primitive_attr matmul_attr;
        matmul_attr.set_post_ops(matmul_ops);

        // Create primitive descriptor.
        auto matmul_pd
                = matmul::primitive_desc(get_dnnl_engine(), input_md, weight_md, bias_md, output_md, matmul_attr);

        // Repack and convert input data.
        auto input_mem = memory(matmul_pd.src_desc(), get_dnnl_engine());
        auto weight_mem = memory(matmul_pd.weights_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(packedB));
        auto bias_mem = memory(matmul_pd.bias_desc(), get_dnnl_engine(), const_cast<float *>(bias));
        auto output_mem = memory(matmul_pd.dst_desc(), get_dnnl_engine(), C);

        // Create the primitive.
        auto matmul_prim = matmul(matmul_pd);
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});

        // Executions.
        reorder(user_input_mem, input_mem).execute(get_dnnl_stream(), user_input_mem, input_mem);
        matmul_prim.execute(get_dnnl_stream(), matmul_args);
        get_dnnl_stream().wait();
    }

    static void ig_amx_sgemm_f32bf16f32_compute_silu(bool transA, int M, int N, int K, float alpha, const float *A,
            int lda, const bfloat16_t *packedB, float beta, float *C, int ldc) {
        TimeLine t("MMHelper.ig_amx_sgemm_f32bf16f32_compute_silu");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        // Source (A), weights (B), and destination (C) matrix dimensions.
        memory::dims input_dims = {M, K};
        memory::dims weight_dims = {K, N};
        memory::dims output_dims = {M, N};

        // Create memory descriptors and memory objects for src, weights, bias, and dst.
        auto user_input_md = memory::desc(input_dims, dt::f32, tag::ab);
        auto user_input_mem = memory(user_input_md, get_dnnl_engine(), const_cast<float *>(A));

        // Create primitive descriptor.
        auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
        auto weight_md = memory::desc(weight_dims, dt::bf16, tag::BA16a64b2a);
        auto output_md = memory::desc(output_dims, dt::f32, tag::ab);

        // Create primitive post-ops (SiLU).
        const float post_alpha = 1.0f;
        const float post_beta = 0.0f;
        post_ops matmul_ops;
        matmul_ops.append_eltwise(algorithm::eltwise_swish, post_alpha, post_beta);
        primitive_attr matmul_attr;
        matmul_attr.set_post_ops(matmul_ops);

        // Create primitive descriptor.
        auto matmul_pd = matmul::primitive_desc(get_dnnl_engine(), input_md, weight_md, output_md, matmul_attr);

        // Repack and convert input data.
        auto input_mem = memory(matmul_pd.src_desc(), get_dnnl_engine());
        auto weight_mem = memory(matmul_pd.weights_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(packedB));
        auto output_mem = memory(matmul_pd.dst_desc(), get_dnnl_engine(), C);

        // Create the primitive.
        auto matmul_prim = matmul(matmul_pd);
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});

        // Executions.
        reorder(user_input_mem, input_mem).execute(get_dnnl_stream(), user_input_mem, input_mem);
        matmul_prim.execute(get_dnnl_stream(), matmul_args);
        get_dnnl_stream().wait();
    }

    // TODO: may be error
    static void ig_amx_sgemm_f32bf16f32_compute_resmul(bool transA, int M, int N, int K, float alpha, const float *A,
            int lda, const bfloat16_t *packedB, float beta, float *C, int ldc, const float *res, int ldres) {
        TimeLine t("MMHelper.ig_amx_sgemm_f32bf16f32_compute_resmul");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        // Source (A), weights (B), and destination (C) matrix dimensions.
        memory::dims input_dims = {M, K};
        memory::dims weight_dims = {K, N};
        memory::dims scale_dims = {M, N};
        memory::dims output_dims = {M, N};

        // Create memory descriptors and memory objects for src, weights, bias, and dst.
        auto user_input_md = memory::desc(input_dims, dt::f32, tag::ab);
        auto user_input_mem = memory(user_input_md, get_dnnl_engine(), const_cast<float *>(A));

        // Create primitive descriptor.
        auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
        auto weight_md = memory::desc(weight_dims, dt::bf16, tag::BA16a64b2a);
        auto scale_md = memory::desc(scale_dims, dt::f32, tag::ab);
        auto output_md = memory::desc(output_dims, dt::f32, tag::ab);

        // Create primitive post-ops (resmul).
        post_ops binary_ops;
        // dst_tmp = dst_tmp * scale
        binary_ops.append_binary(algorithm::binary_mul, scale_md);
        primitive_attr binary_attr;
        binary_attr.set_post_ops(binary_ops);

        // Create primitive descriptor.
        auto matmul_pd = matmul::primitive_desc(get_dnnl_engine(), input_md, weight_md, output_md, binary_attr);

        // Repack and convert input data.
        auto input_mem = memory(matmul_pd.src_desc(), get_dnnl_engine());
        auto weight_mem = memory(matmul_pd.weights_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(packedB));
        auto scale_mem = memory(scale_md, get_dnnl_engine(), const_cast<float *>(res));
        auto output_mem = memory(matmul_pd.dst_desc(), get_dnnl_engine(), C);

        // Create the primitive.
        auto matmul_prim = matmul(matmul_pd);
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, scale_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});

        // Executions.
        reorder(user_input_mem, input_mem).execute(get_dnnl_stream(), user_input_mem, input_mem);
        matmul_prim.execute(get_dnnl_stream(), matmul_args);
        get_dnnl_stream().wait();
    }

    static void ig_amx_sgemm_f32bf16f32_compute_residential(bool transA, int M, int N, int K, float alpha,
            const float *A, int lda, const bfloat16_t *packedB, float beta, float *C, int ldc, const float *bias,
            const float *res, int ldres) {
        TimeLine t("MMHelper.ig_amx_sgemm_f32bf16f32_compute_residential");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        // Source (A), weights (B), and destination (C) matrix dimensions.
        memory::dims input_dims = {M, K};
        memory::dims weight_dims = {K, N};
        memory::dims bias_dims = {1, N};
        memory::dims shift_dims = {M, N};
        memory::dims output_dims = {M, N};

        // Create memory descriptors and memory objects for src, weights, bias, and dst.
        auto user_input_md = memory::desc(input_dims, dt::f32, tag::ab);
        auto user_input_mem = memory(user_input_md, get_dnnl_engine(), const_cast<float *>(A));

        // Create primitive descriptor.
        auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
        auto weight_md = memory::desc(weight_dims, dt::bf16, tag::BA16a64b2a);
        auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
        auto shift_md = memory::desc(shift_dims, dt::f32, tag::ab);
        auto output_md = memory::desc(output_dims, dt::f32, tag::ab);

        // Create primitive post-ops (residential): dst_tmp = dst_tmp + shift
        post_ops matmul_ops;
        matmul_ops.append_binary(algorithm::binary_add, shift_md);
        primitive_attr matmul_attr;
        matmul_attr.set_post_ops(matmul_ops);

        if (bias != nullptr) {
            // Create primitive descriptor.
            auto matmul_pd
                    = matmul::primitive_desc(get_dnnl_engine(), input_md, weight_md, bias_md, output_md, matmul_attr);

            // Repack and convert input data.
            auto input_mem = memory(matmul_pd.src_desc(), get_dnnl_engine());
            auto weight_mem = memory(matmul_pd.weights_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(packedB));
            auto bias_mem = memory(matmul_pd.bias_desc(), get_dnnl_engine(), const_cast<float *>(bias));
            auto shift_mem = memory(shift_md, get_dnnl_engine(), const_cast<float *>(res));
            auto output_mem = memory(matmul_pd.dst_desc(), get_dnnl_engine(), C);

            // Create the primitive.
            auto matmul_prim = matmul(matmul_pd);
            std::unordered_map<int, memory> matmul_args;
            matmul_args.insert({DNNL_ARG_SRC, input_mem});
            matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
            matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
            matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, shift_mem});
            matmul_args.insert({DNNL_ARG_DST, output_mem});

            // Executions.
            reorder(user_input_mem, input_mem).execute(get_dnnl_stream(), user_input_mem, input_mem);
            matmul_prim.execute(get_dnnl_stream(), matmul_args);
            get_dnnl_stream().wait();
        } else {
            // Create primitive descriptor.
            auto matmul_pd = matmul::primitive_desc(get_dnnl_engine(), input_md, weight_md, output_md, matmul_attr);

            // Repack and convert input data.
            auto input_mem = memory(matmul_pd.src_desc(), get_dnnl_engine());
            auto weight_mem = memory(matmul_pd.weights_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(packedB));
            auto shift_mem = memory(shift_md, get_dnnl_engine(), const_cast<float *>(res));
            auto output_mem = memory(matmul_pd.dst_desc(), get_dnnl_engine(), C);

            // Create the primitive.
            auto matmul_prim = matmul(matmul_pd);
            std::unordered_map<int, memory> matmul_args;
            matmul_args.insert({DNNL_ARG_SRC, input_mem});
            matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
            matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, shift_mem});
            matmul_args.insert({DNNL_ARG_DST, output_mem});

            // Executions.
            reorder(user_input_mem, input_mem).execute(get_dnnl_stream(), user_input_mem, input_mem);
            matmul_prim.execute(get_dnnl_stream(), matmul_args);
            get_dnnl_stream().wait();
        }
    }
};
