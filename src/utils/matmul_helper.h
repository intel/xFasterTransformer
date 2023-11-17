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
#include "bfloat16.h"
#include "float16.h"
#include "xdnn.h"
#include "my_types.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_version.h"
#include "split_util.h"
#include "timeline.h"
#include "transformer_ctx.h"

#include <map>
#include <tuple>
#include <cstring>

#define USE_AMX_M 8

class MMHelper {
public:
    // Pack the MatMul weight from 'src(rows, cols)' to 'weight'
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
    static void convertWeight(bool trans, int rows, int cols, const float *src, int splitOffset, int splitSize,
            bool verticalSplit, hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight, bool unused) {
        // FP32 transpose
        if constexpr (std::is_same_v<WeiT, float>) {
            if (verticalSplit) {
                int colsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(colsPerSplit, rows);
                    const float *base = src + splitOffset * quantizedWeight.Stride();
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        memcpy(quantizedWeight.Data() + i * quantizedWeight.Stride(), base + i * rows,
                                quantizedWeight.Cols());
                    }
                } else {
                    quantizedWeight.Resize(rows, colsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        memcpy(quantizedWeight.Data() + i * quantizedWeight.Stride(), src + i * cols + splitOffset,
                                quantizedWeight.Cols());
                    }
                }
            } else {
                int rowsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        memcpy(quantizedWeight.Data() + i * quantizedWeight.Stride(),
                                src + i * rows + splitOffset, quantizedWeight.Cols());
                    }
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    const float *base = src + splitOffset * quantizedWeight.Stride();
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
                int colsPerSplit = splitSize;
                if (trans) { // right
                    quantizedWeight.Resize(colsPerSplit, rows);
                    const float *base = src + splitOffset * quantizedWeight.Stride();
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        float16_t::cvt_float_to_float16(base + i * rows,
                                quantizedWeight.Data() + i * quantizedWeight.Stride(), quantizedWeight.Cols());
                    }
                } else {
                    quantizedWeight.Resize(rows, colsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        float16_t::cvt_float_to_float16(src + i * cols + splitOffset,
                                quantizedWeight.Data() + i * quantizedWeight.Stride(), quantizedWeight.Cols());
                    }
                }
            } else {
                int rowsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        float16_t::cvt_float_to_float16(src + i * rows + splitOffset,
                                quantizedWeight.Data() + i * quantizedWeight.Stride(), quantizedWeight.Cols());
                    }
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    const float *base = src + splitOffset * quantizedWeight.Stride();
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
                int colsPerSplit = splitSize;
                if (trans) { // right
                    quantizedWeight.Resize(colsPerSplit, rows);
                    const float *base = src + splitOffset * quantizedWeight.Stride();
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
                                    = bfloat16_t(src[i * cols + splitOffset + j]);
                        }
                    }
                }
            } else {
                int rowsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        for (int j = 0; j < quantizedWeight.Cols(); ++j) {
                            quantizedWeight.Data()[i * quantizedWeight.Stride() + j]
                                    = bfloat16_t(src[i * rows + splitOffset + j]);
                        }
                    }
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    const float *base = src + splitOffset * quantizedWeight.Stride();
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
                int colsPerSplit = splitSize;
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
                xdnn_sgemm_f32i8f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset),
                        trans ? rows : cols, 0.9999f, quantizedWeight.Data(), trans ? rows : colsPerSplit,
                        scaleWeight.Data(), zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
                xdnn_hgemm_f32i8f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset),
                        trans ? rows : cols, 0.9999f, quantizedWeight.Data(), trans ? rows : colsPerSplit,
                        scaleWeight.Data(), zeroWeight.Data());
#else
                printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
                exit(-1);
#endif
            } else {
                int rowsPerSplit = splitSize;
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
                xdnn_sgemm_f32i8f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols),
                        trans ? rows : cols, 0.9999f, quantizedWeight.Data(), trans ? rowsPerSplit : cols,
                        scaleWeight.Data(), zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
                xdnn_hgemm_f32i8f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols),
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
    static void convertWeight(bool trans, int rows, int cols, const float *src, int numSplit, int splitIdx,
            bool verticalSplit, hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight) {
        int totalSize = verticalSplit ? cols : rows;
        std::pair<int, int> range = SplitUtil::getTaskRange(totalSize, numSplit, splitIdx);

        int splitSize = range.second - range.first;
        int splitOffset = range.first;

        convertWeight(trans, rows, cols, src, splitOffset, splitSize, verticalSplit, quantizedWeight, scaleWeight,
                zeroWeight, true);
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
        int K = trans ? src.Cols() : src.Rows();
        int N = trans ? src.Rows() : src.Cols();

        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            weight.Resize(K, N);
            xdnn_sgemm_packb(trans, N, K, src.Data(), src.Stride(), weight.Data());
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
            weight.Resize(K, N);
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            xdnn_sgemm_f32f16f32_packb(trans, N, K, (const XDNN_FP16 *)src.Data(), src.Stride(), (XDNN_FP16 *)weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            xdnn_hgemm_f32f16f32_packb(trans, N, K, (const XDNN_FP16 *)src.Data(), src.Stride(), (XDNN_FP16 *)weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
            set_amx_data_type(dnnl::memory::format_tag::BA16a64b2a);
            int amx_rows = (int)((K + 15) / 16) * 16;
            int amx_cols = (int)((N + 63) / 64) * 64;
            weight.Resize(amx_rows, amx_cols);
            memset(weight.Data(), 0, amx_rows * amx_cols * sizeof(bfloat16_t));
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            xdnn_sgemm_f32bf16f32_packb(trans, N, K, (const XDNN_BF16 *)src.Data(), src.Stride(), (XDNN_BF16 *)weight.Data(), 16, 64);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            xdnn_bgemm_f32bf16f32_packb(trans, N, K, (const XDNN_BF16 *)src.Data(), src.Stride(), (XDNN_BF16 *)weight.Data(), 16, 64);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
            weight.Resize(K, N);
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            xdnn_sgemm_f32i8f32_packb(trans, N, K, src.Data(), src.Stride(), weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            xdnn_hgemm_f32i8f32_packb(trans, N, K, src.Data(), src.Stride(), weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            TimeLine t("xdnn_sgemm_compute");
            xdnn_sgemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            TimeLine t("xdnn_sgemm_f32f16f32_compute");
            xdnn_sgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            TimeLine t("xdnn_hgemm_f32f16f32_compute");
            xdnn_hgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            TimeLine t("xdnn_sgemm_f32bf16f32_compute");
            xdnn_sgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                TimeLine t("onednn_amx_sgemm_f32bf16f32_compute");
                onednn_amx_sgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
            } else {
                TimeLine t("xdnn_bgemm_f32bf16f32_compute");
                xdnn_bgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            TimeLine t("xdnn_sgemm_f32i8f32_compute");
            xdnn_sgemm_f32i8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            TimeLine t("xdnn_hgemm_f32i8f32_compute");
            xdnn_hgemm_f32i8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_bias(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc,
            const float *bias) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            TimeLine t("xdnn_sgemm_compute_biasadd");
            xdnn_sgemm_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            TimeLine t("xdnn_sgemm_f32f16f32_compute_biasadd");
            xdnn_sgemm_f32f16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            TimeLine t("xdnn_hgemm_f32f16f32_compute_biasadd");
            xdnn_hgemm_f32f16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            TimeLine t("xdnn_sgemm_f32bf16f32_compute_biasadd");
            xdnn_sgemm_f32bf16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_biasadd");
                onednn_amx_sgemm_f32bf16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
            } else {
                TimeLine t("xdnn_bgemm_f32bf16f32_compute_biasadd");
                xdnn_bgemm_f32bf16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            TimeLine t("xdnn_sgemm_f32i8f32_compute_biasadd");
            xdnn_sgemm_f32i8f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            TimeLine t("xdnn_hgemm_f32i8f32_compute_biasadd");
            xdnn_hgemm_f32i8f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_biasadd_relu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc,
            const float *bias) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            TimeLine t("xdnn_sgemm_compute_biasadd_relu");
            xdnn_sgemm_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            TimeLine t("xdnn_sgemm_f32f16f32_compute_biasadd_relu");
            xdnn_sgemm_f32f16f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            TimeLine t("xdnn_hgemm_f32f16f32_compute_biasadd_relu");
            xdnn_hgemm_f32f16f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            TimeLine t("xdnn_sgemm_f32bf16f32_compute_biasadd_relu");
            xdnn_sgemm_f32bf16f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_biasadd_relu");
                onednn_amx_sgemm_f32bf16f32_compute_biasadd_relu(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
            } else {
                TimeLine t("xdnn_bgemm_f32bf16f32_compute_biasadd_relu");
                xdnn_bgemm_f32bf16f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            TimeLine t("xdnn_sgemm_f32i8f32_compute_biasadd_relu");
            xdnn_sgemm_f32i8f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            TimeLine t("xdnn_hgemm_f32i8f32_compute_biasadd_relu");
            xdnn_hgemm_f32i8f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_silu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            TimeLine t("xdnn_sgemm_compute_silu");
            xdnn_sgemm_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            TimeLine t("xdnn_sgemm_f32f16f32_compute_silu");
            xdnn_sgemm_f32f16f32_compute_silu(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            TimeLine t("xdnn_hgemm_f32f16f32_compute_silu");
            xdnn_hgemm_f32f16f32_compute_silu(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            TimeLine t("xdnn_sgemm_f32bf16f32_compute_silu");
            xdnn_sgemm_f32bf16f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_silu");
                onednn_amx_sgemm_f32bf16f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
            } else {
                TimeLine t("xdnn_bgemm_f32bf16f32_compute_silu");
                xdnn_bgemm_f32bf16f32_compute_silu(transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            TimeLine t("xdnn_sgemm_f32i8f32_compute_silu");
            xdnn_sgemm_f32i8f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            TimeLine t("xdnn_hgemm_f32i8f32_compute_silu");
            xdnn_hgemm_f32i8f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_resmul(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc,
            const float *res, int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            TimeLine t("xdnn_sgemm_compute_resmul");
            xdnn_sgemm_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            TimeLine t("xdnn_sgemm_f32f16f32_compute_resmul");
            xdnn_sgemm_f32f16f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            TimeLine t("xdnn_hgemm_f32f16f32_compute_resmul");
            xdnn_hgemm_f32f16f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            TimeLine t("xdnn_sgemm_f32bf16f32_compute_resmul");
            xdnn_sgemm_f32bf16f32_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_resmul");
                onednn_amx_sgemm_f32bf16f32_compute_resmul(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
            } else {
                TimeLine t("xdnn_bgemm_f32bf16f32_compute_resmul");
                xdnn_bgemm_f32bf16f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, res, ldres);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            TimeLine t("xdnn_sgemm_f32i8f32_compute_resmul");
            xdnn_sgemm_f32i8f32_compute_resmul(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            TimeLine t("xdnn_hgemm_f32i8f32_compute_resmul");
            xdnn_hgemm_f32i8f32_compute_resmul(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_residential(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc,
            const float *bias, const float *res, int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            TimeLine t("xdnn_sgemm_compute_residential");
            xdnn_sgemm_compute_residential(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            TimeLine t("xdnn_sgemm_f32f16f32_compute_residential");
            xdnn_sgemm_f32f16f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            TimeLine t("xdnn_hgemm_f32f16f32_compute_residential");
            xdnn_hgemm_f32f16f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            TimeLine t("xdnn_sgemm_f32bf16f32_compute_residential");
            xdnn_sgemm_f32bf16f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_residential");
                onednn_amx_sgemm_f32bf16f32_compute_residential(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
            } else {
                TimeLine t("xdnn_bgemm_f32bf16f32_compute_residential");
                xdnn_bgemm_f32bf16f32_compute_residential(
                        transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias, res, ldres);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            TimeLine t("xdnn_sgemm_f32i8f32_compute_residential");
            xdnn_sgemm_f32i8f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            TimeLine t("xdnn_hgemm_f32i8f32_compute_residential");
            xdnn_hgemm_f32i8f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_resext(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc,
            const float *bias, float gamma, float *res, int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            TimeLine t("xdnn_sgemm_compute_resext");
            xdnn_sgemm_compute_resext(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            TimeLine t("xdnn_sgemm_f32f16f32_compute_resext");
            xdnn_sgemm_f32f16f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias, gamma, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            TimeLine t("xdnn_hgemm_f32f16f32_compute_resext");
            xdnn_hgemm_f32f16f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias, gamma, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            TimeLine t("xdnn_sgemm_f32bf16f32_compute_resext");
            xdnn_sgemm_f32bf16f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_residential");
#pragma omp parallel for collapse(2)
                for (int i = 0; i < M; ++i) {
                    for (int j = 0; j < N; ++j) {
                        res[i * ldres + j] = res[i * ldres + j] * gamma;
                    }
                }
                onednn_amx_sgemm_f32bf16f32_compute_residential(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
            } else {
                TimeLine t("xdnn_bgemm_f32bf16f32_compute_resext");
                xdnn_bgemm_f32bf16f32_compute_resext(
                        transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias, gamma, res, ldres);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            TimeLine t("xdnn_sgemm_f32i8f32_compute_resext");
            xdnn_sgemm_f32i8f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            TimeLine t("xdnn_hgemm_f32i8f32_compute_resext");
            xdnn_hgemm_f32i8f32_compute_resext(
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

    static std::unordered_map<std::string, std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *>> &
    get_dnnl_matmul() {
        static std::unordered_map<std::string, std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *>> matmul;
        return matmul;
    }

    enum matmul_kinds {
        Basic = 0,
        BiasAdd = 1,
        BiasAdd_Relu = 2,
        Silu = 3,
        Resmul = 4,
        Residential = 5,
        Resext = 6,
    };

    static std::string create_key(bool transA, int M, int N, int K, int matmul_kind) {
        std::string key = std::to_string(transA) + "_" + std::to_string(M) + "_" + std::to_string(N) + "_"
                + std::to_string(K) + "_" + std::to_string(matmul_kind);
        return key;
    }

    template <typename Tin, typename Tout>
    static void onednn_amx_sgemm_f32bf16f32_compute(bool transA, int M, int N, int K, float alpha, const Tin *A, int lda,
            const bfloat16_t *packedB, float beta, Tout *C, int ldc) {
        TimeLine t("onednn_amx_sgemm_f32bf16f32_compute");
        TimeLine t1("onednn_amx_sgemm_f32bf16f32_compute.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Basic);
        auto it = get_dnnl_matmul().find(key);
        if (it != get_dnnl_matmul().end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B) and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims output_dims = {M, N};

            // Create memory descriptors and memory objects for src, weights, bias, and dst.
            auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_amx_data_type());
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            }
            else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            }
            else {
                printf(">>> onednn amx output date type not supported.");
            }

            // Create primitive descriptor and primitive.
            matmul_pd = new matmul::primitive_desc(get_dnnl_engine(), input_md, weight_md, output_md);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::Basic);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            get_dnnl_matmul()[key] = value;
        }

        // Repack and convert input data.
        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine());
        }
        else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(A));
        }
        else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(packedB));
        auto output_mem = memory(matmul_pd->dst_desc(), get_dnnl_engine(), C);

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        t1.release();

        // Executions.
        TimeLine t2("onednn_amx_sgemm_f32bf16f32_compute.execute_primitive");
        // Reorder
        if constexpr (std::is_same_v<Tin, float>) {
#pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
            }
        }

        matmul_prim->execute(get_dnnl_stream(), matmul_args);
        get_dnnl_stream().wait();
    }

    template <typename Tin, typename Tout>
    static void onednn_amx_sgemm_f32bf16f32_compute_biasadd(bool transA, int M, int N, int K, float alpha, const Tin *A,
            int lda, const bfloat16_t *packedB, float beta, Tout *C, int ldc, const float *bias) {
        TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_biasadd");
        TimeLine t1("onednn_amx_sgemm_f32bf16f32_compute_biasadd.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::BiasAdd);
        auto it = get_dnnl_matmul().find(key);
        if (it != get_dnnl_matmul().end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B), and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims bias_dims = {1, N};
            memory::dims output_dims = {M, N};

            // Create memory descriptors and memory objects for src, weights, bias, and dst.
            auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_amx_data_type());
            auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            }
            else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            }
            else {
                printf(">>> onednn amx output date type not supported.");
            }

            // Create primitive descriptor & primitive.
            matmul_pd = new matmul::primitive_desc(get_dnnl_engine(), input_md, weight_md, bias_md, output_md);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::BiasAdd);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            get_dnnl_matmul()[key] = value;
        }

        // Repack and convert input data.
        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine());
        }
        else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(A));
        }
        else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(packedB));
        auto bias_mem = memory(matmul_pd->bias_desc(), get_dnnl_engine(), const_cast<float *>(bias));
        auto output_mem = memory(matmul_pd->dst_desc(), get_dnnl_engine(), C);

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        t1.release();

        // Executions.
        TimeLine t2("onednn_amx_sgemm_f32bf16f32_compute_biasadd.execute_primitive");
        // Reorder
        if constexpr (std::is_same_v<Tin, float>) {
#pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
            }
        }

        matmul_prim->execute(get_dnnl_stream(), matmul_args);
        get_dnnl_stream().wait();
    }

    template <typename Tin, typename Tout>
    static void onednn_amx_sgemm_f32bf16f32_compute_biasadd_relu(bool transA, int M, int N, int K, float alpha,
            const Tin *A, int lda, const bfloat16_t *packedB, float beta, Tout *C, int ldc, const float *bias) {
        TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_biasadd_relu");
        TimeLine t1("onednn_amx_sgemm_f32bf16f32_compute_biasadd_relu.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::BiasAdd_Relu);
        auto it = get_dnnl_matmul().find(key);
        if (it != get_dnnl_matmul().end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B), and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims bias_dims = {1, N};
            memory::dims output_dims = {M, N};

            // Create primitive descriptor.
            auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_amx_data_type());
            auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            }
            else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            }
            else {
                printf(">>> onednn amx output date type not supported.");
            }

            // Create primitive post-ops (ReLU).
            const float post_alpha = 0.0f;
            const float post_beta = 0.0f;
            post_ops matmul_ops;
            matmul_ops.append_eltwise(algorithm::eltwise_relu, post_alpha, post_beta);
            primitive_attr matmul_attr;
            matmul_attr.set_post_ops(matmul_ops);

            // Create primitive descriptor & primitive.
            matmul_pd = new matmul::primitive_desc(
                    get_dnnl_engine(), input_md, weight_md, bias_md, output_md, matmul_attr);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::BiasAdd_Relu);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            get_dnnl_matmul()[key] = value;
        }

        // Repack and convert input data.
        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine());
        }
        else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(A));
        }
        else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(packedB));
        auto bias_mem = memory(matmul_pd->bias_desc(), get_dnnl_engine(), const_cast<float *>(bias));
        auto output_mem = memory(matmul_pd->dst_desc(), get_dnnl_engine(), C);

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        t1.release();

        // Executions.
        TimeLine t2("onednn_amx_sgemm_f32bf16f32_compute_biasadd_relu.execute_primitive");
        // Reorder
        if constexpr (std::is_same_v<Tin, float>) {
#pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
            }
        }

        matmul_prim->execute(get_dnnl_stream(), matmul_args);
        get_dnnl_stream().wait();
    }

    template <typename Tin, typename Tout>
    static void onednn_amx_sgemm_f32bf16f32_compute_silu(bool transA, int M, int N, int K, float alpha, const Tin *A,
            int lda, const bfloat16_t *packedB, float beta, Tout *C, int ldc) {
        TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_silu");
        TimeLine t1("onednn_amx_sgemm_f32bf16f32_compute_silu.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Silu);
        auto it = get_dnnl_matmul().find(key);
        if (it != get_dnnl_matmul().end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B), and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims output_dims = {M, N};

            // Create primitive descriptor.
            auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_amx_data_type());
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            }
            else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            }
            else {
                printf(">>> onednn amx output date type not supported.");
            }

            // Create primitive post-ops (SiLU).
            const float post_alpha = 1.0f;
            const float post_beta = 0.0f;
            post_ops matmul_ops;
            matmul_ops.append_eltwise(algorithm::eltwise_swish, post_alpha, post_beta);
            primitive_attr matmul_attr;
            matmul_attr.set_post_ops(matmul_ops);

            // Create primitive descriptor and primitive.
            matmul_pd = new matmul::primitive_desc(get_dnnl_engine(), input_md, weight_md, output_md, matmul_attr);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::Silu);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            get_dnnl_matmul()[key] = value;
        }

        // Repack and convert input data.
        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine());
        }
        else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
           input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(A));
        }
        else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(packedB));
        auto output_mem = memory(matmul_pd->dst_desc(), get_dnnl_engine(), C);

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        t1.release();

        // Executions.
        TimeLine t2("onednn_amx_sgemm_f32bf16f32_compute_silu.execute_primitive");
        // Reorder
        if constexpr (std::is_same_v<Tin, float>) {
#pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
            }
        }

        matmul_prim->execute(get_dnnl_stream(), matmul_args);
        get_dnnl_stream().wait();
    }

    template <typename Tin, typename Tout>
    static void onednn_amx_sgemm_f32bf16f32_compute_resmul(bool transA, int M, int N, int K, float alpha, const Tin *A,
            int lda, const bfloat16_t *packedB, float beta, Tout *C, int ldc, const float *res, int ldres) {
        TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_resmul");
        TimeLine t1("onednn_amx_sgemm_f32bf16f32_compute_resmul.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Resmul);
        auto it = get_dnnl_matmul().find(key);
        if (it != get_dnnl_matmul().end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B), and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims scale_dims = {M, N};
            memory::dims output_dims = {M, N};

            // Create primitive descriptor.
            auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_amx_data_type());
            auto scale_md = memory::desc(scale_dims, dt::f32, tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            }
            else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            }
            else {
                printf(">>> onednn amx output date type not supported.");
            }

            // Create primitive post-ops (resmul).
            post_ops binary_ops;
            // dst_tmp = dst_tmp * scale
            binary_ops.append_binary(algorithm::binary_mul, scale_md);
            primitive_attr matmul_attr;
            matmul_attr.set_post_ops(binary_ops);

            // Create primitive descriptor and primitive.
            matmul_pd = new matmul::primitive_desc(get_dnnl_engine(), input_md, weight_md, output_md, matmul_attr);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::Resmul);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            get_dnnl_matmul()[key] = value;
        }

        // Repack and convert input data.
        memory::dims scale_dims = {M, N};
        auto scale_md = memory::desc(scale_dims, dt::f32, tag::ab);
        dnnl::memory scale_mem;
        if (C == res) {
            scale_mem = memory(scale_md, get_dnnl_engine());
#pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                memcpy((float *)scale_mem.get_data_handle() + i * N, res + i * ldres, N * sizeof(float));
            }
        } else {
            scale_mem = memory(scale_md, get_dnnl_engine(), const_cast<float *>(res));
        }

        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine());
        }
        else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(A));
        }
        else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(packedB));
        auto output_mem = memory(matmul_pd->dst_desc(), get_dnnl_engine(), C);

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, scale_mem});
        t1.release();

        // Executions.
        TimeLine t2("onednn_amx_sgemm_f32bf16f32_compute_resmul.execute_primitive");
        // Reorder
        if constexpr (std::is_same_v<Tin, float>) {
#pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
            }
        }

        matmul_prim->execute(get_dnnl_stream(), matmul_args);
        get_dnnl_stream().wait();
    }

     template <typename Tin, typename Tout>
    static void onednn_amx_sgemm_f32bf16f32_compute_residential(bool transA, int M, int N, int K, float alpha,
            const Tin *A, int lda, const bfloat16_t *packedB, float beta, Tout *C, int ldc, const float *bias,
            const float *res, int ldres) {
        TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_residential");
        TimeLine t1("onednn_amx_sgemm_f32bf16f32_compute_residential.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Residential);
        auto it = get_dnnl_matmul().find(key);
        if (it != get_dnnl_matmul().end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B), and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims bias_dims = {1, N};
            memory::dims shift_dims = {M, N};
            memory::dims output_dims = {M, N};

            // Create primitive descriptor.
            auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_amx_data_type());
            auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
            auto shift_md = memory::desc(shift_dims, dt::f32, tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            }
            else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            }
            else {
                printf(">>> onednn amx output date type not supported.");
            }

            // Create primitive post-ops (residential): dst_tmp = dst_tmp + shift
            post_ops matmul_ops;
            matmul_ops.append_binary(algorithm::binary_add, shift_md);
            primitive_attr matmul_attr;
            matmul_attr.set_post_ops(matmul_ops);

            if (bias != nullptr) {
                // Create primitive descriptor and primitive.
                matmul_pd = new matmul::primitive_desc(
                        get_dnnl_engine(), input_md, weight_md, bias_md, output_md, matmul_attr);
                matmul_prim = new matmul(*matmul_pd);
            } else {
                // Create primitive descriptor and primitive.
                matmul_pd = new matmul::primitive_desc(get_dnnl_engine(), input_md, weight_md, output_md, matmul_attr);
                matmul_prim = new matmul(*matmul_pd);
            }

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::Residential);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            get_dnnl_matmul()[key] = value;
        }

        // Repack and convert input data.
        memory::dims shift_dims = {M, N};
        auto shift_md = memory::desc(shift_dims, dt::f32, tag::ab);

        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine());
        }
        else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(A));
        }
        else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(packedB));
        memory bias_mem;
        auto shift_mem = memory(shift_md, get_dnnl_engine(), const_cast<float *>(res));
        auto output_mem = memory(matmul_pd->dst_desc(), get_dnnl_engine(), C);
        if (bias != nullptr) {
            bias_mem = memory(matmul_pd->bias_desc(), get_dnnl_engine(), const_cast<float *>(bias));
        }

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, shift_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        if (bias != nullptr) { matmul_args.insert({DNNL_ARG_BIAS, bias_mem}); }
        t1.release();

        // Executions.
        TimeLine t2("onednn_amx_sgemm_f32bf16f32_compute_bias_residential.execute_primitive");
        // Reorder
        if constexpr (std::is_same_v<Tin, float>) {
#pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
            }
        }

        matmul_prim->execute(get_dnnl_stream(), matmul_args);
        get_dnnl_stream().wait();
    }

private:
    static dnnl::memory::format_tag &get_amx_data_type() {
        static dnnl::memory::format_tag amx_weight_tag;
        return amx_weight_tag;
    }

    static void set_amx_data_type(dnnl::memory::format_tag tag) { get_amx_data_type() = tag; }
};
