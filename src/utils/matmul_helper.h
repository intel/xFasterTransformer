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
#include "normal_float4x2.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_version.h"
#include "split_util.h"
#include "timeline.h"
#include "transformer_ctx.h"
#include "uint4x2.h"
#include "verbose.h"
#include "xdnn.h"

#include <cstring>
#include <map>
#include <tuple>

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
            hpj::Vector<float> &zeroWeight, hpj::Vector<float> &sumWeight, bool unused) {
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
                        memcpy(quantizedWeight.Data() + i * quantizedWeight.Stride(), src + i * rows + splitOffset,
                                quantizedWeight.Cols());
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

        // FP32 -> INT8/W8A8
        else if constexpr (std::is_same_v<WeiT, int8_t> || std::is_same_v<WeiT, w8a8_t>) {
            if (verticalSplit) {
                int colsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(colsPerSplit, rows);
                    scaleWeight.Resize(colsPerSplit);
                    zeroWeight.Resize(colsPerSplit);
                    sumWeight.Resize(colsPerSplit);
                } else {
                    quantizedWeight.Resize(rows, colsPerSplit);
                    scaleWeight.Resize(colsPerSplit);
                    zeroWeight.Resize(colsPerSplit);
                    sumWeight.Resize(colsPerSplit);
                }
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
                xdnn_sgemm_f32s8f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset), trans ? rows : cols, 0.9999f,
                        (int8_t *)quantizedWeight.Data(), trans ? rows : colsPerSplit, scaleWeight.Data(),
                        zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
                xdnn_hgemm_f32s8f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset), trans ? rows : cols, 0.9999f,
                        (int8_t *)quantizedWeight.Data(), trans ? rows : colsPerSplit, scaleWeight.Data(),
                        zeroWeight.Data());
#else
                printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
                exit(-1);
#endif
                if (trans) {
#pragma omp parallel for
                    for (int i = 0; i < colsPerSplit; i++) {
                        sumWeight.Data()[i] = 0.0f;
                        for (int j = 0; j < rows; j++) {
                            sumWeight.Data()[i] += quantizedWeight.Data()[i * quantizedWeight.Stride() + j];
                        }
                    }
                } else {
#pragma omp parallel for
                    for (int i = 0; i < colsPerSplit; i++) {
                        sumWeight.Data()[i] = 0.0f;
                        for (int j = 0; j < rows; j++) {
                            sumWeight.Data()[i] += quantizedWeight.Data()[j * quantizedWeight.Stride() + i];
                        }
                    }
                }
            } else {
                int rowsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
                    scaleWeight.Resize(cols);
                    zeroWeight.Resize(cols);
                    sumWeight.Resize(cols);
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    scaleWeight.Resize(cols);
                    zeroWeight.Resize(cols);
                    sumWeight.Resize(cols);
                }
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
                xdnn_sgemm_f32s8f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols), trans ? rows : cols, 0.9999f,
                        (int8_t *)quantizedWeight.Data(), trans ? rowsPerSplit : cols, scaleWeight.Data(),
                        zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
                xdnn_hgemm_f32s8f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols), trans ? rows : cols, 0.9999f,
                        (int8_t *)quantizedWeight.Data(), trans ? rowsPerSplit : cols, scaleWeight.Data(),
                        zeroWeight.Data());
#else
                printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
                exit(-1);
#endif
                if (trans) {
#pragma omp parallel for
                    for (int i = 0; i < cols; i++) {
                        sumWeight.Data()[i] = 0.0f;
                        for (int j = 0; j < rowsPerSplit; j++) {
                            sumWeight.Data()[i] += quantizedWeight.Data()[i * quantizedWeight.Stride() + j];
                        }
                    }
                } else {
#pragma omp parallel for
                    for (int i = 0; i < cols; i++) {
                        sumWeight.Data()[i] = 0.0f;
                        for (int j = 0; j < rowsPerSplit; j++) {
                            sumWeight.Data()[i] += quantizedWeight.Data()[j * quantizedWeight.Stride() + i];
                        }
                    }
                }
            }
        }

        // FP32 -> UINT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
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
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
                xdnn_sgemm_f32u4f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset), trans ? rows : cols, 0.9999f,
                        (XDNN_UINT4x2 *)quantizedWeight.Data(), trans ? rows : colsPerSplit, scaleWeight.Data(),
                        zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
                xdnn_hgemm_f32u4f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset), trans ? rows : cols, 0.9999f,
                        (XDNN_UINT4x2 *)quantizedWeight.Data(), trans ? rows : colsPerSplit, scaleWeight.Data(),
                        zeroWeight.Data());
#else
                printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
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
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
                xdnn_sgemm_f32u4f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols), trans ? rows : cols, 0.9999f,
                        (XDNN_UINT4x2 *)quantizedWeight.Data(), trans ? rowsPerSplit : cols, scaleWeight.Data(),
                        zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
                xdnn_hgemm_f32u4f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols), trans ? rows : cols, 0.9999f,
                        (XDNN_UINT4x2 *)quantizedWeight.Data(), trans ? rowsPerSplit : cols, scaleWeight.Data(),
                        zeroWeight.Data());
#else
                printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
                exit(-1);
#endif
            }
        }

        // FP32 -> NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
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
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
                xdnn_sgemm_f32nf4f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset), trans ? rows : cols, 0.9999f,
                        (XDNN_NF4x2 *)quantizedWeight.Data(), trans ? rows : colsPerSplit, scaleWeight.Data(),
                        zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
                xdnn_hgemm_f32nf4f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset), trans ? rows : cols, 0.9999f,
                        (XDNN_NF4x2 *)quantizedWeight.Data(), trans ? rows : colsPerSplit, scaleWeight.Data(),
                        zeroWeight.Data());
#else
                printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
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
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
                xdnn_sgemm_f32nf4f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols), trans ? rows : cols, 0.9999f,
                        (XDNN_NF4x2 *)quantizedWeight.Data(), trans ? rowsPerSplit : cols, scaleWeight.Data(),
                        zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
                xdnn_hgemm_f32nf4f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols), trans ? rows : cols, 0.9999f,
                        (XDNN_NF4x2 *)quantizedWeight.Data(), trans ? rowsPerSplit : cols, scaleWeight.Data(),
                        zeroWeight.Data());
#else
                printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
                exit(-1);
#endif
            }
        }
    }

    template <typename WeiT>
    static void convertWeight(bool trans, int rows, int cols, const float *src, int numSplit, int splitIdx,
            bool verticalSplit, hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight, hpj::Vector<float> &sumWeight) {
        int totalSize = verticalSplit ? cols : rows;
        std::pair<int, int> range = SplitUtil::getTaskRange(totalSize, numSplit, splitIdx);

        int splitSize = range.second - range.first;
        int splitOffset = range.first;

        convertWeight(trans, rows, cols, src, splitOffset, splitSize, verticalSplit, quantizedWeight, scaleWeight,
                zeroWeight, sumWeight, true);
    }

    template <typename WeiT>
    static void convertWeight(bool trans, int rows, int cols, const float *src, hpj::Matrix<WeiT> &quantizedWeight,
            hpj::Vector<float> &scaleWeight, hpj::Vector<float> &zeroWeight, hpj::Vector<float> &sumWeight) {
        convertWeight(trans, rows, cols, src, 1, 0, true, quantizedWeight, scaleWeight, zeroWeight, sumWeight);
    }

    template <typename WeiT>
    static void convertWeight(DecoderContext *ctx, bool trans, int rows, int cols, const float *src, bool verticalSplit,
            hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float> &scaleWeight, hpj::Vector<float> &zeroWeight,
            hpj::Vector<float> &sumWeight) {
        convertWeight(trans, rows, cols, src, ctx->numSplit, ctx->splitIdx, verticalSplit, quantizedWeight, scaleWeight,
                zeroWeight, sumWeight);
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
            xdnn_sgemm_f32f16f32_packb(
                    trans, N, K, (const XDNN_FP16 *)src.Data(), src.Stride(), (XDNN_FP16 *)weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            xdnn_hgemm_f32f16f32_packb(
                    trans, N, K, (const XDNN_FP16 *)src.Data(), src.Stride(), (XDNN_FP16 *)weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
            int amx_rows = (int)((K + 15) / 16) * 16;
            int amx_cols = (int)((N + 63) / 64) * 64;
            weight.Resize(amx_rows, amx_cols);
            memset(weight.Data(), 0, amx_rows * amx_cols * sizeof(bfloat16_t));
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            xdnn_sgemm_f32bf16f32_packb(
                    trans, N, K, (const XDNN_BF16 *)src.Data(), src.Stride(), (XDNN_BF16 *)weight.Data(), 16, 64);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            xdnn_bgemm_f32bf16f32_packb(
                    trans, N, K, (const XDNN_BF16 *)src.Data(), src.Stride(), (XDNN_BF16 *)weight.Data(), 16, 64);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
            weight.Resize(K, N);
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            xdnn_sgemm_f32s8f32_packb(trans, N, K, src.Data(), src.Stride(), weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            xdnn_hgemm_f32s8f32_packb(trans, N, K, src.Data(), src.Stride(), weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            auto tag = trans ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab;
            dnnl::memory B_mem({{K, N}, dnnl::memory::data_type::s8, tag}, get_dnnl_engine(), src.Data());
            dnnl::memory::desc desc({K, N}, dnnl::memory::data_type::s8, get_amx_s8s8s32_weight_data_type());

            // When converting to oneDNN blocked memory format, padded dims can be larger than [K, N]
            // Resize down Matrix does not change underlying buffer.
            // TODO: Add reserve like function in Matrix, as current 2 times Resize has potential risks.
            auto dims = desc.get_padded_dims();
            weight.Resize(dims[0], dims[1]);
            weight.Resize(K, N);

            dnnl::memory packedB_mem(desc, get_dnnl_engine(), weight.Data());
            dnnl::reorder(B_mem, packedB_mem).execute(get_dnnl_stream(), B_mem, packedB_mem);
            get_dnnl_stream().wait();
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
            weight.Resize(K, N);
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            xdnn_sgemm_f32u4f32_packb(
                    trans, N, K, (const XDNN_UINT4x2 *)src.Data(), src.Stride(), (XDNN_UINT4x2 *)weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            xdnn_hgemm_f32u4f32_packb(
                    trans, N, K, (const XDNN_UINT4x2 *)src.Data(), src.Stride(), (XDNN_UINT4x2 *)weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
            weight.Resize(K, N);
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            xdnn_sgemm_f32nf4f32_packb(
                    trans, N, K, (const XDNN_NF4x2 *)src.Data(), src.Stride(), (XDNN_NF4x2 *)weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            xdnn_hgemm_f32nf4f32_packb(
                    trans, N, K, (const XDNN_NF4x2 *)src.Data(), src.Stride(), (XDNN_NF4x2 *)weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE(
                    "xdnn_sgemm_compute", xdnn_sgemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute",
                    xdnn_sgemm_f32f16f32_compute(
                            transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute",
                    xdnn_hgemm_f32f16f32_compute(
                            transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute",
                    xdnn_sgemm_f32bf16f32_compute(
                            transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, beta, C, ldc));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute",
                        onednn_amx_sgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
            } else {
                GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute",
                        xdnn_bgemm_f32bf16f32_compute(
                                transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc));
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute",
                    xdnn_sgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute",
                    xdnn_hgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, nullptr, nullptr, 0, 0.0f, matmul_kinds::Basic));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute",
                    xdnn_sgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB,
                            zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute",
                    xdnn_hgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB,
                            zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute",
                    xdnn_sgemm_f32nf4f32_compute(
                            transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute",
                    xdnn_hgemm_f32nf4f32_compute(
                            transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_bias(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc,
            const float *bias) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_biasadd",
                    xdnn_sgemm_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_biasadd",
                    xdnn_sgemm_f32f16f32_compute_biasadd(
                            transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute_biasadd",
                    xdnn_hgemm_f32f16f32_compute_biasadd(
                            transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute_biasadd",
                    xdnn_sgemm_f32bf16f32_compute_biasadd(
                            transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, beta, C, ldc, bias));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute_biasadd",
                        onednn_amx_sgemm_f32bf16f32_compute_biasadd(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias));
            } else {
                GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_biasadd",
                        xdnn_bgemm_f32bf16f32_compute_biasadd(
                                transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias));
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute_biasadd",
                    xdnn_sgemm_f32s8f32_compute_biasadd(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute_biasadd",
                    xdnn_hgemm_f32s8f32_compute_biasadd(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute_biasadd",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, bias, nullptr, 0, 0.0f, matmul_kinds::BiasAdd));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute_biasadd",
                    xdnn_sgemm_f32u4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute_biasadd",
                    xdnn_hgemm_f32u4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute_biasadd",
                    xdnn_sgemm_f32nf4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute_biasadd",
                    xdnn_hgemm_f32nf4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_biasadd_relu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C,
            int ldc, const float *bias) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_biasadd_relu",
                    xdnn_sgemm_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias));
        }
        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_biasadd_relu",
                    xdnn_sgemm_f32f16f32_compute_biasadd_relu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute_biasadd_relu",
                    xdnn_hgemm_f32f16f32_compute_biasadd_relu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute_biasadd_relu",
                    xdnn_sgemm_f32bf16f32_compute_biasadd_relu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, beta, C, ldc, bias));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute_biasadd_relu",
                        onednn_amx_sgemm_f32bf16f32_compute_biasadd_relu(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias));
            } else {
                GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_biasadd_relu",
                        xdnn_bgemm_f32bf16f32_compute_biasadd_relu(
                                transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias));
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute_biasadd_relu",
                    xdnn_sgemm_f32s8f32_compute_biasadd_relu(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute_biasadd_relu",
                    xdnn_hgemm_f32s8f32_compute_biasadd_relu(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute_biasadd_relu",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, bias, nullptr, 0, 0.0f, matmul_kinds::BiasAdd_Relu));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute_biasadd_relu",
                    xdnn_sgemm_f32u4f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda,
                            (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute_biasadd_relu",
                    xdnn_hgemm_f32u4f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda,
                            (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute_biasadd_relu",
                    xdnn_sgemm_f32nf4f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda,
                            (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute_biasadd_relu",
                    xdnn_hgemm_f32nf4f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda,
                            (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_silu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_silu",
                    xdnn_sgemm_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_silu",
                    xdnn_sgemm_f32f16f32_compute_silu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute_silu",
                    xdnn_hgemm_f32f16f32_compute_silu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute_silu",
                    xdnn_sgemm_f32bf16f32_compute_silu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, beta, C, ldc));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute_silu",
                        onednn_amx_sgemm_f32bf16f32_compute_silu(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
            } else {
                GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_silu",
                        xdnn_bgemm_f32bf16f32_compute_silu(
                                transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc));
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute_silu",
                    xdnn_sgemm_f32s8f32_compute_silu(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute_silu",
                    xdnn_hgemm_f32s8f32_compute_silu(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute_silu",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, nullptr, nullptr, 0, 0.0f, matmul_kinds::Silu));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute_silu",
                    xdnn_sgemm_f32u4f32_compute_silu(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute_silu",
                    xdnn_hgemm_f32u4f32_compute_silu(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute_silu",
                    xdnn_sgemm_f32nf4f32_compute_silu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute_silu",
                    xdnn_hgemm_f32nf4f32_compute_silu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_resmul(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C,
            int ldc, const float *res, int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_resmul",
                    xdnn_sgemm_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_resmul",
                    xdnn_sgemm_f32f16f32_compute_resmul(
                            transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute_resmul",
                    xdnn_hgemm_f32f16f32_compute_resmul(
                            transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute_resmul",
                    xdnn_sgemm_f32bf16f32_compute_resmul(
                            transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, beta, C, ldc, res, ldres));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute_resmul",
                        onednn_amx_sgemm_f32bf16f32_compute_resmul(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres));
            } else {
                GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_resmul",
                        xdnn_bgemm_f32bf16f32_compute_resmul(
                                transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, res, ldres));
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute_resmul",
                    xdnn_sgemm_f32s8f32_compute_resmul(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute_resmul",
                    xdnn_hgemm_f32s8f32_compute_resmul(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute_resmul",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, nullptr, res, ldres, 0.0f, matmul_kinds::Resmul));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute_resmul",
                    xdnn_sgemm_f32u4f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute_resmul",
                    xdnn_hgemm_f32u4f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute_resmul",
                    xdnn_sgemm_f32nf4f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute_resmul",
                    xdnn_hgemm_f32nf4f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_residential(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C,
            int ldc, const float *bias, const float *res, int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_residential",
                    xdnn_sgemm_compute_residential(
                            transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_residential",
                    xdnn_sgemm_f32f16f32_compute_residential(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB,
                            beta, C, ldc, bias, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute_residential",
                    xdnn_hgemm_f32f16f32_compute_residential(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB,
                            beta, C, ldc, bias, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute_residential",
                    xdnn_sgemm_f32bf16f32_compute_residential(transA, M, N, K, alpha, A, lda,
                            (const XDNN_UINT4x2 *)packedB, beta, C, ldc, bias, res, ldres));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute_residential",
                        onednn_amx_sgemm_f32bf16f32_compute_residential(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres));
            } else {
                GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_residential",
                        xdnn_bgemm_f32bf16f32_compute_residential(transA, M, N, K, alpha, A, lda,
                                (const XDNN_BF16 *)packedB, beta, C, ldc, bias, res, ldres));
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute_residential",
                    xdnn_sgemm_f32s8f32_compute_residential(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute_residential",
                    xdnn_hgemm_f32s8f32_compute_residential(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute_residential",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, bias, res, ldres, 0.0f, matmul_kinds::Residential));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute_residential",
                    xdnn_sgemm_f32u4f32_compute_residential(transA, M, N, K, alpha, A, lda,
                            (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute_residential",
                    xdnn_hgemm_f32u4f32_compute_residential(transA, M, N, K, alpha, A, lda,
                            (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute_residential",
                    xdnn_sgemm_f32nf4f32_compute_residential(transA, M, N, K, alpha, A, lda,
                            (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute_residential",
                    xdnn_hgemm_f32nf4f32_compute_residential(transA, M, N, K, alpha, A, lda,
                            (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_resext(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C,
            int ldc, const float *bias, float gamma, float *res, int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_resext",
                    xdnn_sgemm_compute_resext(
                            transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_resext",
                    xdnn_sgemm_f32f16f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB,
                            beta, C, ldc, bias, gamma, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute_resext",
                    xdnn_hgemm_f32f16f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB,
                            beta, C, ldc, bias, gamma, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute_resext",
                    xdnn_sgemm_f32bf16f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            beta, C, ldc, bias, gamma, res, ldres));
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
                GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_resext",
                        xdnn_bgemm_f32bf16f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB,
                                beta, C, ldc, bias, gamma, res, ldres));
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute_resext",
                    xdnn_sgemm_f32s8f32_compute_resext(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C,
                            ldc, bias, gamma, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute_resext",
                    xdnn_hgemm_f32s8f32_compute_resext(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C,
                            ldc, bias, gamma, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute_resext",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, bias, res, ldres, gamma, matmul_kinds::Resext));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute_resext",
                    xdnn_sgemm_f32u4f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute_resext",
                    xdnn_hgemm_f32u4f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute_resext",
                    xdnn_sgemm_f32nf4f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute_resext",
                    xdnn_hgemm_f32nf4f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
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

    static dnnl::memory::format_tag get_amx_f32bf16f32_weight_data_type() {
        return dnnl::memory::format_tag::BA16a64b2a;
    }

    static dnnl::memory::format_tag get_amx_s8s8s32_weight_data_type() { return dnnl::memory::format_tag::BA16a64b4a; }

    template <typename Tin, typename Tout>
    static void onednn_amx_sgemm_f32bf16f32_compute(bool transA, int M, int N, int K, float alpha, const Tin *A,
            int lda, const bfloat16_t *packedB, float beta, Tout *C, int ldc) {
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
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_amx_f32bf16f32_weight_data_type());
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            } else {
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
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(A));
        } else {
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
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_amx_f32bf16f32_weight_data_type());
            auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            } else {
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
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(A));
        } else {
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
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_amx_f32bf16f32_weight_data_type());
            auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            } else {
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
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(A));
        } else {
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
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_amx_f32bf16f32_weight_data_type());
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            } else {
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
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(A));
        } else {
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
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_amx_f32bf16f32_weight_data_type());
            auto scale_md = memory::desc(scale_dims, dt::f32, tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            } else {
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
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(A));
        } else {
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
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_amx_f32bf16f32_weight_data_type());
            auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
            auto shift_md = memory::desc(shift_dims, dt::f32, tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            } else {
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
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine(), const_cast<bfloat16_t *>(A));
        } else {
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

    static void onednn_amx_gemm_s8s8s32(bool transA, int M, int N, int K, float alpha, const int8_t *A, int lda,
            const int8_t *B, float beta, int32_t *C, int ldc) {
        TimeLine t("onednn_amx_gemm_s8s8s32");
        TimeLine t1("onednn_amx_gemm_s8s8s32.create_primitive");
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
            auto input_md = memory::desc(input_dims, dt::s8, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt::s8, get_amx_s8s8s32_weight_data_type());
            memory::desc output_md;
            output_md = memory::desc(output_dims, dt::s32, tag::ab);

            // Create primitive descriptor and primitive.
            matmul_pd = new matmul::primitive_desc(get_dnnl_engine(), input_md, weight_md, output_md);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::Basic);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            get_dnnl_matmul()[key] = value;
        }

        auto input_mem = memory(matmul_pd->src_desc(), get_dnnl_engine(), const_cast<int8_t *>(A));
        auto weight_mem = memory(matmul_pd->weights_desc(), get_dnnl_engine(), const_cast<int8_t *>(B));
        auto output_mem = memory(matmul_pd->dst_desc(), get_dnnl_engine(), C);

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        t1.release();

        // Executions.
        TimeLine t2("onednn_gemm_s8s8s32.execute_primitive");
        matmul_prim->execute(get_dnnl_stream(), matmul_args);
        get_dnnl_stream().wait();
    }

    static void onednn_amx_gemm_f32s8f32_compute(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const int8_t *B, const float *scaleB, const float *zeroB, const float *sumB, float beta, float *C, int ldc,
            const float *bias, const float *res, int ldres, float gamma, matmul_kinds kind) {
        if (transA || (N % 16) != 0 || alpha != 1.0f || beta != 0.0f) {
            printf("%s:%d: Not implemented.\n", __FILE__, __LINE__);
            exit(-1);
        }

        // split M dimension if M*N is too big
        const int max_MN = 4 * 1024 * 1024;
        int numSplit = M * N / max_MN + 1;
        for (int i = 0; i < numSplit; i++) {
            std::pair<int, int> range = SplitUtil::getTaskRange(M, numSplit, i);
            int MB = range.second - range.first;
            int offset = range.first;
            onednn_amx_gemm_f32s8f32_compute_base(transA, MB, N, K, alpha, A + lda * offset, lda, B, scaleB, zeroB,
                    sumB, beta, C + ldc * offset, ldc, bias, res + ldres * offset, ldres, gamma, kind);
        }
    }

    static void onednn_amx_gemm_f32s8f32_compute_base(bool transA, int M, int N, int K, float alpha, const float *A,
            int lda, const int8_t *B, const float *scaleB, const float *zeroB, const float *sumB, float beta, float *C,
            int ldc, const float *bias, const float *res, int ldres, float gamma, matmul_kinds kind) {

#define ALLOC(DATATYPE, VALUE, SIZE)                  \
    std::unique_ptr<DATATYPE, decltype(&free)> VALUE( \
            static_cast<DATATYPE *>(aligned_alloc(64, SIZE * sizeof(DATATYPE))), &free)
        ALLOC(int8_t, quantizedA, M * K);
        ALLOC(float, scaleA, M);
        ALLOC(float, zeroA, M);
        ALLOC(float, sumA, M);
        ALLOC(int32_t, C_int32, M * N);
#undef ALLOC

        TimeLine t1("onednn_amx_gemm_f32s8f32_compute.quantA");
        quantize_s8(M, K, A, lda, quantizedA.get(), K, scaleA.get(), zeroA.get(), sumA.get());
        t1.release();

        onednn_amx_gemm_s8s8s32(transA, M, N, K, alpha, quantizedA.get(), K, B, beta, C_int32.get(), N);

        TimeLine t2("onednn_amx_gemm_f32s8f32_compute.dequantC");
        dequant(M, N, C_int32.get(), N, C, ldc, scaleA.get(), zeroA.get(), sumA.get(), scaleB, zeroB, sumB, bias, res,
                ldres, gamma, kind);
    }

    // Per row quantization of activations
    // src: weight, dst: int8 qweight
    // weight = qweight * scale + zero
    //
    // Also compute per row sums
    // sum = sum_of_one_row(qweight * scale + zero)
    static void quantize_s8(
            int M, int N, const float *src, int lda, int8_t *dst, int ldb, float *scale, float *zero, float *sum) {
#pragma omp parallel for
        for (int i = 0; i < M; i++) {
            __m512 vmax = _mm512_loadu_ps(src + i * lda);
            __m512 vmin = vmax;
            for (int j = 16; j < N; j += 16) {
                int remain = N - j;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 x = _mm512_maskz_loadu_ps(mask, src + i * lda + j);
                vmax = _mm512_mask_max_ps(vmax, mask, vmax, x);
                vmin = _mm512_mask_min_ps(vmin, mask, vmin, x);
            }

            float fmax = _mm512_reduce_max_ps(vmax);
            float fmin = _mm512_reduce_min_ps(vmin);

            //float fscale = (fmax - fmin) / 255.0f;
            //float fzero = (127 * fmin + 128 * fmax) / 255.0f;
            float fzero = (fmin + fmax) / 2.0f;
            float fscale = std::max(std::abs(fmax - fzero), std::abs(fzero - fmin)) / 127.0f;
            scale[i] = fscale;
            zero[i] = fzero;

            // weight = qweight * scale + zero
            // qweight = weight * (1/scale) + (-zero/scale)
            __m512 vscale = _mm512_set1_ps(1.0f / fscale);
            __m512 vzero = _mm512_set1_ps(-fzero / fscale);

            __m512i vsum = _mm512_setzero_epi32();
            for (int j = 0; j < N; j += 16) {
                int remain = N - j;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 x = _mm512_maskz_loadu_ps(mask, src + i * lda + j);
                x = _mm512_maskz_fmadd_ps(mask, x, vscale, vzero);
                __m512i ix = _mm512_maskz_cvtps_epi32(mask, x);
                vsum = _mm512_add_epi32(vsum, ix);
                __m128i i8x = _mm512_maskz_cvtepi32_epi8(mask, ix);
                memcpy(dst + i * ldb + j, &i8x, (remain >= 16 ? 16 : remain));
            }
            sum[i] = _mm512_reduce_add_epi32(vsum) * fscale + N * fzero;
        }
    }

    template <typename DequantOp, typename PostOp>
    static void dequant_base(int M, int N, const int32_t *C_int32, const int ldc_int32, float *C, const int ldc,
            const DequantOp &dequant_op, const PostOp &post_op) {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j += 16) {
                __m512i xi = _mm512_load_epi32(C_int32 + i * ldc_int32 + j);
                __m512 x = dequant_op(xi, i, j);
                x = post_op(x, i, j);
                _mm512_storeu_ps(C + i * ldc + j, x);
            }
        }
    }

    // dequant C_int32 to C
    //
    // sumA = sum(QWeightA * ScaleA + ZeroA)
    // sumB = sum(QWeightB)
    // C_int32 = QWeightA * QWeightB
    //
    // C = WeightA * WeightB
    //   = (QWeightA * ScaleA + ZeroA) * (QWeightB * ScaleB + ZeroB)
    //   = ScaleA * ScaleB * C_int32 + SumB * ScaleB * ZeroA + ZeroB * SumA
    static void dequant(int M, int N, const int32_t *C_int32, const int ldc_int32, float *C, const int ldc,
            const float *scaleA, const float *zeroA, const float *sumA, const float *scaleB, const float *zeroB,
            const float *sumB, const float *bias, const float *res, int ldres, float gamma, matmul_kinds kind) {
        auto dequant_op = [scaleA, zeroA, sumA, scaleB, zeroB, sumB](__m512i &vi, int row, int col) {
            __m512 v = _mm512_cvtepi32_ps(vi);
            __m512 vscaleB = _mm512_loadu_ps(scaleB + col);
            __m512 vzeroB = _mm512_loadu_ps(zeroB + col);
            __m512 vsumB = _mm512_loadu_ps(sumB + col);
            __m512 vscaleA = _mm512_set1_ps(scaleA[row]);
            __m512 vsumA = _mm512_set1_ps(sumA[row]);
            __m512 vzeroA = _mm512_set1_ps(zeroA[row]);
            return v * vscaleA * vscaleB + vsumB * vscaleB * vzeroA + vsumA * vzeroB;
        };

        auto no_post_op = [](__m512 &v, int row, int col) { return v; };
        auto biasadd = [bias](__m512 &v, int row, int col) {
            __m512 vbias = _mm512_loadu_ps(bias + col);
            return _mm512_add_ps(v, vbias);
        };
        auto biasadd_relu = [bias](__m512 &v, int row, int col) {
            __m512 vbias = _mm512_loadu_ps(bias + col);
            return _mm512_max_ps(_mm512_add_ps(v, vbias), _mm512_setzero_ps());
        };
        auto residential = [res, ldres](__m512 &v, int row, int col) {
            __m512 vres = _mm512_loadu_ps(res + row * ldres + col);
            return _mm512_add_ps(v, vres);
        };
        auto biasadd_res = [bias, res, ldres](__m512 &v, int row, int col) {
            __m512 vbias = _mm512_loadu_ps(bias + col);
            __m512 vres = _mm512_loadu_ps(res + row * ldres + col);
            return _mm512_add_ps(_mm512_add_ps(v, vbias), vres);
        };
        auto resext = [res, ldres, gamma](__m512 &v, int row, int col) {
            __m512 vres = _mm512_loadu_ps(res + row * ldres + col);
            __m512 vgamma = _mm512_set1_ps(gamma);
            return _mm512_fmadd_ps(vgamma, vres, v);
        };
        auto biasadd_resext = [bias, res, ldres, gamma](__m512 &v, int row, int col) {
            __m512 vbias = _mm512_loadu_ps(bias + col);
            __m512 vres = _mm512_loadu_ps(res + row * ldres + col);
            __m512 vgamma = _mm512_set1_ps(gamma);
            return _mm512_fmadd_ps(vgamma, vres, _mm512_add_ps(v, vbias));
        };
        auto resmul = [res, ldres](__m512 &v, int row, int col) {
            __m512 vres = _mm512_loadu_ps(res + row * ldres + col);
            return _mm512_mul_ps(v, vres);
        };
        auto silu = [](__m512 &v, int row, int col) {
            __m512 vone = _mm512_set1_ps(1.0f);
            __m512 vp = BertUtil::vexp(v);
            __m512 vrecip = _mm512_rcp14_ps(vp + vone);
            return vp * vrecip * v;
        };

        switch (kind) {
            case matmul_kinds::Basic: dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, no_post_op); break;
            case matmul_kinds::BiasAdd: dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, biasadd); break;
            case matmul_kinds::BiasAdd_Relu:
                dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, biasadd_relu);
                break;
            case matmul_kinds::Silu: dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, silu); break;
            case matmul_kinds::Resmul: dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, resmul); break;
            case matmul_kinds::Residential:
                if (bias) {
                    dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, biasadd_res);
                } else {
                    dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, residential);
                }
                break;
            case matmul_kinds::Resext:
                if (bias) {
                    dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, biasadd_resext);
                } else {
                    dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, resext);
                }
                break;
        }
    }
};
