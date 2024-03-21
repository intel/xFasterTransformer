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
#include "bert_util.h"
#include "bfloat16.h"
#include "copy_util.h"
#include "dtype.h"
#include "float16.h"
#include "gpu_util.h"
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
#include "gpu_layernorm_kernels.h"
#include <CL/sycl.hpp>
#include <dpct/device.hpp>

#define USE_AMX_M 8

class MMHelper {
public:
    MMHelper(xft::DeviceKind device_kind, int idx) {
        if (device_kind == xft::DeviceKind::iCPU) {
            kind = dnnl::engine::kind::cpu;
            engine = new dnnl::engine(kind, idx);
            stream = new dnnl::stream(*engine);
        } else if (device_kind == xft::DeviceKind::iGPU) {
            gpu_index = idx;
            kind = dnnl::engine::kind::gpu;
            engine = new dnnl::engine(kind, idx);
            stream = new dnnl::stream(*engine);
            auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
            gpu_queue = new sycl::queue(devices[engine->get_count(kind) + idx]);
            packedI = sycl::malloc_device<float16_t>(18 * 32000, *gpu_queue);
            packedA = sycl::malloc_device<float16_t>(18 * 32000, *gpu_queue);
            packedC = sycl::malloc_device<float16_t>(18 * 32000, *gpu_queue);
            HostBuf = sycl::malloc_host<float16_t>(18 * 32000, *gpu_queue);
        } else {
            std::cerr << "[Error] Wrong device type." << std::endl;
            std::exit(-1);
        }
    }

    ~MMHelper() {
        if (engine) delete engine;
        if (stream) delete stream;
        sycl::free(packedI, *gpu_queue);
        sycl::free(packedA, *gpu_queue);
        sycl::free(packedC, *gpu_queue);
        sycl::free(HostBuf, *gpu_queue);
    }

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

    template <typename OriWeiT, typename WeiT>
    void convertWeight(bool trans, int rows, int cols, const OriWeiT *weight, const float *scales, const float *zeros,
            int splitOffset, int splitSize, bool verticalSplit, hpj::Matrix<WeiT> &convertedWeight,
            hpj::Vector<float> &scaleWeight, hpj::Vector<float> &zeroWeight, hpj::Vector<float> &sumWeight,
            bool unused) {
        // transform trans cases to no trans cases
        if (trans) {
            std::swap(rows, cols);
            verticalSplit = !verticalSplit;
        }

        int rowOffset, rowSize, colOffset, colSize;
        if (verticalSplit) {
            rowOffset = 0;
            rowSize = rows;
            colOffset = splitOffset;
            colSize = splitSize;
        } else {
            rowOffset = splitOffset;
            rowSize = splitSize;
            colOffset = 0;
            colSize = cols;
        }

        convertedWeight.Resize(rowSize, colSize);

        // FP32 -> FP32
        if constexpr (std::is_same_v<OriWeiT, float> && std::is_same_v<WeiT, float>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                WeiT *dst = convertedWeight.Data() + i * convertedWeight.Stride();
                const OriWeiT *src = weight + (rowOffset + i) * cols + colOffset;
                memcpy(dst, src, colSize * sizeof(WeiT));
            }
        }

        // FP32 -> FP16
        else if constexpr (std::is_same_v<OriWeiT, float> && std::is_same_v<WeiT, float16_t>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                WeiT *dst = convertedWeight.Data() + i * convertedWeight.Stride();
                const OriWeiT *src = weight + (rowOffset + i) * cols + colOffset;
                float16_t::cvt_float_to_float16(src, dst, colSize);
            }
        }

        // FP32 -> BF16
        else if constexpr (std::is_same_v<OriWeiT, float> && std::is_same_v<WeiT, bfloat16_t>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                WeiT *dst = convertedWeight.Data() + i * convertedWeight.Stride();
                const OriWeiT *src = weight + (rowOffset + i) * cols + colOffset;
                bfloat16_t::cvt_float_to_bfloat16(src, dst, colSize);
            }
        }

        // FP32 -> INT8/W8A8
        else if constexpr (std::is_same_v<OriWeiT, float>
                && (std::is_same_v<WeiT, int8_t> || std::is_same_v<WeiT, w8a8_t>)) {
            scaleWeight.Resize(trans ? rowSize : colSize);
            zeroWeight.Resize(trans ? rowSize : colSize);
            const float *src = weight + rowOffset * cols + colOffset;
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            xdnn_sgemm_f32s8f32_quantize(trans, trans ? rowSize : colSize, trans ? colSize : rowSize, src, cols,
                    0.9999f, (int8_t *)convertedWeight.Data(), convertedWeight.Stride(), scaleWeight.Data(),
                    zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            xdnn_hgemm_f32s8f32_quantize(trans, trans ? rowSize : colSize, trans ? colSize : rowSize, src, cols,
                    0.9999f, (int8_t *)convertedWeight.Data(), convertedWeight.Stride(), scaleWeight.Data(),
                    zeroWeight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif

        }

        // FP32 -> UINT4
        else if constexpr (std::is_same_v<OriWeiT, float> && std::is_same_v<WeiT, uint4x2_t>) {
            scaleWeight.Resize(trans ? rowSize : colSize);
            zeroWeight.Resize(trans ? rowSize : colSize);
            const float *src = weight + rowOffset * cols + colOffset;
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            xdnn_sgemm_f32u4f32_quantize(trans, trans ? rowSize : colSize, trans ? colSize : rowSize, src, cols,
                    0.9999f, (XDNN_UINT4x2 *)convertedWeight.Data(), convertedWeight.Stride(), scaleWeight.Data(),
                    zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            xdnn_hgemm_f32u4f32_quantize(trans, trans ? rowSize : colSize, trans ? colSize : rowSize, src, cols,
                    0.9999f, (XDNN_UINT4x2 *)convertedWeight.Data(), convertedWeight.Stride(), scaleWeight.Data(),
                    zeroWeight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // FP32 -> NF4
        else if constexpr (std::is_same_v<OriWeiT, float> && std::is_same_v<WeiT, nf4x2_t>) {
            scaleWeight.Resize(trans ? rowSize : colSize);
            zeroWeight.Resize(trans ? rowSize : colSize);
            const float *src = weight + rowOffset * cols + colOffset;
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            xdnn_sgemm_f32nf4f32_quantize(trans, trans ? rowSize : colSize, trans ? colSize : rowSize, src, cols,
                    0.9999f, (XDNN_NF4x2 *)convertedWeight.Data(), convertedWeight.Stride(), scaleWeight.Data(),
                    zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            xdnn_sgemm_f32nf4f32_quantize(trans, trans ? rowSize : colSize, trans ? colSize : rowSize, src, cols,
                    0.9999f, (XDNN_NF4x2 *)convertedWeight.Data(), convertedWeight.Stride(), scaleWeight.Data(),
                    zeroWeight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8 -> INT8/W8A8
        else if constexpr (std::is_same_v<OriWeiT, int8_t>
                && (std::is_same_v<WeiT, int8_t> || std::is_same_v<WeiT, w8a8_t>)) {
            int size = trans ? rowSize : colSize;
            int offset = trans ? rowOffset : colOffset;
            scaleWeight.Resize(size);
            zeroWeight.Resize(size);
            memcpy(scaleWeight.Data(), scales + offset, size * sizeof(float));
            memcpy(zeroWeight.Data(), zeros + offset, size * sizeof(float));
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                WeiT *dst = convertedWeight.Data() + i * convertedWeight.Stride();
                const OriWeiT *src = weight + (rowOffset + i) * cols + colOffset;
                memcpy(dst, src, colSize * sizeof(WeiT));
            }
        }

        else {
            printf("%s:%d: Do not support this kind of weights datatype convertion.\n", __FILE__, __LINE__);
            exit(-1);
        }

        // Compute per column Sums for W8A8
        if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            sumWeight.Resize(trans ? rowSize : colSize);
#pragma omp parallel for
            for (uint64_t i = 0; i < colSize; i++) {
                sumWeight.Data()[i] = 0.0f;
                for (uint64_t j = 0; j < rowSize; j++) {
                    sumWeight.Data()[i] += convertedWeight.Data()[j * convertedWeight.Stride() + i];
                }
            }
        }
    }

    template <typename OriWeiT, typename WeiT>
    void convertWeight(bool trans, int rows, int cols, const OriWeiT *weight, const float *scales, const float *zeros,
            int numSplit, int splitIdx, bool verticalSplit, hpj::Matrix<WeiT> &quantizedWeight,
            hpj::Vector<float> &scaleWeight, hpj::Vector<float> &zeroWeight, hpj::Vector<float> &sumWeight) {
        int totalSize = verticalSplit ? cols : rows;
        std::pair<int, int> range = SplitUtil::getTaskRange(totalSize, numSplit, splitIdx);

        int splitSize = range.second - range.first;
        int splitOffset = range.first;

        convertWeight(trans, rows, cols, weight, scales, zeros, splitOffset, splitSize, verticalSplit, quantizedWeight,
                scaleWeight, zeroWeight, sumWeight, true);
    }

    template <typename OriWeiT, typename WeiT>
    void convertWeight(bool trans, int rows, int cols, const OriWeiT *weight, const float *scales, const float *zeros,
            hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float> &scaleWeight, hpj::Vector<float> &zeroWeight,
            hpj::Vector<float> &sumWeight) {
        convertWeight(trans, rows, cols, weight, scales, zeros, 1, 0, true, quantizedWeight, scaleWeight, zeroWeight,
                sumWeight);
    }

    template <typename OriWeiT, typename WeiT>
    void convertWeight(DecoderContext *ctx, bool trans, int rows, int cols, const OriWeiT *weight, const float *scales,
            const float *zeros, bool verticalSplit, hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight, hpj::Vector<float> &sumWeight) {
        convertWeight(trans, rows, cols, weight, scales, zeros, ctx->numSplit, ctx->splitIdx, verticalSplit,
                quantizedWeight, scaleWeight, zeroWeight, sumWeight);
    }

    template <typename WeiT>
    void packWeight(bool trans, hpj::Matrix<WeiT> &src, hpj::Matrix<WeiT> &weight) {
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
            memset(weight.Data(), 0, sizeof(bfloat16_t) * amx_rows * amx_cols);
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
            using dt = dnnl::memory::data_type;
            auto tag = trans ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab;
            dnnl::memory B_mem({{K, N}, dt::s8, tag}, *this->engine, src.Data());
            dnnl::memory::desc desc({K, N}, dt::s8, get_onednn_weight_layout(dt::s8));

            // When converting to oneDNN blocked memory format, padded dims can be larger than [K, N]
            // Resize down Matrix does not change underlying buffer.
            // TODO: Add reserve like function in Matrix, as current 2 times Resize has potential risks.
            auto dims = desc.get_padded_dims();
            weight.Resize(dims[0], dims[1]);
            weight.Resize(K, N);

            dnnl::memory packedB_mem(desc, *engine, weight.Data());
            dnnl::reorder(B_mem, packedB_mem).execute(*stream, B_mem, packedB_mem);
            stream->wait();
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

    template <typename WeiT>
    void transposeWeight(bool trans, hpj::Matrix<WeiT> &src, hpj::Matrix<WeiT> &dst) {
        if constexpr (std::is_same_v<WeiT, float16_t>) {
            int K = trans ? src.Cols() : src.Rows();
            int N = trans ? src.Rows() : src.Cols();

            // Reorder weight
            using namespace dnnl;
            using tag = memory::format_tag;
            using dt = memory::data_type;
            dnnl::engine engine(dnnl::engine::kind::cpu, 0);
            dnnl::stream stream(engine);
            auto weight_md = memory::desc({K, N}, dt::f16, trans ? tag::ba : tag::ab);
            auto weight_mem = memory(weight_md, engine, src.Data());
            auto packed_weight_md = memory::desc({K, N}, dt::f16, get_onednn_weight_layout(dt::f16));
            auto packed_weight_mem = memory(packed_weight_md, engine);
            dnnl::reorder(weight_mem, packed_weight_mem).execute(stream, weight_mem, packed_weight_mem);
            stream.wait();
            gpu_queue->memcpy(dst.Data(), packed_weight_mem.get_data_handle(), dst.Rows() * dst.Cols() * sizeof(WeiT))
                    .wait();
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc,
            bool postOp = false, bool copyC2G = true, bool copyG2C = true) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE(
                    "xdnn_sgemm_compute", xdnn_sgemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute",
            //         xdnn_sgemm_f32f16f32_compute(
            //                 transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc));
            GEMMVERBOSE("onednn_sgemm_f32f16f32_compute",
                    onednn_sgemm_f32f16f32_compute(
                            transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, postOp, copyC2G, copyG2C));
            // printf("A:\n");
            // for (int i = 0; i < 6; ++i) {
            //     for (int j = 0; j < 6; ++j) {
            //         printf("%.6f ", A[i * lda + j]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");
            // printf("B:\n");
            // for (int i = 0; i < 6; ++i) {
            //     for (int j = 0; j < 6; ++j) {
            //         printf("%.6f ", float(packedB[i * N + j]));
            //     }
            //     printf("\n");
            // }
            // printf("\n");
            // printf("B:\n");
            // float16_t B_buf[6];
            // gpu_queue->memcpy(B_buf, packedB, sizeof(float16_t) * 6).wait();
            // for (int j = 0; j < 6; ++j) {
            //     printf("%.6f ", float(B_buf[j]));
            // }
            // printf("\n");
            // printf("C:\n");
            // for (int i = 0; i < 6; ++i) {
            //     for (int j = 0; j < 6; ++j) {
            //         printf("%.6f ", C[i * ldc + j]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");
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
                            transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            // TODO: xdnn impl?
            if constexpr (std::is_same_v<InT, bfloat16_t>) {
                GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute",
                        onednn_amx_sgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
            } else {
                if (M > USE_AMX_M) {
                    GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute",
                            onednn_amx_sgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
                } else {
                    GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute",
                            xdnn_bgemm_f32bf16f32_compute(
                                    transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc));
                }
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
    void compute_bias(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
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
            // TODO: xdnn impl?
            if constexpr (std::is_same_v<InT, bfloat16_t>) {
                GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute_biasadd",
                        onednn_amx_sgemm_f32bf16f32_compute_biasadd(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias));
            } else {
                if (M > USE_AMX_M) {
                    GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute_biasadd",
                            onednn_amx_sgemm_f32bf16f32_compute_biasadd(
                                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias));
                } else {
                    GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_biasadd",
                            xdnn_bgemm_f32bf16f32_compute_biasadd(
                                    transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias));
                }
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
    void compute_biasadd_relu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc,
            const float *bias) {
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
    void compute_silu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
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
            if constexpr (std::is_same_v<InT, bfloat16_t>) {
                GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute_silu",
                        onednn_amx_sgemm_f32bf16f32_compute_silu(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
            } else {
                if (M > USE_AMX_M) {
                    GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute_silu",
                            onednn_amx_sgemm_f32bf16f32_compute_silu(
                                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
                } else {
                    GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_silu",
                            xdnn_bgemm_f32bf16f32_compute_silu(
                                    transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc));
                }
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
    void compute_resmul(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc, const InT *res,
            int ldres) {
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
            if constexpr (std::is_same_v<InT, bfloat16_t>) {
                GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute_resmul",
                        onednn_amx_sgemm_f32bf16f32_compute_resmul(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres));
            } else {
                if (M > USE_AMX_M) {
                    GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute_resmul",
                            onednn_amx_sgemm_f32bf16f32_compute_resmul(
                                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres));
                } else {
                    GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_resmul",
                            xdnn_bgemm_f32bf16f32_compute_resmul(transA, M, N, K, alpha, A, lda,
                                    (const XDNN_BF16 *)packedB, beta, C, ldc, res, ldres));
                }
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
    void compute_residential(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc, const float *bias,
            const InT *res, int ldres, bool copyC2G = true, bool copyG2C = true) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_residential",
                    xdnn_sgemm_compute_residential(
                            transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_residential",
            //         xdnn_sgemm_f32f16f32_compute_residential(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB,
            //                 beta, C, ldc, bias, res, ldres));
            GEMMVERBOSE("onednn_sgemm_f32f16f32_compute_residential",
                    onednn_sgemm_f32f16f32_compute_residential(
                            transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres, copyC2G, copyG2C));
            // printf("A:\n");
            // for (int i = 0; i < 6; ++i) {
            //     for (int j = 0; j < 6; ++j) {
            //         printf("%.6f ", A[i * lda + j]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");
            // printf("B:\n");
            // for (int i = 0; i < 6; ++i) {
            //     for (int j = 0; j < 6; ++j) {
            //         printf("%.6f ", float(packedB[i * N + j]));
            //     }
            //     printf("\n");
            // }
            // printf("\n");
            // printf("B:\n");
            // float16_t B_buf[6];
            // gpu_queue->memcpy(B_buf, packedB, sizeof(float16_t) * 6).wait();
            // for (int j = 0; j < 6; ++j) {
            //     printf("%.6f ", float(B_buf[j]));
            // }
            // printf("\n");
            // printf("C:\n");
            // for (int i = 0; i < 6; ++i) {
            //     for (int j = 0; j < 6; ++j) {
            //         printf("%.6f ", C[i * ldc + j]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");
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
            // TODO: xdnn impl?
            if constexpr (std::is_same_v<InT, bfloat16_t>) {
                GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute_residential",
                        onednn_amx_sgemm_f32bf16f32_compute_residential(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres));
            } else {
                if (M > USE_AMX_M) {
                    GEMMVERBOSE("onednn_amx_sgemm_f32bf16f32_compute_residential",
                            onednn_amx_sgemm_f32bf16f32_compute_residential(
                                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres));
                } else {
                    GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_residential",
                            xdnn_bgemm_f32bf16f32_compute_residential(transA, M, N, K, alpha, A, lda,
                                    (const XDNN_BF16 *)packedB, beta, C, ldc, bias, res, ldres));
                }
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
    void compute_resext(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc, const float *bias,
            float gamma, InT *res, int ldres) {
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
            if constexpr (std::is_same_v<InT, bfloat16_t>) {
                TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_residential");
#pragma omp parallel for collapse(2)
                for (int i = 0; i < M; ++i) {
                    for (int j = 0; j < N; ++j) {
                        auto remain = N - j;
                        __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                        auto v = xft::load_avx512(mask, &res[i * ldres + j]);
                        v = _mm512_mul_ps(_mm512_set1_ps(gamma), v);
                        xft::store_avx512(&res[i * ldres + j], mask, v);
                    }
                }
                onednn_amx_sgemm_f32bf16f32_compute_residential(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
            } else {
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
                            xdnn_bgemm_f32bf16f32_compute_resext(transA, M, N, K, alpha, A, lda,
                                    (const XDNN_BF16 *)packedB, beta, C, ldc, bias, gamma, res, ldres));
                }
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

    void computeRotaryPositionEmbedding(
            float *query, float *key, int qStride, int kStride, const int *qkShape, const int *positionIds) {
        const int batchSize = qkShape[0];
        const int seqLen = qkShape[1];
        const int qHeads = qkShape[2];
        const int kHeads = qkShape[4];
        const int head_num = std::max(qHeads, kHeads);
        const int head_size = qkShape[3];
        const int half_head_size = (head_size + 1) / 2;
        using namespace sycl;
        // FunTimer t;

        // Reorder input
        float16_t *packedC_buf = packedC;
        float *embCos = emb_cos;
        float *embSin = emb_sin;
        // float16_t C_buf[batchSize * seqLen * 3 * head_num * head_size];
        // float16_t::cvt_float_to_float16_MT(query, C_buf, batchSize * seqLen * 3 * head_num * head_size);
        // gpu_queue->memcpy(packedC_buf, C_buf, batchSize * seqLen * 3 * head_num * head_size * sizeof(float16_t)).wait();
        // std::cout << "GPU"<< gpu_index << " rope:" <<std::endl;
        // print(batchSize * seqLen, 6, qStride, (float16_t *)packedC_buf, "packedC");

        buffer<int, 1> positionIdsBuf(positionIds, sycl::range<1>(seqLen));
        gpu_queue
                ->submit([&](handler &cgh) {
                    accessor position(positionIdsBuf, cgh, sycl::read_only);
                    range<3> globalSize(batchSize * seqLen, head_num, half_head_size);
                    range<3> workGroupSize(1, 1, 1);

                    cgh.parallel_for<class kernel_rope>(
                            nd_range(globalSize, workGroupSize), [=, this](nd_item<3> item) {
                                rope_kernel(item, embCos, embSin, qHeads, kHeads, seqLen, head_size, half_head_size,
                                        packedC_buf, packedC_buf + head_num * head_size, qStride, kStride, position);
                            });
                })
                .wait();
        // printf("xft_verbose,exec,gpu:%d,%s,%.6lf\n", gpu_index, "rope", t.elapsed());

        // Reorder output
        // FunTimer t2;
        gpu_queue->memcpy(HostBuf, packedC_buf, batchSize * seqLen * 3 * head_num * head_size * sizeof(float16_t))
                .wait();
        float16_t::cvt_float16_to_float_MT(HostBuf, query, batchSize * seqLen * 3 * head_num * head_size);
        // printf("xft_verbose,exec,gpu:%d,%s,%.6lf\n", gpu_index, "memcpy", t2.elapsed());

        // print(batchSize * seqLen, 6, qStride, (float16_t *)packedC_buf, "packedC");
    }

    void print(int rows, int cols, int stride, float16_t *buf, std::string buf_name) {
        std::cout << buf_name.c_str() << ":" << std::endl;
        gpu_queue
                ->submit([&](sycl::handler &cgh) {
                    auto out = sycl::stream(1024, 768, cgh);
                    cgh.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) {
                        int idx_col = item.get_global_id(0);
                        if (idx_col == 0) {
                            for (int row = 0; row < rows; ++row) {
                                for (int col = 0; col < cols; ++col) {
                                    out << buf[row * stride + col] << ", ";
                                }
                                out << sycl::endl;
                            }
                            out << sycl::endl;
                        }
                    });
                })
                .wait();
    }

    void computeRMSNorm(float *output, const float *input, const sycl::half *weight, int rows, int cols) {
        // FunTimer t;
        // float16_t I_buf[rows * cols];
        // float16_t::cvt_float_to_float16_MT(input, I_buf, rows * cols);
        sycl::half *packedI_buf = (sycl::half *)packedI;
        sycl::half *packedA_buf = (sycl::half *)packedA;
        // gpu_queue->memcpy(packedI_buf, I_buf, rows * cols * sizeof(float16_t)).wait();
        // rmsnorm_kernel(packedA_buf, packedI_buf, weight, rows, cols, cols, cols);
        // gpu_queue->memcpy(I_buf, packedA_buf, rows * cols * sizeof(float16_t)).wait();
        // float16_t::cvt_float16_to_float_MT(I_buf, output, rows * cols);
        const float layernorm_eps = 1e-06;
        fastertransformer::invokeGeneralT5LayerNorm(
                packedA_buf, packedI_buf, weight, (const sycl::half *)nullptr, layernorm_eps, rows, cols, gpu_queue);
        // printf("xft_verbose,exec,gpu:%d,%s,%.6lf\n", gpu_index, "rmsnorm", t.elapsed());
        // std::cout << "GPU"<< gpu_index << "rms_norm:" <<std::endl;
        // print(rows, 6, cols, (float16_t *)packedI_buf, "packedI");
        // print(rows, 6, cols, (float16_t *)packedA_buf, "packedA");
    }

    sycl::queue *gpu_queue;
    float *emb_cos;
    float *emb_sin;

    float16_t *packedI;
    float16_t *packedA;
    float16_t *packedC;
    float16_t *HostBuf; // Pinned Memory

private:
    int gpu_index;
    dnnl::engine::kind kind;
    dnnl::engine *engine;
    dnnl::stream *stream;
    std::unordered_map<std::string, std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *>> matmul_hub;

    enum matmul_kinds {
        Basic = 0,
        BiasAdd,
        BiasAdd_Relu,
        Silu,
        Resmul,
        Residential,
        Resext,
    };

    std::string create_key(bool transA, int M, int N, int K, int matmul_kind) {
        std::string key = std::to_string(transA) + "_" + std::to_string(M) + "_" + std::to_string(N) + "_"
                + std::to_string(K) + "_" + std::to_string(matmul_kind);
        return key;
    }

    dnnl::memory::format_tag get_onednn_input_layout(dnnl::memory::data_type dt) {
        if (this->kind == dnnl::engine::kind::cpu) {
            return dnnl::memory::format_tag::undef;
        } else if (this->kind == dnnl::engine::kind::gpu) {
            return dnnl::memory::format_tag::ab;
            // return dnnl::memory::format_tag::AB32a16b;
            // return dnnl::memory::format_tag::any;
        } else {
            printf("[XFT][ERROR] Need a right engine kind in input layout.");
            std::exit(-1);
        }
    }

    dnnl::memory::format_tag get_onednn_weight_layout(dnnl::memory::data_type dt) {
        if (this->kind == dnnl::engine::kind::cpu) {
            if (dt == dnnl::memory::data_type::bf16 || dt == dnnl::memory::data_type::f16) {
                return dnnl::memory::format_tag::BA16a64b2a;
            } else if (dt == dnnl::memory::data_type::s8) {
                return dnnl::memory::format_tag::BA16a64b4a;
            } else {
                printf("[XFT][ERROR] Unsupport your data type in input layout.");
                std::exit(-1);
            }
        } else if (this->kind == dnnl::engine::kind::gpu) {
            // return dnnl::memory::format_tag::ab;
            // return dnnl::memory::format_tag::BA4b8a8b2a;
            // return dnnl::memory::format_tag::any;
            return dnnl::memory::format_tag::ba;
        } else {
            printf("[XFT][ERROR] Need a right engine kind in weight layout.");
            std::exit(-1);
        }
    }

    dnnl::memory::format_tag get_onednn_bias_layout(dnnl::memory::data_type dt) {
        if (this->kind == dnnl::engine::kind::cpu) {
            return dnnl::memory::format_tag::undef;
        } else if (this->kind == dnnl::engine::kind::gpu) {
            return dnnl::memory::format_tag::ab;
        } else {
            printf("[XFT][ERROR] Need a right engine kind in bias layout.");
            std::exit(-1);
        }
    }

    dnnl::memory::format_tag get_onednn_shift_layout(dnnl::memory::data_type dt) {
        if (this->kind == dnnl::engine::kind::cpu) {
            return dnnl::memory::format_tag::undef;
        } else if (this->kind == dnnl::engine::kind::gpu) {
            return dnnl::memory::format_tag::ab;
        } else {
            printf("[XFT][ERROR] Need a right engine kind in shift layout.");
            std::exit(-1);
        }
    }

    dnnl::memory::format_tag get_onednn_output_layout(dnnl::memory::data_type dt) {
        if (this->kind == dnnl::engine::kind::cpu) {
            return dnnl::memory::format_tag::undef;
        } else if (this->kind == dnnl::engine::kind::gpu) {
            return dnnl::memory::format_tag::ab;
            // return dnnl::memory::format_tag::AB32a16b;
            // return dnnl::memory::format_tag::any;
        } else {
            printf("[XFT][ERROR] Need a right engine kind in output layout.");
            std::exit(-1);
        }
    }

    inline void sycl_sigmoid_mul(
            int M, int N, const float16_t *src0, int lds0, const float16_t *src1, int lds1, float16_t *dst, int ldd) {
        gpu_queue
                ->submit([&](sycl::handler &h) {
                    h.parallel_for(M, [=](auto i) {
                        for (int j = 0; j < N; ++j) {
                            dst[i * ldd + j] = (sycl::half)src0[i * lds0 + j]
                                    / ((sycl::half)1.0 + (sycl::half)sycl::native::exp(-src0[i * lds0 + j]))
                                    * src1[i * lds1 + j];
                        }
                    });
                })
                .wait();
    }

    inline void sycl_sigmoid_mul_M1(
            int N, const float16_t *src0, int lds0, const float16_t *src1, int lds1, float16_t *dst, int ldd) {
        gpu_queue
                ->submit([&](sycl::handler &h) {
                    h.parallel_for(N, [=](auto i) {
                        dst[i] = (sycl::half)src0[i] / ((sycl::half)1.0 + (sycl::half)sycl::native::exp(-src0[i]))
                                * src1[i];
                    });
                })
                .wait();
    }

    inline void sycl_memcopy_lines(int M, int N, const float16_t *src, int lds, float16_t *dst, int ldd) {
        gpu_queue
                ->submit([&](sycl::handler &h) {
                    h.parallel_for(M, [=](auto i) {
                        for (int j = 0; j < N; ++j) {
                            dst[i * ldd + j] = src[i * lds + j];
                        }
                    });
                })
                .wait();
    }

    inline void sycl_convert_fp16_to_fp32(int M, int N, const float16_t *src, int lds, float *dst, int ldd) {
        gpu_queue
                ->submit([&](sycl::handler &h) {
                    h.parallel_for(M, [=](auto i) {
                        for (int j = 0; j < N; ++j) {
                            dst[i * ldd + j] = (float)src[i * lds + j];
                        }
                    });
                })
                .wait();
    }

    inline void rmsnorm_kernel(float16_t *device_data_o, const float16_t *device_data_x,
            const float *device_data_weight, int rows, int cols, int iStride, int oStride) {
        int max_work_group_size = 512;
        int elementsPerThread = (cols - 1) / max_work_group_size + 1;
        sycl::buffer<float> part_sum(sycl::range<1>(rows * 512));
        std::vector<float> sum_var(rows, 0.0f);
        sycl::buffer<float> sum_var_buf(sum_var);

        // gpu_queue
        //         ->submit([&](sycl::handler &cgh) {
        //             auto out = sycl::stream(10240, 7680, cgh);
        //             auto part_sum_data = part_sum.get_access<sycl::access::mode::read_write>(cgh);
        //             cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(max_work_group_size), sycl::range<1>(1)), [=](sycl::nd_item<1> item_ct1) {
        //                 for (int row = 0; row < rows; ++row) {
        //                     int idx_col = item_ct1.get_global_id(0);
        //                     float ss = 0.0f;
        //                     for (int i = 0; i < elementsPerThread; i++) {
        //                         float val = (float)device_data_x[row * iStride + idx_col * elementsPerThread + i];
        //                         ss += val * val;
        //                     }
        //                     part_sum_data[row * 512 + idx_col] = ss;
        //                 }
        //             });
        //         }).wait();

        // gpu_queue
        //         ->submit([&](sycl::handler &cgh) {
        //             auto out = sycl::stream(10240, 7680, cgh);
        //             auto part_sum_data = part_sum.get_access<sycl::access::mode::read_write>(cgh);
        //             cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(1)), [=](sycl::nd_item<1> item_ct1) {
        //                 for (int row = 0; row < rows; ++row) {
        //                     int idx_col = item_ct1.get_global_id(0);
        //                     float ss = 0.0f;
        //                     for (int i = 0; i < 16; i++) {
        //                         ss += part_sum_data[row * 512 + idx_col + 32 * i];
        //                     }
        //                     part_sum_data[row * 512 + idx_col] = ss;
        //                 }
        //             });
        //         }).wait();

        // gpu_queue
        //         ->submit([&](sycl::handler &cgh) {
        //             auto out = sycl::stream(10240, 7680, cgh);
        //             auto part_sum_data = part_sum.get_access<sycl::access::mode::read_write>(cgh);
        //             cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(8), sycl::range<1>(1)), [=](sycl::nd_item<1> item_ct1) {
        //                 for (int row = 0; row < rows; ++row) {
        //                     int idx_col = item_ct1.get_global_id(0);
        //                     float ss = 0.0f;
        //                     for (int i = 0; i < 4; i++) {
        //                         ss += part_sum_data[row * 512 + idx_col + 8 * i];
        //                     }
        //                     part_sum_data[row * 512 + idx_col] = ss;
        //                 }
        //             });
        //         }).wait();

        // gpu_queue
        //         ->submit([&](sycl::handler &cgh) {
        //             auto out = sycl::stream(10240, 7680, cgh);
        //             auto part_sum_data = part_sum.get_access<sycl::access::mode::read_write>(cgh);
        //             sycl::accessor accessorVar(sum_var_buf, cgh, sycl::read_write);
        //             cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)), [=](sycl::nd_item<1> item_ct1) {
        //                 for (int row = 0; row < rows; ++row) {
        //                     float ss = part_sum_data[row * 512 + 0] + part_sum_data[row * 512 + 1] + part_sum_data[row * 512 + 2] + part_sum_data[row * 512 + 3] +
        //                                part_sum_data[row * 512 + 4] + part_sum_data[row * 512 + 5] + part_sum_data[row * 512 + 6] + part_sum_data[row * 512 + 7];
        //                     ss /= cols;
        //                     ss += 1e-5f;
        //                     ss = 1.0f / sycl::sqrt(ss);
        //                     accessorVar[row] = ss;
        //                 }
        //             });
        //         }).wait();

        // gpu_queue
        //         ->submit([&](sycl::handler &cgh) {
        //             auto out = sycl::stream(10240, 7680, cgh);
        //             auto part_sum_data = part_sum.get_access<sycl::access::mode::read_write>(cgh);
        //             sycl::accessor accessorVar(sum_var_buf, cgh, sycl::read_write);
        //             cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)), [=](sycl::nd_item<1> item_ct1) {
        //                 for (int row = 0; row < rows; ++row) {
        //                     float ss = 0.0f;
        //                     for (int i = 0; i < 512; i++) {
        //                         ss += part_sum_data[row * 512 + i];
        //                     }
        //                     ss /= cols;
        //                     ss += 1e-5f;
        //                     ss = 1.0f / sycl::sqrt(ss);
        //                     accessorVar[row] = ss;
        //                 }
        //             });
        //         }).wait();

        gpu_queue
                ->submit([&](sycl::handler &cgh) {
                    auto out = sycl::stream(10240, 7680, cgh);
                    auto part_sum_data = part_sum.get_access<sycl::access::mode::read_write>(cgh);
                    sycl::accessor accessorVar(sum_var_buf, cgh, sycl::read_write);
                    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(max_work_group_size), sycl::range<1>(1)),
                            [=](sycl::nd_item<1> item_ct1) {
                                for (int row = 0; row < rows; ++row) {
                                    int idx_col = item_ct1.get_global_id(0);
                                    float ss = 0.0f;
                                    for (int i = 0; i < elementsPerThread; i++) {
                                        float val
                                                = (float)device_data_x[row * iStride + idx_col * elementsPerThread + i];
                                        ss += val * val;
                                    }
                                    part_sum_data[row * 512 + idx_col] = ss;

                                    item_ct1.barrier(sycl::access::fence_space::global_space);
                                    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);

                                    if (idx_col == 0) {
                                        float sss = 0.0f;
                                        for (int i = 0; i < 512; i++) {
                                            sss += part_sum_data[row * 512 + i];
                                        }
                                        sss /= cols;
                                        sss += 1e-5f;
                                        sss = 1.0f / sycl::sqrt(sss);
                                        accessorVar[row] = sss;
                                    }
                                }
                            });
                })
                .wait();

        gpu_queue
                ->submit([&](sycl::handler &cgh) {
                    auto out = sycl::stream(10240, 7680, cgh);
                    auto part_sum_data = part_sum.get_access<sycl::access::mode::read_write>(cgh);
                    sycl::accessor accessorVar(sum_var_buf, cgh, sycl::read_write);
                    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(max_work_group_size), sycl::range<1>(1)),
                            [=](sycl::nd_item<1> item_ct1) {
                                for (int row = 0; row < rows; ++row) {
                                    int idx_col = item_ct1.get_global_id(0);
                                    float ss = accessorVar[row];

                                    // if (idx_col == 0) {
                                    //     out << "cols: " << cols << sycl::endl;
                                    //     out << "elementsPerThread: " << elementsPerThread << sycl::endl;
                                    //     out << "row * iStride + idx_col * elementsPerThread: " << row * iStride + idx_col * elementsPerThread << sycl::endl;
                                    //     out << "ss[" << row << "]: " << ss << sycl::endl;
                                    //     out << "device_data_x[" << row << "]: " << device_data_x[row * iStride + idx_col * elementsPerThread] << sycl::endl;
                                    //     out << "device_data_weight[" << row << "]: " << device_data_weight[idx_col * elementsPerThread] << sycl::endl;
                                    // }

                                    for (int i = 0; i < elementsPerThread; i++) {
                                        float val
                                                = (float)device_data_x[row * iStride + idx_col * elementsPerThread + i];
                                        val *= ss * device_data_weight[idx_col * elementsPerThread + i];
                                        device_data_o[row * oStride + idx_col * elementsPerThread + i]
                                                = (sycl::half)val;
                                    }

                                    // if (idx_col == 0) {
                                    //     out << "device_data_o[" << row << "]: " << device_data_o[row * oStride + idx_col * elementsPerThread] << sycl::endl;
                                    // }
                                }
                            });
                })
                .wait();

        // gpu_queue
        //         ->submit([&](sycl::handler &cgh) {
        //             auto out = sycl::stream(10240, 7680, cgh);
        //             auto part_sum_data = part_sum.get_access<sycl::access::mode::atomic>(cgh);
        //             sycl::local_accessor<float, 0> shared_ss(cgh);
        //             cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(max_work_group_size), sycl::range<1>(1)), [=](sycl::nd_item<1> item_ct1) {
        //                 for (int row = 0; row < rows; ++row) {
        //                     int idx_col = item_ct1.get_global_id(0);
        //                     part_sum_data[idx_col] = 0.0f;
        //                     for (int i = 0; i < elementsPerThread; i++) {
        //                         float val = (float)device_data_x[row * iStride + idx_col * elementsPerThread + i];
        //                         part_sum_data[idx_col] += val * val;
        //                     }
        //                     item_ct1.barrier(sycl::access::fence_space::global_space);
        //                     // part_sum_data[idx_col]  = sycl::reduce_over_group(item_ct1.get_group(), part_sum_data[idx_col] , sycl::plus<float>());
        //                     for(unsigned int s = 1; s < 512; s *= 2) {
        //                         if (idx_col % (2 * s) == 0) {
        //                             part_sum_data[idx_col] += part_sum_data[idx_col + s];
        //                         }
        //                         item_ct1.barrier(sycl::access::fence_space::global_space);
        //                     }
        //                     if (idx_col == 0) {
        //                             part_sum_data[idx_col] /= cols;
        //                             part_sum_data[idx_col] += 1e-5f;
        //                             part_sum_data[idx_col] = 1.0f / sycl::sqrt(part_sum_data[idx_col]);
        //                             shared_ss = part_sum_data[idx_col];
        //                     }
        //                     item_ct1.barrier(sycl::access::fence_space::global_space);

        //                     float ss = shared_ss;
        //                     if (idx_col == 0) {
        //                         out << "cols: " << cols << sycl::endl;
        //                         out << "elementsPerThread: " << elementsPerThread << sycl::endl;
        //                         out << "row * iStride + idx_col * elementsPerThread: " << row * iStride + idx_col * elementsPerThread << sycl::endl;
        //                         out << "ss[" << row << "]: " << ss << sycl::endl;
        //                         out << "device_data_x[" << row << "]: " << device_data_x[row * iStride + idx_col * elementsPerThread] << sycl::endl;
        //                         out << "device_data_weight[" << row << "]: " << device_data_weight[idx_col * elementsPerThread] << sycl::endl;
        //                     }

        //                     for (int i = 0; i < elementsPerThread; i++) {
        //                         float val = (float)device_data_x[row * iStride + idx_col * elementsPerThread + i];
        //                         val *= ss * device_data_weight[idx_col * elementsPerThread + i];
        //                         device_data_o[row * oStride + idx_col * elementsPerThread + i] = (sycl::half)val;
        //                     }
        //                     if (idx_col == 0) {
        //                         out << "device_data_o[" << row << "]: " << device_data_o[row * oStride + idx_col * elementsPerThread] << sycl::endl;
        //                     }
        //                     item_ct1.barrier(sycl::access::fence_space::global_space);
        //                 }
        //             });
        //         })
        //         .wait();
    }

    inline void rope_kernel(sycl::nd_item<3> &item, const float *embCos, const float *embSin, const int qHeads,
            const int kHeads, const int seq_size, const int head_size, const int half, float16_t *query, float16_t *key,
            int qStride, int kStride, const sycl::accessor<int, 1, sycl::access::mode::read> &positionIds) {
        size_t idx_bs_seq = item.get_global_id(0);
        size_t idx_head_num = item.get_global_id(1);
        size_t idx_half_head_dim = item.get_global_id(2);

        size_t pos = positionIds[idx_bs_seq % seq_size];
        const sycl::half cos = (sycl::half)embCos[pos * half + idx_half_head_dim];
        const sycl::half sin = (sycl::half)embSin[pos * half + idx_half_head_dim];

        sycl::half *q = (sycl::half *)query + idx_bs_seq * qStride + idx_head_num * head_size + idx_half_head_dim;
        sycl::half *k = (sycl::half *)key + idx_bs_seq * kStride + idx_head_num * head_size + idx_half_head_dim;

        if (idx_head_num < qHeads) {
            auto q1 = q[0];
            q[0] = q1 * cos - q[half] * sin;
            q[half] = q[half] * cos + q1 * sin;
        }
        if (idx_head_num < kHeads) {
            auto k1 = k[0];
            k[0] = k1 * cos - k[half] * sin;
            k[half] = k[half] * cos + k1 * sin;
        }
    }

    void onednn_sgemm_f32f16f32_compute(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const float16_t *packedB, float beta, float *C, int ldc, bool postOp = false, bool copyC2G = true,
            bool copyG2C = true) {
        TimeLine t("onednn_amx_sgemm_f32bf16f32_compute");
        TimeLine t1("onednn_amx_sgemm_f32bf16f32_compute.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Basic);
        auto it = matmul_hub.find(key);
        if (it != matmul_hub.end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B) and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims output_dims = {M, N};

            // Create memory descriptors and memory objects for src, weights, bias, and dst.
            auto input_md = memory::desc(input_dims, dt::f16, get_onednn_input_layout(dt::f16));
            auto weight_md = memory::desc(weight_dims, dt::f16, get_onednn_weight_layout(dt::f16));
            auto output_md = memory::desc(output_dims, dt::f16, get_onednn_output_layout(dt::f16));

            // Create primitive descriptor and primitive.
            matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::Basic);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            matmul_hub[key] = value;
        }

        // Repack and convert input data.
        auto packed_input_mem = memory(matmul_pd->src_desc(), *engine, packedA);
        auto packed_weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<float16_t *>(packedB));
        auto packed_output_mem = memory(matmul_pd->dst_desc(), *engine, packedC);

        if (copyC2G == true) {
            // Reorder input
            // FunTimer t2;
            float16_t::cvt_float_to_float16_MT(A, HostBuf, M * K);
            gpu_queue->memcpy(packed_input_mem.get_data_handle(), HostBuf, M * K * sizeof(float16_t)).wait();
            // printf("xft_verbose,exec,gpu:%d,%s,%.6lf\n", gpu_index, "memcpy", t2.elapsed());
        }

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, packed_input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, packed_weight_mem});
        matmul_args.insert({DNNL_ARG_DST, packed_output_mem});
        matmul_prim->execute(*stream, matmul_args);
        stream->wait();

        if (postOp == true) {
            // FunTimer t3;
            if (M > 1)
                sycl_sigmoid_mul(M, N / 2, packedC, ldc, packedC + N / 2, ldc, packedA, ldc / 2);
            else if (M == 1)
                sycl_sigmoid_mul_M1(N / 2, packedC, ldc, packedC + N / 2, ldc, packedA, ldc / 2);
            // printf("xft_verbose,exec,gpu:%d,%s,%.6lf\n", gpu_index, "sycl_sigmoid_mul", t3.elapsed());
        }

        if (copyG2C == true) {
            // Reorder output
            // FunTimer t3;
            gpu_queue->memcpy(HostBuf, packed_output_mem.get_data_handle(), M * N * sizeof(float16_t)).wait();
            float16_t::cvt_float16_to_float_MT(HostBuf, C, M * N);
            // printf("xft_verbose,exec,gpu:%d,%s,%.6lf\n", gpu_index, "memcpy", t3.elapsed());
        }

        // std::cout << "GPU"<< gpu_index << "gemm:" << std::endl;
        // print(M, 6, K, (float16_t *)packedA, "packedA");
        // print(M, 6, N, (float16_t *)packedC, "packedC");
    }

    void onednn_sgemm_f32f16f32_compute_residential(bool transA, int M, int N, int K, float alpha, const float *A,
            int lda, const float16_t *packedB, float beta, float *C, int ldc, const float *bias, const float *res,
            int ldres, bool copyC2G = true, bool copyG2C = true) {
        TimeLine t("onednn_sgemm_f32f16f32_compute_residential");
        TimeLine t1("onednn_sgemm_f32f16f32_compute_residential.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Residential);
        auto it = matmul_hub.find(key);
        if (it != matmul_hub.end()) {
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
            auto input_md = memory::desc(input_dims, dt::f16, get_onednn_input_layout(dt::f16));
            auto weight_md = memory::desc(weight_dims, dt::f16, get_onednn_weight_layout(dt::f16));
            auto bias_md = memory::desc(bias_dims, dt::f32, get_onednn_bias_layout(dt::f16));
            auto shift_md = memory::desc(shift_dims, dt::f16, get_onednn_shift_layout(dt::f16));
            auto output_md = memory::desc(output_dims, dt::f16, get_onednn_output_layout(dt::f16));

            // Create primitive post-ops (residential): dst_tmp = dst_tmp + shift
            post_ops matmul_ops;
            matmul_ops.append_binary(algorithm::binary_add, shift_md);
            primitive_attr matmul_attr;
            matmul_attr.set_post_ops(matmul_ops);

            if (bias != nullptr) {
                // Create primitive descriptor and primitive.
                matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, bias_md, output_md, matmul_attr);
                matmul_prim = new matmul(*matmul_pd);
            } else {
                // Create primitive descriptor and primitive.
                matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md, matmul_attr);
                matmul_prim = new matmul(*matmul_pd);
            }

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::Residential);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            matmul_hub[key] = value;
        }

        // Repack and convert input data.
        auto packed_input_mem = memory(matmul_pd->src_desc(), *engine, packedA);
        auto packed_weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<float16_t *>(packedB));
        memory bias_mem;
        if (bias != nullptr) { bias_mem = memory(matmul_pd->bias_desc(), *engine, const_cast<float *>(bias)); }
        auto shift_md = memory::desc({M, N}, dt::f16, get_onednn_shift_layout(dt::f16));
        auto shift_mem = memory(shift_md, *engine, const_cast<float16_t *>(packedI));
        auto packed_output_mem = memory(matmul_pd->dst_desc(), *engine, packedI);

        if (copyC2G == true) {
            // Reorder input
            // FunTimer t2;
            // float16_t::cvt_float_to_float16_MT(A, A_buf, M * K);
#pragma omp parallel for
            for (int i = 0; i < M; ++i)
                float16_t::cvt_float_to_float16(A + i * lda, HostBuf + i * K, K);
            gpu_queue->memcpy(packed_input_mem.get_data_handle(), HostBuf, M * K * sizeof(float16_t)).wait();
            // printf("xft_verbose,exec,gpu:%d,%s,%.6lf\n", gpu_index, "memcpy", t2.elapsed());

            // // Reorder shift
            // FunTimer t3;
            // float16_t shift_buf[M * N];
            // float16_t::cvt_float_to_float16_MT(res, shift_buf, M * N);
            // gpu_queue->memcpy(shift_mem.get_data_handle(), shift_buf, M * N * sizeof(float16_t)).wait();
            // // printf("xft_verbose,exec,gpu:%d,%s,%.6lf\n", gpu_index, "memcpy", t3.elapsed());
        }

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, packed_input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, packed_weight_mem});
        if (bias != nullptr) { matmul_args.insert({DNNL_ARG_BIAS, bias_mem}); }
        matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, shift_mem});
        matmul_args.insert({DNNL_ARG_DST, packed_output_mem});
        matmul_prim->execute(*stream, matmul_args);
        stream->wait();

        if (copyG2C == true) {
            // Reorder output
            // FunTimer t4;
            gpu_queue->memcpy(HostBuf, packed_output_mem.get_data_handle(), M * N * sizeof(float16_t)).wait();
            float16_t::cvt_float16_to_float_MT(HostBuf, C, M * N);
            // printf("xft_verbose,exec,gpu:%d,%s,%.6lf\n", gpu_index, "memcpy", t4.elapsed());
        }

        // std::cout << "GPU"<< gpu_index << "gemm_res:" << std::endl;
        // print(M, 6, K, (float16_t *)packedA, "packedA");
        // print(M, 6, N, (float16_t *)packedI, "packedI");
    }

    template <typename Tin, typename Tout>
    void onednn_amx_sgemm_f32bf16f32_compute(bool transA, int M, int N, int K, float alpha, const Tin *A, int lda,
            const bfloat16_t *packedB, float beta, Tout *C, int ldc) {
        TimeLine t("onednn_amx_sgemm_f32bf16f32_compute");
        TimeLine t1("onednn_amx_sgemm_f32bf16f32_compute.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Basic);
        auto it = matmul_hub.find(key);
        if (it != matmul_hub.end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B) and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims output_dims = {M, N};

            // Create memory descriptors and memory objects for src, weights, bias, and dst.
            auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_onednn_weight_layout(dt::bf16));
            memory::desc output_md;
            if constexpr (std::is_same_v<Tout, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            } else if constexpr (std::is_same_v<Tout, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            } else {
                printf(">>> onednn amx output date type not supported.");
            }

            // Create primitive descriptor and primitive.
            matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::Basic);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            matmul_hub[key] = value;
        }

        // Repack and convert input data.
        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), *engine);
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<bfloat16_t *>(A));
        } else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<bfloat16_t *>(packedB));
        auto output_mem = memory(matmul_pd->dst_desc(), *engine, C);

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
            for (uint64_t i = 0; i < M; ++i) {
                bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
            }
        }

        matmul_prim->execute(*stream, matmul_args);
        stream->wait();
    }

    template <typename Tin, typename Tout>
    void onednn_amx_sgemm_f32bf16f32_compute_biasadd(bool transA, int M, int N, int K, float alpha, const Tin *A,
            int lda, const bfloat16_t *packedB, float beta, Tout *C, int ldc, const float *bias) {
        TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_biasadd");
        TimeLine t1("onednn_amx_sgemm_f32bf16f32_compute_biasadd.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::BiasAdd);
        auto it = matmul_hub.find(key);
        if (it != matmul_hub.end()) {
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
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_onednn_weight_layout(dt::bf16));
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
            matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, bias_md, output_md);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::BiasAdd);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            matmul_hub[key] = value;
        }

        // Repack and convert input data.
        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), *engine);
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<bfloat16_t *>(A));
        } else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<bfloat16_t *>(packedB));
        auto bias_mem = memory(matmul_pd->bias_desc(), *engine, const_cast<float *>(bias));
        auto output_mem = memory(matmul_pd->dst_desc(), *engine, C);

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
            for (uint64_t i = 0; i < M; ++i) {
                bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
            }
        }

        matmul_prim->execute(*stream, matmul_args);
        stream->wait();
    }

    template <typename Tin, typename Tout>
    void onednn_amx_sgemm_f32bf16f32_compute_biasadd_relu(bool transA, int M, int N, int K, float alpha, const Tin *A,
            int lda, const bfloat16_t *packedB, float beta, Tout *C, int ldc, const float *bias) {
        TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_biasadd_relu");
        TimeLine t1("onednn_amx_sgemm_f32bf16f32_compute_biasadd_relu.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::BiasAdd_Relu);
        auto it = matmul_hub.find(key);
        if (it != matmul_hub.end()) {
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
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_onednn_weight_layout(dt::bf16));
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
            matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, bias_md, output_md, matmul_attr);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::BiasAdd_Relu);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            matmul_hub[key] = value;
        }

        // Repack and convert input data.
        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), *engine);
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<bfloat16_t *>(A));
        } else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<bfloat16_t *>(packedB));
        auto bias_mem = memory(matmul_pd->bias_desc(), *engine, const_cast<float *>(bias));
        auto output_mem = memory(matmul_pd->dst_desc(), *engine, C);

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
            for (uint64_t i = 0; i < M; ++i) {
                bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
            }
        }

        matmul_prim->execute(*stream, matmul_args);
        stream->wait();
    }

    template <typename Tin, typename Tout>
    void onednn_amx_sgemm_f32bf16f32_compute_silu(bool transA, int M, int N, int K, float alpha, const Tin *A, int lda,
            const bfloat16_t *packedB, float beta, Tout *C, int ldc) {
        TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_silu");
        TimeLine t1("onednn_amx_sgemm_f32bf16f32_compute_silu.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Silu);
        auto it = matmul_hub.find(key);
        if (it != matmul_hub.end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B), and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims output_dims = {M, N};

            // Create primitive descriptor.
            auto input_md = memory::desc(input_dims, dt::bf16, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_onednn_weight_layout(dt::bf16));
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
            matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md, matmul_attr);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::Silu);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            matmul_hub[key] = value;
        }

        // Repack and convert input data.
        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), *engine);
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<bfloat16_t *>(A));
        } else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<bfloat16_t *>(packedB));
        auto output_mem = memory(matmul_pd->dst_desc(), *engine, C);

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
            for (uint64_t i = 0; i < M; ++i) {
                bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
            }
        }

        matmul_prim->execute(*stream, matmul_args);
        stream->wait();
    }

    template <typename Tin, typename Tout>
    void onednn_amx_sgemm_f32bf16f32_compute_resmul(bool transA, int M, int N, int K, float alpha, const Tin *A,
            int lda, const bfloat16_t *packedB, float beta, Tout *C, int ldc, const Tin *res, int ldres) {
        TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_resmul");
        TimeLine t1("onednn_amx_sgemm_f32bf16f32_compute_resmul.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Resmul);
        auto it = matmul_hub.find(key);
        if (it != matmul_hub.end()) {
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
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_onednn_weight_layout(dt::bf16));
            auto scale_md = memory::desc(scale_dims,
                    std::is_same_v<Tin, float> ? dt::f32 : (std::is_same_v<Tin, bfloat16_t> ? dt::bf16 : dt::undef),
                    tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tout, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            } else if constexpr (std::is_same_v<Tout, bfloat16_t>) {
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
            matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md, matmul_attr);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::Resmul);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            matmul_hub[key] = value;
        }

        // Repack and convert input data.
        memory::dims scale_dims = {M, N};
        auto scale_md = memory::desc(scale_dims,
                std::is_same_v<Tin, float> ? dt::f32 : (std::is_same_v<Tin, bfloat16_t> ? dt::bf16 : dt::undef),
                tag::ab);
        dnnl::memory scale_mem;
        if (C == res) {
            scale_mem = memory(scale_md, *engine);
#pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                memcpy((Tin *)scale_mem.get_data_handle() + i * N, res + i * ldres, N * sizeof(Tin));
            }
        } else {
            scale_mem = memory(scale_md, *engine, (void *)res);
        }

        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), *engine);
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<bfloat16_t *>(A));
        } else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<bfloat16_t *>(packedB));
        auto output_mem = memory(matmul_pd->dst_desc(), *engine, C);

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
            for (uint64_t i = 0; i < M; ++i) {
                bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
            }
        }

        matmul_prim->execute(*stream, matmul_args);
        stream->wait();
    }

    template <typename Tin, typename Tout>
    void onednn_amx_sgemm_f32bf16f32_compute_residential(bool transA, int M, int N, int K, float alpha, const Tin *A,
            int lda, const bfloat16_t *packedB, float beta, Tout *C, int ldc, const float *bias, const Tin *res,
            int ldres) {
        TimeLine t("onednn_amx_sgemm_f32bf16f32_compute_residential");
        TimeLine t1("onednn_amx_sgemm_f32bf16f32_compute_residential.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Residential);
        auto it = matmul_hub.find(key);
        if (it != matmul_hub.end()) {
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
            auto weight_md = memory::desc(weight_dims, dt::bf16, get_onednn_weight_layout(dt::bf16));
            auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
            auto shift_md = memory::desc(shift_dims,
                    std::is_same_v<Tin, float> ? dt::f32 : (std::is_same_v<Tout, bfloat16_t> ? dt::bf16 : dt::undef),
                    tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tout, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            } else if constexpr (std::is_same_v<Tout, bfloat16_t>) {
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
                matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, bias_md, output_md, matmul_attr);
                matmul_prim = new matmul(*matmul_pd);
            } else {
                // Create primitive descriptor and primitive.
                matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md, matmul_attr);
                matmul_prim = new matmul(*matmul_pd);
            }

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::Residential);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            matmul_hub[key] = value;
        }

        // Repack and convert input data.
        memory::dims shift_dims = {M, N};
        auto shift_md = memory::desc(shift_dims,
                std::is_same_v<Tin, float> ? dt::f32 : (std::is_same_v<Tout, bfloat16_t> ? dt::bf16 : dt::undef),
                tag::ab);

        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), *engine);
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<bfloat16_t *>(A));
        } else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<bfloat16_t *>(packedB));
        memory bias_mem;
        auto shift_mem = memory(shift_md, *engine, (void *)res);
        auto output_mem = memory(matmul_pd->dst_desc(), *engine, C);
        if (bias != nullptr) { bias_mem = memory(matmul_pd->bias_desc(), *engine, const_cast<float *>(bias)); }

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
            for (uint64_t i = 0; i < M; ++i) {
                bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
            }
        }

        matmul_prim->execute(*stream, matmul_args);
        stream->wait();
    }

    void onednn_amx_gemm_s8s8s32(bool transA, int M, int N, int K, float alpha, const int8_t *A, int lda,
            const int8_t *B, float beta, int32_t *C, int ldc) {
        TimeLine t("onednn_amx_gemm_s8s8s32");
        TimeLine t1("onednn_amx_gemm_s8s8s32.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Basic);
        auto it = matmul_hub.find(key);
        if (it != matmul_hub.end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B) and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims output_dims = {M, N};

            // Create memory descriptors and memory objects for src, weights, bias, and dst.
            auto input_md = memory::desc(input_dims, dt::s8, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt::s8, get_onednn_weight_layout(dt::s8));
            memory::desc output_md;
            output_md = memory::desc(output_dims, dt::s32, tag::ab);

            // Create primitive descriptor and primitive.
            matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::Basic);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            matmul_hub[key] = value;
        }

        auto input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<int8_t *>(A));
        auto weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<int8_t *>(B));
        auto output_mem = memory(matmul_pd->dst_desc(), *engine, C);

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        t1.release();

        // Executions.
        TimeLine t2("onednn_gemm_s8s8s32.execute_primitive");
        matmul_prim->execute(*stream, matmul_args);
        stream->wait();
    }

    void onednn_amx_gemm_f32s8f32_compute(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
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
            uint64_t offset = range.first;
            onednn_amx_gemm_f32s8f32_compute_base(transA, MB, N, K, alpha, A + offset * lda, lda, B, scaleB, zeroB,
                    sumB, beta, C + offset * ldc, ldc, bias, res + offset * ldres, ldres, gamma, kind);
        }
    }

    void onednn_amx_gemm_f32s8f32_compute_base(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const int8_t *B, const float *scaleB, const float *zeroB, const float *sumB, float beta, float *C, int ldc,
            const float *bias, const float *res, int ldres, float gamma, matmul_kinds kind) {

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
    void quantize_s8(
            int M, int N, const float *src, int lda, int8_t *dst, int ldb, float *scale, float *zero, float *sum) {
#pragma omp parallel for
        for (uint16_t i = 0; i < M; i++) {
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
    void dequant_base(int M, int N, const int32_t *C_int32, const int ldc_int32, float *C, const int ldc,
            const DequantOp &dequant_op, const PostOp &post_op) {
#pragma omp parallel for collapse(2)
        for (uint64_t i = 0; i < M; i++) {
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
    void dequant(int M, int N, const int32_t *C_int32, const int ldc_int32, float *C, const int ldc,
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
