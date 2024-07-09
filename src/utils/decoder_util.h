// Copyright (c) 2023-2024 Intel Corporation
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
#include <string>

#include <mkl.h>
#include "bert_util.h"
#include "bfloat16.h"
#include "compile_util.h"
#include "float16.h"
#include "intrinsics_util.h"
#include "matmul_helper.h"
#include "my_types.h"
#include "timeline.h"
#include "transformer_ctx.h"
#include "xdnn.h"

#ifdef XFT_GPU
#include <CL/sycl.hpp>
#endif

extern int getFlashThresh();
extern bool enableCATMLP();

class DecoderUtil {
public:
#if __AVX512F__
    static void rmsNorm(xft::Matrix<float> &x, xft::Matrix<float> &y, xft::Vector<float> &normWeight, float epsilon) {
        TimeLine t("DecoderUtil::rmsNorm");
        float *pweight = normWeight.Data();
        int size = x.Cols();

#pragma omp parallel for
        for (int r = 0; r < x.Rows(); ++r) {
            float *px = x.Row(r);
            float *py = y.Row(r);

            float squareSum = 0;

            __m512 vsqare = _mm512_set1_ps(0);

            int col = 0;
            for (; col + 15 < size; col += 16) {
                // SUM(x*x)
                __m512 vx = _mm512_loadu_ps(px + col);
                __m512 tmp = _mm512_mul_ps(vx, vx);
                vsqare = _mm512_add_ps(vsqare, tmp);
            }
            if (col < size) {
                __mmask16 mask = (1 << (size - col)) - 1;
                __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
                __m512 tmp = _mm512_mul_ps(vx, vx);
                vsqare = _mm512_add_ps(vsqare, tmp);
            }

            squareSum = _mm512_reduce_add_ps(vsqare);

            // Variance
            float var = 1 / sqrt(squareSum / size + epsilon);
            __m512 vvar = _mm512_set1_ps(var);

            for (col = 0; col + 15 < size; col += 16) {
                __m512 vx = _mm512_loadu_ps(px + col);
                __m512 vw = _mm512_loadu_ps(pweight + col);
                __m512 vy = vx * vvar * vw;
                _mm512_storeu_ps(py + col, vy);
            }
            if (col < size) {
                __mmask16 mask = (1 << (size - col)) - 1;
                __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
                __m512 vw = _mm512_maskz_loadu_ps(mask, pweight + col);
                __m512 vy = vx * vvar * vw;
                _mm512_mask_storeu_ps(py + col, mask, vy);
            }
        } // end for rows
    }

    static void layerNorm(
            xft::Matrix<float> &x, xft::Matrix<float> &y, xft::Vector<float> &gamma, xft::Vector<float> &beta) {
        TimeLine t("DecoderUtil::layerNorm");
        float *pgamma = gamma.Data();
        float *pbeta = beta.Data();
        int size = x.Cols();

        if (x.Rows() == 1) {
            LayerNormOneRow(x, y, pgamma, pbeta, size);
            return;
        }

#pragma omp parallel for
        for (int r = 0; r < x.Rows(); ++r) {
            float *px = x.Row(r);
            float *py = y.Row(r);

            float sum = 0;
            float squareSum = 0;

            __m512 vsum = _mm512_set1_ps(0);
            __m512 vsqare = _mm512_set1_ps(0);

            for (int col = 0; col < size; col += 16) {
                int remain = size - col;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                // SUM(x)
                __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
                vsum = _mm512_add_ps(vsum, vx);

                // SUM(x*x)
                __m512 tmp = _mm512_mul_ps(vx, vx);
                vsqare = _mm512_add_ps(vsqare, tmp);
            }

            sum = _mm512_reduce_add_ps(vsum);
            squareSum = _mm512_reduce_add_ps(vsqare);

            // Mean
            float mean = sum / size;
            __m512 vmean = _mm512_set1_ps(mean);

            // Variance
            const float epsilon = 1e-5; // use the default value in PyTorch
            float var = 1 / sqrt(squareSum / size - mean * mean + epsilon);
            __m512 vvar = _mm512_set1_ps(var);

            for (int col = 0; col < size; col += 16) {
                int remain = size - col;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
                __m512 vgamma = _mm512_maskz_loadu_ps(mask, pgamma + col);
                __m512 vbeta = _mm512_maskz_loadu_ps(mask, pbeta + col);
                __m512 vy = (vx - vmean) * vgamma * vvar + vbeta;
                _mm512_mask_storeu_ps(py + col, mask, vy);
            }
        }
    }

    // Layer norm for small matrix with just a one Rrow
    static void LayerNormOneRow(xft::Matrix<float> &x, xft::Matrix<float> &y, float *pgamma, float *pbeta, int size) {
        TimeLine t("DecoderUtil::LayerNormOneRow");
        constexpr int BLKSIZE = 128;
        const int splitSize = (size > BLKSIZE && size % BLKSIZE == 0) ? BLKSIZE : size; // size of each split
        const int splitNum = (size + splitSize - 1) / splitSize;

        float sum = 0;
        float squareSum = 0;
        float *px = x.Row(0);
        float *py = y.Row(0);

#pragma omp parallel for reduction(+ : sum) reduction(+ : squareSum)
        for (int s = 0; s < splitNum; ++s) {
            __m512 vsum = _mm512_set1_ps(0);
            __m512 vsqare = _mm512_set1_ps(0);

            int end = (s + 1) * splitSize;
            if (end > size) end = size;
            for (int col = s * splitSize; col < end; col += 16) {
                int remain = end - col;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                // SUM(x)
                __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
                vsum = _mm512_add_ps(vsum, vx);

                // SUM(x*x)
                __m512 tmp = _mm512_mul_ps(vx, vx);
                vsqare = _mm512_add_ps(vsqare, tmp);
            }

            sum += _mm512_reduce_add_ps(vsum);
            squareSum += _mm512_reduce_add_ps(vsqare);
        }

        // Mean
        float mean = sum / size;

        // Variance
        const float epsilon = 1e-5; // use the default value in PyTorch
        float var = 1 / sqrt(squareSum / size - mean * mean + epsilon);

#pragma omp parallel for
        for (int s = 0; s < splitNum; ++s) {
            __m512 vmean = _mm512_set1_ps(mean);
            __m512 vvar = _mm512_set1_ps(var);

            int end = (s + 1) * splitSize;
            if (end > size) end = size;
            for (int col = s * splitSize; col < end; col += 16) {
                int remain = size - col;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
                __m512 vgamma = _mm512_maskz_loadu_ps(mask, pgamma + col);
                __m512 vbeta = _mm512_maskz_loadu_ps(mask, pbeta + col);
                __m512 vy = (vx - vmean) * vgamma * vvar + vbeta;
                _mm512_mask_storeu_ps(py + col, mask, vy);
            }
        }
    }
#else
    static void layerNorm(DecoderContext *ctx, xft::Matrix<float> &x, xft::Matrix<float> &y, xft::Vector<float> &gamma,
            xft::Vector<float> &beta) {
        TimeLine t("DecoderUtil::layerNorm");
        assert(x.Rows() == ctx->batchSize * ctx->inputSeqLen);
        assert(x.Cols() == ctx->hiddenSize);

        float *pgamma = gamma.Data();
        float *pbeta = beta.Data();

#pragma omp parallel for
        for (int i = 0; i < x.Rows(); ++i) {
            float sum = 0;
            float *px = x.Row(i);
            float *py = y.Row(i);
#pragma omp simd
            for (int j = 0; j < x.Cols(); ++j) {
                sum += px[j];
            }
            float mean = sum / ctx->hiddenSize;

            sum = 0;
#pragma omp simd
            for (int j = 0; j < x.Cols(); ++j) {
                float delta = (px[j] - mean);
                sum += delta * delta;
            }
            float tmp = sum / ctx->hiddenSize + 9.999999960041972e-13;
            float rvariance = 1.0f / sqrt(tmp);

#pragma omp simd
            for (int j = 0; j < x.Cols(); ++j) {
                py[j] = (px[j] - mean) * rvariance * pgamma[j] + pbeta[j];
            }
        }
    }
#endif

    // General version
    static void computeSoftmax(float *data, const float *attnMask, int size, float scale) {
        int vecs = (size + 15) / 16; // how many avx512 vectors
        __mmask16 tailMask = (size % 16 == 0 ? 0xffff : (1 << (size % 16)) - 1); // mask of last vector

        __m512 vsum = _mm512_set1_ps(0);

        // maxVal is used to avoid exp(x) = inf
        float maxVal = std::numeric_limits<float>::lowest();
        __m512 vmax = _mm512_set1_ps(maxVal);
        __m512 vfactor = _mm512_set1_ps(scale);

        int i = 0;
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vx = _mm512_maskz_loadu_ps(k, data + i * 16);
            __m512 vmask = _mm512_maskz_loadu_ps(k, attnMask + i * 16);
            vmax = _mm512_mask_max_ps(vmax, k, vmax, vx * vfactor + vmask);
        }

        maxVal = _mm512_reduce_max_ps(vmax);
        vmax = _mm512_set1_ps(maxVal);

        // Compute vexp(vx - vmax) and sum it
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vx = _mm512_maskz_loadu_ps(k, data + i * 16);
            __m512 vmask = _mm512_maskz_loadu_ps(k, attnMask + i * 16);
            vx = BertUtil::vexp(vx * vfactor + vmask - vmax);
            _mm512_mask_storeu_ps(data + i * 16, k, vx);
            vsum = _mm512_mask_add_ps(vsum, k, vsum, vx);
        }

        float sum = _mm512_reduce_add_ps(vsum);
        __m512 vrsum = _mm512_set1_ps(1.0f / sum);

        // Compute exp/sum(exp) and store
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vx = _mm512_maskz_loadu_ps(k, data + i * 16);
            vx = vx * vrsum;
            _mm512_mask_storeu_ps(data + i * 16, k, vx);
        }
    }

    // General version
    static void computeSoftmax(float *data, int size) {
        int vecs = (size + 15) / 16; // how many avx512 vectors
        __mmask16 tailMask = (size % 16 == 0 ? 0xffff : (1 << (size % 16)) - 1); // mask of last vector

        __m512 vsum = _mm512_set1_ps(0);

        // maxVal is used to avoid exp(x) = inf
        float maxVal = std::numeric_limits<float>::lowest();
        __m512 vmax = _mm512_set1_ps(maxVal);

        int i = 0;
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vx = _mm512_maskz_loadu_ps(k, data + i * 16);
            vmax = _mm512_mask_max_ps(vmax, k, vmax, vx);
        }

        maxVal = _mm512_reduce_max_ps(vmax);
        vmax = _mm512_set1_ps(maxVal);

        // Compute vexp(vx - vmax) and sum it
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vx = _mm512_maskz_loadu_ps(k, data + i * 16);
            vx = BertUtil::vexp(vx - vmax);
            _mm512_mask_storeu_ps(data + i * 16, k, vx);
            vsum = _mm512_mask_add_ps(vsum, k, vsum, vx);
        }

        float sum = _mm512_reduce_add_ps(vsum);
        __m512 vrsum = _mm512_set1_ps(1.0f / sum);

        // Compute exp/sum(exp) and store
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vx = _mm512_maskz_loadu_ps(k, data + i * 16);
            vx = vx * vrsum;
            _mm512_mask_storeu_ps(data + i * 16, k, vx);
        }
    }

    // General version
    static void computeSoftmax(float16_t *data, float scale, int size) {
        int vecs = (size + 15) / 16; // how many avx512 vectors
        __mmask16 tailMask = (size % 16 == 0 ? 0xffff : (1 << (size % 16)) - 1); // mask of last vector

        __m512 vsum = _mm512_set1_ps(0);

        // maxVal is used to avoid exp(x) = inf
        float maxVal = std::numeric_limits<float>::lowest();
        __m512 vmax = _mm512_set1_ps(maxVal);
        __m512 vfactor = _mm512_set1_ps(scale);

        int i = 0;
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vx = xft::load_avx512(k, data + i * 16);
            vmax = _mm512_mask_max_ps(vmax, k, vmax, vx * vfactor);
        }

        maxVal = _mm512_reduce_max_ps(vmax);
        vmax = _mm512_set1_ps(maxVal);

        // Compute vexp(vx - vmax) and sum it
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vx = xft::load_avx512(k, data + i * 16);
            vx = BertUtil::vexp(vx * vfactor - vmax);
            xft::store_avx512(data + i * 16, k, vx);
            vsum = _mm512_mask_add_ps(vsum, k, vsum, vx);
        }

        float sum = _mm512_reduce_add_ps(vsum);
        __m512 vrsum = _mm512_set1_ps(1.0f / sum);

        // Compute exp/sum(exp) and store
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vx = xft::load_avx512(k, data + i * 16);
            vx = vx * vrsum;
            xft::store_avx512(data + i * 16, k, vx);
        }
    }

    // Softmax: skip the calculation when attention mask is the lowest value
    static void softmaxSkipMask(float *data, const float *attnMask, int size, float scale) {
        int vecs = (size + 15) / 16; // how many avx512 vectors
        __mmask16 tailMask = (size % 16 == 0 ? 0xffff : (1 << (size % 16)) - 1); // mask of last vector

        __m512 vzero = _mm512_set1_ps(0);
        __m512 vsum = _mm512_set1_ps(0);

        // maxVal is used to avoid exp(x) = inf
        float maxVal = std::numeric_limits<float>::lowest();
        __m512 vlowest = _mm512_set1_ps(maxVal);
        __m512 vmax = _mm512_set1_ps(maxVal);
        __m512 vfactor = _mm512_set1_ps(scale);

        int i = 0;
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vmask = _mm512_maskz_loadu_ps(k, attnMask + i * 16);

            // masked out
            if (_mm512_cmpeq_ps_mask(vmask, vlowest) == 0xffff) { continue; }

            __m512 vx = _mm512_maskz_loadu_ps(k, data + i * 16);
            vmax = _mm512_mask_max_ps(vmax, k, vmax, vx * vfactor + vmask);
        }

        maxVal = _mm512_reduce_max_ps(vmax);
        vmax = _mm512_set1_ps(maxVal);

        // Compute vexp(vx - vmax) and sum it
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vmask = _mm512_maskz_loadu_ps(k, attnMask + i * 16);

            // masked out
            if (_mm512_cmpeq_ps_mask(vmask, vlowest) == 0xffff) { continue; }

            __m512 vx = _mm512_maskz_loadu_ps(k, data + i * 16);
            vx = BertUtil::vexp(vx * vfactor + vmask - vmax);
            _mm512_mask_storeu_ps(data + i * 16, k, vx);
            vsum = _mm512_mask_add_ps(vsum, k, vsum, vx);
        }

        float sum = _mm512_reduce_add_ps(vsum);
        __m512 vrsum = _mm512_set1_ps(1.0f / sum);

        // Compute exp/sum(exp) and store
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vmask = _mm512_maskz_loadu_ps(k, attnMask + i * 16);

            // masked out
            if (_mm512_cmpeq_ps_mask(vmask, vlowest) == 0xffff) {
                _mm512_mask_storeu_ps(data + i * 16, k, vzero);
                continue;
            }

            __m512 vx = _mm512_maskz_loadu_ps(k, data + i * 16);
            vx = vx * vrsum;
            _mm512_mask_storeu_ps(data + i * 16, k, vx);
        }
    }

    // Same implementation with softmax, but:
    // Return max value, and the sum value of exp
    static std::pair<float, float> softmaxWithStats(float *data, const float *attnMask, int size, float scale) {
        int vecs = (size + 15) / 16; // how many avx512 vectors
        __mmask16 tailMask = (size % 16 == 0 ? 0xffff : (1 << (size % 16)) - 1); // mask of last vector

        __m512 vsum = _mm512_set1_ps(0);

        // maxVal is used to avoid exp(x) = inf
        float maxVal = std::numeric_limits<float>::lowest();
        __m512 vmax = _mm512_set1_ps(maxVal);
        __m512 vfactor = _mm512_set1_ps(scale);

        int i = 0;
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vx = _mm512_maskz_loadu_ps(k, data + i * 16);
            __m512 vmask = _mm512_maskz_loadu_ps(k, attnMask + i * 16);
            vmax = _mm512_mask_max_ps(vmax, k, vmax, vx * vfactor + vmask);
        }

        maxVal = _mm512_reduce_max_ps(vmax);
        vmax = _mm512_set1_ps(maxVal);

        // Compute vexp(vx - vmax) and sum it
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vx = _mm512_maskz_loadu_ps(k, data + i * 16);
            __m512 vmask = _mm512_maskz_loadu_ps(k, attnMask + i * 16);
            vx = BertUtil::vexp(vx * vfactor + vmask - vmax);
            _mm512_mask_storeu_ps(data + i * 16, k, vx);
            vsum = _mm512_mask_add_ps(vsum, k, vsum, vx);
        }

        float sum = _mm512_reduce_add_ps(vsum);
        __m512 vrsum = _mm512_set1_ps(1.0f / sum);

        // Compute exp/sum(exp) and store
        for (i = 0; i < vecs; ++i) {
            __mmask16 k = (i == vecs - 1 ? tailMask : 0xffff);
            __m512 vx = _mm512_maskz_loadu_ps(k, data + i * 16);
            vx = vx * vrsum;
            _mm512_mask_storeu_ps(data + i * 16, k, vx);
        }

        return std::make_pair(maxVal, sum);
    }

#ifdef XFT_GPU
    template <typename T1, typename T2>
    static void siluSum(xft::Matrix<T1> &src, xft::Matrix<T2> &dst, void *device = nullptr) {
        int M = src.Rows();
        int lds = src.Stride();
        int N = lds / 2;
        int ldd = dst.Stride();

        if (device != nullptr) {
            sycl::queue *gpu_queue = static_cast<sycl::queue *>(device);

            if constexpr (std::is_same_v<T1, float16_t> && std::is_same_v<T2, float16_t>) {
                sycl::half *src0 = (sycl::half *)src.Data();
                sycl::half *src1 = (sycl::half *)(src.Data() + N);
                sycl::half *dest = (sycl::half *)dst.Data();

                gpu_queue
                        ->submit([&](sycl::handler &h) {
                            h.parallel_for(M * N, [=](auto i) {
                                int32_t row = i / N;
                                int32_t col = i % N;
                                sycl::half tmp0 = src0[row * lds + col];
                                sycl::half tmp1 = src1[row * lds + col];
                                dest[row * ldd + col] = tmp0 * tmp1
                                        / ((sycl::half)1.0f + (sycl::half)sycl::native::exp(tmp0 * -1.0f));
                            });
                        })
                        .wait();
            }
        }
    }
#else
    // compute silu on the left half and then add it with the right half
    template <typename T1, typename T2>
    static void siluSum(xft::Matrix<T1> &src, xft::Matrix<T2> &dst, void *device = nullptr) {
        __m512 one = _mm512_set1_ps(1.f);
        __m512 negOne = _mm512_set1_ps(-1.f);
        int M = src.Rows();
        int stride = src.Cols();
        int N = stride / 2;
        int ldd = dst.Stride();

#pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; j += 16) {
                int remain = N - j;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                auto left = xft::load_avx512(mask, src.Data() + i * stride + j);
                auto right = xft::load_avx512(mask, src.Data() + i * stride + j + N);
                auto x0 = BertUtil::vexp(_mm512_mul_ps(left, negOne));
                auto x1 = _mm512_add_ps(one, x0);
                auto x2 = _mm512_div_ps(left, x1);
                auto res = _mm512_mul_ps(right, x2);
                xft::store_avx512(dst.Data() + i * ldd + j, mask, res);
            }
        }
    }
#endif

    // compute gelu on the left half and then add it with the right half
    template <typename T1, typename T2>
    static void geluSum(xft::Matrix<T1> &src, xft::Matrix<T2> &dst, void *device = nullptr) {
        const __m512 c1 = _mm512_set1_ps(0.044715f);
        const __m512 c2 = _mm512_set1_ps(0.7978845608f);
        const __m512 vone = _mm512_set1_ps(1.0f);
        const __m512 vtwo = _mm512_set1_ps(2.0f);
        const __m512 vhalf = _mm512_set1_ps(0.5f);
        const int M = src.Rows();
        const int stride = src.Cols();
        const int N = stride / 2;
        const int ldd = dst.Stride();

#pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; j += 16) {
                int remain = N - j;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                auto vx = xft::load_avx512(mask, src.Data() + i * stride + j);
                auto right = xft::load_avx512(mask, src.Data() + i * stride + j + N);
                __m512 vt = c2 * (vx + c1 * vx * vx * vx);
                vt = BertUtil::vexp(vt * vtwo);
                vt = vone - vtwo * _mm512_rcp14_ps(vt + vone); // tanh
                __m512 vy = vx * (vone + vt) * vhalf;
                auto res = _mm512_mul_ps(right, vy);
                xft::store_avx512(dst.Data() + i * ldd + j, mask, res);
            }
        }
    }

    // C = A * B
    // bTranspose: B need to be transposed or not
    // xdnn_sgemm_single_thread(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    template <typename T>
    static void sgemm(const T *A, const T *B, float *C, int m, int n, int k, int lda, int ldb, int ldc, bool transa,
            bool transb) {
        float alpha = 1;
        float beta = 0;

        if constexpr (std::is_same_v<T, float>) {
            char ta[] = "N";
            char tb[] = "N";
            if (transa) ta[0] = 'T';
            if (transb) tb[0] = 'T';

            dnnl_sgemm(ta[0], tb[0], m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

        } else {
            CBLAS_TRANSPOSE ta, tb;
            ta = transa ? CblasTrans : CblasNoTrans;
            tb = transb ? CblasTrans : CblasNoTrans;

            if (std::is_same_v<T, bfloat16_t>) {
                cblas_gemm_bf16bf16f32(CblasRowMajor, ta, tb, m, n, k, alpha, (const MKL_BF16 *)(A), lda,
                        (const MKL_BF16 *)(B), ldb, beta, C, ldc);
            } else if (std::is_same_v<T, float16_t>) {
                static int mkl_enable_inst = -1;
                if (mkl_enable_inst == -1) {
#ifdef AMX_FP16_WEIGHT_ONLY_FP16
                    // AMX FP16
                    mkl_enable_inst = mkl_enable_instructions(MKL_ENABLE_AVX512_E5);
#else
                    // AVX512_FP16, skip E4 avoiding illegal instruction error
                    mkl_enable_inst = mkl_enable_instructions(MKL_ENABLE_AVX512_E3);
#endif
                }
                cblas_gemm_f16f16f32(CblasRowMajor, ta, tb, m, n, k, alpha, (const MKL_F16 *)(A), lda,
                        (const MKL_F16 *)(B), ldb, beta, C, ldc);
            } else {
                printf("Datatype Not supported yet\n");
                exit(-1);
            }
        }
    }

    // need to do for res.
    template <typename ImT>
    static void softmaxTile(float *AB, ImT *ABout, float *sum, float *max, float *preSum, float *preMax, float scale,
            const float *attnMask, int m, int k, int attnMskStride) {
        float maxVal = std::numeric_limits<float>::lowest();
        __m512 vscale = _mm512_set1_ps(scale);
        for (int i = 0; i < m; ++i) {
            float *buf = AB + i * k;
            ImT *obuf = ABout + i * k;
            const float *attnMsk = attnMask + i * attnMskStride;
            // max val for avoiding inf and nan
            __m512 vmax = _mm512_set1_ps(maxVal);
            for (int off = 0; off < k; off += 16) {
                int remain = k - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 vx = xft::load_avx512(mask, buf + off);
                __m512 vmask = xft::load_avx512(mask, attnMsk + off);

                vmax = _mm512_mask_max_ps(vmax, mask, vmax, vx * vscale + vmask);
            }
            float _max = _mm512_reduce_max_ps(vmax);

            _max = _max > max[i] ? _max : max[i];
            __m512 merr = _mm512_set1_ps(max[i] - _max);
            merr = BertUtil::vexp(merr);
            max[i] = _max;

            // exp and get sum
            __m512 vsum = _mm512_set1_ps(0);
            vmax = _mm512_set1_ps(_max);
            for (int off = 0; off < k; off += 16) {
                int remain = k - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                __m512 vx = xft::load_avx512(mask, buf + off);
                __m512 vmask = xft::load_avx512(mask, attnMsk + off);
                vx = BertUtil::vexp(vx * vscale + vmask - vmax);

                xft::store_avx512(obuf + off, mask, vx);

                vsum = _mm512_mask_add_ps(vsum, mask, vsum, vx);
            }
            float _sum = _mm512_reduce_add_ps(vsum);
            float fac = _mm512_cvtss_f32(merr);
            sum[i] = sum[i] * fac + _sum;
            _sum = sum[i];

            // Compute exp/sum(exp) and store
            __m512 vrsum = _mm512_set1_ps(1.0f / _sum);
            for (int off = 0; off < k; off += 16) {
                int remain = k - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                __m512 vx = xft::load_avx512(mask, obuf + off);
                vx = vx * vrsum;

                xft::store_avx512(obuf + off, mask, vx);
            }
        }
    }

    template <typename ImT>
    static void alibiSoftmax(ImT *buf, float scale, float headSlope, int elements) {
        float maxVal = std::numeric_limits<float>::lowest();
        __m512 vpos = _mm512_set_ps(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        __m512 vmax = _mm512_set1_ps(maxVal);
        __m512 vscale = _mm512_set1_ps(scale);
        for (int off = 0; off < elements; off += 16) {
            int remain = elements - off;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
            __m512 vx = xft::load_avx512(mask, buf + off);
            // compute avx512 var vmask that is pos * alibiSlopes[hidx]
            __m512 vpositions = _mm512_add_ps(vpos, _mm512_set1_ps(off));
            __m512 vmask = _mm512_mul_ps(vpositions, _mm512_set1_ps(headSlope));
            vmax = _mm512_mask_max_ps(vmax, mask, vmax, vx * vscale + vmask);
        }
        float _max = _mm512_reduce_max_ps(vmax);

        // exp and get sum
        __m512 vsum = _mm512_set1_ps(0);
        vmax = _mm512_set1_ps(_max);
        for (int off = 0; off < elements; off += 16) {
            int remain = elements - off;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = xft::load_avx512(mask, buf + off);
            // compute avx512 var vmask that is pos * alibiSlopes[hidx]
            __m512 vpositions = _mm512_add_ps(vpos, _mm512_set1_ps(off));
            __m512 vmask = _mm512_mul_ps(vpositions, _mm512_set1_ps(headSlope));
            vx = BertUtil::vexp(vx * vscale + vmask - vmax);

            xft::store_avx512(buf + off, mask, vx);

            vsum = _mm512_mask_add_ps(vsum, mask, vsum, vx);
        }
        float _sum = _mm512_reduce_add_ps(vsum);

        // Compute exp/sum(exp) and store
        __m512 vrsum = _mm512_set1_ps(1.0f / _sum);
        for (int off = 0; off < elements; off += 16) {
            int remain = elements - off;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = xft::load_avx512(mask, buf + off);
            vx = vx * vrsum;

            xft::store_avx512(buf + off, mask, vx);
        }
    }

    template <typename ImT>
    static void softmaxTileCausal(float *AB, ImT *ABout, float *sum, float *max, float *preSum, float *preMax,
            float scale, float headSlope, int qLoc, int kLoc, int tRows, int tCols) {
        // build-in mask softmax computing
        float maxVal = std::numeric_limits<float>::lowest();
        __m512 vscale = _mm512_set1_ps(scale);

        __m512 vpos = _mm512_set_ps(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        vpos = _mm512_add_ps(vpos, _mm512_set1_ps(kLoc));

        for (int i = 0; i < tRows; ++i) {
            float *buf = AB + i * tCols;
            ImT *obuf = ABout + i * tCols;
            int k = qLoc + i + 1 - kLoc;
            k = std::max(k, 0);
            k = std::min(k, tCols);
            // max val for avoiding inf and nan
            __m512 vmax = _mm512_set1_ps(maxVal);
            for (int off = 0; off < k; off += 16) {
                int remain = k - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 vx = xft::load_avx512(mask, buf + off);

                if (headSlope != 0) {
                    // compute avx512 var vmask that is pos * alibiSlopes[hidx]
                    __m512 vpositions = _mm512_add_ps(vpos, _mm512_set1_ps(off));
                    __m512 vmask = _mm512_mul_ps(vpositions, _mm512_set1_ps(headSlope));
                    vmax = _mm512_mask_max_ps(vmax, mask, vmax, vx * vscale + vmask);
                } else {
                    vmax = _mm512_mask_max_ps(vmax, mask, vmax, vx * vscale);
                }
            }
            float _max = _mm512_reduce_max_ps(vmax);

            _max = _max > max[i] ? _max : max[i];
            __m512 merr = _mm512_set1_ps(max[i] - _max);
            merr = BertUtil::vexp(merr);
            max[i] = _max;

            // exp and get sum
            __m512 vsum = _mm512_set1_ps(0);
            vmax = _mm512_set1_ps(_max);
            for (int off = 0; off < k; off += 16) {
                int remain = k - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                __m512 vx = xft::load_avx512(mask, buf + off);
                if (headSlope != 0) {
                    // compute avx512 var vmask that is pos * alibiSlopes[hidx]
                    __m512 vpositions = _mm512_add_ps(vpos, _mm512_set1_ps(off));
                    __m512 vmask = _mm512_mul_ps(vpositions, _mm512_set1_ps(headSlope));
                    vx = BertUtil::vexp(vx * vscale + vmask - vmax);
                } else {
                    vx = BertUtil::vexp(vx * vscale - vmax);
                }

                xft::store_avx512(obuf + off, mask, vx);

                vsum = _mm512_mask_add_ps(vsum, mask, vsum, vx);
            }
            float _sum = _mm512_reduce_add_ps(vsum);
            float fac = _mm512_cvtss_f32(merr);
            sum[i] = sum[i] * fac + _sum;
            _sum = sum[i];

            // Compute exp/sum(exp) and store
            __m512 vrsum = _mm512_set1_ps(1.0f / _sum);
            for (int off = 0; off < k; off += 16) {
                int remain = k - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                __m512 vx = xft::load_avx512(mask, obuf + off);
                vx = vx * vrsum;

                xft::store_avx512(obuf + off, mask, vx);
            }
            if (tCols > k) { memset(obuf + k, 0, (tCols - k) * sizeof(ImT)); }
        }
    }

    template <typename T>
    static void updateOutTile(T *output, const float *expABC, float *preSum, float *sum, float *preMax, float *max,
            int m, int n, int stride) {
        for (int i = 0; i < m; ++i) {
            const float *buf = expABC + i * n;
            T *outbuf = output + i * stride;
            __m512 merr = _mm512_set1_ps(preMax[i] - max[i]);
            merr = BertUtil::vexp(merr);
            __m512 vfac = _mm512_set1_ps(preSum[i] / sum[i]);
            for (int off = 0; off < n; off += 16) {
                int remain = n - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 vout = xft::load_avx512(mask, (const T *)outbuf + off);
                __m512 vabc = _mm512_maskz_loadu_ps(mask, buf + off);
                __m512 vupt = vout * merr * vfac + vabc;
                xft::store_avx512(outbuf + off, mask, vupt);
            }
            preSum[i] = sum[i];
            preMax[i] = max[i];
        }
    }

    // hard code: axis = 1
    // sum += sum(exp(A[i]))
    // output = output * preSum / sum + (exp(A) / sum) x B
    // preSum = sum
    template <typename T, typename ImT>
    static void incrementalTileAttention(const T *A, const T *B, const T *C, const float *attnMask, int m, int n, int k,
            int attnMskStride, float *preSum, float *sum, float *preMax, float *max, float scale, float *AB,
            float *expABC, ImT *output, int qStride, int kStride, int vStride, int stride) {
        sgemm(A, B, AB, m, k, n, qStride, kStride, k, false, true);
        // TODO:optimize
        softmaxTile(AB, (T *)AB, sum, max, preSum, preMax, scale, attnMask, m, k, attnMskStride);

        sgemm((T *)AB, C, expABC, m, n, k, k, vStride, n, false, false);
        updateOutTile(output, expABC, preSum, sum, preMax, max, m, n, stride);
    }

    template <typename T, typename ImT>
    static void incrementalTileAttentionCausal(const T *A, const T *B, const T *C, float headSlope, int srcLoc,
            int tgtLoc, int m, int n, int k, float *preSum, float *sum, float *preMax, float *max, float scale,
            float *AB, float *expABC, ImT *output, int qStride, int kStride, int vStride, int stride) {
        sgemm(A, B, AB, m, k, n, qStride, kStride, k, false, true);
        // TODO:optimize
        softmaxTileCausal(AB, (T *)AB, sum, max, preSum, preMax, scale, headSlope, srcLoc, tgtLoc, m, k);

        sgemm((T *)AB, C, expABC, m, n, k, k, vStride, n, false, false);
        updateOutTile(output, expABC, preSum, sum, preMax, max, m, n, stride);
    }
};
