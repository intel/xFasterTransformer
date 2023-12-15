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
#include <string>

#include <mkl.h>
#include "bert_util.h"
#include "bfloat16.h"
#include "compile_util.h"
#include "float16.h"
#include "matmul_helper.h"
#include "my_types.h"
#include "timeline.h"
#include "transformer_ctx.h"
#include "xdnn.h"

class DecoderUtil {
public:
    // Dense without bias
    template <typename WeiT>
    static void dense(hpj::Matrix<float> &x, hpj::Matrix<WeiT> &weight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight, hpj::Vector<float> &sumWeight, hpj::Matrix<float> &result) {
        MMHelper::compute(false, x.Rows(), weight.Cols(), x.Cols(), 1.0f, x.Data(), x.Stride(), weight.Data(),
                scaleWeight.Data(), zeroWeight.Data(), sumWeight.Data(), 0.0f, result.Data(), result.Stride());
    }

    template <typename WeiT>
    static void dense(hpj::Matrix<float> &x, hpj::Matrix<WeiT> &weight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight, hpj::Vector<float> &sumWeight, hpj::Vector<float> &bias,
            hpj::Matrix<float> &result) {
        REQUIRES(x.Cols() == weight.Rows(), "dense error: x.Cols (%d) != weight.Rows (%d)", x.Cols(), weight.Rows());
        REQUIRES(x.Rows() == result.Rows(), "dense error: x.Rows (%d) != result.Rows (%d)", x.Rows(), result.Rows());
        REQUIRES(weight.Cols() == result.Cols(), "dense error: weight.Cols (%d) != result.Cols (%d)", weight.Cols(),
                result.Cols());

        // Bias is empty
        if (bias.Size() == 0) {
            dense(x, weight, scaleWeight, zeroWeight, sumWeight, result);
            return;
        }

        MMHelper::compute_bias(false, x.Rows(), weight.Cols(), x.Cols(), 1.0f, x.Data(), x.Stride(), weight.Data(),
                scaleWeight.Data(), zeroWeight.Data(), sumWeight.Data(), 0.0f, result.Data(), result.Stride(),
                bias.Data());
    }

    // result = x * weight + bias + input
    template <typename WeiT>
    static void denseWithSum(hpj::Matrix<float> &x, hpj::Matrix<WeiT> &weight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight, hpj::Vector<float> &sumWeight, hpj::Vector<float> &bias,
            hpj::Matrix<float> &input, hpj::Matrix<float> &result) {
        REQUIRES(x.Cols() == weight.Rows(), "denseWithSum error: x.Cols (%d) != weight.Rows (%d)", x.Cols(),
                weight.Rows());
        REQUIRES(x.Rows() == result.Rows(), "denseWithSum error: x.Rows (%d) != result.Rows (%d)", x.Rows(),
                result.Rows());
        REQUIRES(weight.Cols() == result.Cols(), "denseWithSum error: weight.Cols (%d) != result.Cols(%d)",
                weight.Cols(), result.Cols());
        REQUIRES(input.Rows() == result.Rows(), "denseWithSum error: input.Rows (%d) != result.Rows (%d)", input.Rows(),
                result.Rows());
        REQUIRES(input.Cols() == result.Cols(), "denseWithSum error: input.Cols (%d) != result.Cols (%d)", input.Cols(),
                result.Cols());

        // Make sure use the correct bias
        float *pbias = bias.Data();
        if (bias.Size() == 0) { pbias = nullptr; }

        MMHelper::compute_residential(false, x.Rows(), weight.Cols(), x.Cols(), 1.0f, x.Data(), x.Stride(),
                weight.Data(), scaleWeight.Data(), zeroWeight.Data(), sumWeight.Data(), 0.0f, result.Data(),
                result.Stride(), pbias, input.Data(), input.Stride());
    }

    // result = x * weight + bias + gamma * input
    // TODO: some path is commented
    template <typename WeiT>
    static void denseWithScaledSum(hpj::Matrix<float> &x, hpj::Matrix<WeiT> &weight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight, hpj::Vector<float> &sumWeight, hpj::Vector<float> &bias, float gamma,
            hpj::Matrix<float> &input, hpj::Matrix<float> &result) {
        REQUIRES(x.Cols() == weight.Rows(), "Error: x.Cols() != weight.Rows()");
        REQUIRES(x.Rows() == result.Rows(), "Error: x.Rows() != result.Rows()");
        REQUIRES(weight.Cols() == result.Cols(), "Error: weight.Cols() != result.Cols()");
        REQUIRES(input.Rows() == result.Rows(), "Error: input.Rows() != result.Rows()");
        REQUIRES(input.Cols() == result.Cols(), "Error: input.Cols() != result.Cols()");

        // Make sure use the correct bias
        float *pbias = bias.Data();
        if (bias.Size() == 0) { pbias = nullptr; }

        MMHelper::compute_resext(false, x.Rows(), weight.Cols(), x.Cols(), 1.0f, x.Data(), x.Stride(), weight.Data(),
                scaleWeight.Data(), zeroWeight.Data(), sumWeight.Data(), 0.0f, result.Data(), result.Stride(), pbias,
                gamma, input.Data(), input.Stride());
    }

#if __AVX512F__
    static void rmsNorm(hpj::Matrix<float> &x, hpj::Matrix<float> &y, hpj::Vector<float> &normWeight, float epsilon) {
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
            hpj::Matrix<float> &x, hpj::Matrix<float> &y, hpj::Vector<float> &gamma, hpj::Vector<float> &beta) {
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
    static void LayerNormOneRow(hpj::Matrix<float> &x, hpj::Matrix<float> &y, float *pgamma, float *pbeta, int size) {
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
    static void layerNorm(DecoderContext *ctx, hpj::Matrix<float> &x, hpj::Matrix<float> &y, hpj::Vector<float> &gamma,
            hpj::Vector<float> &beta) {
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
    static void computeSoftmax(DecoderContext *ctx, float *data, const float *attnMask, int size) {
        int vecs = (size + 15) / 16; // how many avx512 vectors
        __mmask16 tailMask = (size % 16 == 0 ? 0xffff : (1 << (size % 16)) - 1); // mask of last vector

        __m512 vsum = _mm512_set1_ps(0);

        // maxVal is used to avoid exp(x) = inf
        float maxVal = std::numeric_limits<float>::lowest();
        __m512 vmax = _mm512_set1_ps(maxVal);
        __m512 vfactor = _mm512_set1_ps(ctx->attFactor);

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

    // Softmax: skip the calculation when attention mask is the lowest value
    static void softmaxSkipMask(DecoderContext *ctx, float *data, const float *attnMask, int size) {
        int vecs = (size + 15) / 16; // how many avx512 vectors
        __mmask16 tailMask = (size % 16 == 0 ? 0xffff : (1 << (size % 16)) - 1); // mask of last vector

        __m512 vzero = _mm512_set1_ps(0);
        __m512 vsum = _mm512_set1_ps(0);

        // maxVal is used to avoid exp(x) = inf
        float maxVal = std::numeric_limits<float>::lowest();
        __m512 vlowest = _mm512_set1_ps(maxVal);
        __m512 vmax = _mm512_set1_ps(maxVal);
        __m512 vfactor = _mm512_set1_ps(ctx->attFactor);

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

    // input and output are both in qkScores
    // attnMask: attention mask with the shape of (bs, 1, queryLen, keyLen)
    // Note: the source has the shape of (bs, attHeadNum/num_spit, queryLen, keyLen)
    static void computeSoftmax(DecoderContext *ctx, const float *attnMask, int queryLen, int keyLen, int stride = -1) {
        TimeLine t("DecoderUtil::computeSoftmax");
        const int batchStride = queryLen * keyLen;
        if (stride == -1) { stride = keyLen; }

        auto range = SplitUtil::getTaskRange(ctx->attHeadNum, ctx->numSplit, ctx->splitIdx);
        int responsibleHeads = range.second - range.first;

#pragma omp parallel for collapse(2)
        for (int b = 0; b < ctx->batchSize; ++b) {
            for (int i = 0; i < responsibleHeads; ++i) {
                int idx = b * responsibleHeads + i;
                float *result = ctx->qkScores + idx * queryLen * stride;

                for (int seq = 0; seq < queryLen; ++seq) {
                    computeSoftmax(ctx, result, attnMask + b * batchStride + seq * keyLen, keyLen);
                    result += stride;
                }
            }
        }
    }

    // Same implementation with softmax, but:
    // Return max value, and the sum value of exp
    static std::pair<float, float> softmaxWithStats(DecoderContext *ctx, float *data, const float *attnMask, int size) {
        int vecs = (size + 15) / 16; // how many avx512 vectors
        __mmask16 tailMask = (size % 16 == 0 ? 0xffff : (1 << (size % 16)) - 1); // mask of last vector

        __m512 vsum = _mm512_set1_ps(0);

        // maxVal is used to avoid exp(x) = inf
        float maxVal = std::numeric_limits<float>::lowest();
        __m512 vmax = _mm512_set1_ps(maxVal);
        __m512 vfactor = _mm512_set1_ps(ctx->attFactor);

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

    template <typename T, typename Tt>
    static void arrayCpy(T *dst, const Tt *src, int n) {
#pragma omp simd
        for (int i = 0; i < n; i++) {
            dst[i] = static_cast<T>(src[i]);
        }
    }

    template <typename T>
    static void single_thread_cvt2bf16_inplace(T *buf, int m, int n, int stride) {
        if (std::is_same_v<T, float>)
            for (int i = 0; i < m; ++i)
                bfloat16_t::cvt_float_to_bfloat16(buf + i * stride, (bfloat16_t *)buf + i * stride, n);
    }

    // compute silu on the left half and then add it with the right half
    static void siluSum(hpj::Matrix<float> &src) {
        __m512 one = _mm512_set1_ps(1.f);
        __m512 negOne = _mm512_set1_ps(-1.f);
        int M = src.Rows();
        int stride = src.Cols();
        int N = stride / 2;

#pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; j += 16) {
                int remain = N - j;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                auto left = _mm512_maskz_loadu_ps(mask, src.Data() + i * stride + j);
                auto right = _mm512_maskz_loadu_ps(mask, src.Data() + i * stride + j + N);
                auto x0 = BertUtil::vexp(_mm512_mul_ps(left, negOne));
                auto x1 = _mm512_add_ps(one, x0);
                auto x2 = _mm512_div_ps(left, x1);
                auto res = _mm512_mul_ps(right, x2);
                _mm512_mask_storeu_ps(src.Data() + i * stride + j, mask, res);
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

        } else if (std::is_same_v<T, bfloat16_t>) {
            CBLAS_TRANSPOSE ta, tb;
            ta = transa ? CblasTrans : CblasNoTrans;
            tb = transb ? CblasTrans : CblasNoTrans;

            cblas_gemm_bf16bf16f32(CblasRowMajor, ta, tb, m, n, k, alpha, (const MKL_BF16 *)(A), lda,
                    (const MKL_BF16 *)(B), ldb, beta, C, ldc);
        } else {
            printf("Datatype Not supported yet\n");
            exit(-1);
        }
    }

    // need to do for res.
    static void softmaxTile(float *AB, float *sum, float *max, float *preSum, float *preMax, float refac,
            const float *attnMask, int m, int k, int attnMskStride) {
        float maxVal = std::numeric_limits<float>::lowest();
        __m512 vrefac = _mm512_set1_ps(refac);
        for (int i = 0; i < m; ++i) {
            float *buf = AB + i * k;
            const float *attnMsk = attnMask + i * attnMskStride;
            // max val for avoiding inf and nan
            __m512 vmax = _mm512_set1_ps(maxVal);
            for (int off = 0; off < k; off += 16) {
                int remain = k - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 vx = _mm512_maskz_loadu_ps(mask, buf + off);
                __m512 vmask = _mm512_maskz_loadu_ps(mask, attnMsk + off);

                vmax = _mm512_mask_max_ps(vmax, mask, vmax, vx * vrefac + vmask);
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

                __m512 vx = _mm512_maskz_loadu_ps(mask, buf + off);
                __m512 vmask = _mm512_maskz_loadu_ps(mask, attnMsk + off);
                vx = BertUtil::vexp(vx * vrefac + vmask - vmax);

                _mm512_mask_storeu_ps(buf + off, mask, vx);

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

                __m512 vx = _mm512_maskz_loadu_ps(mask, buf + off);
                vx = vx * vrsum;

                _mm512_mask_storeu_ps(buf + off, mask, vx);
            }
        }
    }

    static void updateOutTile(float *output, const float *expABC, float *preSum, float *sum, float *preMax, float *max,
            int m, int n, int stride) {
        for (int i = 0; i < m; ++i) {
            const float *buf = expABC + i * n;
            float *outbuf = output + i * stride;
            __m512 merr = _mm512_set1_ps(preMax[i] - max[i]);
            merr = BertUtil::vexp(merr);
            __m512 vfac = _mm512_set1_ps(preSum[i] / sum[i]);
            for (int off = 0; off < n; off += 16) {
                int remain = n - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 vout = _mm512_maskz_loadu_ps(mask, outbuf + off);
                __m512 vabc = _mm512_maskz_loadu_ps(mask, buf + off);
                __m512 vupt = vout * merr * vfac + vabc;
                _mm512_mask_storeu_ps(outbuf + off, mask, vupt);
            }
            preSum[i] = sum[i];
            preMax[i] = max[i];
        }
    }

    // hard code: axis = 1
    // sum += sum(exp(A[i]))
    // output = output * preSum / sum + (exp(A) / sum) x B
    // preSum = sum
    template <typename T>
    static void incrementalTileAttention(const T *A, const T *B, const T *C, const float *attnMask, int m, int n, int k,
            int attnMskStride, float *preSum, float *sum, float *preMax, float *max, float refac, float *AB,
            float *expABC, float *output, int qStride, int kStride, int vStride, int stride) {
        sgemm(A, B, AB, m, k, n, qStride, kStride, k, false, true);
        softmaxTile(AB, sum, max, preSum, preMax, refac, attnMask, m, k, attnMskStride);

        single_thread_cvt2bf16_inplace(AB, m, k, k);
        sgemm((T *)AB, C, expABC, m, n, k, k, vStride, n, false, false);
        updateOutTile(output, expABC, preSum, sum, preMax, max, m, n, stride);
    }
};
