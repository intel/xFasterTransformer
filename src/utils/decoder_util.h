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

#include "bert_util.h"
#include "bfloat16.h"
#include "compile_util.h"
#include "float16.h"
#include "matmul_helper.h"
#include "my_types.h"
#include "timeline.h"
#include "transformer_ctx.h"

class DecoderUtil {
public:
    // Dense without bias
    template <typename WeiT>
    static void dense(hpj::Matrix<float> &x, hpj::Matrix<WeiT> &weight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight, hpj::Matrix<float> &result) {
        MMHelper::compute(false, x.Rows(), weight.Cols(), x.Cols(), 1.0f, x.Data(), x.Stride(), weight.Data(),
                scaleWeight.Data(), zeroWeight.Data(), 0.0f, result.Data(), result.Stride());
    }

    template <typename WeiT>
    static void dense(hpj::Matrix<float> &x, hpj::Matrix<WeiT> &weight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight, hpj::Vector<float> &bias, hpj::Matrix<float> &result) {
        REQUIRES(x.Cols() == weight.Rows(), "dense error: x.Cols (%d) != weight.Rows (%d)", x.Cols(), weight.Rows());
        REQUIRES(x.Rows() == result.Rows(), "dense error: x.Rows (%d) != result.Rows (%d)", x.Rows(), result.Rows());
        REQUIRES(weight.Cols() == result.Cols(), "dense error: weight.Cols (%d) != result.Cols (%d)", weight.Cols(),
                result.Cols());

        // Bias is empty
        if (bias.Size() == 0) {
            dense(x, weight, scaleWeight, zeroWeight, result);
            return;
        }

        MMHelper::compute_bias(false, x.Rows(), weight.Cols(), x.Cols(), 1.0f, x.Data(), x.Stride(), weight.Data(),
                scaleWeight.Data(), zeroWeight.Data(), 0.0f, result.Data(), result.Stride(), bias.Data());
    }

    // result = x * weight + bias + input
    template <typename WeiT>
    static void denseWithSum(hpj::Matrix<float> &x, hpj::Matrix<WeiT> &weight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight, hpj::Vector<float> &bias, hpj::Matrix<float> &input,
            hpj::Matrix<float> &result) {
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
                weight.Data(), scaleWeight.Data(), zeroWeight.Data(), 0.0f, result.Data(), result.Stride(), pbias,
                input.Data(), input.Stride());
    }

    // result = x * weight + bias + gamma * input
    // TODO: some path is commented
    template <typename WeiT>
    static void denseWithScaledSum(hpj::Matrix<float> &x, hpj::Matrix<WeiT> &weight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight, hpj::Vector<float> &bias, float gamma, hpj::Matrix<float> &input,
            hpj::Matrix<float> &result) {
        REQUIRES(x.Cols() == weight.Rows(), "Error: x.Cols() != weight.Rows()");
        REQUIRES(x.Rows() == result.Rows(), "Error: x.Rows() != result.Rows()");
        REQUIRES(weight.Cols() == result.Cols(), "Error: weight.Cols() != result.Cols()");
        REQUIRES(input.Rows() == result.Rows(), "Error: input.Rows() != result.Rows()");
        REQUIRES(input.Cols() == result.Cols(), "Error: input.Cols() != result.Cols()");

        // Make sure use the correct bias
        float *pbias = bias.Data();
        if (bias.Size() == 0) { pbias = nullptr; }

        MMHelper::compute_resext(false, x.Rows(), weight.Cols(), x.Cols(), 1.0f, x.Data(), x.Stride(), weight.Data(),
                scaleWeight.Data(), zeroWeight.Data(), 0.0f, result.Data(), result.Stride(), pbias, gamma, input.Data(),
                input.Stride());
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

    template <typename T, typename Tt>
    static void arrayCpy(T* dst, const Tt* src, int n) {
#pragma omp simd
        for (int i = 0; i < n; i++) {
            dst[i] = static_cast<T>(src[i]);
        }
    }
    
    // batchs x seqlen x 3 x head x heads ->  3 x batchs x head x seqlen x heads (2
    // 0 3 1 4)
    template <typename T, typename Tt>
    static void transposeQKV(const T *qkvBuffer, Tt *qkvTransBuffer, int batchSize, int seqLen, int headQNum,
            int headKVNum, int headSize) {
        int hiddenQSize = headQNum * headSize;
        int hiddenKVSize = headKVNum * headSize;
        int hiddenQKVSize = hiddenQSize + hiddenKVSize * 2;

        int blockSize = hiddenQKVSize * seqLen;
    
        const T *qBuffer = qkvBuffer;
        const T *kBuffer = qkvBuffer + hiddenQSize;
        const T *vBuffer = qkvBuffer + hiddenQSize + hiddenKVSize;
    
        Tt *qTransBuffer = qkvTransBuffer;
        Tt *kTransBuffer = qkvTransBuffer + batchSize * hiddenQSize * seqLen;
        Tt *vTransBuffer = qkvTransBuffer + batchSize * (hiddenQSize + hiddenKVSize) * seqLen;
    
#pragma omp parallel for collapse(3)
        for (int i = 0; i < batchSize; i++) {
            for (int k = 0; k < headQNum; k++) { // assume headQNum >= headKVNum
                for (int j = 0; j < seqLen; j++) {
                    const float *qSrcEachBatch =
                        reinterpret_cast<const T*>(qBuffer) + blockSize * i;
                    const float *kSrcEachBatch =
                        reinterpret_cast<const T*>(kBuffer) + blockSize * i;
                    const float *vSrcEachBatch =
                        reinterpret_cast<const T*>(vBuffer) + blockSize * i;
    
                    int dstOffEachHead = k * seqLen * headSize;
                    int srcOffEachLine = k * headSize;
    
                    int dstOffEachLine = j * headSize;
                    int srcOffEachHead = j * hiddenQKVSize;
    
                    Tt *qDstEachLine = qTransBuffer + i * hiddenQSize * seqLen +
                                          dstOffEachHead + dstOffEachLine;
                    const T* qSrcEachLine =
                        qSrcEachBatch + srcOffEachHead + srcOffEachLine;
    
                    Tt *kDstEachLine = kTransBuffer + i * hiddenKVSize * seqLen +
                                          dstOffEachHead + dstOffEachLine;
                    const T *kSrcEachLine =
                        kSrcEachBatch + srcOffEachHead + srcOffEachLine;
    
                    Tt *vDstEachLine = vTransBuffer + i * hiddenKVSize * seqLen +
                                          dstOffEachHead + dstOffEachLine;
                    const T *vSrcEachLine =
                        vSrcEachBatch + srcOffEachHead + srcOffEachLine;
                    arrayCpy<Tt, T>(qDstEachLine, qSrcEachLine, headSize);
                    if (k < headKVNum) {
                        arrayCpy<Tt, T>(kDstEachLine, kSrcEachLine, headSize);
                        arrayCpy<Tt, T>(vDstEachLine, vSrcEachLine, headSize);
                    }
                }
            }
        }
    }
    
    // batchs x head x seqlen x heads -> batchs x seqlen x head x heads (0 2 1 3)
    template <typename T, typename Tt>
    static void transposeAttnResult(T *Buffer, Tt *TransBuffer, int batchSize, int seqLen, int headNum,
            int headSize, int dstStride) {
        int hiddenSize = headNum * headSize;
        int blockSize = seqLen * hiddenSize;  // dst buffer stride in each batch
    
#pragma omp parallel for collapse(2)
        for (int i = 0; i < batchSize; i++) {
            for (int k = 0; k < seqLen; k++) {
                int srcOffEachHead = k * headSize;
                int dstOffEachLine = k * dstStride;
    
                for (int j = 0; j < headNum; j++) {
                    int srcOffEachLine = j * seqLen * headSize;
                    int dstOffEachHead = j * headSize;
    
                    Tt *qDstEachLine = TransBuffer + dstOffEachHead +
                                  dstOffEachLine + i * seqLen * dstStride;
                    const T *qSrcEachLine = Buffer + srcOffEachLine +
                                       srcOffEachHead + i * blockSize;
    
                    arrayCpy<Tt, T>(qDstEachLine, qSrcEachLine, headSize);
                }
            }
        }
    }
    
    // C = A * B
    // bTranspose: B need to be transposed or not
    // ig_sgemm_single_thread(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    static void sgemm(const float* A, const float* B, float* C, int m, int n, int k,
            bool transa, bool transb) {
        int lda = (transa ? m : k);
        int ldb = (transb ? k : n);
        int ldc = n;
        float alpha = 1;
        float beta = 0;
        char ta[] = "N";
        char tb[] = "N";
        if (transa) ta[0] = 'T';
        if (transb) tb[0] = 'T';
    
        dnnl_sgemm(ta[0], tb[0], m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    // need to do for res.
    static void softmaxTile(float *AB, float *sum, float *max, float *preSum, float *preMax, float refac,
            const float *attnMask, int m, int k, int attnMskStride) {
        float maxVal = std::numeric_limits<float>::lowest();
        __m512 vrefac = _mm512_set1_ps(refac);
        for (int i = 0; i < m; ++i) {
            float* buf = AB + i * k;
            const float* attnMsk = attnMask + i * attnMskStride;
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
    
    static void updateOutTile(float* output, const float* expABC, float* preSum, float* sum, float* preMax,
            float* max, int m, int n) {
        for (int i = 0; i < m; ++i) {
            const float* buf = expABC + i * n;
            float* outbuf = output + i * n;
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
    static void incrementalTileAttention(const float* A, const float* B, const float* C, const float* attnMask,
            int m, int n, int k, int attnMskStride, float* preSum, float* sum, float* preMax, float* max,
            float refac, float* AB, float* expABC, float* output) {
        sgemm(A, B, AB, m, k, n, false, true);
        softmaxTile(AB, sum, max, preSum, preMax, refac, attnMask, m, k, attnMskStride);
        sgemm(AB, C, expABC, m, n, k, false, false);
        updateOutTile(output, expABC, preSum, sum, preMax, max, m, n);
    }
    
};
