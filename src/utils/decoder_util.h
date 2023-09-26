#pragma once
#include <immintrin.h>
#include <string>

#include "bert_util.h"
#include "bfloat16.h"
#include "compile_util.h"
#include "float16.h"
#include "matmul_helper.h"
#include "my_types.h"
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
        REQUIRES(x.Cols() == weight.Rows(), "Error: x.Cols() != weight.Rows()");
        REQUIRES(x.Rows() == result.Rows(), "Error: x.Rows() != result.Rows()");
        REQUIRES(weight.Cols() == result.Cols(), "Error: weight.Cols() != result.Cols()");

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
        REQUIRES(x.Cols() == weight.Rows(), "Error: x.Cols() != weight.Rows()");
        REQUIRES(x.Rows() == result.Rows(), "Error: x.Rows() != result.Rows()");
        REQUIRES(weight.Cols() == result.Cols(), "Error: weight.Cols() != result.Cols()");
        REQUIRES(input.Rows() == result.Rows(), "Error: input.Rows() != result.Rows()");
        REQUIRES(input.Cols() == result.Cols(), "Error: input.Cols() != result.Cols()");

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
        const int splitSize = 128; // size of each split
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
        __m512 vsum = _mm512_set1_ps(0);

        // max_val is used to avoid exp(x) = inf
        float max_val = std::numeric_limits<float>::lowest();
        __m512 vmax = _mm512_set1_ps(max_val);

        for (int off = 0; off < size; off += 16) {
            int remain = size - off;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = _mm512_maskz_loadu_ps(mask, data + off);
            __m512 vmask = _mm512_maskz_loadu_ps(mask, attnMask + off);
            vmax = _mm512_mask_max_ps(vmax, mask, vmax, vx + vmask);
        }

        max_val = _mm512_reduce_max_ps(vmax);
        vmax = _mm512_set1_ps(max_val * ctx->attFactor);
        __m512 vfactor = _mm512_set1_ps(ctx->attFactor);

        // Compute vexp(vx - vmax) and sum it
        for (int off = 0; off < size; off += 16) {
            int remain = size - off;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = _mm512_maskz_loadu_ps(mask, data + off);
            __m512 vmask = _mm512_maskz_loadu_ps(mask, attnMask + off);
            vx = BertUtil::vexp(vx * vfactor + vmask - vmax);

            _mm512_mask_storeu_ps(data + off, mask, vx);

            vsum = _mm512_mask_add_ps(vsum, mask, vsum, vx);
        }

        float sum = _mm512_reduce_add_ps(vsum);
        __m512 vrsum = _mm512_set1_ps(1.0f / sum);

        // Compute exp/sum(exp) and store
        for (int off = 0; off < size; off += 16) {
            int remain = size - off;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = _mm512_maskz_loadu_ps(mask, data + off);
            vx = vx * vrsum;

            _mm512_mask_storeu_ps(data + off, mask, vx);
        }
    }

    // input and output are both in qkScores
    // attnMask: attention mask with the shape of (bs, 1, queryLen, keyLen)
    // Note: the source has the shape of (bs, attHeadNum/num_spit, queryLen, keyLen)
    static void computeSoftmax(DecoderContext *ctx, const float *attnMask, int queryLen, int keyLen, int stride = -1) {
        const int batchStride = queryLen * keyLen;
        if (stride == -1) { stride = keyLen; }

#pragma omp parallel for collapse(2)
        for (int b = 0; b < ctx->batchSize; ++b) {
            for (int i = 0; i < ctx->attHeadNum / ctx->numSplit; ++i) {
                int idx = b * ctx->attHeadNum / ctx->numSplit + i;
                float *result = ctx->qkScores + idx * queryLen * stride;

                for (int seq = 0; seq < queryLen; ++seq) {
                    computeSoftmax(ctx, result, attnMask + b * batchStride + seq * keyLen, keyLen);
                    result += stride;
                }
            }
        }
    }
};
