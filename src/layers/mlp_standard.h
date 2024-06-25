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
#include "bert_util.h"
#include "debugger.h"
#include "decoder_util.h"
#include "matmul_helper.h"
#include "split_util.h"

// WeiT: weight data type
// INPUT_AS_RESID: input as residential or not, most models use input as residential,
//                 but there are exceptions like ChatGLM use values after layernorm as residential
template <typename WeiT, typename InT = float, typename ImT = float, typename OutT = float, bool INPUT_AS_RESID = true>
class MLP {
public:
    MLP(DecoderContext *ctx) {}

    // OriWeiT: float
    template <typename OriWeiT>
    void setWeights(DecoderContext *ctx, const OriWeiT *_imWeight, const float * /*unused*/, const float * /*unused*/,
            const float *_imBias, const OriWeiT *_outputWeight, const float * /*unused*/, const float * /*unused*/,
            const float *_outputBias, const float *_gamma2, const float *_beta2, const OriWeiT * /*unused*/,
            const float * /*unused*/, const float * /*unused*/, bool trans = true) {
        int hiddenSize = ctx->hiddenSize;
        int intermediateSize = ctx->intermediateSize;

        // Vertically split intermediate(FC1) weight
        xft::Matrix<WeiT> quantizedIntermediateWeight;
        ctx->mmHelper->convertWeight(ctx, trans, hiddenSize, intermediateSize, _imWeight, nullptr, nullptr, true,
                quantizedIntermediateWeight, intermediateWeightScale, intermediateWeightZero, intermediateWeightSum);
        ctx->mmHelper->packWeight(trans, quantizedIntermediateWeight, intermediateWeight);

        // Intermediate bias
        auto range = SplitUtil::getTaskRange(intermediateSize, ctx->numSplit, ctx->splitIdx);
        int colsPerSplit = range.second - range.first;
        intermediateBias.Resize(colsPerSplit);
        memcpy(intermediateBias.Data(), _imBias + colsPerSplit * ctx->splitIdx, sizeof(float) * colsPerSplit);

        // Horizontally split the output(FC2) weight
        xft::Matrix<WeiT> quantizedOutputWeight;
        ctx->mmHelper->convertWeight(ctx, trans, intermediateSize, hiddenSize, _outputWeight, nullptr, nullptr, false,
                quantizedOutputWeight, outputWeightScale, outputWeightZero, outputWeightSum);
        ctx->mmHelper->packWeight(trans, quantizedOutputWeight, outputWeight);

        // Output bias
        outputBias.Resize(hiddenSize);
        if (ctx->splitIdx == 0) {
            memcpy(outputBias.Data(), _outputBias, sizeof(float) * hiddenSize);
        } else { // For other splits, set bias to 0, to avoid duplicated calculation
            memset(outputBias.Data(), 0, sizeof(float) * hiddenSize);
        }

        // gamma and beta for layer norm
        if (_gamma2 && _beta2) {
            gamma2.Resize(hiddenSize);
            beta2.Resize(hiddenSize);
            memcpy(gamma2.Data(), _gamma2, sizeof(float) * hiddenSize);
            memcpy(beta2.Data(), _beta2, sizeof(float) * hiddenSize);
        }
    }

#ifdef XFT_DEBUG
    void setDebugger(const Debugger &debugger) { this->dbg = debugger; }
#endif

    // Forward for FFN (Feed Forward Network)
    void forward(DecoderContext *ctx, float *input, float *output, int iStride, int oStride, bool doLnBefore,
            int totInSeqLen = 0) {
        TimeLine t("StandardMLP");
        int M = totInSeqLen == 0 ? ctx->batchSize * ctx->inputSeqLen : totInSeqLen;
        xft::Matrix<float> outBuffer(output, M, ctx->hiddenSize, ctx->hiddenSize);

        auto &resultBuffer1 = outBuffer;
        auto &resultBuffer2 = ctx->tmpBuf;
        auto &imBuffer = ctx->imOut;

        // When doLnBefore=true, conduct layernorm in the beginning
        // Note: input is resultBuffer2 as of some history reason
        if (doLnBefore) {
            // Need to keep input unmodified
            if constexpr (INPUT_AS_RESID) {
                DecoderUtil::layerNorm(resultBuffer2, resultBuffer1, gamma2, beta2);
            } else {
                DecoderUtil::layerNorm(resultBuffer2, resultBuffer2, gamma2, beta2);
            }
        }

        auto &imInput = doLnBefore ? (INPUT_AS_RESID ? resultBuffer1 : resultBuffer2) : resultBuffer2;

#ifdef XFT_DEBUG
        dbg.debugPrint("layer norm after attention:\n");
        dbg.dumpMatrix(imInput, false, ctx->device);
#endif

        // intermediate
        switch (ctx->actType) {
            case DecoderContext::RELU: intermediate_relu(ctx, imInput, imBuffer); break;
            case DecoderContext::GELU: intermediate_gelu(ctx, imInput, imBuffer); break;
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("intermediate:\n");
        dbg.dumpMatrix(imBuffer, false, ctx->device);
#endif

        // dense in output
        if (ctx->splitIdx == 0) {
            float gamma = getResidentialScale();

            // denseWithScaledSum is enough, but as the perf of denseWithScaledSum is not verified, so denseWithSum is still here
            if (gamma == 1) {
                float *pbias = outputBias.Data();
                if (outputBias.Size() == 0) { pbias = nullptr; }
                ctx->mmHelper->compute_residential(false, imBuffer.Rows(), outputWeight.Cols(), imBuffer.Cols(), 1.0f,
                        imBuffer.Data(), imBuffer.Stride(), outputWeight.Data(), outputWeightScale.Data(),
                        outputWeightZero.Data(), outputWeightSum.Data(), 0.0f, resultBuffer1.Data(),
                        resultBuffer1.Stride(), pbias, resultBuffer2.Data(), resultBuffer2.Stride());
            } else {
                float *pbias = outputBias.Data();
                if (outputBias.Size() == 0) { pbias = nullptr; }
                ctx->mmHelper->compute_resext(false, imBuffer.Rows(), outputWeight.Cols(), imBuffer.Cols(), 1.0f,
                        imBuffer.Data(), imBuffer.Stride(), outputWeight.Data(), outputWeightScale.Data(),
                        outputWeightZero.Data(), outputWeightSum.Data(), 0.0f, resultBuffer1.Data(),
                        resultBuffer1.Stride(), pbias, gamma, resultBuffer2.Data(), resultBuffer2.Stride());
            }
        } else {
            if (outputBias.Size() == 0) {
                ctx->mmHelper->compute(false, imBuffer.Rows(), outputWeight.Cols(), imBuffer.Cols(), 1.0f,
                        imBuffer.Data(), imBuffer.Stride(), outputWeight.Data(), outputWeightScale.Data(),
                        outputWeightZero.Data(), outputWeightSum.Data(), 0.0f, resultBuffer1.Data(),
                        resultBuffer1.Stride());
            } else {
                ctx->mmHelper->compute_bias(false, imBuffer.Rows(), outputWeight.Cols(), imBuffer.Cols(), 1.0f,
                        imBuffer.Data(), imBuffer.Stride(), outputWeight.Data(), outputWeightScale.Data(),
                        outputWeightZero.Data(), outputWeightSum.Data(), 0.0f, resultBuffer1.Data(),
                        resultBuffer1.Stride(), outputBias.Data());
            }
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("output:\n");
        dbg.dumpMatrix(resultBuffer1, false, ctx->device);
#endif

        // layerNorm
        if (!doLnBefore) { DecoderUtil::layerNorm(resultBuffer1, resultBuffer1, gamma2, beta2); }

#ifdef XFT_DEBUG
        dbg.debugPrint("final output:\n");
        dbg.dumpMatrix(resultBuffer1, false, ctx->device);
#endif
    }

protected:
    void intermediate_relu(DecoderContext *ctx, xft::Matrix<float> &input, xft::Matrix<float> &output) {
        ctx->mmHelper->compute_biasadd_relu(false, input.Rows(), output.Cols(), input.Cols(), 1.0f, input.Data(),
                input.Stride(), intermediateWeight.Data(), intermediateWeightScale.Data(),
                intermediateWeightZero.Data(), intermediateWeightSum.Data(), 0.0f, output.Data(), output.Stride(),
                intermediateBias.Data());
    }

    void intermediate_gelu(DecoderContext *ctx, xft::Matrix<float> &input, xft::Matrix<float> &output) {
        ctx->mmHelper->compute(false, input.Rows(), output.Cols(), input.Cols(), 1.0f, input.Data(), input.Stride(),
                intermediateWeight.Data(), intermediateWeightScale.Data(), intermediateWeightZero.Data(),
                intermediateWeightSum.Data(), 0.0f, output.Data(), output.Stride());

        float *pbias = intermediateBias.Data();
        float factor = 0.7978845608; // np.sqrt(2 / np.pi)

#pragma omp parallel for
        for (int i = 0; i < output.Rows(); ++i) {
            // int tid = omp_get_thread_num();
            // float *pout = output.Row(i);
            // #pragma omp simd
            // for (int j = 0; j < output.Cols(); ++j) {
            //     float x = pout[j] + pbias[j];
            //     ctx->erf_buffer[tid][j] = x;
            //     pout[j] = factor * (x + 0.044715f * x * x * x);
            // }
            // vsTanh(output.Cols(), pout, pout);
            // #pragma omp simd
            // for (int j = 0; j < output.Cols(); ++j) {
            //     pout[j] = ctx->erf_buffer[tid][j] * 0.5f * (1 + pout[j]);
            // }
            float *pout = output.Row(i);
            const __m512 c1 = _mm512_set1_ps(0.044715f);
            const __m512 c2 = _mm512_set1_ps(factor);
            const __m512 vone = _mm512_set1_ps(1);
            const __m512 vtwo = _mm512_set1_ps(2);
            const __m512 vhalf = _mm512_set1_ps(0.5f);

            for (int off = 0; off < output.Cols(); off += 16) {
                int remain = output.Cols() - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                __m512 vx = _mm512_maskz_loadu_ps(mask, pout + off);
                vx = vx + _mm512_maskz_loadu_ps(mask, pbias + off);

                __m512 vt = c2 * (vx + c1 * vx * vx * vx);
                vt = BertUtil::vexp(vt * vtwo);
                vt = vone - vtwo * _mm512_rcp14_ps(vt + vone); // tanh
                __m512 vy = vx * (vone + vt) * vhalf;

                _mm512_mask_storeu_ps(pout + off, mask, vy);
            }
        }
    }

    //    protected:
    virtual float getResidentialScale() {
        return 1; // directly add the residential
    }

    //    private:
    xft::Matrix<WeiT> intermediateWeight;
    xft::Vector<float> intermediateWeightScale;
    xft::Vector<float> intermediateWeightZero;
    xft::Vector<float> intermediateWeightSum;
    xft::Vector<float> intermediateBias;

    xft::Matrix<WeiT> outputWeight;
    xft::Vector<float> outputWeightScale;
    xft::Vector<float> outputWeightZero;
    xft::Vector<float> outputWeightSum;
    xft::Vector<float> outputBias;

    // layerNorm param
    xft::Vector<float> gamma2, beta2;

#ifdef XFT_DEBUG
    Debugger dbg;
#endif
};
