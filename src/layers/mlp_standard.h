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
template <typename WeiT, bool INPUT_AS_RESID = true>
class MLP {
public:
    MLP(DecoderContext *ctx) {}

    // The inerface is for PyTorch, thus the weights are already transposed
    void setWeights(DecoderContext *ctx, std::vector<float *> &params, bool trans = true) {
        int hiddenSize = ctx->hiddenSize;
        int intermediateSize = ctx->intermediateSize;

        const float *_imWeight = params[0];
        const float *_imBias = params[1];
        const float *_outputWeight = params[2];
        const float *_outputBias = params[3];
        const float *_gamma2 = params[4];
        const float *_beta2 = params[5];

        // Vertically split intermediate(FC1) weight
        hpj::Matrix<WeiT> quantizedIntermediateWeight;
        MMHelper::convertWeight(ctx, trans, hiddenSize, intermediateSize, _imWeight, true, quantizedIntermediateWeight,
                intermediateWeightScale, intermediateWeightZero, intermediateWeightSum);
        MMHelper::packWeight(trans, quantizedIntermediateWeight, intermediateWeight);

        // Intermediate bias
        auto range = SplitUtil::getTaskRange(intermediateSize, ctx->numSplit, ctx->splitIdx);
        int colsPerSplit = range.second - range.first;
        intermediateBias.Resize(colsPerSplit);
        memcpy(intermediateBias.Data(), _imBias + colsPerSplit * ctx->splitIdx, sizeof(float) * colsPerSplit);

        // Horizontally split the output(FC2) weight
        hpj::Matrix<WeiT> quantizedOutputWeight;
        MMHelper::convertWeight(ctx, trans, intermediateSize, hiddenSize, _outputWeight, false, quantizedOutputWeight,
                outputWeightScale, outputWeightZero, outputWeightSum);
        MMHelper::packWeight(trans, quantizedOutputWeight, outputWeight);

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

#ifdef DEBUG
    void setDebugger(const Debugger &debugger) { this->dbg = debugger; }
#endif

    // Forward for FFN (Feed Forward Network)
    void forward(DecoderContext *ctx, float *input, float *output, int iStride, int oStride, bool doLnBefore) {
        TimeLine t("StandardMLP");
        int M = ctx->batchSize * ctx->inputSeqLen;
        hpj::Matrix<float> outBuffer(output, M, ctx->hiddenSize, ctx->hiddenSize);

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

#ifdef DEBUG
        dbg.debugPrint("layer norm after attention:\n");
        dbg.dumpMatrix(imInput);
#endif

        // intermediate
        switch (ctx->actType) {
            case DecoderContext::RELU: intermediate_relu(imInput, imBuffer); break;
            case DecoderContext::GELU: intermediate_gelu(imInput, imBuffer); break;
        }

#ifdef DEBUG
        dbg.debugPrint("intermediate:\n");
        dbg.dumpMatrix(imBuffer);
#endif

        // dense in output
        if (ctx->splitIdx == 0) {
            float gamma = getResidentialScale();

            // denseWithScaledSum is enough, but as the perf of denseWithScaledSum is not verified, so denseWithSum is still here
            if (gamma == 1) {
                DecoderUtil::denseWithSum(imBuffer, outputWeight, outputWeightScale, outputWeightZero, outputWeightSum,
                        outputBias, resultBuffer2, resultBuffer1);
            } else {
                DecoderUtil::denseWithScaledSum(imBuffer, outputWeight, outputWeightScale, outputWeightZero,
                        outputWeightSum, outputBias, gamma, resultBuffer2, resultBuffer1);
            }
        } else {
            DecoderUtil::dense(imBuffer, outputWeight, outputWeightScale, outputWeightZero, outputWeightSum, outputBias,
                    resultBuffer1);
        }

#ifdef DEBUG
        dbg.debugPrint("output:\n");
        dbg.dumpMatrix(resultBuffer1);
#endif

        // layerNorm
        if (!doLnBefore) { DecoderUtil::layerNorm(resultBuffer1, resultBuffer1, gamma2, beta2); }

#ifdef DEBUG
        dbg.debugPrint("final output:\n");
        dbg.dumpMatrix(resultBuffer1);
#endif
    }

protected:
    void intermediate_relu(hpj::Matrix<float> &input, hpj::Matrix<float> &output) {
        MMHelper::compute_biasadd_relu(false, input.Rows(), output.Cols(), input.Cols(), 1.0f, input.Data(),
                input.Stride(), intermediateWeight.Data(), intermediateWeightScale.Data(),
                intermediateWeightZero.Data(), intermediateWeightSum.Data(), 0.0f, output.Data(), output.Stride(),
                intermediateBias.Data());
    }

    void intermediate_gelu(hpj::Matrix<float> &input, hpj::Matrix<float> &output) {
        MMHelper::compute(false, input.Rows(), output.Cols(), input.Cols(), 1.0f, input.Data(), input.Stride(),
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
            __m512 c1 = _mm512_set1_ps(0.044715f);
            __m512 c2 = _mm512_set1_ps(factor);
            __m512 vone = _mm512_set1_ps(1);
            __m512 vtwo = _mm512_set1_ps(2);
            __m512 vhalf = _mm512_set1_ps(0.5f);

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
    hpj::Matrix<WeiT> intermediateWeight;
    hpj::Vector<float> intermediateWeightScale;
    hpj::Vector<float> intermediateWeightZero;
    hpj::Vector<float> intermediateWeightSum;
    hpj::Vector<float> intermediateBias;

    hpj::Matrix<WeiT> outputWeight;
    hpj::Vector<float> outputWeightScale;
    hpj::Vector<float> outputWeightZero;
    hpj::Vector<float> outputWeightSum;
    hpj::Vector<float> outputBias;

    // layerNorm param
    hpj::Vector<float> gamma2, beta2;

#ifdef DEBUG
    Debugger dbg;
#endif
};
