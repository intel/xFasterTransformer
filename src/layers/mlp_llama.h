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
#include "singleton.h"
#include "timeline.h"

extern bool enableCATMLP;
extern bool enableCBLASMLP;
void setMLPOPTConfig();
// C++ implementation for the python code in modeling_llama.py:
// residual = hidden_states
// hidden_states = self.post_attention_layernorm(hidden_states)
// hidden_states = self.mlp(hidden_states)
// hidden_states = residual + hidden_states
//
// While LlamaMLP is like:
// def forward(self, x):
//         return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
// But please also be noted: we extended the MLP to include layer norm
template <typename WeiT>
class LlamaMLP : public SingletonBase<LlamaMLP<WeiT>> {
public:
    LlamaMLP() {}

    LlamaMLP(DecoderContext *ctx) {}

    // The inerface is for PyTorch, thus the weights are already transposed
    void setWeights(DecoderContext *ctx, std::vector<float *> &params, bool trans = true) {
        int hiddenSize = ctx->hiddenSize;
        int imSize = ctx->intermediateSize;

        // Refer to CommonDecoder for parameters order
        const float *gateW = params[0];
        const float *upW = params[2];
        const float *normW = params[4];
        const float *downW = params[6];

        REQUIRES(ctx->actType == DecoderContext::SILU, "unsupported activation.");

        // Vertically split the gate weight and up weight
        hpj::Matrix<WeiT> quantizedGateWeight, quantizedUpWeight, quantizedDownWeight;

        auto it = SplitUtil::getTaskRange(imSize, ctx->numSplit, ctx->splitIdx);
        downWeight.Resize(it.second - it.first, hiddenSize);

        MMHelper::convertWeight(ctx, trans, hiddenSize, imSize, gateW, true, quantizedGateWeight, gateWeightScale,
                gateWeightZero, gateWeightSum);
        MMHelper::convertWeight(
                ctx, trans, hiddenSize, imSize, upW, true, quantizedUpWeight, upWeightScale, upWeightZero, upWeightSum);

        setMLPOPTConfig();
        if (!enableCATMLP) {
            gateWeight.Resize(hiddenSize, it.second - it.first);
            upWeight.Resize(hiddenSize, it.second - it.first);
            MMHelper::packWeight(trans, quantizedGateWeight, gateWeight);
            MMHelper::packWeight(trans, quantizedUpWeight, upWeight);
        } else {
            hpj::Matrix<WeiT> quantizedCatWeights;
            catGateUpWeights(quantizedGateWeight, quantizedUpWeight, gateWeightScale, gateWeightZero, gateWeightSum,
                    upWeightScale, upWeightZero, upWeightSum, quantizedCatWeights, catWeightsScale, catWeightsZero,
                    catWeightsSum);
            quantizedGateWeight.Release();
            quantizedUpWeight.Release();
            catWeights.Resize(quantizedCatWeights.Rows(), quantizedCatWeights.Cols());
            MMHelper::packWeight(trans, quantizedCatWeights, catWeights);
        }
        // Horizontally split the down weight
        if (enableCBLASMLP && std::is_same_v<WeiT, bfloat16_t>) {
            MMHelper::convertWeight(ctx, trans, imSize, hiddenSize, downW, false, downWeight, downWeightScale,
                    downWeightZero, downWeightSum);
        } else {
            MMHelper::convertWeight(ctx, trans, imSize, hiddenSize, downW, false, quantizedDownWeight, downWeightScale,
                    downWeightZero, downWeightSum);
            MMHelper::packWeight(trans, quantizedDownWeight, downWeight);
        }

#ifdef DEBUG
        dbg.debugPrint("quantizedGateWeight:\n");
        dbg.dumpMatrix(quantizedGateWeight);

        dbg.debugPrint("quantizedUpWeight:\n");
        dbg.dumpMatrix(quantizedUpWeight);

        dbg.debugPrint("quantizedDownWeight:\n");
        dbg.dumpMatrix(quantizedDownWeight);
#endif

        // LlamaRMSNorm
        if (normW) {
            normWeight.Resize(hiddenSize);
            memcpy(normWeight.Data(), normW, sizeof(float) * hiddenSize);
        }
    }

#ifdef DEBUG
    void setDebugger(const Debugger &debugger) { this->dbg = debugger; }
#endif

    // Forward for FFN (Feed Forward Network)
    void forward(DecoderContext *ctx, float *input, float *output, int iStride, int oStride,
            bool doLnBefore = true /*not used*/) {
        TimeLine t("LlamaMLP");
        const int M = ctx->batchSize * ctx->inputSeqLen;
        const int hiddenSize = ctx->hiddenSize;

        hpj::Matrix<float> inBuffer(input, M, hiddenSize, iStride);
        hpj::Matrix<float> outBuffer(output, M, hiddenSize, oStride);
        auto &normBuffer = ctx->normBuf;

        if (doLnBefore == true) { DecoderUtil::rmsNorm(inBuffer, normBuffer, normWeight, 1e-6); }

#ifdef DEBUG
        dbg.debugPrint("LayerNorm before MLP:\n");
        dbg.dumpMatrix(normBuffer);
#endif

        if (!enableCATMLP) {
            auto &imBuffer = ctx->imOut;
            gateProj(doLnBefore ? normBuffer : inBuffer, imBuffer);

#ifdef DEBUG
            dbg.debugPrint("gateWeight:\n");
            dbg.dumpMatrix(gateWeight);
            dbg.debugPrint("gate output:\n");
            dbg.dumpMatrix(imBuffer);
#endif

            upProj(doLnBefore ? normBuffer : inBuffer, imBuffer);

#ifdef DEBUG
            dbg.debugPrint("upWeight:\n");
            dbg.dumpMatrix(upWeight);
            dbg.debugPrint("up output:\n");
            dbg.dumpMatrix(imBuffer);
#endif
            downProj(imBuffer, outBuffer, inBuffer, ctx->splitIdx == 0);

        } else {
            hpj::Matrix<float> imBuffer(ctx->imOut.Data(), normBuffer.Rows(), catWeights.Cols(), catWeights.Cols());
            catGateUpProj(doLnBefore ? normBuffer : inBuffer, imBuffer);

#ifdef DEBUG
            dbg.debugPrint("catWeights:\n");
            dbg.dumpMatrix(catWeights);
            dbg.debugPrint("gateUp output:\n");
            dbg.dumpMatrix(imBuffer);
#endif
            downProj(imBuffer, outBuffer, inBuffer, ctx->splitIdx == 0);
        }

#ifdef DEBUG
        dbg.debugPrint("downWeight:\n");
        dbg.dumpMatrix(downWeight);
        dbg.debugPrint("residential:\n");
        dbg.dumpMatrix(inBuffer);
        dbg.debugPrint("final output:\n");
        dbg.dumpMatrix(outBuffer);
#endif
    }

private:
    void gateProj(hpj::Matrix<float> &input, hpj::Matrix<float> &output) {
        TimeLine t("GateProj");

        assert(input.Rows() == output.Rows());
        assert(input.Cols() == gateWeight.Rows());
        assert(gateWeight.Cols() == output.Cols());

        int M = input.Rows(), N = output.Cols(), K = input.Cols();
        int lda = input.Stride(), ldc = output.Stride();

        const float *A = input.Data();
        const WeiT *B = gateWeight.Data();
        const float *scaleB = gateWeightScale.Data();
        const float *zeroB = gateWeightZero.Data();
        const float *sumB = gateWeightSum.Data();
        float *C = output.Data();

        MMHelper::compute_silu(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc);
    }

    void upProj(hpj::Matrix<float> &input, hpj::Matrix<float> &output) {
        TimeLine t("UpProj");

        assert(input.Rows() == output.Rows());
        assert(input.Cols() == upWeight.Rows());
        assert(upWeight.Cols() == output.Cols());

        int M = input.Rows(), N = output.Cols(), K = input.Cols();
        int lda = input.Stride(), ldc = output.Stride();

        const float *A = input.Data();
        const WeiT *B = upWeight.Data();
        const float *scaleB = upWeightScale.Data();
        const float *zeroB = upWeightZero.Data();
        const float *sumB = upWeightSum.Data();
        float *C = output.Data();

        MMHelper::compute_resmul(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc, C, ldc);
    }

    void downProj(
            hpj::Matrix<float> &input, hpj::Matrix<float> &output, hpj::Matrix<float> &residential, bool isMaster) {
        TimeLine t("DownProj");

        assert(input.Rows() == output.Rows());
        assert(input.Cols() == downWeight.Rows());
        assert(downWeight.Cols() == output.Cols());

        int M = input.Rows(), N = output.Cols(), K = downWeight.Rows();
        int lda = input.Stride(), ldc = output.Stride(), ldr = residential.Stride();

        const float *A = input.Data();
        const WeiT *B = downWeight.Data();
        const float *scaleB = downWeightScale.Data();
        const float *zeroB = downWeightZero.Data();
        const float *sumB = downWeightSum.Data();
        float *C = output.Data();
        const float *R = residential.Data();

        if (isMaster) {
            if (enableCBLASMLP && std::is_same_v<WeiT, bfloat16_t>) {
                compute_proj_bf16(A, B, C, M, N, K, lda, ldc, ldc, R, ldr);
            } else {
                MMHelper::compute_residential(
                        false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc, NULL, R, ldr);
            }
        } else {
            if (enableCBLASMLP && std::is_same_v<WeiT, bfloat16_t>) {
                compute_proj_bf16(A, B, C, M, N, K, lda, ldc, ldc, nullptr, 0);
            } else {
                MMHelper::compute(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc);
            }
        }
    }

    void compute_proj_bf16(const float *A, const WeiT *B, float *C, int M, int N, int K, int lda, int ldb, int ldc,
            const float *R, int ldr) {
        int alpha = 1.0;
        int beta = 0.0;
        if (R != nullptr) {
#pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                memcpy(C + i * ldc, R + i * ldr, N * sizeof(float));
            }
            beta = 1.0;
        }
        int ldaH = lda * 2;
#pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)A + i * ldaH, K);
        }
        cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, (const MKL_BF16 *)(A), ldaH,
                (const MKL_BF16 *)(B), ldb, beta, C, ldc);
    }

    void catGateUpProj(hpj::Matrix<float> &input, hpj::Matrix<float> &output) {
        TimeLine t("catGateUpProj");

        assert(input.Rows() == output.Rows());
        assert(input.Cols() == catWeights.Rows());
        assert(catWeights.Cols() == output.Cols());

        int M = input.Rows(), N = output.Cols(), K = input.Cols();
        int lda = input.Stride(), ldc = output.Stride();

        const float *A = input.Data();
        const WeiT *B = catWeights.Data();
        const float *scaleB = catWeightsScale.Data();
        const float *zeroB = catWeightsZero.Data();
        const float *sumB = catWeightsSum.Data();
        float *C = output.Data();

        MMHelper::compute(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc);
        // compute silu on the left half and then add it with the right half
        DecoderUtil::siluSum(output);
    }

    void catGateUpWeights(hpj::Matrix<WeiT> &gateWeight, hpj::Matrix<WeiT> &upWeight,
            hpj::Vector<float> &gateWeightScale, hpj::Vector<float> &gateWeightZero, hpj::Vector<float> &gateWeightSum,
            hpj::Vector<float> &upWeightScale, hpj::Vector<float> &upWeightZero, hpj::Vector<float> &upWeightSum,
            hpj::Matrix<WeiT> &catWeights, hpj::Vector<float> &catWeightsScale, hpj::Vector<float> &catWeightsZero,
            hpj::Vector<float> &catWeightsSum) {
        catWeights.Resize(gateWeight.Rows(), gateWeight.Cols() + upWeight.Cols());
        catWeightsScale.Resize(gateWeightScale.Size() + upWeightScale.Size());
        catWeightsZero.Resize(gateWeightZero.Size() + upWeightZero.Size());
        catWeightsSum.Resize(gateWeightSum.Size() + upWeightSum.Size());

        int M = catWeights.Rows();
        int Stride = catWeights.Cols();
        int N = gateWeight.Cols();
#pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            memcpy(catWeights.Data() + i * Stride, gateWeight.Data() + i * N, N * sizeof(WeiT));
            memcpy(catWeights.Data() + i * Stride + N, upWeight.Data() + i * N, N * sizeof(WeiT));
        }

        M = gateWeightScale.Size();
        N = upWeightScale.Size();
        memcpy(catWeightsScale.Data(), gateWeightScale.Data(), M * sizeof(float));
        memcpy(catWeightsScale.Data() + M, upWeightScale.Data(), N * sizeof(float));
        memcpy(catWeightsZero.Data(), gateWeightZero.Data(), M * sizeof(float));
        memcpy(catWeightsZero.Data() + M, upWeightZero.Data(), N * sizeof(float));
        memcpy(catWeightsSum.Data(), gateWeightSum.Data(), M * sizeof(float));
        memcpy(catWeightsSum.Data() + M, upWeightSum.Data(), N * sizeof(float));
    }

protected:
    hpj::Matrix<WeiT> gateWeight;
    hpj::Vector<float> gateWeightScale; // For int8_t weight
    hpj::Vector<float> gateWeightZero; // For int8_t weight
    hpj::Vector<float> gateWeightSum; // For int8_t weight
    hpj::Matrix<WeiT> upWeight;
    hpj::Vector<float> upWeightScale; // For int8_t weight
    hpj::Vector<float> upWeightZero; // For int8_t weight
    hpj::Vector<float> upWeightSum; // For int8_t weight
    hpj::Matrix<WeiT> catWeights;
    hpj::Vector<float> catWeightsScale; // For int8_t weight
    hpj::Vector<float> catWeightsZero; // For int8_t weight
    hpj::Vector<float> catWeightsSum; // For int8_t weight
    hpj::Matrix<WeiT> downWeight;
    hpj::Vector<float> downWeightScale; // For int8_t weight
    hpj::Vector<float> downWeightZero; // For int8_t weight
    hpj::Vector<float> downWeightSum; // For int8_t weight

    // LlamaRMSNorm param
    hpj::Vector<float> normWeight;

#ifdef DEBUG
    Debugger dbg;
#endif
};
