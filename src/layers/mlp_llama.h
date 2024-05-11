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

#ifdef UNDEBUG
#undef NDEBUG
#endif

#include "bert_util.h"
#include "copy_util.h"
#include "debugger.h"
#include "decoder_util.h"
#include "matmul_helper.h"
#include "rmsnorm_kernels.h"
#include "simple_mem_pool.h"
#include "singleton.h"
#include "timeline.h"

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
template <typename WeiT, typename InT = float, typename ImT = float, typename OutT = float>
class LlamaMLP : public SingletonBase<LlamaMLP<WeiT>> {
public:
    LlamaMLP() {}

    LlamaMLP(DecoderContext *ctx) {}

    // OriWeiT: float, int8_t or uint4x2_t
    template <typename OriWeiT>
    void setWeights(DecoderContext *ctx, const OriWeiT *gateW, const float *gateS, const float *gateZ,
            const float * /*unused*/, const OriWeiT *upW, const float *upS, const float *upZ, const float * /*unused*/,
            const float *normW, const float * /*unused*/, const OriWeiT *downW, const float *downS, const float *downZ,
            bool trans = true) {
        int hiddenSize = ctx->hiddenSize;
        int imSize = ctx->intermediateSize;

        REQUIRES(ctx->actType == DecoderContext::SILU || ctx->actType == DecoderContext::GELU,
                "unsupported activation.");

        // Vertically split the gate weight and up weight
        xft::Matrix<WeiT> quantizedGateWeight, quantizedUpWeight, quantizedDownWeight;

        auto it = SplitUtil::getTaskRange(imSize, ctx->numSplit, ctx->splitIdx);
        downWeight.Resize(it.second - it.first, hiddenSize);

        ctx->mmHelper->convertWeight(ctx, trans, hiddenSize, imSize, gateW, gateS, gateZ, true, quantizedGateWeight,
                gateWeightScale, gateWeightZero, gateWeightSum);
        ctx->mmHelper->convertWeight(ctx, trans, hiddenSize, imSize, upW, upS, upZ, true, quantizedUpWeight,
                upWeightScale, upWeightZero, upWeightSum);

        if (!enableCATMLP()) {
            gateWeight.Resize(hiddenSize, it.second - it.first);
            upWeight.Resize(hiddenSize, it.second - it.first);
            ctx->mmHelper->packWeight(trans, quantizedGateWeight, gateWeight);
            ctx->mmHelper->packWeight(trans, quantizedUpWeight, upWeight);
        } else {
            xft::Matrix<WeiT> quantizedCatWeights;
            catGateUpWeights(quantizedGateWeight, quantizedUpWeight, gateWeightScale, gateWeightZero, gateWeightSum,
                    upWeightScale, upWeightZero, upWeightSum, quantizedCatWeights, catWeightsScale, catWeightsZero,
                    catWeightsSum);
            quantizedGateWeight.Release();
            quantizedUpWeight.Release();
            catWeights.Resize(quantizedCatWeights.Rows(), quantizedCatWeights.Cols());
            ctx->mmHelper->packWeight(trans, quantizedCatWeights, catWeights);
        }
        // Horizontally split the down weight
        ctx->mmHelper->convertWeight(ctx, trans, imSize, hiddenSize, downW, downS, downZ, false, quantizedDownWeight,
                downWeightScale, downWeightZero, downWeightSum);
        ctx->mmHelper->packWeight(trans, quantizedDownWeight, downWeight);

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
    void forward(DecoderContext *ctx, InT *input, OutT *output, int iStride, int oStride,
            bool doLnBefore = true /*not used*/, int totInSeqLen = 0) {
        TimeLine t("LlamaMLP");

        const int M = totInSeqLen == 0 ? ctx->batchSize * ctx->inputSeqLen : totInSeqLen;
        const int hiddenSize = ctx->hiddenSize;

        static_assert(sizeof(ctx->normBuf.Data()[0]) >= sizeof(ImT), "normBuff is not big enough!");

        xft::Matrix<InT> inBuffer(input, M, hiddenSize, iStride);
        xft::Matrix<OutT> outBuffer(output, M, hiddenSize, oStride);
        xft::Matrix<ImT> normBuffer(
                (ImT *)ctx->normBuf.Data(), ctx->normBuf.Rows(), ctx->normBuf.Cols(), ctx->normBuf.Stride());

        if (doLnBefore == true) {
            xft::rmsNorm(normBuffer.Data(), inBuffer.Data(), normWeight.Data(), M, hiddenSize, inBuffer.Stride(),
                    normBuffer.Stride(), 1e-6);
        }

#ifdef DEBUG
        dbg.debugPrint("LayerNorm before MLP:\n");
        dbg.dumpMatrix(normBuffer);
        dbg.debugPrint(">>> residential: [%d, %d] (%d)\n", inBuffer.Rows(), inBuffer.Cols(), inBuffer.Stride());
        dbg.dumpMatrix(inBuffer);
#endif

        if (!enableCATMLP()) {
            xft::Matrix<ImT> imBuffer(
                    (ImT *)ctx->imOut.Data(), ctx->imOut.Rows(), ctx->imOut.Cols(), ctx->imOut.Stride());
            gateProj(ctx, doLnBefore ? normBuffer : inBuffer, imBuffer);

#ifdef DEBUG
            dbg.debugPrint(
                    ">>> gateWeight: [%d, %d] (%d)\n", gateWeight.Rows(), gateWeight.Cols(), gateWeight.Stride());
            dbg.dumpMatrix(gateWeight);
            dbg.debugPrint(">>> gate output: [%d, %d] (%d)\n", imBuffer.Rows(), imBuffer.Cols(), imBuffer.Stride());
            dbg.dumpMatrix(imBuffer);
#endif

            upProj(ctx, doLnBefore ? normBuffer : inBuffer, imBuffer);

#ifdef DEBUG
            dbg.debugPrint(">>> upWeight: [%d, %d] (%d)\n", upWeight.Rows(), upWeight.Cols(), upWeight.Stride());
            dbg.dumpMatrix(upWeight);
            dbg.debugPrint(">>> up output: [%d, %d] (%d)\n", imBuffer.Rows(), imBuffer.Cols(), imBuffer.Stride());
            dbg.dumpMatrix(imBuffer);
#endif
            downProj(ctx, imBuffer, outBuffer, inBuffer, ctx->splitIdx == 0);

        } else {
            auto M = normBuffer.Rows();
            auto N = catWeights.Cols();
            xft::Matrix<ImT> imBuffer((ImT *)ctx->imOut.Data(), M, N, N);

            // Need to allocate extra buffer as oneDNN does not support the case of stride > cols
            const int cols = N / 2;
            auto bufSize = sizeof(ImT) * M * cols;
            ImT *t = (ImT *)SimpleMemPool::instance().getBuffer("mlp_silu", bufSize);
            xft::Matrix<ImT> siluBuf(t, M, cols, cols);
#ifdef DEBUG
            dbg.debugPrint(
                    ">>> enableCATMLP imBuffer: [%d, %d] (%d)\n", imBuffer.Rows(), imBuffer.Cols(), imBuffer.Stride());
            dbg.dumpMatrix(imBuffer);
            dbg.debugPrint(">>> residential: [%d, %d] (%d)\n", inBuffer.Rows(), inBuffer.Cols(), inBuffer.Stride());
            dbg.dumpMatrix(inBuffer);
#endif
            catGateUpProj(ctx, doLnBefore ? normBuffer : inBuffer, imBuffer, siluBuf);
#ifdef DEBUG
            dbg.debugPrint("catWeights:\n");
            dbg.dumpMatrix(catWeights);
            dbg.debugPrint("gateUp output:\n");
            dbg.dumpMatrix(siluBuf);
            dbg.debugPrint(">>> residential: [%d, %d] (%d)\n", inBuffer.Rows(), inBuffer.Cols(), inBuffer.Stride());
            dbg.dumpMatrix(inBuffer);
#endif
            downProj(ctx, siluBuf, outBuffer, inBuffer, ctx->splitIdx == 0);
        }

#ifdef DEBUG
        dbg.debugPrint(">>> downWeight: [%d, %d] (%d)\n", downWeight.Rows(), downWeight.Cols(), downWeight.Stride());
        dbg.dumpMatrix(downWeight);
        dbg.debugPrint(">>> residential: [%d, %d] (%d)\n", inBuffer.Rows(), inBuffer.Cols(), inBuffer.Stride());
        dbg.dumpMatrix(inBuffer);
        dbg.debugPrint(">>> final output: [%d, %d] (%d)\n", outBuffer.Rows(), outBuffer.Cols(), outBuffer.Stride());
        dbg.dumpMatrix(outBuffer);
#endif
    }

private:
    void gateProj(DecoderContext *ctx, xft::Matrix<InT> &input, xft::Matrix<ImT> &output) {
        TimeLine t("GateProj");

        assert(input.Rows() == output.Rows());
        assert(input.Cols() == gateWeight.Rows());
        assert(gateWeight.Cols() == output.Cols());

        int M = input.Rows(), N = output.Cols(), K = input.Cols();
        int lda = input.Stride(), ldc = output.Stride();

        const InT *A = input.Data();
        const WeiT *B = gateWeight.Data();
        const float *scaleB = gateWeightScale.Data();
        const float *zeroB = gateWeightZero.Data();
        const float *sumB = gateWeightSum.Data();
        ImT *C = output.Data();

        if (ctx->actType == DecoderContext::SILU) {
            ctx->mmHelper->compute_silu(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc);
        } else if (ctx->actType == DecoderContext::SWIGLU) { // chatglm2/3
            ctx->mmHelper->compute_silu(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc);
        } else if (ctx->actType == DecoderContext::GELU) { // gemma
            ctx->mmHelper->compute_gelu(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc);
        } else {
            printf("ERROR: unsupported activation in MLP.\n");
            exit(-1);
        }
    }

    void upProj(DecoderContext *ctx, xft::Matrix<InT> &input, xft::Matrix<ImT> &output) {
        TimeLine t("UpProj");

        assert(input.Rows() == output.Rows());
        assert(input.Cols() == upWeight.Rows());
        assert(upWeight.Cols() == output.Cols());

        int M = input.Rows(), N = output.Cols(), K = input.Cols();
        int lda = input.Stride(), ldc = output.Stride();

        const InT *A = input.Data();
        const WeiT *B = upWeight.Data();
        const float *scaleB = upWeightScale.Data();
        const float *zeroB = upWeightZero.Data();
        const float *sumB = upWeightSum.Data();
        ImT *C = output.Data();

        ctx->mmHelper->compute_resmul(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc, C, ldc);
    }

    void downProj(DecoderContext *ctx, xft::Matrix<ImT> &input, xft::Matrix<OutT> &output,
            xft::Matrix<InT> &residential, bool isMaster) {
        TimeLine t("DownProj");

        assert(input.Rows() == output.Rows());
        assert(input.Cols() == downWeight.Rows());
        assert(downWeight.Cols() == output.Cols());

        int M = input.Rows(), N = output.Cols(), K = downWeight.Rows();
        int lda = input.Stride(), ldc = output.Stride(), ldr = residential.Stride();

        const ImT *A = input.Data();
        const WeiT *B = downWeight.Data();
        const float *scaleB = downWeightScale.Data();
        const float *zeroB = downWeightZero.Data();
        const float *sumB = downWeightSum.Data();
        OutT *C = output.Data();
        const InT *R = residential.Data();

        if (isMaster) {
            ctx->mmHelper->compute_residential(
                    false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc, NULL, R, ldr);
        } else {
            ctx->mmHelper->compute(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc);
        }
    }

    template <typename T1, typename T2>
    void catGateUpProj(DecoderContext *ctx, xft::Matrix<T1> &input, xft::Matrix<T2> &output, xft::Matrix<T2> &siluBuf) {
        TimeLine t("catGateUpProj");

        assert(input.Rows() == output.Rows());
        assert(input.Cols() == catWeights.Rows());
        assert(catWeights.Cols() == output.Cols());

        int M = input.Rows(), N = output.Cols(), K = input.Cols();
        int lda = input.Stride(), ldc = output.Stride();

        const T1 *A = input.Data();
        const WeiT *B = catWeights.Data();
        const float *scaleB = catWeightsScale.Data();
        const float *zeroB = catWeightsZero.Data();
        const float *sumB = catWeightsSum.Data();
        T2 *C = output.Data();

        ctx->mmHelper->compute(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc);

        // Compute silu on the left half and then add it with the right half
        if (ctx->actType == DecoderContext::SILU) {
            DecoderUtil::siluSum(output, siluBuf);
        } else if (ctx->actType == DecoderContext::SWIGLU) { // chatglm2/3
            DecoderUtil::siluSum(output, siluBuf);
        } else if (ctx->actType == DecoderContext::GELU) { // gemma
            DecoderUtil::geluSum(output, siluBuf);
        } else {
            printf("ERROR: unsupported activation in MLP.\n");
            exit(-1);
        }
    }

    void catGateUpWeights(xft::Matrix<WeiT> &gateWeight, xft::Matrix<WeiT> &upWeight,
            xft::Vector<float> &gateWeightScale, xft::Vector<float> &gateWeightZero, xft::Vector<float> &gateWeightSum,
            xft::Vector<float> &upWeightScale, xft::Vector<float> &upWeightZero, xft::Vector<float> &upWeightSum,
            xft::Matrix<WeiT> &catWeights, xft::Vector<float> &catWeightsScale, xft::Vector<float> &catWeightsZero,
            xft::Vector<float> &catWeightsSum) {
        catWeights.Resize(gateWeight.Rows(), gateWeight.Cols() + upWeight.Cols());
        catWeightsScale.Resize(gateWeightScale.Size() + upWeightScale.Size());
        catWeightsZero.Resize(gateWeightZero.Size() + upWeightZero.Size());
        catWeightsSum.Resize(gateWeightSum.Size() + upWeightSum.Size());

        int M = catWeights.Rows();
        int Stride = catWeights.Cols();
        int N = gateWeight.Cols();
        if (std::is_same_v<WeiT, uint4x2_t> || std::is_same_v<WeiT, nf4x2_t>) {
            // two values are packed into one byte
            Stride /= 2;
            N /= 2;
        }
#pragma omp parallel for
        for (uint64_t i = 0; i < M; ++i) {
            memcpy(catWeights.Data() + i * Stride, gateWeight.Data() + i * N, N * sizeof(WeiT));
            memcpy(catWeights.Data() + i * Stride + N, upWeight.Data() + i * N, N * sizeof(WeiT));
        }

        M = gateWeightScale.Size();
        N = upWeightScale.Size();
        memcpy(catWeightsScale.Data(), gateWeightScale.Data(), M * sizeof(float));
        memcpy(catWeightsScale.Data() + M, upWeightScale.Data(), N * sizeof(float));
        memcpy(catWeightsZero.Data(), gateWeightZero.Data(), M * sizeof(float));
        memcpy(catWeightsZero.Data() + M, upWeightZero.Data(), N * sizeof(float));
        M = gateWeightSum.Size();
        N = upWeightSum.Size();
        memcpy(catWeightsSum.Data(), gateWeightSum.Data(), M * sizeof(float));
        memcpy(catWeightsSum.Data() + M, upWeightSum.Data(), N * sizeof(float));
    }

protected:
    xft::Matrix<WeiT> gateWeight;
    xft::Vector<float> gateWeightScale; // For int8_t weight
    xft::Vector<float> gateWeightZero; // For int8_t weight
    xft::Vector<float> gateWeightSum; // For int8_t weight
    xft::Matrix<WeiT> upWeight;
    xft::Vector<float> upWeightScale; // For int8_t weight
    xft::Vector<float> upWeightZero; // For int8_t weight
    xft::Vector<float> upWeightSum; // For int8_t weight
    xft::Matrix<WeiT> catWeights;
    xft::Vector<float> catWeightsScale; // For int8_t weight
    xft::Vector<float> catWeightsZero; // For int8_t weight
    xft::Vector<float> catWeightsSum; // For int8_t weight
    xft::Matrix<WeiT> downWeight;
    xft::Vector<float> downWeightScale; // For int8_t weight
    xft::Vector<float> downWeightZero; // For int8_t weight
    xft::Vector<float> downWeightSum; // For int8_t weight

    // LlamaRMSNorm param
    xft::Vector<float> normWeight;

#ifdef DEBUG
    Debugger dbg;
#endif
};
