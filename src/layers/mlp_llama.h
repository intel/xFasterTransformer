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
template <typename WeiT>
class LlamaMLP {
public:
    LlamaMLP(DecoderContext *ctx) {
    }

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
        gateWeight.Resize(hiddenSize, it.second - it.first);
        upWeight.Resize(hiddenSize, it.second - it.first);
        downWeight.Resize(it.second - it.first, hiddenSize);

        MMHelper::convertWeight(
                ctx, trans, hiddenSize, imSize, gateW, true, quantizedGateWeight, gateWeightScale, gateWeightZero);
        MMHelper::packWeight(trans, quantizedGateWeight, gateWeight);

        MMHelper::convertWeight(
                ctx, trans, hiddenSize, imSize, upW, true, quantizedUpWeight, upWeightScale, upWeightZero);
        MMHelper::packWeight(trans, quantizedUpWeight, upWeight);

        // Horizontally split the down weight
        MMHelper::convertWeight(
                ctx, trans, imSize, hiddenSize, downW, false, quantizedDownWeight, downWeightScale, downWeightZero);
        MMHelper::packWeight(trans, quantizedDownWeight, downWeight);

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
    void setDebugger(const Debugger &debugger) {
        this->dbg = debugger;
    }
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
        auto &imBuffer = ctx->imOut;

        DecoderUtil::rmsNorm(inBuffer, normBuffer, normWeight, 1e-6);

#ifdef DEBUG
        dbg.debugPrint("LayerNorm before MLP:\n");
        dbg.dumpMatrix(normBuffer);
#endif

        gateProj(normBuffer, imBuffer);

#ifdef DEBUG
        dbg.debugPrint("gateWeight:\n");
        dbg.dumpMatrix(gateWeight);
        dbg.debugPrint("gate output:\n");
        dbg.dumpMatrix(imBuffer);
#endif

        upProj(normBuffer, imBuffer);

#ifdef DEBUG
        dbg.debugPrint("upWeight:\n");
        dbg.dumpMatrix(upWeight);
        dbg.debugPrint("up output:\n");
        dbg.dumpMatrix(imBuffer);
#endif

        downProj(imBuffer, outBuffer, inBuffer, ctx->splitIdx == 0);

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
        float *C = output.Data();

        MMHelper::compute_silu(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, 0.0f, C, ldc);
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
        float *C = output.Data();

        MMHelper::compute_resmul(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, 0.0f, C, ldc, C, ldc);
    }

    void downProj(
            hpj::Matrix<float> &input, hpj::Matrix<float> &output, hpj::Matrix<float> &residential, bool isMaster) {
        TimeLine t("DownProj");

        assert(input.Rows() == output.Rows());
        assert(input.Cols() == downWeight.Rows());
        assert(downWeight.Cols() == output.Cols());

        int M = input.Rows(), N = output.Cols(), K = input.Cols();
        int lda = input.Stride(), ldc = output.Stride(), ldr = residential.Stride();

        const float *A = input.Data();
        const WeiT *B = downWeight.Data();
        const float *scaleB = downWeightScale.Data();
        const float *zeroB = downWeightZero.Data();
        float *C = output.Data();
        const float *R = residential.Data();

        if (isMaster) {
            MMHelper::compute_residential(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, 0.0f, C, ldc, NULL, R, ldr);
        } else {
            MMHelper::compute(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, 0.0f, C, ldc);
        }
    }

protected:
    hpj::Matrix<WeiT> gateWeight;
    hpj::Vector<float> gateWeightScale; // For int8_t weight
    hpj::Vector<float> gateWeightZero; // For int8_t weight
    hpj::Matrix<WeiT> upWeight;
    hpj::Vector<float> upWeightScale; // For int8_t weight
    hpj::Vector<float> upWeightZero; // For int8_t weight
    hpj::Matrix<WeiT> downWeight;
    hpj::Vector<float> downWeightScale; // For int8_t weight
    hpj::Vector<float> downWeightZero; // For int8_t weight

    // LlamaRMSNorm param
    hpj::Vector<float> normWeight;

#ifdef DEBUG
    Debugger dbg;
#endif
};
