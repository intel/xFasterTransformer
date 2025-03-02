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
#include "copy_util.h"
#include "add_util.h"
#include "datatypes.h"
#include "debugger.h"
#include "decoder_util.h"
#include "dtype.h"
#include "llm_params.h"
#include "logger.h"
#include "matmul_helper.h"
#include "rms_norm.h"
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
template <typename WeiT, typename InT = float, typename ImT = float, typename OutT = float,
        typename NORM_CLS = xft::RmsNorm>
class LlamaMLP {
public:
    LlamaMLP(int layerId, DecoderContext *ctx) : layerId(layerId), norm(ctx) {}

    static xft::DataType getWeightDataType() { return xft::getDataType<WeiT>(); }

    // OriWeiT: float, int8_t or uint4x2_t
    template <typename OriWeiT>
    void setWeights(DecoderContext *ctx, const OriWeiT *gateW, const float *gateS, const float *gateZ,
            const float * /*unused*/, const OriWeiT *upW, const float *upS, const float *upZ, const float * /*unused*/,
            const float *normW, const float * /*unused*/, const OriWeiT *downW, const float *downS, const float *downZ,
            const float *downB, bool trans = true) {
        int hiddenSize = ctx->hiddenSize;
        int imSize = ctx->intermediateSize;
        if (layerId >= ctx->firstKDenseReplace && ctx->moeIntermediateSize > 0) {
            // for each expert MLP in MoE
            imSize = ctx->moeIntermediateSize;
        }

        REQUIRES(ctx->actType == DecoderContext::SILU || ctx->actType == DecoderContext::GELU,
                "unsupported activation.");

        // Vertically split the gate weight and up weight
        xft::Matrix<WeiT> quantizedGateWeight, quantizedUpWeight, quantizedDownWeight;

        // for e4m3_t, size should be multiple of 128 (64 * 2)
        int gran = std::is_same_v<WeiT, e4m3_t> ? 2 : 1;
        auto it = SplitUtil::getTaskRange(imSize, gran, ctx->numSplit, ctx->splitIdx);

        ctx->mmHelper->convertWeight(ctx, trans, hiddenSize, imSize, gateW, gateS, gateZ, true, quantizedGateWeight,
                gateWeightScale, gateWeightZero, gateWeightSum);
        ctx->mmHelper->convertWeight(ctx, trans, hiddenSize, imSize, upW, upS, upZ, true, quantizedUpWeight,
                upWeightScale, upWeightZero, upWeightSum);

        if (!Env::getInstance().getMlpCatEnabled()) {
            if (std::is_same_v<WeiT, e4m3_t>) {
                xft::Logger::error("Internal Error: Not support split-GateUp MLP for fp8_e4m3.");
                exit(-1);
            }
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

#ifdef XFT_GPU
            xft::Matrix<WeiT> catWeightsT;
            int catWeiRows = quantizedCatWeights.Rows();
            int catWeiCols = quantizedCatWeights.Cols();
            catWeightsT.Resize(catWeiRows, catWeiCols);
            ctx->mmHelper->transposeWeight(trans, quantizedCatWeights, catWeightsT);

            WeiT *catWeiData = (WeiT *)xft::alloc(catWeiRows * catWeiCols * sizeof(WeiT), ctx->device);
            catWeights.Assign(catWeiData, catWeiRows, catWeiCols, catWeiCols);
            xft::memcopy(catWeights.Data(), catWeightsT.Data(), catWeiRows * catWeiCols * sizeof(WeiT), ctx->device);
#else
            catWeights.Resize(quantizedCatWeights.Rows(), quantizedCatWeights.Cols());
            ctx->mmHelper->packWeight(trans, quantizedCatWeights, catWeights);
#endif
        }
        // Horizontally split the down weight
        ctx->mmHelper->convertWeight(ctx, trans, imSize, hiddenSize, downW, downS, downZ, false, quantizedDownWeight,
                downWeightScale, downWeightZero, downWeightSum);
#ifdef XFT_GPU
        xft::Matrix<WeiT> downWeightT;
        int downWeiRows = it.second - it.first;
        int downWeiCols = hiddenSize;
        downWeightT.Resize(downWeiRows, downWeiCols);
        ctx->mmHelper->transposeWeight(trans, quantizedDownWeight, downWeightT);

        WeiT *downWeiData = (WeiT *)xft::alloc(downWeiRows * downWeiCols * sizeof(WeiT), ctx->device);
        downWeight.Assign(downWeiData, downWeiRows, downWeiCols, downWeiCols);
        xft::memcopy(downWeight.Data(), downWeightT.Data(), downWeiRows * downWeiCols * sizeof(WeiT), ctx->device);
#else
        downWeight.Resize(it.second - it.first, hiddenSize);
        ctx->mmHelper->packWeight(trans, quantizedDownWeight, downWeight);
#endif
        // Down bias
        // For other splits, not init bias to avoid duplicated calculation
        if (downB != nullptr && ctx->splitIdx == 0) {
            downBias.Resize(hiddenSize);
            memcpy(downBias.Data(), downB, sizeof(float) * hiddenSize);
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("quantizedGateWeight:\n");
        dbg.dumpMatrix(quantizedGateWeight);

        dbg.debugPrint("quantizedUpWeight:\n");
        dbg.dumpMatrix(quantizedUpWeight);

        dbg.debugPrint("quantizedDownWeight:\n");
        dbg.dumpMatrix(quantizedDownWeight);
#endif

        // LlamaRMSNorm
        if (normW) { norm.setWeight(normW, nullptr, hiddenSize); }
    }

    template <typename WType>
    void setWeights(DecoderContext *ctx, xft::FFNParams *ffnParams) {
        auto *llamaFFN = dynamic_cast<xft::LlamaFFNParams *>(ffnParams);
        if (llamaFFN == nullptr) {
            xft::Logger::error("Cannot cast FFNParams to LlamaFFNParams.");
            exit(-1);
        }

        setWeights(ctx, (WType *)llamaFFN->gate.weight, llamaFFN->gate.weight_scale, llamaFFN->gate.weight_zp,
                llamaFFN->gate.bias, (WType *)llamaFFN->up.weight, llamaFFN->up.weight_scale, llamaFFN->up.weight_zp,
                llamaFFN->up.bias, llamaFFN->norm.gamma, llamaFFN->norm.beta, (WType *)llamaFFN->down.weight,
                llamaFFN->down.weight_scale, llamaFFN->down.weight_zp, llamaFFN->down.bias, false);

        if (std::is_same_v<WeiT, e4m3_t>) {
            prepareFP8Scales(ctx, llamaFFN->gate, llamaFFN->up, catGUScales, true);
            prepareFP8Scales(ctx, llamaFFN->down, downScales, false);
        }
    }

    template <typename WType>
    void setWeights(DecoderContext *ctx, xft::ExpertParams &ffn) {
        setWeights(ctx, (WType *)ffn.gate.weight, ffn.gate.weight_scale, ffn.gate.weight_zp, ffn.gate.bias,
                (WType *)ffn.up.weight, ffn.up.weight_scale, ffn.up.weight_zp, ffn.up.bias, nullptr, nullptr,
                (WType *)ffn.down.weight, ffn.down.weight_scale, ffn.down.weight_zp, ffn.down.bias, false);

        if (std::is_same_v<WeiT, e4m3_t>) {
            prepareFP8Scales(ctx, ffn.gate, ffn.up, catGUScales, true);
            prepareFP8Scales(ctx, ffn.down, downScales, false);
        }
    }

#ifdef XFT_DEBUG
    void setDebugger(const Debugger &debugger) { this->dbg = debugger; }
#endif

    // Forward for FFN (Feed Forward Network)
    // forceNoResidual: true when it is called by MOE MLP like MixtralMLP
    void forward(DecoderContext *ctx, InT *input, OutT *output, int iStride, int oStride, bool doLnBefore = true,
            int totInSeqLen = 0, bool forceNoResidual = false) {
        TimeLine t("LlamaMLP");

        const int M = totInSeqLen == 0 ? ctx->batchSize * ctx->inputSeqLen : totInSeqLen;
        const int hiddenSize = ctx->hiddenSize;

        static_assert(sizeof(ctx->normBuf.Data()[0]) >= sizeof(ImT), "normBuff is not big enough!");

        xft::Matrix<InT> inBuffer(input, M, hiddenSize, iStride);
        xft::Matrix<OutT> outBuffer(output, M, hiddenSize, oStride);
        xft::Matrix<ImT> normBuffer(
                (ImT *)ctx->normBuf.Data(), ctx->normBuf.Rows(), ctx->normBuf.Cols(), ctx->normBuf.Stride());

        if (doLnBefore == true) {
            norm.forward(inBuffer.Data(), normBuffer.Data(), M, inBuffer.Stride(), normBuffer.Stride(), 1e-6);
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("LayerNorm before MLP:\n");
        dbg.dumpMatrix(normBuffer, false, ctx->device);
        dbg.debugPrint(">>> residential: [%d, %d] (%d)\n", inBuffer.Rows(), inBuffer.Cols(), inBuffer.Stride());
        dbg.dumpMatrix(inBuffer, false, ctx->device);
#endif

        if (!Env::getInstance().getMlpCatEnabled()) {
            xft::Matrix<ImT> imBuffer(
                    (ImT *)ctx->imOut.Data(), ctx->imOut.Rows(), ctx->imOut.Cols(), ctx->imOut.Stride());
            gateProj(ctx, doLnBefore ? normBuffer : inBuffer, imBuffer);

#ifdef XFT_DEBUG
            dbg.debugPrint(
                    ">>> gateWeight: [%d, %d] (%d)\n", gateWeight.Rows(), gateWeight.Cols(), gateWeight.Stride());
            dbg.dumpMatrix(gateWeight, false, ctx->device);
            dbg.debugPrint(">>> gate output: [%d, %d] (%d)\n", imBuffer.Rows(), imBuffer.Cols(), imBuffer.Stride());
            dbg.dumpMatrix(imBuffer, false, ctx->device);
#endif

            upProj(ctx, doLnBefore ? normBuffer : inBuffer, imBuffer);

#ifdef XFT_DEBUG
            dbg.debugPrint(">>> upWeight: [%d, %d] (%d)\n", upWeight.Rows(), upWeight.Cols(), upWeight.Stride());
            dbg.dumpMatrix(upWeight, false, ctx->device);
            dbg.debugPrint(">>> up output: [%d, %d] (%d)\n", imBuffer.Rows(), imBuffer.Cols(), imBuffer.Stride());
            dbg.dumpMatrix(imBuffer, false, ctx->device);
#endif
            bool residential = !forceNoResidual && ctx->splitIdx == 0;
            downProj(ctx, imBuffer, outBuffer, inBuffer, residential);

        } else {
            auto M = inBuffer.Rows();
            auto N = catWeights.Cols();
            xft::Matrix<ImT> imBuffer((ImT *)ctx->imOut.Data(), M, N, N);

            // Need to allocate extra buffer as oneDNN does not support the case of stride > cols
            const int cols = N / 2;
            auto bufSize = sizeof(ImT) * M * cols;
            ImT *t = (ImT *)SimpleMemPool::instance().getBuffer("mlp_silu", bufSize, ctx->device);
            xft::Matrix<ImT> siluBuf(t, M, cols, cols);
#ifdef XFT_DEBUG
            dbg.debugPrint(
                    ">>> enableCATMLP imBuffer: [%d, %d] (%d)\n", imBuffer.Rows(), imBuffer.Cols(), imBuffer.Stride());
            dbg.dumpMatrix(imBuffer, false, ctx->device);
            dbg.debugPrint(">>> residential: [%d, %d] (%d)\n", inBuffer.Rows(), inBuffer.Cols(), inBuffer.Stride());
            dbg.dumpMatrix(inBuffer, false, ctx->device);
#endif
            catGateUpProj(ctx, doLnBefore ? normBuffer : inBuffer, imBuffer, siluBuf);
#ifdef XFT_DEBUG
            dbg.debugPrint("catWeights:\n");
            dbg.dumpMatrix(catWeights, false, ctx->device);
            dbg.debugPrint("gateUp output:\n");
            dbg.dumpMatrix(imBuffer, false, ctx->device);
            dbg.debugPrint("gateUpSilu output:\n");
            dbg.dumpMatrix(siluBuf, false, ctx->device);
            dbg.debugPrint(">>> residential: [%d, %d] (%d)\n", inBuffer.Rows(), inBuffer.Cols(), inBuffer.Stride());
            dbg.dumpMatrix(inBuffer, false, ctx->device);
#endif
            bool residential = !forceNoResidual && ctx->splitIdx == 0;
            downProj(ctx, siluBuf, outBuffer, inBuffer, residential);
        }

#ifdef XFT_DEBUG
        dbg.debugPrint(">>> downWeight: [%d, %d] (%d)\n", downWeight.Rows(), downWeight.Cols(), downWeight.Stride());
        dbg.dumpMatrix(downWeight, false, ctx->device);
        dbg.debugPrint(">>> residential: [%d, %d] (%d)\n", inBuffer.Rows(), inBuffer.Cols(), inBuffer.Stride());
        dbg.dumpMatrix(inBuffer, false, ctx->device);
        dbg.debugPrint(">>> final output: [%d, %d] (%d)\n", outBuffer.Rows(), outBuffer.Cols(), outBuffer.Stride());
        dbg.dumpMatrix(outBuffer, false, ctx->device);
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
        int lds = (std::is_same_v<WeiT, e4m3_t> ? downScales.Stride() : 1);

        const ImT *A = input.Data();
        const WeiT *B = downWeight.Data();
        const float *scaleB = (std::is_same_v<WeiT, e4m3_t> ? downScales.Data() : downWeightScale.Data());
        const float *zeroB = downWeightZero.Data();
        const float *sumB = downWeightSum.Data();
        OutT *C = output.Data();
        const InT *R = residential.Data();

        if (isMaster) {
            float *pbias = downBias.Data();
            if (downBias.Size() == 0) { pbias = nullptr; }
            //if (std::is_same_v<WeiT, e4m3_t>) {
            //    ctx->mmHelper->compute(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc, lds);
//#pragma omp parallel for
            //    for (size_t i = 0; i < M; ++i)
            //        xft::addto(C + i * ldc, R + i * ldr, 1.0, N);
            //} else {
            ctx->mmHelper->compute_residential(
                    false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc, pbias, R, ldr, lds);
	    //}
        } else {
            ctx->mmHelper->compute(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc, lds);
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
        int lds = (std::is_same_v<WeiT, e4m3_t> ? catGUScales.Stride() : 1);

        const T1 *A = input.Data();
        const WeiT *B = catWeights.Data();
        const float *scaleB = (std::is_same_v<WeiT, e4m3_t> ? catGUScales.Data() : catWeightsScale.Data());
        const float *zeroB = catWeightsZero.Data();
        const float *sumB = catWeightsSum.Data();
        T2 *C = output.Data();

        ctx->mmHelper->compute(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, sumB, 0.0f, C, ldc, lds);

        // Compute silu on the left half and then add it with the right half
        if (ctx->actType == DecoderContext::SILU) {
            DecoderUtil::siluSum(output, siluBuf, ctx->device);
        } else if (ctx->actType == DecoderContext::SWIGLU) { // chatglm2/3
            DecoderUtil::siluSum(output, siluBuf, ctx->device);
        } else if (ctx->actType == DecoderContext::GELU) { // gemma
            DecoderUtil::geluSum(output, siluBuf, ctx->device);
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

    void prepareFP8Scales(DecoderContext *ctx, xft::DenseLayerParams &param, xft::Matrix<float> &scales, bool bVerticalSplit) {
        // Check weight type
        if (param.wtype != xft::ParamType::FP8_E4M3) {
            xft::Logger::error("The weight type is not FP8_E4M3, no scales.");
            exit(-1);
        }

        int stride = (param.output_dim + param.block_size1 - 1) / param.block_size1;
        int splitTarget = bVerticalSplit ? param.output_dim : param.input_dim;
        // for e4m3_t, size should be multiple of 128 (64 * 2)
        auto it = SplitUtil::getTaskRange(splitTarget, 2, ctx->numSplit, ctx->splitIdx);
        int splitSize = it.second - it.first;
        int splitOffset = it.first;
        int rows, cols;
        if (bVerticalSplit) {
            rows = (param.input_dim + param.block_size0 - 1) / param.block_size0;
            cols = (splitSize + param.block_size1 - 1) / param.block_size1;
            splitOffset = (splitOffset + param.block_size1 - 1) / param.block_size1;
        } else {
            rows = (splitSize + param.block_size0 - 1) / param.block_size0;
            cols = (param.output_dim + param.block_size1 - 1) / param.block_size1;
            splitOffset = (splitOffset + param.block_size0 - 1) / param.block_size0 * cols;
        }
        //scales.Resize(rows, cols);

        //for (int i = 0; i < rows; ++i) {
        //    memcpy(scales.Row(i), param.weight_scale + i * cols, cols * sizeof(float));
        //}
        // transpose for xddn fp8 kernel
        scales.Resize(cols, rows);

#pragma omp parallel for collapse(2)
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < rows; ++j) {
                memcpy(scales.Row(i) + j, param.weight_scale + splitOffset + j * stride + i, sizeof(float));
            }
        }
    }

    // Prepare FP8 scales by merging 2 scales into 1
    void prepareFP8Scales(DecoderContext *ctx, xft::DenseLayerParams &param1, xft::DenseLayerParams &param2, xft::Matrix<float> &scales, bool bVerticalSplit) {
        // Check weight type
        if (param1.wtype != xft::ParamType::FP8_E4M3 || param2.wtype != xft::ParamType::FP8_E4M3) {
            xft::Logger::error("Internal Error: The weight type is not FP8_E4M3, no scales.");
            exit(-1);
        }

        if (param1.input_dim != param2.input_dim) {
            xft::Logger::error("Internal Error: The input dim of two params are not matched.");
            exit(-1);
        }

        assert(param1.input_dim == param2.input_dim);
        assert(param1.output_dim == param2.output_dim);

        int splitTarget = bVerticalSplit ? param1.output_dim : param1.input_dim;
        int stride = (param1.output_dim + param1.block_size1 - 1) / param1.block_size1;
        // for e4m3_t, size should be multiple of 128 (64 * 2)
        auto it = SplitUtil::getTaskRange(splitTarget, 2, ctx->numSplit, ctx->splitIdx);
        int splitSize = it.second - it.first;
        int splitOffset = it.first;
        int rows, cols1, cols2;
        if (bVerticalSplit) {
            rows = (param1.input_dim + param1.block_size0 - 1) / param1.block_size0;
            cols1 = (splitSize + param1.block_size1 - 1) / param1.block_size1;
            splitOffset = (splitOffset + param1.block_size1 - 1) / param1.block_size1;
        } else {
            rows = (splitSize + param1.block_size0 - 1) / param1.block_size0;
            cols1 = (param1.output_dim + param1.block_size1 - 1) / param1.block_size1;
            splitOffset = (splitOffset + param1.block_size0 - 1) / param1.block_size0 * cols1;
        }

        cols2 = cols1;
        // transpose for xddn fp8 kernel
        scales.Resize(cols1 + cols2, rows);

#pragma omp parallel for collapse(2)
        for (int i = 0; i < cols1; ++i) {
            for (int j = 0; j < rows; ++j) {
                memcpy(scales.Row(i) + j, param1.weight_scale + splitOffset + j * stride + i, sizeof(float));
            }
        }
#pragma omp parallel for collapse(2)
        for (int i = 0; i < cols2; ++i) {
            for (int j = 0; j < rows; ++j) {
                memcpy(scales.Row(i + cols1) + j, param2.weight_scale + splitOffset + j * stride + i, sizeof(float));
            }
        }
    }

protected:
    xft::Matrix<WeiT> gateWeight;
    xft::Vector<float> gateWeightScale; // For int8_t weight
    xft::Vector<float> gateWeightZero; // For int8_t weight
    xft::Vector<float> gateWeightSum; // For int8_t weight
    xft::Matrix<float> gateScales; // For fp8_e4m3 weight

    xft::Matrix<WeiT> upWeight;
    xft::Vector<float> upWeightScale; // For int8_t weight
    xft::Vector<float> upWeightZero; // For int8_t weight
    xft::Vector<float> upWeightSum; // For int8_t weight
    xft::Matrix<float> upScales; // For fp8_e4m3 weight

    xft::Matrix<WeiT> catWeights;
    xft::Vector<float> catWeightsScale; // For int8_t weight
    xft::Vector<float> catWeightsZero; // For int8_t weight
    xft::Vector<float> catWeightsSum; // For int8_t weight
    xft::Matrix<float> catGUScales; // For fp8_e4m3 weight

    xft::Matrix<WeiT> downWeight;
    xft::Vector<float> downBias;
    xft::Vector<float> downWeightScale; // For int8_t weight
    xft::Vector<float> downWeightZero; // For int8_t weight
    xft::Vector<float> downWeightSum; // For int8_t weight
    xft::Matrix<float> downScales; // For fp8_e4m3 weight

    // LlamaRMSNorm param
    NORM_CLS norm;
    
    int layerId;

#ifdef XFT_DEBUG
    Debugger dbg;
#endif
};
