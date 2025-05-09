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
#include <cmath>
#include "mlp_llama.h"

template <typename WeiT, typename InT, typename ImT, typename OutT, typename NORM_CLS, bool INPUT_AS_RESID>
class ChatGLM2MLP : public LlamaMLP<WeiT, InT, ImT, OutT> {
public:
    ChatGLM2MLP(int layerId, DecoderContext *ctx) : LlamaMLP<WeiT, InT, ImT, OutT>(layerId, ctx) {}

    // OriWeiT: float
    template <typename OriWeiT>
    void setWeights(DecoderContext *ctx, const OriWeiT *gate_upW, const float * /*unused*/, const float * /*unused*/,
            const float * /*unused*/, const OriWeiT *downW, const float * /*unused*/, const float * /*unused*/,
            const float * /*unused*/, const float *normW, const float * /*unused*/, const OriWeiT * /*unused*/,
            const float * /*unused*/, const float * /*unused*/, const float * /*unused*/, bool trans = true) {
        int hiddenSize = ctx->hiddenSize;
        int intermediateSize = ctx->intermediateSize;

        REQUIRES(ctx->actType == DecoderContext::SWIGLU, "unsupported activation.");

        // Vertically split the gate weight and up weight
        xft::Matrix<WeiT> convertedGateWeight, convertedUpWeight, convertedDownWeight;

        auto range = SplitUtil::getTaskRange(intermediateSize, ctx->numSplit, ctx->splitIdx);
        int colSplit = range.second - range.first;

        if (!Env::getInstance().getMlpCatEnabled()) {
            OriWeiT *gateW = (OriWeiT *)malloc(hiddenSize * colSplit * sizeof(OriWeiT));
            OriWeiT *upW = (OriWeiT *)malloc(hiddenSize * colSplit * sizeof(OriWeiT));
            if (trans) {
                int blockSize = colSplit * hiddenSize;
                memcpy(gateW, gate_upW + range.first * hiddenSize, blockSize * sizeof(OriWeiT));
                memcpy(upW, gate_upW + intermediateSize * hiddenSize + range.first * hiddenSize,
                        blockSize * sizeof(OriWeiT));
            } else {
                const OriWeiT *weightPTR = gate_upW;
                for (int i = 0; i < hiddenSize; i++) {
                    memcpy(gateW + i * colSplit, weightPTR + range.first, colSplit * sizeof(OriWeiT));
                    weightPTR += intermediateSize;
                    memcpy(upW + i * colSplit, weightPTR + range.first, colSplit * sizeof(OriWeiT));
                    weightPTR += intermediateSize;
                }
            }

            ctx->mmHelper->convertWeight(trans, hiddenSize, colSplit, gateW, nullptr, nullptr, convertedGateWeight,
                    this->gateWeightScale, this->gateWeightZero, this->gateWeightSum);
            ctx->mmHelper->packWeight(trans, convertedGateWeight, this->gateWeight);

            ctx->mmHelper->convertWeight(trans, hiddenSize, colSplit, upW, nullptr, nullptr, convertedUpWeight,
                    this->upWeightScale, this->upWeightZero, this->upWeightSum);
            ctx->mmHelper->packWeight(trans, convertedUpWeight, this->upWeight);

            free(gateW);
            free(upW);

        } else {
            if (trans) {
                printf("Trans GateUpW Not supported yet.\n");
                exit(-1);
            } else {
                int colSplitStride = colSplit * 2;
                OriWeiT *gateUpW = (OriWeiT *)malloc(hiddenSize * colSplitStride * sizeof(OriWeiT));
                const OriWeiT *weightPTR = gate_upW;
                for (int i = 0; i < hiddenSize; i++) {
                    memcpy(gateUpW + i * colSplitStride, weightPTR + range.first, colSplit * sizeof(OriWeiT));
                    weightPTR += intermediateSize;
                    memcpy(gateUpW + colSplit + i * colSplitStride, weightPTR + range.first,
                            colSplit * sizeof(OriWeiT));
                    weightPTR += intermediateSize;
                }
                xft::Matrix<WeiT> quantizedCatWeights;
                ctx->mmHelper->convertWeight(trans, hiddenSize, colSplitStride, gateUpW, nullptr, nullptr,
                        quantizedCatWeights, this->catWeightsScale, this->catWeightsZero, this->catWeightsSum);
                this->catWeights.Resize(quantizedCatWeights.Rows(), quantizedCatWeights.Cols());
                ctx->mmHelper->packWeight(trans, quantizedCatWeights, this->catWeights);
                free(gateUpW);
            }
        }
        // Horizontally split the down weight
        ctx->mmHelper->convertWeight(ctx, trans, intermediateSize, hiddenSize, downW, nullptr, nullptr, false,
                convertedDownWeight, this->downWeightScale, this->downWeightZero, this->downWeightSum);
        ctx->mmHelper->packWeight(trans, convertedDownWeight, this->downWeight);
#ifdef XFT_DEBUG
        this->dbg.debugPrint("convertedGateWeight [%d, %d](%d):\n", convertedGateWeight.Rows(),
                convertedGateWeight.Cols(), convertedGateWeight.Stride());
        this->dbg.dumpMatrix(convertedGateWeight);

        this->dbg.debugPrint("packed convertedGateWeight [%d, %d](%d):\n", this->gateWeight.Rows(),
                this->gateWeight.Cols(), this->gateWeight.Stride());
        this->dbg.dumpMatrix(this->gateWeight, false, ctx->device);

        this->dbg.debugPrint("convertedUpWeight [%d, %d](%d):\n", convertedUpWeight.Rows(), convertedUpWeight.Cols(),
                convertedUpWeight.Stride());
        this->dbg.dumpMatrix(convertedUpWeight);

        this->dbg.debugPrint("packed convertedUpWeight [%d, %d](%d):\n", this->upWeight.Rows(), this->upWeight.Cols(),
                this->upWeight.Stride());
        this->dbg.dumpMatrix(this->upWeight, false, ctx->device);

        this->dbg.debugPrint("convertedDownWeight [%d, %d](%d):\n", convertedDownWeight.Rows(),
                convertedDownWeight.Cols(), convertedDownWeight.Stride());
        this->dbg.dumpMatrix(convertedDownWeight);

        this->dbg.debugPrint("packed convertedDownWeight [%d, %d](%d):\n", this->downWeight.Rows(),
                this->downWeight.Cols(), this->downWeight.Stride());
        this->dbg.dumpMatrix(this->downWeight, false, ctx->device);
#endif
        // norm.setWeight(normW, NULL, hiddenSize);

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
    }

private:
    using LlamaMLP<WeiT, InT, ImT, OutT>::norm;
};
