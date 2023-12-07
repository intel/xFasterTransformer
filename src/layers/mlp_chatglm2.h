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

template <typename WeiT, typename NORM_CLS, bool INPUT_AS_RESID>
class ChatGLM2MLP : public LlamaMLP<WeiT> {
public:
    ChatGLM2MLP(DecoderContext *ctx) : LlamaMLP<WeiT>(ctx) {}

    // The inerface is for PyTorch, thus the weights are already transposed
    void setWeights(DecoderContext *ctx, std::vector<float *> &params, bool trans = true) {
        int hiddenSize = ctx->hiddenSize;
        int intermediateSize = ctx->intermediateSize;

        const float *gate_upW = params[0];
        const float *downW = params[2];
        const float *normW = params[4];

        REQUIRES(ctx->actType == DecoderContext::SWIGLU, "unsupported activation.");

        // Vertically split the gate weight and up weight
        hpj::Matrix<WeiT> convertedGateWeight, convertedUpWeight, convertedDownWeight;

        auto range = SplitUtil::getTaskRange(intermediateSize, ctx->numSplit, ctx->splitIdx);
        int colSplit = range.second - range.first;

        setMLPOPTConfig();
        if (!enableCATMLP) {
            float *gateW = (float *)malloc(hiddenSize * colSplit * sizeof(float));
            float *upW = (float *)malloc(hiddenSize * colSplit * sizeof(float));
            if (trans) {
                int blockSize = colSplit * hiddenSize;
                memcpy(gateW, gate_upW + ctx->splitIdx * blockSize, blockSize * sizeof(float));
                memcpy(upW, gate_upW + intermediateSize * hiddenSize + ctx->splitIdx * blockSize,
                        blockSize * sizeof(float));
            } else {
                const float *weightPTR = gate_upW;
                for (int i = 0; i < hiddenSize; i++) {
                    memcpy(gateW + i * colSplit, weightPTR + ctx->splitIdx * colSplit, colSplit * sizeof(float));
                    weightPTR += intermediateSize;
                    memcpy(upW + i * colSplit, weightPTR + ctx->splitIdx * colSplit, colSplit * sizeof(float));
                    weightPTR += intermediateSize;
                }
            }

            MMHelper::convertWeight(trans, hiddenSize, colSplit, gateW, convertedGateWeight, this->gateWeightScale,
                    this->gateWeightZero, this->gateWeightSum);
            MMHelper::packWeight(trans, convertedGateWeight, this->gateWeight);

            MMHelper::convertWeight(trans, hiddenSize, colSplit, upW, convertedUpWeight, this->upWeightScale,
                    this->upWeightZero, this->upWeightSum);
            MMHelper::packWeight(trans, convertedUpWeight, this->upWeight);

            free(gateW);
            free(upW);

        } else {
            if (trans) {
                printf("Trans GateUpW Not supported yet.\n");
                exit(-1);
            } else {
                int colSplitStride = colSplit * 2;
                float *gateUpW = (float *)malloc(hiddenSize * colSplitStride * sizeof(float));
                const float *weightPTR = gate_upW;
                for (int i = 0; i < hiddenSize; i++) {
                    memcpy(gateUpW + i * colSplitStride, weightPTR + ctx->splitIdx * colSplit,
                            colSplit * sizeof(float));
                    weightPTR += intermediateSize;
                    memcpy(gateUpW + colSplit + i * colSplitStride, weightPTR + ctx->splitIdx * colSplit,
                            colSplit * sizeof(float));
                    weightPTR += intermediateSize;
                }
                hpj::Matrix<WeiT> quantizedCatWeights;
                MMHelper::convertWeight(trans, hiddenSize, colSplitStride, gateUpW, quantizedCatWeights,
                        this->catWeightsScale, this->catWeightsZero, this->catWeightsSum);
                this->catWeights.Resize(quantizedCatWeights.Rows(), quantizedCatWeights.Cols());
                MMHelper::packWeight(trans, quantizedCatWeights, this->catWeights);
                free(gateUpW);
            }
        }
        // Horizontally split the down weight
        if (enableCBLASMLP && std::is_same_v<WeiT, bfloat16_t>) {
            MMHelper::convertWeight(ctx, trans, intermediateSize, hiddenSize, downW, false, this->downWeight,
                    this->downWeightScale, this->downWeightZero, this->gateWeightSum);
        } else {
            MMHelper::convertWeight(ctx, trans, intermediateSize, hiddenSize, downW, false, convertedDownWeight,
                    this->downWeightScale, this->downWeightZero, this->downWeightSum);
            MMHelper::packWeight(trans, convertedDownWeight, this->downWeight);
        }
#ifdef DEBUG
        this->dbg.debugPrint("convertedGateWeight [%d, %d](%d):\n", convertedGateWeight.Rows(),
                convertedGateWeight.Cols(), convertedGateWeight.Stride());
        this->dbg.dumpMatrix(convertedGateWeight);

        this->dbg.debugPrint("packed convertedGateWeight [%d, %d](%d):\n", this->gateWeight.Rows(),
                this->gateWeight.Cols(), this->gateWeight.Stride());
        this->dbg.dumpMatrix(this->gateWeight);

        this->dbg.debugPrint("convertedUpWeight [%d, %d](%d):\n", convertedUpWeight.Rows(), convertedUpWeight.Cols(),
                convertedUpWeight.Stride());
        this->dbg.dumpMatrix(convertedUpWeight);

        this->dbg.debugPrint("packed convertedUpWeight [%d, %d](%d):\n", this->upWeight.Rows(), this->upWeight.Cols(),
                this->upWeight.Stride());
        this->dbg.dumpMatrix(this->upWeight);

        this->dbg.debugPrint("convertedDownWeight [%d, %d](%d):\n", convertedDownWeight.Rows(),
                convertedDownWeight.Cols(), convertedDownWeight.Stride());
        this->dbg.dumpMatrix(convertedDownWeight);

        this->dbg.debugPrint("packed convertedDownWeight [%d, %d](%d):\n", this->downWeight.Rows(),
                this->downWeight.Cols(), this->downWeight.Stride());
        this->dbg.dumpMatrix(this->downWeight);
#endif
        // norm.setWeight(normW, NULL, hiddenSize);
        if (normW) {
            this->normWeight.Resize(hiddenSize);
            memcpy(this->normWeight.Data(), normW, sizeof(float) * hiddenSize);
        }
    }
};
