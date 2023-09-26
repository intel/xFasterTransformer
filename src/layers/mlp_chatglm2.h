#pragma once
#include <cmath>
#include "mlp_llama.h"

template <typename WeiT, typename NORM_CLS, bool INPUT_AS_RESID>
class ChatGLM2MLP : public LlamaMLP<WeiT> {
public:
    ChatGLM2MLP(DecoderContext *ctx) : LlamaMLP<WeiT>(ctx) { }

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

        int colSplit = intermediateSize / ctx->numSplit;
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

        MMHelper::convertWeight(
                trans, hiddenSize, colSplit, gateW, convertedGateWeight, this->gateWeightScale, this->gateWeightZero);
        MMHelper::packWeight(trans, convertedGateWeight, this->gateWeight);

        MMHelper::convertWeight(trans, hiddenSize, colSplit, upW, convertedUpWeight, this->upWeightScale, this->upWeightZero);
        MMHelper::packWeight(trans, convertedUpWeight, this->upWeight);

        free(gateW);
        free(upW);

        // Horizontally split the down weight
        MMHelper::convertWeight(ctx, trans, intermediateSize, hiddenSize, downW, false, convertedDownWeight,
                this->downWeightScale, this->downWeightZero);
        MMHelper::packWeight(trans, convertedDownWeight, this->downWeight);
#ifdef DEBUG
        this->dbg.debugPrint("convertedGateWeight [%d, %d](%d):\n", convertedGateWeight.Rows(), convertedGateWeight.Cols(),
                convertedGateWeight.Stride());
        this->dbg.dumpMatrix(convertedGateWeight);

        this->dbg.debugPrint("packed convertedGateWeight [%d, %d](%d):\n", this->gateWeight.Rows(), this->gateWeight.Cols(),
                this->gateWeight.Stride());
        this->dbg.dumpMatrix(this->gateWeight);

        this->dbg.debugPrint("convertedUpWeight [%d, %d](%d):\n", convertedUpWeight.Rows(), convertedUpWeight.Cols(),
                convertedUpWeight.Stride());
        this->dbg.dumpMatrix(convertedUpWeight);

        this->dbg.debugPrint("packed convertedUpWeight [%d, %d](%d):\n", this->upWeight.Rows(), this->upWeight.Cols(), this->upWeight.Stride());
        this->dbg.dumpMatrix(this->upWeight);

        this->dbg.debugPrint("convertedDownWeight [%d, %d](%d):\n", convertedDownWeight.Rows(), convertedDownWeight.Cols(),
                convertedDownWeight.Stride());
        this->dbg.dumpMatrix(convertedDownWeight);

        this->dbg.debugPrint("packed convertedDownWeight [%d, %d](%d):\n", this->downWeight.Rows(), this->downWeight.Cols(),
                this->downWeight.Stride());
        this->dbg.dumpMatrix(this->downWeight);
#endif
        // norm.setWeight(normW, NULL, hiddenSize);
        if (normW) {
            this->normWeight.Resize(hiddenSize);
            memcpy(this->normWeight.Data(), normW, sizeof(float) * hiddenSize);
        }
    }
};
