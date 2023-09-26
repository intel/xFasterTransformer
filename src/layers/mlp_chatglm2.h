#pragma once
#include <cmath>
#include "mlp_standard.h"

template <typename WeiT, typename NORM_CLS, bool INPUT_AS_RESID>
class ChatGLM2MLP {
public:
    ChatGLM2MLP(DecoderContext *ctx) {
        //residScale = std::sqrt(2 * ctx->layers);
    }

    // The inerface is for PyTorch, thus the weights are already transposed
    void setWeights(DecoderContext *ctx, std::vector<float *> &params, bool trans = true) {
        int hiddenSize = ctx->hiddenSize;
        int intermediateSize = ctx->intermediateSize;

        // printf("setWeights in ChatGLM2 MLP\n");
        const float *gate_upW = params[0];
        const float *downW = params[2];
        const float *normW = params[4];

        REQUIRES(ctx->actType == DecoderContext::SWIGLU, "unsupported activation.");

        // Vertically split the gate weight and up weight
        hpj::Matrix<WeiT> quantizedGateWeight, quantizedUpWeight, quantizedDownWeight;
        // static void quantizeWeight(bool trans, int rows, int cols, const float *src, int numSplit, int splitIdx,
        //    bool verticalSplit, hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float16_t> &scaleWeight, hpj::Vector<float16_t> &zeroWeight)
        MMHelper::convertWeight(trans, hiddenSize, intermediateSize * 2, gate_upW, 2, 0, true, quantizedGateWeight,
                gateWeightScale, gateWeightZero);
        MMHelper::packWeight(trans, quantizedGateWeight, gateWeight);

        MMHelper::convertWeight(trans, hiddenSize, intermediateSize * 2, gate_upW, 2, 1, true, quantizedUpWeight,
                upWeightScale, upWeightZero);
        MMHelper::packWeight(trans, quantizedUpWeight, upWeight);

        // Horizontally split the down weight
        MMHelper::convertWeight(ctx, trans, intermediateSize, hiddenSize, downW, false, quantizedDownWeight,
                downWeightScale, downWeightZero);
        MMHelper::packWeight(trans, quantizedDownWeight, downWeight);

        // gamma and beta for layer norm
        // this->gamma2.Resize(hiddenSize);
        // memcpy(this->gamma2.Data(), _gamma2, sizeof(float) * hiddenSize);
        norm.setWeight(normW, NULL, hiddenSize);
    }

#ifdef DEBUG
    void setDebugger(const Debugger &debugger) { this->dbg = debugger; }
#endif
    // Forward for FFN (Feed Forward Network)
    void forward(DecoderContext *ctx, float *input, float *output, int iStride, int oStride, bool doLnBefore,
            bool forPT = true) {
        // Forward for FFN (Feed Forward Network)
        const int M = ctx->batchSize * ctx->inputSeqLen;
        const int hiddenSize = ctx->hiddenSize;
        const float epsilon = ctx->epsilon;

        hpj::Matrix<float> inBuffer(input, M, hiddenSize, iStride);
        hpj::Matrix<float> outBuffer(output, M, hiddenSize, oStride);
        auto &normBuffer = ctx->normBuf;
        auto &imBuffer = ctx->imOut;

        // norm.forward(inBuffer, normBuffer, normWeight, 1e-6);
        norm.forward(
                inBuffer.Data(), normBuffer.Data(), inBuffer.Rows(), inBuffer.Stride(), normBuffer.Stride(), epsilon);

#ifdef DEBUG
        dbg.debugPrint("LayerNorm before MLP:\n");
        dbg.dumpMatrix(normBuffer);
#endif
        //first part of dense_1_to_4
        gateProj(normBuffer, imBuffer);

#ifdef DEBUG
        dbg.debugPrint("gate output:\n");
        dbg.dumpMatrix(imBuffer);
#endif

        upProj(normBuffer, imBuffer);

#ifdef DEBUG
        dbg.debugPrint("up output:\n");
        dbg.dumpMatrix(imBuffer);
#endif

        // downProj(imBuffer, outBuffer, inBuffer);
        downProj(imBuffer, outBuffer, inBuffer, ctx->splitIdx == 0);

#ifdef DEBUG
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

private:
    hpj::Matrix<WeiT> gateWeight;
    hpj::Vector<float> gateWeightScale; // For int8_t weight
    hpj::Vector<float> gateWeightZero; // For int8_t weight
    hpj::Matrix<WeiT> upWeight;
    hpj::Vector<float> upWeightScale; // For int8_t weight
    hpj::Vector<float> upWeightZero; // For int8_t weight
    hpj::Matrix<WeiT> downWeight;
    hpj::Vector<float> downWeightScale; // For int8_t weight
    hpj::Vector<float> downWeightZero; // For int8_t weight

    NORM_CLS norm;
#ifdef DEBUG
    Debugger dbg;
#endif
};