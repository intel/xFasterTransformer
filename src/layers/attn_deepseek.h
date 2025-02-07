// Copyright (c) 2025 Intel Corporation
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

#include "attention.h"
#include "llm_params.h"
#include "logger.h"
#include "my_types.h"
#include "rms_norm.h"

template <typename WeiT, typename InT, typename ImT, typename OutT>
class DeepSeekAttention {
public:
    DeepSeekAttention(int layerId, DecoderContext *ctx) {}

#ifdef XFT_DEBUG
    void setDebugger(const Debugger &debugger) { this->dbg = debugger; }
#endif

    template <typename OriWeiT>
    void setWeights(DecoderContext *ctx, const OriWeiT *queryWeight, const float *queryScale, const float *queryZero,
            const float *queryBias, const OriWeiT *keyWeight, const float *keyScale, const float *keyZero,
            const float *keyBias, const OriWeiT *valueWeight, const float *valueScale, const float *valueZero,
            const float *valueBias, const OriWeiT *attnOutWeight, const float *attnOutScale, const float *attnOutZero,
            const float *attnOutBias, bool doLNorm, const float *gamma1, const float *beta1, bool trans = true) {
        xft::Logger::error("Cannot use the old API to set weights in DeepSeekAttention.");
        exit(-1);
    }

    void setWeights(DecoderContext *ctx, xft::AttnParams *attnParams) {
        xft::MLAttnParams *mlap = dynamic_cast<xft::MLAttnParams *>(attnParams);
        if (mlap == nullptr) {
            xft::Logger::error("Cannot cast AttnParams to MLAttnParams.");
            exit(-1);
        }

        // Suppose the weight is not transposed (report error if it is transposed)
        if (mlap->q_a_proj.wtrans || mlap->kv_a_proj.wtrans) {
            xft::Logger::error("The weights should not be transposed.");
            exit(-1);
        }

        // Check the data type, so that we can safely merge or copy
        if (!isSameWeiType(mlap->q_a_proj.wtype) || !isSameWeiType(mlap->kv_a_proj.wtype)
                || !isSameWeiType(mlap->q_b_proj.wtype) || !isSameWeiType(mlap->kv_b_proj.wtype)) {
            xft::Logger::error("The weight type is not the same.");
            exit(-1);
        }

        // Merge q_a and kv_a
        int mergedDim = mlap->q_a_proj.output_dim + mlap->kv_a_proj.output_dim;
        WeiT *buffer = (WeiT *)aligned_alloc(64, ctx->hiddenSize * mergedDim * sizeof(WeiT));

#pragma omp parallel for
        for (int i = 0; i < ctx->hiddenSize; ++i) {
            memcpy(buffer + i * mergedDim, mlap->q_a_proj.weight + i * mlap->q_a_proj.output_dim,
                    mlap->q_a_proj.output_dim * sizeof(WeiT));
            memcpy(buffer + i * mergedDim + mlap->q_a_proj.output_dim,
                    mlap->kv_a_proj.weight + i * mlap->kv_a_proj.output_dim, mlap->kv_a_proj.output_dim * sizeof(WeiT));
        }

        // Pack the merged weights
        qkvAWeights.Resize(ctx->hiddenSize, mergedDim);
        xft::Matrix mergedW(buffer, ctx->hiddenSize, mergedDim, mergedDim);
        ctx->mmHelper->packWeight(mlap->q_a_proj.wtrans, mergedW, qkvAWeights);

        free(buffer);

        // Pack the weights for q_b and kv_b
        packDenseWeights(ctx, mlap->q_b_proj, qBWeights);
        packDenseWeights(ctx, mlap->kv_b_proj, kvBWeights);

        // Pack the weights for output
        packDenseWeights(ctx, mlap->o_proj, outWeights);

        // Norm params
        this->inputNorm.setWeight(mlap->input_norm.gamma, nullptr, ctx->hiddenSize);
        this->qANorm.setWeight(mlap->q_a_norm.gamma, nullptr, ctx->qLoraRank);
        this->kvANorm.setWeight(mlap->kv_a_norm.gamma, nullptr, ctx->kvLoraRank);
    }

    template <typename KVCacheT>
    void forward(DecoderContext *ctx, InT *input, ImT *imBuf, OutT *output, const float *attnMask,
            KVCacheTensor<KVCacheT> &presentKey, KVCacheTensor<KVCacheT> &presentValue, int inputSeqLen, int pastSeqLen,
            bool useSelfAttn, bool doLnBefore, bool doLnAfter, int *positionIds = nullptr) {
        xft::Logger::error("Cannot use the old API to forward in DeepSeekAttention.");
        exit(-1);
    }

    template <typename KVCacheT>
    void forward(DecoderContext *ctx, std::vector<xft::SequenceMeta *> &seqs, InT *input, OutT *output,
            size_t totInSeqLen, std::vector<KVCacheTensor<KVCacheT> *> &keyCaches,
            std::vector<KVCacheTensor<KVCacheT> *> &valueCaches, bool doLnBefore = true) {
        auto hiddenSize = ctx->hiddenSize;
        xft::Matrix<InT> inputBuffer(input, totInSeqLen, hiddenSize, hiddenSize);

        // Norm buffer
        xft::Matrix<ImT> normBuffer((ImT *)ctx->normBuf.Data(), totInSeqLen, hiddenSize, hiddenSize);

        // Lora buffer (Down projection)
        auto qkvACols = qkvAWeights.Cols();
        ImT *loraBuf = (ImT *)ctx->getBuffer<ImT>("tmp", totInSeqLen * qkvACols, ctx->device);
        xft::Matrix<ImT> loraBuffer(loraBuf, totInSeqLen, qkvACols, qkvACols);

        // Up projection buffer
        // TODO: make sure buffer is big enough
        auto &qkvMatMul = ctx->qkvMatMul;
        auto qCols = ctx->attHeadNum * (ctx->nopeDim + ctx->ropeDim);
        xft::Matrix<ImT> qBuffer((ImT *)qkvMatMul.Data(), totInSeqLen, qCols, qCols);

        auto kvCols = ctx->attHeadNum * (ctx->nopeDim + ctx->vHeadDim);
        uint64_t kvOffset = (uint64_t)qCols * totInSeqLen;
        xft::Matrix<ImT> kvBuffer(qBuffer.Data() + kvOffset, totInSeqLen, kvCols, kvCols);

        // Output buffer
        xft::Matrix<OutT> outBuffer(output, totInSeqLen, hiddenSize, hiddenSize);

        float epsilon = ctx->epsilon;
        int headSize = ctx->attHeadSize;

#ifdef XFT_DEBUG
        dbg.debugPrint("---- DecoderLayer.forward ----\n");
        dbg.debugPrint("input:\n");
        dbg.dumpMatrix(inputBuffer, false, ctx->device);
#endif

        if (doLnBefore) {
            TimeLine t("input.layer_norm");
            inputNorm.forward(inputBuffer.Data(), normBuffer.Data(), inputBuffer.Rows(), inputBuffer.Stride(),
                    normBuffer.Stride(), epsilon);
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("layer norm:\n");
        dbg.dumpMatrix(normBuffer, false, ctx->device);
#endif

        // Apply qkv_down/lora
        {
            TimeLine t("qkv_down_projection");
            ctx->mmHelper->compute(false, normBuffer.Rows(), qkvAWeights.Cols(), normBuffer.Cols(), 1.0f,
                    normBuffer.Data(), normBuffer.Stride(), qkvAWeights.Data(), nullptr, nullptr, nullptr, 0.0f,
                    loraBuffer.Data(), loraBuffer.Stride());
        }

        // Q_A_Norm
        {
            TimeLine t("q_a_norm");
            qANorm.forward(loraBuffer.Data(), loraBuffer.Data(), loraBuffer.Rows(), loraBuffer.Stride(),
                    normBuffer.Stride(), epsilon);
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("q_a_norm:\n");
        dbg.dumpMatrix(loraBuffer, false, ctx->device);
#endif

        // Q up projection
        {
            TimeLine t("q_up_projection");
            ctx->mmHelper->compute(false, loraBuffer.Rows(), qBWeights.Cols(), qBWeights.Rows(), 1.0f, loraBuffer.Data(),
                    loraBuffer.Stride(), qBWeights.Data(), nullptr, nullptr, nullptr, 0.0f, qBuffer.Data(),
                    qBuffer.Stride());
        }

        // Apply rotary embedding
        // ...
    }

private:
    void packDenseWeights(DecoderContext *ctx, xft::DenseLayerParams &dense, xft::Matrix<WeiT> &packedW) {
        xft::Matrix<WeiT> w(dense.weight, dense.input_dim, dense.output_dim, dense.output_dim);
        packedW.Resize(dense.input_dim, dense.output_dim);
        ctx->mmHelper->packWeight(dense.wtrans, w, packedW);
    }

    bool isSameWeiType(xft::ParamType type) {
        if constexpr (std::is_same_v<WeiT, int8_t>) {
            return type == xft::ParamType::Int8;
        } else if constexpr (std::is_same_v<WeiT, float16_t>) {
            return type == xft::ParamType::FP16;
        } else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
            return type == xft::ParamType::BF16;
        } else if constexpr (std::is_same_v<WeiT, float>) {
            return type == xft::ParamType::FP32;
        }
        return false;
    }

private:
    xft::Matrix<WeiT> qkvAWeights; // merged q_a and kv_a
    xft::Matrix<WeiT> qBWeights;
    xft::Matrix<WeiT> kvBWeights;

    xft::Matrix<WeiT> outWeights;

    xft::RmsNorm inputNorm;
    xft::RmsNorm qANorm;
    xft::RmsNorm kvANorm;

#ifdef XFT_DEBUG
    Debugger dbg;
#endif
};