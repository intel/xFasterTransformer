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

#include "attention.h"

template <typename WeiT, typename QKPO_CLS, typename NORM_CLS, bool INPUT_AS_RESID>
class ChatGLM2Attention : public Attention<WeiT, QKPO_CLS, NORM_CLS, INPUT_AS_RESID> {
public:
    ChatGLM2Attention(int layerId, DecoderContext *ctx)
        : Attention<WeiT, QKPO_CLS, NORM_CLS, INPUT_AS_RESID>(layerId, ctx) {}
    virtual ~ChatGLM2Attention() { }

public:
    template <typename KVCacheT>
    void forward(DecoderContext *ctx, float *input, float *output, const float *attnMask, KVCacheTensor<KVCacheT> &presentKey,
            KVCacheTensor<KVCacheT> &presentValue, int inputSeqLen, int pastSeqLen, bool useSelfAttn, bool doLnBefore,
            bool returnAttn, bool returnKVs, bool forPT = true, int *positionIds = nullptr) {
        if (forPT) {
            printf("For better perf, need to manage cached key/vaues by ourself, PyTorch extension is not supported "
                   "any more.\n");
            exit(-1);
        }

        KVCacheT *presentKeys = presentKey.getData();
        KVCacheT *presentValues = presentValue.getData();

        hpj::Matrix<float> inputBuffer(input, ctx->batchSize * inputSeqLen, ctx->hiddenSize, ctx->hiddenSize);
        hpj::Matrix<float> outBuffer(output, ctx->batchSize * inputSeqLen, ctx->hiddenSize, ctx->hiddenSize);

        auto hiddenSize = ctx->hiddenSize;
        auto &qkvMatMul = ctx->qkvMatMul;
        auto &resultBuffer1 = (ctx->numSplit == 1 ? outBuffer : ctx->normBuf);
        auto &resultBuffer2 = ctx->tmpBuf;
        float epsilon = ctx->epsilon;

        // //init group_qkvBuffer
        int attHeadSize = ctx->attHeadSize;
        int qkvRows = ctx->batchSize * inputSeqLen;
        // multi query attention
        int q_cols = (this->endQHead - this->startQHead) * attHeadSize;
        int kv_cols = (this->endKVHead - this->startKVHead) * attHeadSize;
        int qkCols = q_cols + kv_cols;
        int qkvCols = qkCols + kv_cols;

        int qkvStride = qkvCols;
        hpj::Matrix<float> qkvGroupMatMul(qkvMatMul.Data(), qkvRows, qkvCols, qkvStride);

#ifdef DEBUG
        this->dbg.debugPrint("---- GLM2 DecoderLayer.forward (useSelfAttn=%d) ----\n", useSelfAttn);
        this->dbg.debugPrint("input [%d, %d, %d]:\n", ctx->batchSize * inputSeqLen, ctx->hiddenSize, ctx->hiddenSize);
        this->dbg.dumpMatrix(inputBuffer);
#endif

        if (doLnBefore) {
            TimeLine t1("input.layer_norm");
            this->norm.forward(inputBuffer.Data(), resultBuffer1.Data(), inputBuffer.Rows(), inputBuffer.Stride(),
                    resultBuffer1.Stride(), epsilon);
        }
#ifdef DEBUG
        this->dbg.debugPrint(
                "layer norm [%d, %d, %d]:\n", ctx->batchSize * inputSeqLen, ctx->hiddenSize, ctx->hiddenSize);
        this->dbg.dumpMatrix(resultBuffer1);
#endif
        // Query, Key, Value computed together
        TimeLine t2("QKV.linear");
        DecoderUtil::dense(resultBuffer1, this->qkvWeight, this->qkvWeightScale, this->qkvWeightZero, this->qkvBias,
                qkvGroupMatMul);
        t2.release();

#ifdef DEBUG
        this->dbg.debugPrint("dense [%d, %d, %d]:\n", ctx->batchSize * inputSeqLen, ctx->hiddenSize, qkvCols);
        this->dbg.dumpMatrix(qkvGroupMatMul);
#endif

        // Apply post operattions on query and key
        TimeLine t3("QKPO");
        if (positionIds != nullptr) {
            this->qkpo.forward(qkvGroupMatMul.Data(), qkvGroupMatMul.Stride(), ctx->batchSize, inputSeqLen, qkCols,
                    attHeadSize, positionIds);
        } else {
            std::vector<int> position_ids(ctx->inputSeqLen);
            if (inputSeqLen == 1) {
                position_ids[0] = pastSeqLen;
            } else {
                std::iota(position_ids.begin(), position_ids.end(), 0);
            }
            this->qkpo.forward(qkvGroupMatMul.Data(), qkvGroupMatMul.Stride(), ctx->batchSize, inputSeqLen, qkCols,
                    attHeadSize, position_ids.data());
        }
        t3.release();

#ifdef DEBUG
        this->dbg.debugPrint("qkpo [%d, %d, %d]:\n", ctx->batchSize * inputSeqLen, ctx->hiddenSize, qkvCols);
        this->dbg.dumpMatrix(qkvGroupMatMul);
#endif

        // this->expand_to_qkv(qkvMatMul, qkvGroupMatMul, qkvRows, q_cols, kv_cols, kvHeadNum);
        // printf("q_cols=%d, kv_cols=%d, qk_cols=%d\n", q_cols, kv_cols, qkCols);
        hpj::Matrix<float> query(qkvGroupMatMul, 0, inputBuffer.Rows(), 0, q_cols);
        hpj::Matrix<float> key(qkvGroupMatMul, 0, inputBuffer.Rows(), q_cols, kv_cols);
        hpj::Matrix<float> value(qkvGroupMatMul, 0, inputBuffer.Rows(), qkCols, kv_cols);

#ifdef DEBUG
        this->dbg.debugPrint("Q [%d, %d]:\n", query.Rows(), query.Cols());
        this->dbg.dumpMatrix(query);
        this->dbg.debugPrint("K [%d, %d]:\n", key.Rows(), key.Cols());
        this->dbg.dumpMatrix(key);
        this->dbg.debugPrint("V [%d, %d]:\n", value.Rows(), value.Cols());
        this->dbg.dumpMatrix(value);
#endif

        if (this->getScalingCoeff() != 0) { ctx->attFactor = this->getScalingCoeff(); }
        TimeLine t4("MHA");
        if constexpr (!INPUT_AS_RESID) {
            auto presult = resultBuffer1.Data();
            int rows = resultBuffer1.Rows(), cols = resultBuffer1.Cols(), stride = resultBuffer1.Stride();
            resultBuffer1.Assign(inputBuffer.Data(), inputBuffer.Rows(), inputBuffer.Cols(), inputBuffer.Stride());
            inputBuffer.Assign(presult, rows, cols, stride);
        }
        if (ctx->inputSeqLen > 1024 && pastSeqLen == 0)
            this->flashAttention(
                    ctx, qkvGroupMatMul, resultBuffer2, resultBuffer1, presentKey, presentValue, attnMask, pastSeqLen);
        else
            this->fusedAttention(
                    ctx, query, key, value, resultBuffer1, presentKey, presentValue, attnMask, pastSeqLen);
        t4.release();
        hpj::Matrix<float> attnSplit(resultBuffer1.Data(), resultBuffer1.Rows(), resultBuffer1.Cols() / ctx->numSplit,
                resultBuffer1.Stride());

#ifdef DEBUG
        this->dbg.debugPrint("attention_%d (softmax * value):\n", ctx->splitIdx);
        this->dbg.dumpMatrix(attnSplit);
#endif

        TimeLine t5("Output");
        // Output/projection in attention, only add the input in the first split
        if (ctx->splitIdx == 0) {
            float gamma = this->getResidentialScale();

            // denseWithScaledSum should be enough, but as the performance of denseWithScaledSum is not verified,
            // So here still use denseWithSum
            if (gamma == 1) {
                DecoderUtil::denseWithSum(attnSplit, this->attnOutputWeight, this->attnOutputWeightScale,
                        this->attnOutputWeightZero, this->attnOutputBias, inputBuffer, resultBuffer2);
            } else {
                DecoderUtil::denseWithScaledSum(attnSplit, this->attnOutputWeight, this->attnOutputWeightScale,
                        this->attnOutputWeightZero, this->attnOutputBias, gamma, inputBuffer, resultBuffer2);
            }
        } else {
            DecoderUtil::dense(attnSplit, this->attnOutputWeight, this->attnOutputWeightScale,
                    this->attnOutputWeightZero, this->attnOutputBias, resultBuffer2);
        }
        t5.release();

#ifdef DEBUG
        this->dbg.debugPrint("attention output/projection: [%d, %d] (%d)\n", resultBuffer2.Rows(), resultBuffer2.Cols(),
                resultBuffer2.Stride());
        this->dbg.dumpMatrix(resultBuffer2);
#endif

        if (!doLnBefore) {
            TimeLine t6("result.layer_norm");
            this->norm.forward(resultBuffer2.Data(), resultBuffer1.Data(), resultBuffer2.Rows(), resultBuffer2.Stride(),
                    resultBuffer1.Stride());
        }
    }
};
