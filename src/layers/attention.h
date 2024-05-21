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
#include <numeric>

#include "aligned_type.h"
#include "attention_kernels.h"
#include "bfloat16.h"
#include "copy_util.h"
#include "debugger.h"
#include "decoder_util.h"
#include "float16.h"
#include "gemm_kernel_ext.h"
#include "kvcache_tensor.h"
#include "matmul_helper.h"
#include "sequence.h"
#include "simple_mem_pool.h"
#include "transformer_ctx.h"
#include "transformer_util.h"

#include "rotary_embedding.h"

/**
 * WeiT: weight data type
 * InT: input data type
 * ImT: intermediate data type
 * OutT: output data type
 * QKPO_CLS: class for post operation of query/key, it is generally the rotary embedding
 * NORM_CLS: class for layernorm or other norms
 * INPUT_AS_RESID: input as residential or not, most models use input as residential,
 *                 but there are exceptions like ChatGLM use values after layernorm as residential
*/
template <typename WeiT, typename QKPO_CLS, typename NORM_CLS, typename InT = float, typename ImT = float,
        typename OutT = float, bool INPUT_AS_RESID = true>
class Attention {
public:
    Attention(int layerId, DecoderContext *ctx) : layerId(layerId), qkpo(ctx->attHeadSize, ctx->maxPosEmbed) {

        //todo(marvin): clear this code after all rotary_emb refactor
        if constexpr (std::is_same<QKPO_CLS, LlamaRotaryEmbedding>::value) { qkpo = LlamaRotaryEmbedding(ctx); }

        // Group attention or multi-head attention (multi-head attn is a special case of group attn)
        if (ctx->attHeadNum % ctx->kvHeadNum == 0) {
            // We are responsible for the range [startQHead, endQHead)
            auto range = getTaskRange(ctx->attHeadNum, ctx->numSplit, ctx->splitIdx);
            this->startQHead = range.first;
            this->endQHead = range.second;

            int expandFactor = ctx->attHeadNum / ctx->kvHeadNum;
            this->startKVHead = startQHead / expandFactor;
            this->endKVHead = (this->endQHead - 1) / expandFactor + 1;
        }

        // Unexpected case
        else {
            printf("Not supported yet: QHeads=%d, KVHeads=%d\n", ctx->attHeadNum, ctx->kvHeadNum);
            exit(-1);
        }

        alibiSlopes = nullptr;
    }

    // The inerface is for PyTorch, thus the weights are already transposed
    // OriWeiT: float, int8_t or uint4x2_t
    template <typename OriWeiT>
    void setWeights(DecoderContext *ctx, const OriWeiT *queryWeight, const float *queryScale, const float *queryZero,
            const float *queryBias, const OriWeiT *keyWeight, const float *keyScale, const float *keyZero,
            const float *keyBias, const OriWeiT *valueWeight, const float *valueScale, const float *valueZero,
            const float *valueBias, const OriWeiT *attnOutWeight, const float *attnOutScale, const float *attnOutZero,
            const float *attnOutBias, bool doLNorm, const float *gamma1, const float *beta1, bool trans = true) {
        int hiddenSize = ctx->hiddenSize;
        int headSize = ctx->attHeadSize;

        // Merged weights, dimension is like: hiddenSize * (hiddenSize + 2 * kvHiddenSize)
        // Vertically split the QKV weights
        int qResponsibleCols = (this->endQHead - this->startQHead) * headSize;
        int kvResponsibleCols = (this->endKVHead - this->startKVHead) * headSize;
        int responsibleCols = qResponsibleCols + 2 * kvResponsibleCols;
        qkvWeight.Resize(hiddenSize, responsibleCols);

        constexpr int sizeFactor = std::is_same_v<OriWeiT, uint4x2_t> ? 2 : 1;

        OriWeiT *concatBuf = (OriWeiT *)malloc(hiddenSize * responsibleCols * sizeof(OriWeiT) / sizeFactor);
        if (trans) {
            memcpy(concatBuf, queryWeight + this->startQHead * headSize * hiddenSize / sizeFactor,
                    hiddenSize * qResponsibleCols * sizeof(OriWeiT) / sizeFactor);
            memcpy(concatBuf + hiddenSize * qResponsibleCols / sizeFactor,
                    keyWeight + this->startKVHead * headSize * hiddenSize / sizeFactor,
                    hiddenSize * kvResponsibleCols * sizeof(OriWeiT) / sizeFactor);
            memcpy(concatBuf + hiddenSize * (qResponsibleCols + kvResponsibleCols) / sizeFactor,
                    valueWeight + this->startKVHead * headSize * hiddenSize / sizeFactor,
                    hiddenSize * kvResponsibleCols * sizeof(OriWeiT) / sizeFactor);
        } else {
            int qkvStride = (ctx->attHeadNum + ctx->kvHeadNum + ctx->kvHeadNum) * ctx->attHeadSize;
#pragma omp parallel for
            for (int i = 0; i < hiddenSize; ++i) {
                memcpy(concatBuf + i * responsibleCols / sizeFactor,
                        queryWeight + i * qkvStride / sizeFactor + this->startQHead * headSize / sizeFactor,
                        qResponsibleCols * sizeof(OriWeiT) / sizeFactor);
                memcpy(concatBuf + i * responsibleCols / sizeFactor + qResponsibleCols / sizeFactor,
                        keyWeight + i * qkvStride / sizeFactor + this->startKVHead * headSize / sizeFactor,
                        kvResponsibleCols * sizeof(OriWeiT) / sizeFactor);
                memcpy(concatBuf + i * responsibleCols / sizeFactor + qResponsibleCols / sizeFactor
                                + kvResponsibleCols / sizeFactor,
                        valueWeight + i * qkvStride / sizeFactor + this->startKVHead * headSize / sizeFactor,
                        kvResponsibleCols * sizeof(OriWeiT) / sizeFactor);
            }
        }
        float *concatScale = nullptr;
        float *concatZero = nullptr;
        if constexpr (std::is_same_v<OriWeiT, int8_t> || std::is_same_v<OriWeiT, uint4x2_t>) {
            concatScale = (float *)malloc(responsibleCols * sizeof(float));
            concatZero = (float *)malloc(responsibleCols * sizeof(float));
            memcpy(concatScale, queryScale + this->startQHead * headSize, qResponsibleCols * sizeof(float));
            memcpy(concatScale + qResponsibleCols, keyScale + this->startKVHead * headSize,
                    kvResponsibleCols * sizeof(float));
            memcpy(concatScale + qResponsibleCols + kvResponsibleCols, valueScale + this->startKVHead * headSize,
                    kvResponsibleCols * sizeof(float));
            memcpy(concatZero, queryZero + this->startQHead * headSize, qResponsibleCols * sizeof(float));
            memcpy(concatZero + qResponsibleCols, keyZero + this->startKVHead * headSize,
                    kvResponsibleCols * sizeof(float));
            memcpy(concatZero + qResponsibleCols + kvResponsibleCols, valueZero + this->startKVHead * headSize,
                    kvResponsibleCols * sizeof(float));
        }

        xft::Matrix<WeiT> convertedqkvWeight;
        ctx->mmHelper->convertWeight(trans, hiddenSize, responsibleCols, concatBuf, concatScale, concatZero,
                convertedqkvWeight, qkvWeightScale, qkvWeightZero, qkvWeightSum);
        ctx->mmHelper->packWeight(trans, convertedqkvWeight, qkvWeight);

        free(concatBuf);
        free(concatScale);
        free(concatZero);

#ifdef DEBUG
        dbg.debugPrint("attention qkv weight: [%d, %d] (%d)\n", convertedqkvWeight.Rows(), convertedqkvWeight.Cols(),
                convertedqkvWeight.Stride());
        dbg.dumpMatrix(convertedqkvWeight);
        dbg.debugPrint(
                "attention qkv packed weight: [%d, %d] (%d)\n", qkvWeight.Rows(), qkvWeight.Cols(), qkvWeight.Stride());
        dbg.dumpMatrix(qkvWeight);
#endif

        // Merged bias
        if (queryBias && keyBias && valueBias) {
            qkvBias.Resize(responsibleCols);
            memcpy(qkvBias.Data(), queryBias + ctx->splitIdx * qResponsibleCols, sizeof(float) * qResponsibleCols);
            memcpy(qkvBias.Data() + qResponsibleCols, keyBias + this->startKVHead * headSize,
                    sizeof(float) * kvResponsibleCols);
            memcpy(qkvBias.Data() + qResponsibleCols + kvResponsibleCols, valueBias + this->startKVHead * headSize,
                    sizeof(float) * kvResponsibleCols);
        }

        // Weights for attention output
        // Horizontally split the weight, as the source (PyTorch weight) is transposed, thus looks like vertically
        xft::Matrix<WeiT> convertedWeight;
        ctx->mmHelper->convertWeight(trans, ctx->attHeadNum * ctx->attHeadSize, hiddenSize, attnOutWeight, attnOutScale,
                attnOutZero, this->startQHead * headSize, qResponsibleCols, false, convertedWeight,
                attnOutputWeightScale, attnOutputWeightZero, attnOutputWeightSum, true);
        ctx->mmHelper->packWeight(trans, convertedWeight, attnOutputWeight);

#ifdef DEBUG
        dbg.debugPrint(">>> attention output weight: [%d, %d] (%d)\n", convertedWeight.Rows(), convertedWeight.Cols(),
                convertedWeight.Stride());
        dbg.dumpMatrix(convertedWeight);
        dbg.debugPrint("attention output packed weight: [%d, %d] (%d)\n", attnOutputWeight.Rows(),
                attnOutputWeight.Cols(), attnOutputWeight.Stride());
        dbg.dumpMatrix(attnOutputWeight);
#endif

        // Attention output bias
        if (attnOutBias) {
            this->attnOutputBias.Resize(hiddenSize);
            if (ctx->splitIdx == 0) {
                memcpy(this->attnOutputBias.Data(), attnOutBias, sizeof(float) * hiddenSize);
            } else { // For other splits, set bias to 0, to avoid duplicated calculation
                memset(this->attnOutputBias.Data(), 0, sizeof(float) * hiddenSize);
            }
        }

        // LayerNorm
        if (doLNorm) this->norm.setWeight(gamma1, beta1, hiddenSize);
    }

#ifdef DEBUG
    void setDebugger(const Debugger &debugger) { this->dbg = debugger; }
#endif

    /**
     * Forward computing for the whole encoder/decoder layer
     * Inputs:
     * - input: (bs * seq_len) x hidden_size (input buffer)
     * - imBuf: (bs * seq_len) x hidden_size (intermediate buffer)
     * - output: (bs * seq_len) x hidden_size (output buffer)
     * - attnMask: (bs, 1, tgt_len, src_len) (tgt_len is the length of query, src_len is the length of key)
     * - presentKeys, presentValues: past key/values concats current key/values
     * - pastSeqLen: the sequence length in pastKeys and pastValues
     * - useSelfAttn: use self attention or not, self attention is used to gen first token
     * - doLnBefore: Do layer norm before or not. If true, will do layer norm as the first step
     *               currently only support doLnBefore=true
     * - doLnAfter: Do layer norm before or not. If true, will do layer norm as the first step
     *  _________                _________                _________                _________                _________                
     * |_________|------------->|_________|------------->|_________|------------->|_________|------------->|_________|
     *              layerNorm                QKV Linear                  MHA                   out Linear             
     *    input                   imBuffer                qkvMatMul                 imBuffer                  output
    */
    template <typename KVCacheT>
    void forward(DecoderContext *ctx, InT *input, ImT *imBuf, OutT *output, const float *attnMask,
            KVCacheTensor<KVCacheT> &presentKey, KVCacheTensor<KVCacheT> &presentValue, int inputSeqLen, int pastSeqLen,
            bool useSelfAttn, bool doLnBefore, bool doLnAfter, int *positionIds = nullptr) {

        auto hiddenSize = ctx->hiddenSize;
        xft::Matrix<InT> inputBuffer(input, ctx->batchSize * inputSeqLen, hiddenSize, hiddenSize);
        xft::Matrix<ImT> imBuffer(imBuf, ctx->batchSize * inputSeqLen, hiddenSize, hiddenSize);
        xft::Matrix<OutT> outBuffer(output, ctx->batchSize * inputSeqLen, hiddenSize, hiddenSize);

        float epsilon = ctx->epsilon;
        int headSize = ctx->attHeadSize;
        int qkvRows = ctx->batchSize * inputSeqLen;
        int qCols = (this->endQHead - this->startQHead) * headSize;
        int kvCols = (this->endKVHead - this->startKVHead) * headSize;
        int qkCols = qCols + kvCols;
        int qkvCols = qkCols + kvCols;

        int qkvStride = qkvCols;
        auto &qkvMatMul = ctx->qkvMatMul;
        xft::Matrix<ImT> qkvGroupMatMul((ImT *)qkvMatMul.Data(), qkvRows, qkvCols, qkvStride);

#ifdef DEBUG
        dbg.debugPrint("---- DecoderLayer.forward (useSelfAttn=%d) ----\n", useSelfAttn);
        dbg.debugPrint("input:\n");
        dbg.dumpMatrix(inputBuffer);
#endif

        if (doLnBefore) {
            TimeLine t1("input.layer_norm");
            norm.forward(inputBuffer.Data(), imBuffer.Data(), inputBuffer.Rows(), inputBuffer.Stride(),
                    imBuffer.Stride(), epsilon);
        }
#ifdef DEBUG
        dbg.debugPrint("layer norm:\n");
        dbg.dumpMatrix(imBuffer);
        dbg.debugPrint("qkvWeight [%d, %d]:\n", this->qkvWeight.Rows(), this->qkvWeight.Cols());
        dbg.dumpMatrix(this->qkvWeight);
#endif

        // Query, Key, Value computed together
        TimeLine t2("QKV.linear");
        if (qkvBias.Size() == 0) {
            ctx->mmHelper->compute(false, imBuffer.Rows(), qkvWeight.Cols(), imBuffer.Cols(), 1.0f, imBuffer.Data(),
                    imBuffer.Stride(), qkvWeight.Data(), qkvWeightScale.Data(), qkvWeightZero.Data(),
                    qkvWeightSum.Data(), 0.0f, qkvGroupMatMul.Data(), qkvGroupMatMul.Stride());
        } else {
            ctx->mmHelper->compute_bias(false, imBuffer.Rows(), qkvWeight.Cols(), imBuffer.Cols(), 1.0f,
                    imBuffer.Data(), imBuffer.Stride(), qkvWeight.Data(), qkvWeightScale.Data(), qkvWeightZero.Data(),
                    qkvWeightSum.Data(), 0.0f, qkvGroupMatMul.Data(), qkvGroupMatMul.Stride(), qkvBias.Data());
        }
        t2.release();

        xft::Matrix<ImT> query(qkvGroupMatMul, 0, inputBuffer.Rows(), 0, qCols);
        xft::Matrix<ImT> key(qkvGroupMatMul, 0, inputBuffer.Rows(), qCols, kvCols);
        xft::Matrix<ImT> value(qkvGroupMatMul, 0, inputBuffer.Rows(), qkCols, kvCols);

#ifdef DEBUG
        dbg.debugPrint("Q[%d,%d](%d):\n", query.Rows(), query.Cols(), query.Stride());
        dbg.dumpMatrix(query);
        dbg.debugPrint("K[%d,%d](%d):\n", key.Rows(), key.Cols(), key.Stride());
        dbg.dumpMatrix(key);
        dbg.debugPrint("V[%d,%d](%d):\n", value.Rows(), value.Cols(), value.Stride());
        dbg.dumpMatrix(value);
#endif

        // Apply post operations on query and key
        TimeLine t3("QKPO");
        int qheads = this->endQHead - this->startQHead;
        int kheads = this->endKVHead - this->startKVHead;
        int qkShape[7] = {ctx->batchSize, ctx->inputSeqLen, qheads, headSize, kheads, ctx->maxSeqLength, pastSeqLen};
        if (positionIds != nullptr) {
            qkpo.forward(query.Data(), key.Data(), query.Stride(), key.Stride(), qkShape, positionIds);
        } else if (ctx->maxPosEmbed > 0) {
            // Use the default position ids
            std::vector<int> posIds(ctx->inputSeqLen);
            if (inputSeqLen == 1) {
                posIds[0] = pastSeqLen;
            } else {
                std::iota(posIds.begin(), posIds.end(), pastSeqLen);
            }
            qkpo.forward(query.Data(), key.Data(), query.Stride(), key.Stride(), qkShape, posIds.data());
        }
        t3.release();

#ifdef DEBUG
        dbg.debugPrint("Q[%d,%d](%d) after post op:\n", query.Rows(), query.Cols(), query.Stride());
        dbg.dumpMatrix(query);
        dbg.debugPrint("K[%d,%d](%d) after post op:\n", key.Rows(), key.Cols(), key.Stride());
        dbg.dumpMatrix(key);
#endif

        // Revise attnFactor before softmax (for some models, attnFactor may be not the default value)
        // We initially introduced the code for ChatGLM, but eventually found it has no difference and was unnecessary.
        // However, we have chosen to keep it in the codebase in case it becomes useful for future models.
        if (getScalingCoeff() != 0) { ctx->attFactor = getScalingCoeff(); }

        TimeLine t4("MHA");
        if constexpr (!INPUT_AS_RESID) { // Swap inputBuffer and imBuffer
            auto tmp = imBuffer.Data();
            int rows = imBuffer.Rows(), cols = imBuffer.Cols(), stride = imBuffer.Stride();
            imBuffer.Assign(inputBuffer.Data(), inputBuffer.Rows(), inputBuffer.Cols(), inputBuffer.Stride());
            inputBuffer.Assign(tmp, rows, cols, stride);
        }

        // For multiple nodes inference, not the whole result buffer
        xft::Matrix<ImT> attnSplit(imBuffer.Data(), imBuffer.Rows(), qCols, qCols);

        if (pastSeqLen == 0) {
            if (ctx->inputSeqLen > getFlashThresh()) {
                flashAttention(ctx, query, key, value, attnSplit, presentKey, presentValue, attnMask, pastSeqLen);
            } else if constexpr (std::is_same_v<InT, bfloat16_t> && std::is_same_v<OutT, bfloat16_t>) {
                selfAttentionBF16(ctx, query, key, value, attnSplit, presentKey, presentValue);
            } else {
                fusedAttention(ctx, query, key, value, attnSplit, presentKey, presentValue, attnMask, pastSeqLen);
            }
        } else {
            fusedAttention(ctx, query, key, value, attnSplit, presentKey, presentValue, attnMask, pastSeqLen);
        }
        t4.release();

#ifdef DEBUG
        dbg.debugPrint(">>> attention_%d (softmax * value): [%d, %d] (%d)\n", ctx->splitIdx, attnSplit.Rows(),
                attnSplit.Cols(), attnSplit.Stride());
        dbg.dumpMatrix(attnSplit);
#endif

        TimeLine t5("Output");
        // Output/projection in attention, only add the input in the first split
        if (ctx->splitIdx == 0) {
            float gamma = getResidentialScale();

            // denseWithScaledSum should be enough, but as the performance of denseWithScaledSum is not verified,
            // So here still use denseWithSum
            if (gamma == 1) {
                float *pbias = attnOutputBias.Data();
                if (attnOutputBias.Size() == 0) { pbias = nullptr; }
                ctx->mmHelper->compute_residential(false, attnSplit.Rows(), attnOutputWeight.Cols(), attnSplit.Cols(),
                        1.0f, attnSplit.Data(), attnSplit.Stride(), attnOutputWeight.Data(),
                        attnOutputWeightScale.Data(), attnOutputWeightZero.Data(), attnOutputWeightSum.Data(), 0.0f,
                        outBuffer.Data(), outBuffer.Stride(), pbias, inputBuffer.Data(), inputBuffer.Stride());
            } else {
                float *pbias = attnOutputBias.Data();
                if (attnOutputBias.Size() == 0) { pbias = nullptr; }
                ctx->mmHelper->compute_resext(false, attnSplit.Rows(), attnOutputWeight.Cols(), attnSplit.Cols(), 1.0f,
                        attnSplit.Data(), attnSplit.Stride(), attnOutputWeight.Data(), attnOutputWeightScale.Data(),
                        attnOutputWeightZero.Data(), attnOutputWeightSum.Data(), 0.0f, outBuffer.Data(),
                        outBuffer.Stride(), pbias, gamma, inputBuffer.Data(), inputBuffer.Stride());
            }
        } else {
            if (attnOutputBias.Size() == 0) {
                ctx->mmHelper->compute(false, attnSplit.Rows(), attnOutputWeight.Cols(), attnSplit.Cols(), 1.0f,
                        attnSplit.Data(), attnSplit.Stride(), attnOutputWeight.Data(), attnOutputWeightScale.Data(),
                        attnOutputWeightZero.Data(), attnOutputWeightSum.Data(), 0.0f, outBuffer.Data(),
                        outBuffer.Stride());
            } else {
                ctx->mmHelper->compute_bias(false, attnSplit.Rows(), attnOutputWeight.Cols(), attnSplit.Cols(), 1.0f,
                        attnSplit.Data(), attnSplit.Stride(), attnOutputWeight.Data(), attnOutputWeightScale.Data(),
                        attnOutputWeightZero.Data(), attnOutputWeightSum.Data(), 0.0f, outBuffer.Data(),
                        outBuffer.Stride(), attnOutputBias.Data());
            }
        }
        t5.release();

#ifdef DEBUG
        dbg.debugPrint(">>> attention output/projection[%d, %d] (%d):\n", outBuffer.Rows(), outBuffer.Cols(),
                outBuffer.Stride());
        dbg.dumpMatrix(outBuffer);
#endif

        if (doLnAfter) {
            TimeLine t6("result.layer_norm");
            norm.forward(outBuffer.Data(), outBuffer.Data(), outBuffer.Rows(), outBuffer.Stride(), outBuffer.Stride());
#ifdef DEBUG
            dbg.debugPrint("LayerNorm after attention: [%d, %d] (%d)\n", outBuffer.Rows(), outBuffer.Cols(),
                    outBuffer.Stride());
            dbg.dumpMatrix(outBuffer);
#endif
        }
    }

    /**
     * Forward computing for the whole Attention layer (QKV MatMul + MHA/GQA + Output MatMul)
     */
    template <typename KVCacheT>
    void forward(DecoderContext *ctx, std::vector<xft::SequenceMeta *> &seqs, InT *input, OutT *output,
            size_t totInSeqLen, std::vector<KVCacheTensor<KVCacheT> *> &keyCaches,
            std::vector<KVCacheTensor<KVCacheT> *> &valueCaches, bool doLnBefore = true) {

        auto hiddenSize = ctx->hiddenSize;
        xft::Matrix<InT> inputBuffer(input, totInSeqLen, hiddenSize, hiddenSize);
        ImT *imBuf = (ImT *)ctx->getBuffer<ImT>("tmp", totInSeqLen * hiddenSize);
        xft::Matrix<ImT> imBuffer(imBuf, totInSeqLen, hiddenSize, hiddenSize);
        xft::Matrix<OutT> outBuffer(output, totInSeqLen, hiddenSize, hiddenSize);

        float epsilon = ctx->epsilon;
        int headSize = ctx->attHeadSize;
        auto qkvRows = totInSeqLen;
        int qCols = (this->endQHead - this->startQHead) * headSize;
        int kvCols = (this->endKVHead - this->startKVHead) * headSize;
        int qkCols = qCols + kvCols;
        int qkvCols = qkCols + kvCols;

        int qkvStride = qkvCols;
        auto &qkvMatMul = ctx->qkvMatMul;
        xft::Matrix<ImT> qkvGroupMatMul((ImT *)qkvMatMul.Data(), qkvRows, qkvCols, qkvStride);

#ifdef DEBUG
        dbg.debugPrint("---- DecoderLayer.forward ----\n");
        dbg.debugPrint("input:\n");
        dbg.dumpMatrix(inputBuffer);
#endif

        if (doLnBefore) {
            TimeLine t1("input.layer_norm");
            norm.forward(inputBuffer.Data(), imBuffer.Data(), inputBuffer.Rows(), inputBuffer.Stride(),
                    imBuffer.Stride(), epsilon);
        }
#ifdef DEBUG
        dbg.debugPrint("layer norm:\n");
        dbg.dumpMatrix(imBuffer);
        dbg.debugPrint("qkvWeight [%d, %d]:\n", this->qkvWeight.Rows(), this->qkvWeight.Cols());
        dbg.dumpMatrix(this->qkvWeight);
#endif

        // Query, Key, Value computed together
        TimeLine t2("QKV.linear");
        if (qkvBias.Size() == 0) {
            ctx->mmHelper->compute(false, imBuffer.Rows(), qkvWeight.Cols(), imBuffer.Cols(), 1.0f, imBuffer.Data(),
                    imBuffer.Stride(), qkvWeight.Data(), qkvWeightScale.Data(), qkvWeightZero.Data(),
                    qkvWeightSum.Data(), 0.0f, qkvGroupMatMul.Data(), qkvGroupMatMul.Stride());
        } else {
            ctx->mmHelper->compute_bias(false, imBuffer.Rows(), qkvWeight.Cols(), imBuffer.Cols(), 1.0f,
                    imBuffer.Data(), imBuffer.Stride(), qkvWeight.Data(), qkvWeightScale.Data(), qkvWeightZero.Data(),
                    qkvWeightSum.Data(), 0.0f, qkvGroupMatMul.Data(), qkvGroupMatMul.Stride(), qkvBias.Data());
        }
        t2.release();

        xft::Matrix<ImT> query(qkvGroupMatMul, 0, inputBuffer.Rows(), 0, qCols);
        xft::Matrix<ImT> key(qkvGroupMatMul, 0, inputBuffer.Rows(), qCols, kvCols);
        xft::Matrix<ImT> value(qkvGroupMatMul, 0, inputBuffer.Rows(), qkCols, kvCols);

#ifdef DEBUG
        dbg.debugPrint("Q[%d,%d](%d):\n", query.Rows(), query.Cols(), query.Stride());
        dbg.dumpMatrix(query);
        dbg.debugPrint("K[%d,%d](%d):\n", key.Rows(), key.Cols(), key.Stride());
        dbg.dumpMatrix(key);
        dbg.debugPrint("V[%d,%d](%d):\n", value.Rows(), value.Cols(), value.Stride());
        dbg.dumpMatrix(value);
#endif

        // Apply post operations on query and key
        TimeLine t3("QKPO");
        if (ctx->maxPosEmbed > 0) {
            int qheads = this->endQHead - this->startQHead;
            int kheads = this->endKVHead - this->startKVHead;
            int totInputSeqLen = 0;
            for (auto seq : seqs) {
                totInputSeqLen += seq->getInputSeqLen();
            }
            // Use the default position ids
            std::vector<int> posIds(totInputSeqLen);
            int loc = 0;
            for (auto seq : seqs) {
                std::iota(posIds.begin() + loc, posIds.begin() + loc + seq->getInputSeqLen(), seq->getPastSeqLen());
                loc += seq->getInputSeqLen();
            }
            qkpo.forward(query.Data(), key.Data(), totInputSeqLen, query.Stride(), key.Stride(), qheads, kheads,
                    posIds.data());
        }
        t3.release();

#ifdef DEBUG
        dbg.debugPrint("Q[%d,%d](%d) after post op:\n", query.Rows(), query.Cols(), query.Stride());
        dbg.dumpMatrix(query);
        dbg.debugPrint("K[%d,%d](%d) after post op:\n", key.Rows(), key.Cols(), key.Stride());
        dbg.dumpMatrix(key);
#endif

        // Revise attnFactor before softmax (for some models, attnFactor may be not the default value)
        // We initially introduced the code for ChatGLM, but eventually found it has no difference and was unnecessary.
        // However, we have chosen to keep it in the codebase in case it becomes useful for future models.
        if (getScalingCoeff() != 0) { ctx->attFactor = getScalingCoeff(); }

        TimeLine t4("MHA");
        if constexpr (!INPUT_AS_RESID) { // Swap inputBuffer and imBuffer
            auto tmp = imBuffer.Data();
            int rows = imBuffer.Rows(), cols = imBuffer.Cols(), stride = imBuffer.Stride();
            imBuffer.Assign(inputBuffer.Data(), inputBuffer.Rows(), inputBuffer.Cols(), inputBuffer.Stride());
            inputBuffer.Assign(tmp, rows, cols, stride);
        }

        // For multiple nodes inference, not the whole result buffer
        xft::Matrix<ImT> attnSplit(imBuffer.Data(), imBuffer.Rows(), qCols, qCols);

        if (seqs[0]->getStep() == 0) { // First token generation
            if (totInSeqLen > getFlashThresh() * seqs.size()) {
                flashAttention(ctx, query, key, value, attnSplit, keyCaches, valueCaches, seqs);
            } else if constexpr (std::is_same_v<InT, bfloat16_t> && std::is_same_v<OutT, bfloat16_t>) {
                selfAttentionBF16(ctx, query, key, value, attnSplit, keyCaches, valueCaches, seqs);
            } else {
                fusedAttention(ctx, query, key, value, attnSplit, keyCaches, valueCaches, seqs);
            }
        } else {
            fusedAttention(ctx, query, key, value, attnSplit, keyCaches, valueCaches, seqs);
        }
        t4.release();

#ifdef DEBUG
        dbg.debugPrint(">>> attention_%d (softmax * value): [%d, %d] (%d)\n", ctx->splitIdx, attnSplit.Rows(),
                attnSplit.Cols(), attnSplit.Stride());
        dbg.dumpMatrix(attnSplit);
#endif

        TimeLine t5("Output");
        // Output/projection in attention, only add the input in the first split
        if (ctx->splitIdx == 0) {
            float gamma = getResidentialScale();

            // denseWithScaledSum should be enough, but as the performance of denseWithScaledSum is not verified,
            // So here still use denseWithSum
            if (gamma == 1) {
                float *pbias = attnOutputBias.Data();
                if (attnOutputBias.Size() == 0) { pbias = nullptr; }
                ctx->mmHelper->compute_residential(false, attnSplit.Rows(), attnOutputWeight.Cols(), attnSplit.Cols(),
                        1.0f, attnSplit.Data(), attnSplit.Stride(), attnOutputWeight.Data(),
                        attnOutputWeightScale.Data(), attnOutputWeightZero.Data(), attnOutputWeightSum.Data(), 0.0f,
                        outBuffer.Data(), outBuffer.Stride(), pbias, inputBuffer.Data(), inputBuffer.Stride());
            } else {
                float *pbias = attnOutputBias.Data();
                if (attnOutputBias.Size() == 0) { pbias = nullptr; }
                ctx->mmHelper->compute_resext(false, attnSplit.Rows(), attnOutputWeight.Cols(), attnSplit.Cols(), 1.0f,
                        attnSplit.Data(), attnSplit.Stride(), attnOutputWeight.Data(), attnOutputWeightScale.Data(),
                        attnOutputWeightZero.Data(), attnOutputWeightSum.Data(), 0.0f, outBuffer.Data(),
                        outBuffer.Stride(), pbias, gamma, inputBuffer.Data(), inputBuffer.Stride());
            }
        } else {
            if (attnOutputBias.Size() == 0) {
                ctx->mmHelper->compute(false, attnSplit.Rows(), attnOutputWeight.Cols(), attnSplit.Cols(), 1.0f,
                        attnSplit.Data(), attnSplit.Stride(), attnOutputWeight.Data(), attnOutputWeightScale.Data(),
                        attnOutputWeightZero.Data(), attnOutputWeightSum.Data(), 0.0f, outBuffer.Data(),
                        outBuffer.Stride());
            } else {
                ctx->mmHelper->compute_bias(false, attnSplit.Rows(), attnOutputWeight.Cols(), attnSplit.Cols(), 1.0f,
                        attnSplit.Data(), attnSplit.Stride(), attnOutputWeight.Data(), attnOutputWeightScale.Data(),
                        attnOutputWeightZero.Data(), attnOutputWeightSum.Data(), 0.0f, outBuffer.Data(),
                        outBuffer.Stride(), attnOutputBias.Data());
            }
        }
        t5.release();

#ifdef DEBUG
        dbg.debugPrint(">>> attention output/projection[%d, %d] (%d):\n", outBuffer.Rows(), outBuffer.Cols(),
                outBuffer.Stride());
        dbg.dumpMatrix(outBuffer);
#endif

        if (!doLnBefore) {
            TimeLine t6("result.layer_norm");
            norm.forward(outBuffer.Data(), outBuffer.Data(), outBuffer.Rows(), outBuffer.Stride(), outBuffer.Stride());
#ifdef DEBUG
            dbg.debugPrint("LayerNorm after attention: [%d, %d] (%d)\n", outBuffer.Rows(), outBuffer.Cols(),
                    outBuffer.Stride());
            dbg.dumpMatrix(outBuffer);
#endif
        }
    }

protected:
    template <typename KVCacheT>
    void selfAttentionBF16(DecoderContext *ctx, xft::Matrix<bfloat16_t> &query, xft::Matrix<bfloat16_t> &key,
            xft::Matrix<bfloat16_t> &value, xft::Matrix<bfloat16_t> &result, KVCacheTensor<KVCacheT> &presentKey,
            KVCacheTensor<KVCacheT> &presentValue) {
        int responsibleQHeads = this->endQHead - this->startQHead;
        int responsibleKVHeads = this->endKVHead - this->startKVHead;

        int tokenSizes[ctx->batchSize];
        for (int i = 0; i < ctx->batchSize; ++i) {
            tokenSizes[i] = ctx->inputSeqLen;
        }

        xft::selfAttention(
                result.Data(), query.Data(), key.Data(), value.Data(), responsibleQHeads, responsibleKVHeads,
                ctx->attHeadSize, result.Stride(), query.Stride(), key.Stride(), ctx->batchSize, tokenSizes,
                ctx->attFactor, alibiSlopes, ctx->numThreads,
                [&](int b, int headIdx, int seqIdx) { return presentKey.getSequence(seqIdx, b, headIdx); },
                [&](int b, int headIdx, int seqIdx) { return presentValue.getSequence(seqIdx, b, headIdx); });
    }

    template <typename KVCacheT>
    void selfAttentionBF16(DecoderContext *ctx, xft::Matrix<bfloat16_t> &query, xft::Matrix<bfloat16_t> &key,
            xft::Matrix<bfloat16_t> &value, xft::Matrix<bfloat16_t> &result,
            std::vector<KVCacheTensor<KVCacheT> *> &keyCaches, std::vector<KVCacheTensor<KVCacheT> *> &valueCaches,
            std::vector<xft::SequenceMeta *> &seqs) {
        int responsibleQHeads = this->endQHead - this->startQHead;
        int responsibleKVHeads = this->endKVHead - this->startKVHead;

        int batchSize = seqs.size();
        int tokenSizes[batchSize];
        for (int i = 0; i < batchSize; ++i) {
            tokenSizes[i] = seqs[i]->getInputSeqLen();
        }

        xft::selfAttention(
                result.Data(), query.Data(), key.Data(), value.Data(), responsibleQHeads, responsibleKVHeads,
                ctx->attHeadSize, result.Stride(), query.Stride(), key.Stride(), batchSize, tokenSizes, ctx->attFactor,
                alibiSlopes, ctx->numThreads,
                [&](int b, int headIdx, int seqIdx) { return keyCaches[b]->getSequence(seqIdx, 0, headIdx); },
                [&](int b, int headIdx, int seqIdx) { return valueCaches[b]->getSequence(seqIdx, 0, headIdx); });
    }

    template <typename T, typename KVCacheT>
    void fusedAttention(DecoderContext *ctx, xft::Matrix<T> &query, xft::Matrix<T> &key, xft::Matrix<T> &value,
            xft::Matrix<T> &result, std::vector<KVCacheTensor<KVCacheT> *> &keyCaches,
            std::vector<KVCacheTensor<KVCacheT> *> &valueCaches, std::vector<xft::SequenceMeta *> &seqs) {
        int responsibleQHeads = this->endQHead - this->startQHead;
        int responsibleKVHeads = this->endKVHead - this->startKVHead;

        int batchSize = seqs.size();

        // TODO: move to AttentionBlock
        int inputSeqLens[batchSize];
        int pastSeqLens[batchSize];
        for (int i = 0; i < batchSize; ++i) {
            inputSeqLens[i] = seqs[i]->getInputSeqLen();
            pastSeqLens[i] = seqs[i]->getPastSeqLen();
        }

        // TODO: non-causal case handle
        xft::crossAttnByHead<T, KVCacheT>(
                result.Data(), query.Data(), key.Data(), value.Data(), responsibleQHeads, responsibleKVHeads,
                ctx->attHeadSize, result.Stride(), query.Stride(), key.Stride(), batchSize, inputSeqLens, pastSeqLens,
                true, ctx->attFactor, alibiSlopes, ctx->numThreads,
                [&](int b, int headIdx) { return keyCaches[b]->getHead(0, headIdx); },
                [&](int b, int headIdx) { return valueCaches[b]->getHead(0, headIdx); });
    }

    int getMBlockSize(int inputSeqLen, int headSize, int minVal = 6) {
        // Special case
        if (inputSeqLen == 1) { return 1; }

        const int l2CacheSize = 2 * 1024 * 1024; // TODO: get it dynamically
        const int qkvSize = inputSeqLen * headSize;
        const int scoreSize = inputSeqLen * inputSeqLen;

        // As we do the split along M dimension, so to make sure:
        // All data visited in BMM1 (Q * K -> Score) and BMM2 (Score * V -> output) are in L2
        // (qSize / splits) + kSize + (scoreSize / splits) + vSize + (outSize / splits) <= cacheSize
        int capacity = l2CacheSize / sizeof(ImT);
        int splits = 1;
        if (capacity <= 2 * qkvSize) { // Always cannot cache accessed data
            splits = 1;
        } else {
            splits = std::ceil(1.0f * (2 * qkvSize + scoreSize) / (capacity - 2 * qkvSize));
        }
        if (splits <= 0) { splits = 1; }
        int mBlockSize = (inputSeqLen + splits - 1) / splits;
        if (mBlockSize <= 0) {
            mBlockSize = inputSeqLen > minVal ? minVal : inputSeqLen;
        } else if (mBlockSize > inputSeqLen) {
            mBlockSize = inputSeqLen;
        }

        return mBlockSize;
    }

    // Copy all keys and values to KV cache
    template <typename KVCacheT>
    void copyKVCache(DecoderContext *ctx, xft::Matrix<ImT> &key, xft::Matrix<ImT> &value,
            KVCacheTensor<KVCacheT> &presentKey, KVCacheTensor<KVCacheT> &presentValue, int pastSeqLen) {
        int batchSize = ctx->batchSize;
        int headSize = ctx->attHeadSize;

#pragma omp parallel for collapse(3)
        for (int b = 0; b < batchSize; ++b) {
            for (int h = 0; h < (this->endKVHead - this->startKVHead); ++h) {
                // Copy current key/value to cached keys/values
                // Re-layout is needed: (bs, seq=1, hidden_size) -> (seq=1, bs, hidden_size)
                // Be noted: for group attention, the key/value is less than query
                for (int seq = 0; seq < ctx->inputSeqLen; ++seq) {
                    auto srcK = key.Row(b * ctx->inputSeqLen + seq) + h * headSize;
                    auto dstK = presentKey.getSequence(pastSeqLen + seq, b, h);

                    auto srcV = value.Row(b * ctx->inputSeqLen + seq) + h * headSize;
                    auto dstV = presentValue.getSequence(pastSeqLen + seq, b, h);

                    xft::storeKVCache(dstK, srcK, headSize);
                    xft::storeKVCache(dstV, srcV, headSize);
                }
            }
        }
    }

    template <typename KVCacheT>
    void copyKVCache(DecoderContext *ctx, xft::Matrix<ImT> &key, xft::Matrix<ImT> &value,
            std::vector<KVCacheTensor<KVCacheT> *> &keyCaches, std::vector<KVCacheTensor<KVCacheT> *> &valueCaches,
            std::vector<xft::SequenceMeta *> &seqs) {
        int batchSize = seqs.size();
        int headSize = ctx->attHeadSize;

        int maxSeqLen = 0;
        int inputSeqLens[batchSize], pastSeqLens[batchSize], seqStartLoc[batchSize + 1];
        seqStartLoc[0] = 0;
        for (int i = 0; i < batchSize; ++i) {
            inputSeqLens[i] = seqs[i]->getInputSeqLen();
            pastSeqLens[i] = seqs[i]->getPastSeqLen();
            seqStartLoc[i + 1] = seqStartLoc[i] + inputSeqLens[i];
            maxSeqLen = std::max(maxSeqLen, inputSeqLens[i]);
        }

#pragma omp parallel for collapse(3)
        for (int b = 0; b < batchSize; ++b) {
            for (int h = 0; h < (this->endKVHead - this->startKVHead); ++h) {
                // Copy current key/value to cached keys/values
                // Re-layout is needed: (bs, seq=1, hidden_size) -> (seq=1, bs, hidden_size)
                // Be noted: for group attention, the key/value is less than query
                for (int seq = 0; seq < maxSeqLen; ++seq) {
                    if (seq >= inputSeqLens[b]) continue;
                    auto srcK = key.Row(seqStartLoc[b] + seq) + h * headSize;
                    auto dstK = keyCaches[b]->getSequence(pastSeqLens[b] + seq, 0, h);

                    auto srcV = value.Row(seqStartLoc[b] + seq) + h * headSize;
                    auto dstV = valueCaches[b]->getSequence(pastSeqLens[b] + seq, 0, h);

                    xft::storeKVCache(dstK, srcK, headSize);
                    xft::storeKVCache(dstV, srcV, headSize);
                }
            }
        }
    }

    // Copy one head from key or value to K cache or V cache
    // bdx: batch index; hdx: head index
    template <typename KVCacheT>
    void copyKVCache(DecoderContext *ctx, xft::Matrix<ImT> &kv, KVCacheTensor<KVCacheT> &presentKV, int pastSeqLen,
            int bdx, int hdx) {
        for (int seq = 0; seq < ctx->inputSeqLen; ++seq) {
            auto src = kv.Row(bdx * ctx->inputSeqLen + seq) + hdx * ctx->attHeadSize;
            auto dst = presentKV.getSequence(pastSeqLen + seq, bdx, hdx);
            xft::storeKVCache(dst, src, ctx->attHeadSize);
        }
    }

    // query: M * headSize, key: N * headSize, score: M * N
    // ldq: leading dimension of query; lds: LD of score
    template <typename T1, typename T2, typename T3>
    void gemm1(T1 *query, const std::tuple<T2 *, int, float *> &keyMat, T3 *score, int M, int N, int headSize, int ldq,
            int lds) {
        auto A = query;
        auto B = std::get<0>(keyMat);
        auto C = score;
        const int K = headSize;
        const int ldb = std::get<1>(keyMat);
        const float *scale = std::get<2>(keyMat);
        if constexpr (std::is_same_v<T2, int8_t>) {
            small_gemm_transb(A, B, scale, C, M, N, K, ldq, ldb, lds);
        } else {
            small_gemm_transb(A, B, C, M, N, K, ldq, ldb, lds);
        }
    }

    // Softmax between 2 BMM
    template <typename T1, typename T2>
    void softmax(DecoderContext *ctx, T1 *score, const T2 *mask, int rows, int cols, int lds, int startSeq) {
        const int keyLen = cols;
        for (int seq = 0; seq < rows; ++seq) {
            DecoderUtil::computeSoftmax(score + seq * lds, mask + (seq + startSeq) * keyLen, keyLen, ctx->attFactor);
        }
    }

    // score: M * K(keyLen), value: K * headSize, output: M * headSize
    template <typename T1, typename T2, typename T3>
    void gemm2(T1 *score, const std::tuple<T2 *, int, float *> &valueMat, T3 *output, int M, int headSize, int K,
            int lds, int ldo) {
        auto A = score;
        T2 *B = std::get<0>(valueMat);
        auto C = output;
        const int N = headSize;
        const int ldv = std::get<1>(valueMat);
        if constexpr (std::is_same_v<T2, int8_t>) {
            xft::small_gemm(A, B, std::get<2>(valueMat), C, M, N, K, lds, ldv, ldo);
        } else {
            xft::small_gemm(A, B, C, M, N, K, lds, ldv, ldo);
        }
    }

    // Note: the result here is still the intermediate result from the whole attention scope
    template <typename KVCacheT>
    void fusedAttention(DecoderContext *ctx, xft::Matrix<ImT> &query, xft::Matrix<ImT> &key, xft::Matrix<ImT> &value,
            xft::Matrix<ImT> &result, KVCacheTensor<KVCacheT> &presentKey, KVCacheTensor<KVCacheT> &presentValue,
            const float *attnMask, int pastSeqLen) {
        // How many heads this task should do
        int responsibleHeads = this->endQHead - this->startQHead;
        int batchSize = ctx->batchSize;
        int headSize = ctx->attHeadSize;
        int groupNum = ctx->attHeadNum / ctx->kvHeadNum;

        // If M_dimension/input_seq_len is big (1K, 2K, etc), need to split it to make sure the intermediate result in cache
        // to make sure it works better (the logic here is trying to make sure each head of BMM result [seq * seq] in cache)
        // WARN: reserve field in context is used to make it effective for all layers, do not change it in other places
        int &mBlockSize = ctx->reserved1;
        if (layerId % (ctx->layers / ctx->ppSize) == 0) {
            if (pastSeqLen == 0) {
                mBlockSize = getMBlockSize(ctx->inputSeqLen, ctx->attHeadSize);
            }
            // When pastSeqLen > 0, whether for generation or verification, input seq length is small
            else {
                mBlockSize = ctx->inputSeqLen;
            }
        }

        // If total tasks are too small (compared to total thread number), need to shard the head
        bool shardHead = (ctx->inputSeqLen == 1) && (ctx->numThreads >= batchSize * responsibleHeads * 2);

        // Need to copy current key/values to cache seperately if:
        // (1) For group attention (#kvHeads != #qHeads)
        // (2) When M dimension is split, multiple tasks per copy, so do copy seperately
        // (3) When head is sharded, also multiple tasks per copy
        bool kvCopied = false;
        if (ctx->kvHeadNum < ctx->attHeadNum || mBlockSize != ctx->inputSeqLen || shardHead) {
            copyKVCache(ctx, key, value, presentKey, presentValue, pastSeqLen);
            kvCopied = true;
        }

        if (!shardHead) {
            return slimAttention(ctx, query, key, value, result, presentKey, presentValue, attnMask, pastSeqLen,
                    mBlockSize, kvCopied);
        } else { // Seperate impl. when head is sharded
            return crossAttnShardHead(ctx, query, key, value, result, presentKey, presentValue, attnMask, pastSeqLen);
        }
    }

    template <typename KVCacheT>
    void slimAttention(DecoderContext *ctx, xft::Matrix<ImT> &query, xft::Matrix<ImT> &key, xft::Matrix<ImT> &value,
            xft::Matrix<ImT> &result, KVCacheTensor<KVCacheT> &presentKey, KVCacheTensor<KVCacheT> &presentValue,
            const float *attnMask, int pastSeqLen, int mBlockSize, bool kvCopied) {
        // How many heads this task should do
        int responsibleHeads = this->endQHead - this->startQHead;
        int batchSize = ctx->batchSize;
        int headSize = ctx->attHeadSize;
        int groupNum = ctx->attHeadNum / ctx->kvHeadNum;

        // How many blocks in M dimension
        int mBlockNum = (ctx->inputSeqLen + mBlockSize - 1) / mBlockSize;

        // To get score buffer according to openmp thread num
        int scoreStride = pastSeqLen > 0 ? (pastSeqLen + ctx->inputSeqLen + 15) / 16 * 16 : ctx->inputSeqLen;
        auto bufSizeRequired = ctx->numThreads * mBlockSize * scoreStride;
        float *scoreBuf = (float *)SimpleMemPool::instance().getBuffer("scoreBuf", sizeof(float) * bufSizeRequired);

#pragma omp parallel for collapse(3)
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < responsibleHeads; ++i) {
                for (int mb = 0; mb < mBlockNum; ++mb) {
                    const int startSeq = mb * mBlockSize;
                    const int endSeq
                            = startSeq + mBlockSize < ctx->inputSeqLen ? startSeq + mBlockSize : ctx->inputSeqLen;

                    // Copy current key to cached keys
                    if (!kvCopied) { copyKVCache(ctx, key, presentKey, pastSeqLen, b, i); }

                    // Q * K
                    auto keyMatInfo = presentKey.getHead(b, i / groupNum);
                    int m = endSeq - startSeq;
                    int k = ctx->attHeadSize;
                    int n = pastSeqLen + ctx->inputSeqLen;
                    int lda = query.Stride();
                    int ldb = std::get<1>(keyMatInfo);
                    int ldc = scoreStride;
                    auto A = query.Row(b * ctx->inputSeqLen + startSeq) + i * ctx->attHeadSize; // updated
                    auto B = std::get<0>(keyMatInfo);
                    auto C = scoreBuf + omp_get_thread_num() * mBlockSize * scoreStride;

                    const int queryLen = ctx->inputSeqLen;
                    const int keyLen = pastSeqLen + ctx->inputSeqLen;

                    this->gemm1(A, keyMatInfo, C, m, n, headSize, lda, ldc);

#ifdef DEBUG
                    if (b == 0 && i == 0) {
                        dbg.debugPrint("Q * K, first head:\n");
                        auto p = scoreBuf;
                        dbg.debugPrint("%f, %f, %f ... %f %f %f\n", p[0] * ctx->attFactor, p[1] * ctx->attFactor,
                                p[2] * ctx->attFactor, p[n - 3] * ctx->attFactor, p[n - 2] * ctx->attFactor,
                                p[n - 1] * ctx->attFactor);
                    }
#endif

                    // Softmax(Q * K)
                    this->softmax(ctx, C, getMask(attnMask, b, i, queryLen, keyLen), m, n, ldc, startSeq);

#ifdef DEBUG
                    if (b == 0 && i == 0) {
                        dbg.debugPrint("Softmax(Q * K), first head:\n");
                        auto p = scoreBuf;
                        dbg.debugPrint("%f, %f, %f ... %f %f %f\n", p[0], p[1], p[2], p[keyLen - 3], p[keyLen - 2],
                                p[keyLen - 1]);
                    }
#endif

                    // Copy current value to cached values
                    // Re-layout is needed: (bs, seq, hidden_size) -> (seq, bs, hidden_size)
                    if (!kvCopied) { copyKVCache(ctx, value, presentValue, pastSeqLen, b, i); }

                    // Softmax * V
                    auto valueMat = presentValue.getHead(b, i / groupNum);
                    auto output = result.Row(b * ctx->inputSeqLen + startSeq) + i * ctx->attHeadSize;
                    this->gemm2(C, valueMat, output, m, headSize, keyLen, scoreStride, result.Stride());

#ifdef DEBUG
                    if (b == 0 && i == 0) {
                        dbg.debugPrint("Softmax(Q * K) * V, first head:\n");
                        auto p = output;
                        dbg.debugPrint("%f, %f, %f ... %f %f %f\n", p[0], p[1], p[2], p[ctx->attHeadSize - 3],
                                p[ctx->attHeadSize - 2], p[ctx->attHeadSize - 1]);
                    }
#endif
                } // end for mb
            } // end for i
        } // end for b
    }

    // When #heads is very few, need to shard each head to use more resources
    template <typename KVCacheT>
    void crossAttnShardHead(DecoderContext *ctx, xft::Matrix<ImT> &query, xft::Matrix<ImT> &key,
            xft::Matrix<ImT> &value, xft::Matrix<ImT> &result, KVCacheTensor<KVCacheT> &presentKey,
            KVCacheTensor<KVCacheT> &presentValue, const float *attnMask, int pastSeqLen) {
        const int responsibleHeads = this->endQHead - this->startQHead;
        const int batchSize = ctx->batchSize;
        const int presentSeqLen = pastSeqLen + ctx->inputSeqLen;
        const int groupNum = ctx->attHeadNum / ctx->kvHeadNum;

        xft::crossAttnShardedHead<ImT, KVCacheT>(
                result.Data(), query.Data(), ctx->inputSeqLen, presentSeqLen, responsibleHeads, ctx->attHeadSize,
                result.Stride(), query.Stride(), batchSize, ctx->attFactor, ctx->numThreads,
                [&](int b, int qHeadIdx) { return presentKey.getHead(b, qHeadIdx / groupNum); },
                [&](int b, int qHeadIdx) { return presentValue.getHead(b, qHeadIdx / groupNum); },
                [&](int b, int qHeadIdx, int srcLen, int tgtLen) {
                    return getMask(attnMask, b, qHeadIdx, srcLen, tgtLen);
                });
    }

    template <typename KVCacheT>
    void flashAttention(DecoderContext *ctx, xft::Matrix<ImT> &query, xft::Matrix<ImT> &key, xft::Matrix<ImT> &value,
            xft::Matrix<ImT> &result, KVCacheTensor<KVCacheT> &presentKey, KVCacheTensor<KVCacheT> &presentValue,
            const float *attnMask, int pastSeqLen) {
#if defined(AVX512_BF16_WEIGHT_ONLY_BF16)
        using AttnT = bfloat16_t;
#else
        using AttnT = float;
#endif
        // How many heads this task should do
        int batchSize = ctx->batchSize;
        int respQHeads = this->endQHead - this->startQHead;
        int respKVHeads = this->endKVHead - this->startKVHead;
        int headSize = ctx->attHeadSize;
        int qCols = respQHeads * headSize;
        int kvCols = respKVHeads * headSize;
        int qkvCols = qCols + kvCols * 2;
        float scale = ctx->attFactor;

        int totalTokenSize = 0;
        int inputSeqLens[batchSize], pastSeqLens[batchSize];
        for (int i = 0; i < batchSize; ++i) {
            inputSeqLens[i] = ctx->inputSeqLen;
            pastSeqLens[i] = pastSeqLen;
            totalTokenSize += inputSeqLens[i];
        }

        // TODO: kv dtype conversion for prefixSharing
        AttnT *k, *v;
        int kvStride;
        // convert to AttnT forcely for accelerating purpose
        if constexpr (!std::is_same_v<AttnT, ImT>) {
            kvStride = kvCols * 2;
            AttnT *kvBuf = (AttnT *)SimpleMemPool::instance().getBuffer(
                    "flashKVBuf", totalTokenSize * kvStride * sizeof(AttnT));
#pragma omp parallel for collapse(2)
            for (uint64_t seq = 0; seq < totalTokenSize; ++seq)
                for (uint64_t i = 0; i < kvCols * 2; i += headSize) {
                    const ImT *srcPtr = key.Data() + seq * qkvCols + i;
                    AttnT *dstPtr = kvBuf + seq * kvStride + i;
                    if constexpr (std::is_same_v<AttnT, bfloat16_t> && std::is_same_v<ImT, float>) {
                        bfloat16_t::cvt_float_to_bfloat16(srcPtr, dstPtr, headSize);
                    } else if constexpr (std::is_same_v<AttnT, float> && std::is_same_v<ImT, bfloat16_t>) {
                        bfloat16_t::cvt_bfloat16_to_float(srcPtr, dstPtr, headSize);
                    } else {
                        printf("Not supported Type in Flash Attention yet\n");
                        exit(-1);
                    }
                }

            k = kvBuf;
            v = kvBuf + kvCols;
        } else {
            kvStride = qkvCols;
            k = key.Data();
            v = value.Data();
        }

        // [batch, src, head, headsize]
        xft::selfScaledDpAttention<ImT, AttnT>(result.Data(), query.Data(), k, v, respQHeads, respKVHeads, headSize,
                result.Stride(), query.Stride(), kvStride, batchSize, inputSeqLens, pastSeqLens, true, alibiSlopes,
                attnMask, scale, ctx->numThreads);

        // copy current key/values to cache
        copyKVCache(ctx, key, value, presentKey, presentValue, pastSeqLen);
    }

    template <typename KVCacheT>
    void flashAttention(DecoderContext *ctx, xft::Matrix<ImT> &query, xft::Matrix<ImT> &key, xft::Matrix<ImT> &value,
            xft::Matrix<ImT> &result, std::vector<KVCacheTensor<KVCacheT> *> &keyCaches,
            std::vector<KVCacheTensor<KVCacheT> *> &valueCaches, std::vector<xft::SequenceMeta *> &seqs) {
#if defined(AVX512_BF16_WEIGHT_ONLY_BF16)
        using AttnT = bfloat16_t;
#else
        using AttnT = float;
#endif
        // How many heads this task should do
        int batchSize = seqs.size();
        int respQHeads = this->endQHead - this->startQHead;
        int respKVHeads = this->endKVHead - this->startKVHead;
        int headSize = ctx->attHeadSize;
        int qCols = respQHeads * headSize;
        int kvCols = respKVHeads * headSize;
        int qkvCols = qCols + kvCols * 2;
        float scale = ctx->attFactor;

        int totalTokenSize = 0;
        int inputSeqLens[batchSize], pastSeqLens[batchSize];
        for (int i = 0; i < batchSize; ++i) {
            inputSeqLens[i] = seqs[i]->getInputSeqLen();
            pastSeqLens[i] = seqs[i]->getPastSeqLen();
            totalTokenSize += inputSeqLens[i];
        }

        // TODO: kv dtype conversion for prefixSharing
        AttnT *k, *v;
        int kvStride;
        // convert to AttnT forcely for accelerating purpose
        if constexpr (!std::is_same_v<AttnT, ImT>) {
            kvStride = kvCols * 2;
            AttnT *kvBuf = (AttnT *)SimpleMemPool::instance().getBuffer(
                    "flashKVBuf", totalTokenSize * kvStride * sizeof(AttnT));
#pragma omp parallel for collapse(2)
            for (uint64_t seq = 0; seq < totalTokenSize; ++seq)
                for (uint64_t i = 0; i < kvCols * 2; i += headSize) {
                    const ImT *srcPtr = key.Data() + seq * qkvCols + i;
                    AttnT *dstPtr = kvBuf + seq * kvStride + i;
                    if constexpr (std::is_same_v<AttnT, bfloat16_t> && std::is_same_v<ImT, float>) {
                        bfloat16_t::cvt_float_to_bfloat16(srcPtr, dstPtr, headSize);
                    } else if constexpr (std::is_same_v<AttnT, float> && std::is_same_v<ImT, bfloat16_t>) {
                        bfloat16_t::cvt_bfloat16_to_float(srcPtr, dstPtr, headSize);
                    } else {
                        printf("Not supported Type in Flash Attention yet\n");
                        exit(-1);
                    }
                }

            k = kvBuf;
            v = kvBuf + kvCols;
        } else {
            kvStride = qkvCols;
            k = key.Data();
            v = value.Data();
        }

        // [batch, src, head, headsize]
        xft::selfScaledDpAttention<ImT, AttnT>(result.Data(), query.Data(), k, v, respQHeads, respKVHeads, headSize,
                result.Stride(), query.Stride(), kvStride, batchSize, inputSeqLens, pastSeqLens, true, alibiSlopes,
                nullptr, scale, ctx->numThreads);

        // copy current key/values to cache
        copyKVCache(ctx, key, value, keyCaches, valueCaches, seqs);
    }

private:
    std::pair<int, int> getTaskRange(int N, int splits, int splitIdx) {
        int startId, endId;

        if (N % splits == 0) {
            int tasksPerSplit = N / splits;
            startId = splitIdx * tasksPerSplit;
            endId = startId + tasksPerSplit;
        } else {
            int baseTasksPerSplit = N / splits;
            int remainingTasks = N % splits;

            // Each split has (baseTasksPerSplit + 1) tasks
            if (splitIdx < remainingTasks) {
                int tasksPerSplit = baseTasksPerSplit + 1;
                startId = splitIdx * tasksPerSplit;
                endId = startId + tasksPerSplit;
            }
            // Each split has 'baseTasksPerSplit' tasks
            else {
                int taskOffset = (baseTasksPerSplit + 1) * remainingTasks;
                startId = taskOffset + (splitIdx - remainingTasks) * baseTasksPerSplit;
                endId = startId + baseTasksPerSplit;
            }
        }

        return std::make_pair(startId, endId);
    }

protected:
    virtual float getResidentialScale() {
        return 1; // directly add the residential
    }

    // Used in computeSoftmax
    virtual float getScalingCoeff() {
        return 0; // 0 means using the default value
    }

    virtual const float *getMask(const float *attnMask, int bId, int hId, int srcLen, int tgtLen) {
        // Would mask be different for each sample in one batch?
        return attnMask + bId * srcLen * tgtLen;
    }

    // query, key, value weighs
    xft::Matrix<WeiT> qkvWeight;
    xft::Vector<float> qkvWeightScale; // if weight is int8
    xft::Vector<float> qkvWeightZero; // if weight is int8
    xft::Vector<float> qkvWeightSum; // if weight is int8
    // query, key, value bias
    xft::Vector<float> qkvBias;

    xft::Matrix<WeiT> attnOutputWeight;
    xft::Vector<float> attnOutputWeightScale; // if weight is int8
    xft::Vector<float> attnOutputWeightZero; // if weight is int8
    xft::Vector<float> attnOutputWeightSum; // if weight is int8
    xft::Vector<float> attnOutputBias;

    // Query/Key post op
    QKPO_CLS qkpo;

    // layerNorm param
    NORM_CLS norm;
    int layerId;

    // Alibi Slopes
    float *alibiSlopes;

    // The responsible head in the global view
    // If in single instance, startQHead=startKVHead=0, and endQHead-startQHead=qHeadNum
    int startQHead;
    int endQHead;
    int startKVHead;
    int endKVHead;
#ifdef DEBUG
    Debugger dbg;
#endif
};
