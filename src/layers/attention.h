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

#include "bfloat16.h"
#include "debugger.h"
#include "decoder_util.h"
#include "float16.h"
#include "gemm_kernel_ext.h"
#include "kvcache_tensor.h"
#include "matmul_helper.h"
#include "transformer_ctx.h"
#include "transformer_util.h"

// WeiT: weight data type
// QKPO_CLS: class for post operation of query/key, it is generally the rotary embedding
// NORM_CLS: class for layernorm or other norms
// INPUT_AS_RESID: input as residential or not, most models use input as residential,
//                 but there are exceptions like ChatGLM use values after layernorm as residential
template <typename WeiT, typename QKPO_CLS, typename NORM_CLS, bool INPUT_AS_RESID = true>
class Attention {
public:
    Attention(int layerId, DecoderContext *ctx) : layerId(layerId), qkpo(ctx->attHeadSize) {
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
    }

    // The inerface is for PyTorch, thus the weights are already transposed
    void setWeights(DecoderContext *ctx, const float *queryWeight, const float *queryBias, const float *keyWeight,
            const float *keyBias, const float *valueWeight, const float *valueBias, const float *attnOutWeight,
            const float *attnOutBias, const float *gamma1, const float *beta1, bool trans = true) {
        int hiddenSize = ctx->hiddenSize;
        int headSize = ctx->attHeadSize;

        // Merged weights, dimension is like: hiddenSize * (hiddenSize + 2 * kvHiddenSize)
        // Vertically split the QKV weights
        int qResponsibleCols = (this->endQHead - this->startQHead) * headSize;
        int kvResponsibleCols = (this->endKVHead - this->startKVHead) * headSize;
        int responsibleCols = qResponsibleCols + 2 * kvResponsibleCols;
        qkvWeight.Resize(hiddenSize, responsibleCols);

        float *concatBuf = (float *)malloc(hiddenSize * responsibleCols * sizeof(float));
        if (trans) {
            memcpy(concatBuf, queryWeight + this->startQHead * headSize * hiddenSize,
                    hiddenSize * qResponsibleCols * sizeof(float));
            memcpy(concatBuf + hiddenSize * qResponsibleCols, keyWeight + this->startKVHead * headSize * hiddenSize,
                    hiddenSize * kvResponsibleCols * sizeof(float));
            memcpy(concatBuf + hiddenSize * (qResponsibleCols + kvResponsibleCols),
                    valueWeight + this->startKVHead * headSize * hiddenSize,
                    hiddenSize * kvResponsibleCols * sizeof(float));
        } else {
            int qkvStride = (ctx->attHeadNum + ctx->kvHeadNum + ctx->kvHeadNum) * ctx->attHeadSize;
            for (int i = 0; i < hiddenSize; ++i) {
                memcpy(concatBuf + i * responsibleCols, queryWeight + i * qkvStride + this->startQHead * headSize,
                        qResponsibleCols * sizeof(float));
            }
            for (int i = 0; i < hiddenSize; ++i) {
                memcpy(concatBuf + i * responsibleCols + qResponsibleCols,
                        keyWeight + i * qkvStride + this->startKVHead * headSize, kvResponsibleCols * sizeof(float));
            }
            for (int i = 0; i < hiddenSize; ++i) {
                memcpy(concatBuf + i * responsibleCols + qResponsibleCols + kvResponsibleCols,
                        valueWeight + i * qkvStride + this->startKVHead * headSize, kvResponsibleCols * sizeof(float));
            }
        }

        hpj::Matrix<WeiT> convertedqkvWeight;
        MMHelper::convertWeight(
                trans, hiddenSize, responsibleCols, concatBuf, convertedqkvWeight, qkvWeightScale, qkvWeightZero);
        MMHelper::packWeight(trans, convertedqkvWeight, qkvWeight);

        free(concatBuf);

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
            //responsibleCols = hiddenSize / ctx->numSplit;
            qkvBias.Resize(responsibleCols);
            memcpy(qkvBias.Data(), queryBias + ctx->splitIdx * qResponsibleCols, sizeof(float) * qResponsibleCols);
            memcpy(qkvBias.Data() + qResponsibleCols, keyBias + this->startKVHead * headSize,
                    sizeof(float) * kvResponsibleCols);
            memcpy(qkvBias.Data() + qResponsibleCols + kvResponsibleCols, valueBias + this->startKVHead * headSize,
                    sizeof(float) * kvResponsibleCols);
        }

        // Weights for attention output
        // Horizontally split the weight, as the source (PyTorch weight) is transposed, thus looks like vertically
        hpj::Matrix<WeiT> convertedWeight;
        MMHelper::convertWeight(ctx, trans, hiddenSize, hiddenSize, attnOutWeight, false, convertedWeight,
                attnOutputWeightScale, attnOutputWeightZero);
        MMHelper::packWeight(trans, convertedWeight, attnOutputWeight);

#ifdef DEBUG
        dbg.debugPrint("attention output weight: [%d, %d] (%d)\n", convertedWeight.Rows(), convertedWeight.Cols(),
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
        this->norm.setWeight(gamma1, beta1, hiddenSize);
    }

#ifdef DEBUG
    void setDebugger(const Debugger &debugger) { this->dbg = debugger; }
#endif

    /**
     * Forward computing for the whole encoder/decoder layer
     * Inputs:
     * - input: (bs * seq_len) x hidden_size (input buffer)
     * - imBuf: (bs * seq_len) x hidden_size (intermediate buffer, output is in ctx->tmpBuf)
     * - attnMask: (bs, 1, tgt_len, src_len) (tgt_len is the length of query, src_len is the length of key)
     * - presentKeys, presentValues: past key/values concats current key/values
     * - pastSeqLen: the sequence length in pastKeys and pastValues
     * - useSelfAttn: use self attention or not, self attention is used to gen first token
     * - doLnBefore: Do layer norm before or not. If true, will do layer norm as the first step
     * - returnAttn: return attention values or not (this option is not used any more)
     * - returnKVs: return present key/values or not (this option is not used any more)
     * - forPT: is it for PyTorch or not (this option is not used any more), now cached keys/values are always controlled by us
     * Internal Buffers:
     *  _________                _________                _________                _________                _________                
     * |         |------------->|         |------------->|         |------------->|         |------------->|         |
     * ```````````  layerNorm   ```````````  QKV Linear  ```````````     MHA      ```````````  out Linear  ```````````
     *    input                resultBuffer1              qkvMatMul               resultBuffer1            resultBuffer2
    */
    template <typename KVCacheT>
    void forward(DecoderContext *ctx, float *input, float *imBuf, const float *attnMask,
            KVCacheTensor<KVCacheT> &presentKey, KVCacheTensor<KVCacheT> &presentValue, int inputSeqLen, int pastSeqLen,
            bool useSelfAttn, bool doLnBefore, bool returnAttn, bool returnKVs, bool forPT = true,
            int *positionIds = nullptr) {
        if (forPT) {
            printf("For better perf, need to manage cached key/vaues by ourself, PyTorch extension is not supported "
                   "any more.\n");
            exit(-1);
        }

        hpj::Matrix<float> inputBuffer(input, ctx->batchSize * inputSeqLen, ctx->hiddenSize, ctx->hiddenSize);
        hpj::Matrix<float> imBuffer(imBuf, ctx->batchSize * inputSeqLen, ctx->hiddenSize, ctx->hiddenSize);

        auto hiddenSize = ctx->hiddenSize;
        auto &qkvMatMul = ctx->qkvMatMul;
        auto &resultBuffer1 = imBuffer;
        auto &resultBuffer2 = ctx->tmpBuf;

#ifdef DEBUG
        dbg.debugPrint("---- DecoderLayer.forward (useSelfAttn=%d) ----\n", useSelfAttn);
        dbg.debugPrint("input:\n");
        dbg.dumpMatrix(inputBuffer);
#endif

        if (doLnBefore) {
            TimeLine t1("input.layer_norm");
            norm.forward(inputBuffer.Data(), resultBuffer1.Data(), inputBuffer.Rows(), inputBuffer.Stride(),
                    resultBuffer1.Stride());
        }
#ifdef DEBUG
        dbg.debugPrint("layer norm:\n");
        dbg.dumpMatrix(resultBuffer1);
        dbg.debugPrint("qkvWeight [%d, %d]:\n", this->qkvWeight.Rows(), this->qkvWeight.Cols());
        dbg.dumpMatrix(this->qkvWeight);
#endif

        // Query, Key, Value computed together
        TimeLine t2("QKV.linear");
        DecoderUtil::dense(resultBuffer1, qkvWeight, qkvWeightScale, qkvWeightZero, qkvBias, qkvMatMul);
        t2.release();

        int cols = hiddenSize / ctx->numSplit;
        hpj::Matrix<float> query(qkvMatMul, 0, inputBuffer.Rows(), 0, cols);
        hpj::Matrix<float> key(qkvMatMul, 0, inputBuffer.Rows(), cols, cols);
        hpj::Matrix<float> value(qkvMatMul, 0, inputBuffer.Rows(), cols * 2, cols);

#ifdef DEBUG
        dbg.debugPrint("Q:\n");
        dbg.dumpMatrix(query);
        dbg.debugPrint("K:\n");
        dbg.dumpMatrix(key);
#endif

        // Apply post operattions on query and key
        TimeLine t3("QKPO");
        int qk_shape[4] = {ctx->batchSize, ctx->inputSeqLen, ctx->attHeadNum / ctx->numSplit, ctx->attHeadSize};
        if (positionIds != nullptr) {
            qkpo.forward(query.Data(), key.Data(), query.Stride(), key.Stride(), qk_shape, positionIds);
        } else {
            // Use the default position ids
            std::vector<int> position_ids(ctx->inputSeqLen);
            if (inputSeqLen == 1) {
                position_ids[0] = pastSeqLen;
            } else {
                std::iota(position_ids.begin(), position_ids.end(), 0);
            }
            qkpo.forward(query.Data(), key.Data(), query.Stride(), key.Stride(), qk_shape, position_ids.data());
        }
        t3.release();

#ifdef DEBUG
        dbg.debugPrint("Q after post op:\n");
        dbg.dumpMatrix(query);
        dbg.debugPrint("K after post op:\n");
        dbg.dumpMatrix(key);
#endif

        // Revise attnFactor before softmax (for some models, attnFactor may be not the default value)
        // We initially introduced the code for ChatGLM, but eventually found it has no difference and was unnecessary.
        // However, we have chosen to keep it in the codebase in case it becomes useful for future models.
        if (getScalingCoeff() != 0) { ctx->attFactor = getScalingCoeff(); }

        TimeLine t4("MHA");
        if constexpr (!INPUT_AS_RESID) {
            auto presult = resultBuffer1.Data();
            int rows = resultBuffer1.Rows(), cols = resultBuffer1.Cols(), stride = resultBuffer1.Stride();
            resultBuffer1.Assign(inputBuffer.Data(), inputBuffer.Rows(), inputBuffer.Cols(), inputBuffer.Stride());
            inputBuffer.Assign(presult, rows, cols, stride);
        }
        if (ctx->inputSeqLen > 256 && pastSeqLen == 0)
            flashAttention(ctx, qkvMatMul, resultBuffer2, resultBuffer1, presentKey, presentValue, attnMask, pastSeqLen);
        else
            fusedAttention(ctx, query, key, value, resultBuffer1, presentKey, presentValue, attnMask, pastSeqLen);
        t4.release();

        // For multiple nodes inference, not the whole result buffer
        hpj::Matrix<float> attnSplit(resultBuffer1.Data(), resultBuffer1.Rows(), resultBuffer1.Cols() / ctx->numSplit,
                resultBuffer1.Stride());

#ifdef DEBUG
        dbg.debugPrint("attention_%d (softmax * value): [%d, %d] (%d)\n", ctx->splitIdx, attnSplit.Rows(),
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
                DecoderUtil::denseWithSum(attnSplit, attnOutputWeight, attnOutputWeightScale, attnOutputWeightZero,
                        attnOutputBias, inputBuffer, resultBuffer2);
            } else {
                DecoderUtil::denseWithScaledSum(attnSplit, attnOutputWeight, attnOutputWeightScale,
                        attnOutputWeightZero, attnOutputBias, gamma, inputBuffer, resultBuffer2);
            }
        } else {
            DecoderUtil::dense(attnSplit, attnOutputWeight, attnOutputWeightScale, attnOutputWeightZero, attnOutputBias,
                    resultBuffer2);
        }
        t5.release();

#ifdef DEBUG
        dbg.debugPrint("attention output/projection:\n");
        dbg.dumpMatrix(resultBuffer2);
#endif

        if (!doLnBefore) {
            TimeLine t6("result.layer_norm");
            norm.forward(resultBuffer2.Data(), resultBuffer2.Data(), resultBuffer2.Rows(), resultBuffer2.Stride(),
                    resultBuffer2.Stride());
#ifdef DEBUG
            dbg.debugPrint("LayerNorm after attention: [%d, %d] (%d)\n", resultBuffer2.Rows(), resultBuffer2.Cols(),
                    resultBuffer2.Stride());
            dbg.dumpMatrix(resultBuffer2);
#endif
        }
    }

protected:
    template <typename KVCacheT>
    void fusedAttention(DecoderContext *ctx, hpj::Matrix<float> &query, hpj::Matrix<float> &key,
            hpj::Matrix<float> &value, hpj::Matrix<float> &result, KVCacheTensor<KVCacheT> &presentKey,
            KVCacheTensor<KVCacheT> &presentValue, const float *attnMask, int pastSeqLen) {
        // How many heads this task should do
        int responsibleHeads = this->endQHead - this->startQHead;
        int batchSize = ctx->batchSize;
        int groupNum = ctx->attHeadNum / ctx->kvHeadNum;

        // If M_dimension/input_seq_len is big (1K, 2K, etc), need to split it to make sure the intermediate result in cache
        // to make sure it works better (the logic here is trying to make sure each head of BMM result [seq * seq] in cache)
        // WARN: reserve field in context is used to make it effective for all layers, do not change it in other places
        int &mBlockSize = ctx->reserved1;
        if (layerId == 0) {
            if (pastSeqLen == 0) {
                const int l2CacheSize = 2 * 1024 * 1024; // TODO: get it dynamically
                const int sizeA = ctx->inputSeqLen * ctx->attHeadSize;
                const int sizeB = ctx->inputSeqLen * ctx->attHeadSize;
                const int sizeC = ctx->inputSeqLen * ctx->inputSeqLen;

                // As we do the split along M dimension, so to make sure:
                // (sizeA / splits) + sizeB + (sizeC / splits) <= cacheSize
                int splits = std::ceil(1.0f * (sizeA + sizeC) / (l2CacheSize / sizeof(float) - sizeB));
                mBlockSize = (ctx->inputSeqLen + splits - 1) / splits;
                if (mBlockSize <= 0) {
                    mBlockSize = ctx->inputSeqLen > 6 ? 6 : ctx->inputSeqLen;
                } else if (mBlockSize > ctx->inputSeqLen) {
                    mBlockSize = ctx->inputSeqLen;
                }
            } else {
                mBlockSize = ctx->inputSeqLen;
            }
        }

        // How many blocks in M dimension
        int mBlockNum = (ctx->inputSeqLen + mBlockSize - 1) / mBlockSize;

        // To get score buffer according to openmp thread ID or not (see below)
        int scoreBufSize = batchSize * (ctx->attHeadNum / ctx->numSplit) * ctx->inputSeqLen * ctx->inputSeqLen;
        bool scoreBufByThread = (ctx->numThreads * mBlockSize * (pastSeqLen + ctx->inputSeqLen) <= scoreBufSize);

        // For group attention, as #kvHeads != #qHeads, need to copy current key/values to cache seperately
        // When M dimension is split, also multiple tasks per copy, so do copy seperately
        bool kvCopied = false;
        if (ctx->kvHeadNum < ctx->attHeadNum || mBlockSize != ctx->inputSeqLen) {
#pragma omp parallel for collapse(3)
            for (int b = 0; b < batchSize; ++b) {
                for (int i = 0; i < (this->endKVHead - this->startKVHead); ++i) {
                    // Copy current key/value to cached keys/values
                    // Re-layout is needed: (bs, seq=1, hidden_size) -> (seq=1, bs, hidden_size)
                    // Be noted: for group attention, the key/value is less than query
                    for (int seq = 0; seq < ctx->inputSeqLen; ++seq) {
                        auto srcK = key.Row(b * ctx->inputSeqLen + seq) + i * ctx->attHeadSize;
                        auto dstK = presentKey.getSequence(pastSeqLen + seq, b, i);

                        auto srcV = value.Row(b * ctx->inputSeqLen + seq) + i * ctx->attHeadSize;
                        auto dstV = presentValue.getSequence(pastSeqLen + seq, b, i);

                        if constexpr (std::is_same_v<KVCacheT, float>) {
                            memcpy(dstK, srcK, ctx->attHeadSize * sizeof(float));
                            memcpy(dstV, srcV, ctx->attHeadSize * sizeof(float));
                        } else if constexpr (std::is_same_v<KVCacheT, float16_t>) {
                            float16_t::cvt_float_to_float16(srcK, dstK, ctx->attHeadSize);
                            float16_t::cvt_float_to_float16(srcV, dstV, ctx->attHeadSize);
                        }
                    }
                }
            }
            kvCopied = true;
        }

#pragma omp parallel for collapse(3)
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < responsibleHeads; ++i) {
                for (int mb = 0; mb < mBlockNum; ++mb) {
                    const int startSeq = mb * mBlockSize;
                    const int endSeq
                            = startSeq + mBlockSize < ctx->inputSeqLen ? startSeq + mBlockSize : ctx->inputSeqLen;

                    // Copy current key to cached keys
                    // Re-layout is needed: (bs, seq=1, hidden_size) -> (seq=1, bs, hidden_size)
                    if (!kvCopied) {
                        for (int seq = 0; seq < ctx->inputSeqLen; ++seq) {
                            auto src = key.Row(b * ctx->inputSeqLen + seq) + i * ctx->attHeadSize;
                            auto dst = presentKey.getSequence(pastSeqLen + seq, b, i);
                            if constexpr (std::is_same_v<KVCacheT, float>) {
                                memcpy(dst, src, ctx->attHeadSize * sizeof(float));
                            } else if constexpr (std::is_same_v<KVCacheT, float16_t>) {
                                float16_t::cvt_float_to_float16(src, dst, ctx->attHeadSize);
                            }
                        }
                    }

                    // Q * K
                    auto keyMatInfo = presentKey.getHead(b, i / groupNum);
                    int m = endSeq - startSeq;
                    int k = ctx->attHeadSize;
                    int n = pastSeqLen + ctx->inputSeqLen;
                    int lda = query.Stride();
                    int ldb = keyMatInfo.second;
                    int strideC = pastSeqLen > 0 ? (pastSeqLen + ctx->inputSeqLen + 15) / 16 * 16 : ctx->inputSeqLen;
                    int ldc = strideC;
                    auto A = query.Row(b * ctx->inputSeqLen + startSeq) + i * ctx->attHeadSize; // updated
                    auto B = keyMatInfo.first;
                    // Some special case, maximum required buffer size: max_thread_num * (mBlockSize * strideC)
                    // = bs * heads * mBlockNum * (mBlockSize * strideC)
                    // = bs * heads * (~seqLen) * seqLen, may be bigger than bs * heads * seqLen * seqLen
                    auto C = scoreBufByThread ? ctx->qkScores + omp_get_thread_num() * mBlockSize * strideC
                                              : ctx->qkScores + (b * responsibleHeads + i) * ctx->inputSeqLen * strideC
                                    + startSeq * strideC;

                    const int queryLen = ctx->inputSeqLen;
                    const int keyLen = pastSeqLen + ctx->inputSeqLen;

                    small_gemm_transb(
                            attnMask + b * queryLen * keyLen + startSeq * keyLen, A, B, C, m, n, k, lda, ldb, ldc);

#ifdef DEBUG
                    if (b == 0 && i == 0) {
                        dbg.debugPrint("Q * K, first head:\n");
                        auto p = ctx->qkScores;
                        dbg.debugPrint("%f, %f, %f ... %f %f %f\n", p[0] * ctx->attFactor, p[1] * ctx->attFactor,
                                p[2] * ctx->attFactor, p[keyLen - 3] * ctx->attFactor, p[keyLen - 2] * ctx->attFactor,
                                p[keyLen - 1] * ctx->attFactor);
                    }
#endif

                    // Softmax(Q * K)
                    if (pastSeqLen == 0)
                        for (int seq = 0; seq < endSeq - startSeq; ++seq) {
                            DecoderUtil::softmaxSkipMask(ctx, C + seq * strideC,
                                    attnMask + b * queryLen * keyLen + (seq + startSeq) * keyLen, keyLen);
                        }
                    else
                        for (int seq = 0; seq < endSeq - startSeq; ++seq) {
                            DecoderUtil::computeSoftmax(ctx, C + seq * strideC,
                                    attnMask + b * queryLen * keyLen + (seq + startSeq) * keyLen, keyLen);
                        }

#ifdef DEBUG
                    if (b == 0 && i == 0) {
                        dbg.debugPrint("Softmax(Q * K), first head:\n");
                        auto p = ctx->qkScores;
                        dbg.debugPrint("%f, %f, %f ... %f %f %f\n", p[0], p[1], p[2], p[keyLen - 3], p[keyLen - 2],
                                p[keyLen - 1]);
                    }
#endif

                    // Copy current value to cached values
                    // Re-layout is needed: (bs, seq, hidden_size) -> (seq, bs, hidden_size)
                    if (!kvCopied) {
                        for (int seq = 0; seq < ctx->inputSeqLen; ++seq) {
                            auto src = value.Row(b * ctx->inputSeqLen + seq) + i * ctx->attHeadSize;
                            auto dst = presentValue.getSequence(pastSeqLen + seq, b, i);
                            if constexpr (std::is_same_v<KVCacheT, float>) {
                                memcpy(dst, src, ctx->attHeadSize * sizeof(float));
                            } else if constexpr (std::is_same_v<KVCacheT, float16_t>) {
                                float16_t::cvt_float_to_float16(src, dst, ctx->attHeadSize);
                            }
                        }
                    }

                    // Softmax * V
                    auto valueMatInfo = presentValue.getHead(b, i / groupNum);
                    std::swap(k, n);
                    lda = strideC;
                    ldb = valueMatInfo.second;
                    ldc = result.Stride();
                    A = C;
                    B = valueMatInfo.first;
                    C = result.Row(b * ctx->inputSeqLen + startSeq) + i * ctx->attHeadSize;

                    if constexpr (std::is_same_v<KVCacheT, float>) {
                        ig_sgemm_single_thread(false, false, m, n, k, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
                    } else if constexpr (std::is_same_v<KVCacheT, float16_t>) {
                        ig_sgemm_f32f16f32_single_thread(false, false, m, n, k, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
                    }

#ifdef DEBUG
                    if (b == 0 && i == 0) {
                        dbg.debugPrint("Softmax(Q * K) * V, first head:\n");
                        auto p = C;
                        dbg.debugPrint("%f, %f, %f ... %f %f %f\n", p[0], p[1], p[2], p[ctx->attHeadSize - 3],
                                p[ctx->attHeadSize - 2], p[ctx->attHeadSize - 1]);
                    }
#endif
                } // end for mb
            } // end for i
        } // end for b
    }

    template <typename KVCacheT>
    void flashAttention(DecoderContext *ctx, hpj::Matrix<float> &qkvMatMul, hpj::Matrix<float> &tmpRes,
            hpj::Matrix<float> &result, KVCacheTensor<KVCacheT> &presentKey, KVCacheTensor<KVCacheT> &presentValue,
            const float *attnMask, int pastSeqLen) {

        // How many heads this task should do
        int batchSize = ctx->batchSize;
        int respQHeads = this->endQHead - this->startQHead;
        int respKVHeads = this->endKVHead - this->startKVHead;
        int qkvCols = respQHeads + respKVHeads * 2;
        int headSize = ctx->attHeadSize;
        float scale = ctx->attFactor;
        int srcLen = ctx->inputSeqLen;
        int tgtLen = pastSeqLen + srcLen;

        float *transQKV = (float*)malloc(sizeof(float) * batchSize * qkvCols * srcLen * headSize);

        DecoderUtil::transposeQKV(qkvMatMul.Data(), transQKV, batchSize, srcLen, respQHeads, respKVHeads, headSize);

        float *query = transQKV;
        float *key = transQKV + batchSize * respQHeads * srcLen * headSize;
        float *value = transQKV + batchSize * (respQHeads + respKVHeads) * srcLen * headSize;

        DecoderUtil::scaledDpAttention(query, key, value, attnMask, scale, batchSize, srcLen, tgtLen, respQHeads,
                respKVHeads, headSize, tmpRes.Data());
        DecoderUtil::transposeAttnResult(tmpRes.Data(), result.Data(), batchSize, srcLen, respQHeads, headSize,
                result.Stride());
        
        // For group attention, as #kvHeads != #qHeads, need to copy current key/values to cache seperately
        // When M dimension is split, also multiple tasks per copy, so do copy seperately
#pragma omp parallel for collapse(3)
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < (this->endKVHead - this->startKVHead); ++i) {
                // Copy current key/value to cached keys/values
                // Re-layout is needed: (bs, seq=1, hidden_size) -> (seq=1, bs, hidden_size)
                // Be noted: for group attention, the key/value is less than query
                for (int seq = 0; seq < tgtLen; ++seq) {
                    auto srcK = key + b * respKVHeads * tgtLen * headSize + i * tgtLen * headSize + seq * headSize;
                    auto dstK = presentKey.getSequence(pastSeqLen + seq, b, i);

                    auto srcV = value + b * respKVHeads * tgtLen * headSize + i * tgtLen * headSize + seq * headSize;
                    auto dstV = presentValue.getSequence(pastSeqLen + seq, b, i);

                    if constexpr (std::is_same_v<KVCacheT, float>) {
                        memcpy(dstK, srcK, headSize * sizeof(float));
                        memcpy(dstV, srcV, headSize * sizeof(float));
                    } else if constexpr (std::is_same_v<KVCacheT, float16_t>) {
                        float16_t::cvt_float_to_float16(srcK, dstK, headSize);
                        float16_t::cvt_float_to_float16(srcV, dstV, headSize);
                    }
                }
            }
        }
        free(transQKV);
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

    // query, key, value weighs
    hpj::Matrix<WeiT> qkvWeight;
    hpj::Vector<float> qkvWeightScale; // if weighs is int8
    hpj::Vector<float> qkvWeightZero; // if weighs is int8
    // query, key, value bias
    hpj::Vector<float> qkvBias;

    hpj::Matrix<WeiT> attnOutputWeight;
    hpj::Vector<float> attnOutputWeightScale; // if weighs is int8
    hpj::Vector<float> attnOutputWeightZero; // if weighs is int8
    hpj::Vector<float> attnOutputBias;

    // Query/Key post op
    QKPO_CLS qkpo;

    // layerNorm param
    NORM_CLS norm;
    int layerId;

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
