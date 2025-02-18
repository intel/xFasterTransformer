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
#include "split_util.h"

template <typename WeiT, typename QKPO_CLS, typename InT, typename ImT, typename OutT>
class DeepSeekAttention {
public:
    DeepSeekAttention(int layerId, DecoderContext *ctx) : rope(ctx) {
        if (ctx->attHeadNum == ctx->kvHeadNum) {
            // We are responsible for the range [startHead, endHead)
            auto range = SplitUtil::getHeadRange(ctx->attHeadNum, ctx->kvHeadNum, ctx->numSplit, ctx->splitIdx);
            auto headRange = range.first;
            this->startHead = headRange.first;
            this->endHead = headRange.second;
        }

        // Unexpected case
        else {
            xft::Logger::error("Un expected config: QHeads=%d, KVHeads=%d\n", ctx->attHeadNum, ctx->kvHeadNum);
            exit(-1);
        }
    }

    static xft::DataType getWeightDataType() { return xft::getDataType<WeiT>(); }

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
        // assume the weight is BF16 or FP16
        // TODO: data type convert to support other data types

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
            memcpy(buffer + i * mergedDim, (WeiT *)mlap->q_a_proj.weight + i * mlap->q_a_proj.output_dim,
                    mlap->q_a_proj.output_dim * sizeof(WeiT));
            memcpy(buffer + i * mergedDim + mlap->q_a_proj.output_dim,
                    (WeiT *)mlap->kv_a_proj.weight + i * mlap->kv_a_proj.output_dim,
                    mlap->kv_a_proj.output_dim * sizeof(WeiT));
        }

        // Pack the merged weights
        qkvAWeights.Resize(ctx->hiddenSize, mergedDim);
        xft::Matrix mergedW(buffer, ctx->hiddenSize, mergedDim, mergedDim);
        ctx->mmHelper->packWeight(mlap->q_a_proj.wtrans, mergedW, qkvAWeights);

        free(buffer);

        // Pack the weights for q_b and kv_b
        packDenseWeights(ctx, mlap->q_b_proj, qBWeights, true);
        packDenseWeights(ctx, mlap->kv_b_proj, kvBWeights, true);

        // Pack the weights for output
        packDenseWeights(ctx, mlap->o_proj, outWeights, false);

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
        bool bPrefill = (seqs[0]->getStep() == 0);
        auto hiddenSize = ctx->hiddenSize;
        auto headsOnDuty = this->endHead - this->startHead;
        xft::Matrix<InT> inputBuffer(input, totInSeqLen, hiddenSize, hiddenSize);

        int batchSize = seqs.size();
        int totAccSeqLen = 0; // total input sequence length + past sequence length
        int tokenSizes[batchSize];
        int pastSeqLens[batchSize];
        for (int i = 0; i < batchSize; ++i) {
            tokenSizes[i] = seqs[i]->getInputSeqLen();
            pastSeqLens[i] = seqs[i]->getPastSeqLen();
            totAccSeqLen += tokenSizes[i] + pastSeqLens[i];
        }

        // Norm buffer & MHA output
        // Note: many buffers are reused, need to make sure the buffer is large enough, refactor needed
        xft::Matrix<ImT> normBuffer((ImT *)ctx->normBuf.Data(), totInSeqLen, hiddenSize, hiddenSize);
        xft::Matrix<ImT> qaNormBuffer((ImT *)ctx->tmpBuf.Data(), totInSeqLen, ctx->qLoraRank, ctx->qLoraRank);
        xft::Matrix<ImT> kvNormBuffer((ImT *)ctx->tmpBuf.Data(), totInSeqLen, ctx->kvLoraRank, ctx->kvLoraRank); // reuse
        int mhaOutSize = headsOnDuty * ctx->vHeadDim;
        xft::Matrix<ImT> mhaOutBuffer((ImT *)ctx->normBuf.Data(), totInSeqLen, mhaOutSize, mhaOutSize);

        // Lora buffer (Down projection)
        auto qkvACols = qkvAWeights.Cols();
        ImT *loraBuf = (ImT *)ctx->getBuffer<ImT>("tmp", totInSeqLen * qkvACols, ctx->device);
        xft::Matrix<ImT> loraBuffer(loraBuf, totInSeqLen, qkvACols, qkvACols);

        // Up projection buffer
        auto &qkvMatMul = ctx->qkvMatMul;
        auto qCols = headsOnDuty * (ctx->nopeDim + ctx->ropeDim);
        xft::Matrix<ImT> qBuffer((ImT *)qkvMatMul.Data(), totInSeqLen, qCols, qCols);

        auto kvCols = headsOnDuty * (ctx->nopeDim + ctx->vHeadDim);
        uint64_t kvOffset = (uint64_t)qCols * totInSeqLen;
        xft::Matrix<ImT> kvBuffer(qBuffer.Data() + kvOffset, totAccSeqLen, kvCols, kvCols);

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

#ifdef XFT_DEBUG
        dbg.debugPrint("q_a_projection:\n");
        xft::Matrix<ImT> _qDown(loraBuffer.Data(), loraBuffer.Rows(), ctx->qLoraRank, loraBuffer.Stride());
        dbg.dumpMatrix(_qDown, false, ctx->device);
        dbg.debugPrint("compressed_kv:\n");
        xft::Matrix<ImT> _kvDown(loraBuffer.Data() + ctx->qLoraRank, loraBuffer.Rows(), ctx->kvLoraRank, loraBuffer.Stride());
        dbg.dumpMatrix(_kvDown, false, ctx->device);
#endif

        // Q_A_Norm
        {
            TimeLine t("q_a_norm");
            qANorm.forward(loraBuffer.Data(), qaNormBuffer.Data(), loraBuffer.Rows(), loraBuffer.Stride(),
                    qaNormBuffer.Stride(), epsilon);
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("q_a_norm:\n");
        dbg.dumpMatrix(qaNormBuffer, false, ctx->device);
#endif

        // Q up projection
        {
            TimeLine t("q_up_projection");
            ctx->mmHelper->compute(false, loraBuffer.Rows(), qBWeights.Cols(), qBWeights.Rows(), 1.0f,
                    qaNormBuffer.Data(), qaNormBuffer.Stride(), qBWeights.Data(), nullptr, nullptr, nullptr, 0.0f,
                    qBuffer.Data(), qBuffer.Stride());
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("Query (nope+rope):\n");
        dbg.dumpMatrix(qBuffer, false, ctx->device);
        dbg.debugPrint("Query rope (before pe, first head):\n");
        xft::Matrix<ImT> qFirstHead(qBuffer.Data() + ctx->nopeDim, qBuffer.Rows(), ctx->nopeDim + ctx->ropeDim, qBuffer.Stride());
        dbg.dumpMatrix(qFirstHead, false, ctx->device);
        dbg.debugPrint("Query rope (before pe, second head):\n");
        xft::Matrix<ImT> qSecondHead(qBuffer.Data() + ctx->nopeDim + ctx->ropeDim + ctx->nopeDim, qBuffer.Rows(),
                ctx->ropeDim, qBuffer.Stride());
        dbg.dumpMatrix(qSecondHead, false, ctx->device);
        dbg.debugPrint("keyRope (before pe):\n");
        xft::Matrix<ImT> _keyRope(loraBuffer.Data() + ctx->qLoraRank + ctx->kvLoraRank, totInSeqLen, ctx->ropeDim,
                loraBuffer.Stride());
        dbg.dumpMatrix(_keyRope, false, ctx->device);
#endif

        // Apply rotary embedding
        {
            TimeLine t("rope");
            std::vector<int> posIds(totInSeqLen);
            int loc = 0;
            for (auto seq : seqs) {
                std::iota(posIds.begin() + loc, posIds.begin() + loc + seq->getInputSeqLen(), seq->getPastSeqLen());
                loc += seq->getInputSeqLen();
            }
            ImT *query = qBuffer.Data() + ctx->nopeDim;
            ImT *key = loraBuffer.Data() + ctx->qLoraRank + ctx->kvLoraRank;
            rope.forward(
                    query, key, totInSeqLen, qBuffer.Stride(), loraBuffer.Stride(), headsOnDuty, 1, posIds.data());

#ifdef XFT_DEBUG
            dbg.debugPrint("query rope (after pe, first head):\n");
            xft::Matrix<ImT> _qRopeFirst(query, qBuffer.Rows(), ctx->ropeDim, qBuffer.Stride());
            dbg.dumpMatrix(_qRopeFirst, false, ctx->device);
            
            dbg.debugPrint("query rope (after pe, second head):\n");
            xft::Matrix<ImT> _qRopeSecond(
                    query + ctx->nopeDim + ctx->ropeDim, qBuffer.Rows(), ctx->ropeDim, qBuffer.Stride());
            dbg.dumpMatrix(_qRopeSecond, false, ctx->device);

            dbg.debugPrint("key rope:\n");
            xft::Matrix<ImT> _keyRope(key, totInSeqLen, ctx->ropeDim, loraBuffer.Stride());
            dbg.dumpMatrix(_keyRope, false, ctx->device);
#endif
        }

        // KV_A_Norm
        {
            TimeLine t("kv_a_norm");
            ImT *compressedKV = loraBuffer.Data() + ctx->qLoraRank;
            kvANorm.forward(compressedKV, kvNormBuffer.Data(), loraBuffer.Rows(), loraBuffer.Stride(),
                    kvNormBuffer.Stride(), epsilon);

#ifdef XFT_DEBUG
            dbg.debugPrint("kv_a_norm:\n");
            dbg.dumpMatrix(kvNormBuffer, false, ctx->device);
#endif
        }

        // Copy to KV cache (store the compressed KV)
        // TODO: fuse it with the above computation
        {
            TimeLine t("kv_cache");
            auto getK = [&](int b, int headIdx, int seqIdx) { return keyCaches[b]->getSequence(seqIdx, 0, headIdx); };
            auto getV = [&](int b, int headIdx, int seqIdx) { return valueCaches[b]->getSequence(seqIdx, 0, headIdx); };
#pragma omp parallel for
            for (int i = 0; i < totInSeqLen; ++i) {
                int b = 0, s = i;
                while (s >= tokenSizes[b]) {
                    s -= tokenSizes[b];
                    b++;
                }
                int seqIdx = s + pastSeqLens[b];
                // KeyRope
                xft::copy(getK(b, 0, seqIdx).first, loraBuffer.Row(i) + ctx->qLoraRank + ctx->kvLoraRank, ctx->ropeDim);
                // Compressed KV
                xft::copy(getV(b, 0, seqIdx).first, kvNormBuffer.Row(i), ctx->kvLoraRank);
            }
        }

        // KV up projection
        if (bPrefill) {
            TimeLine t("kv_up_projection");
            ImT *compressedKV = loraBuffer.Data() + ctx->qLoraRank;
            ctx->mmHelper->compute(false, loraBuffer.Rows(), kvBWeights.Cols(), kvBWeights.Rows(), 1.0f,
                    kvNormBuffer.Data(), kvNormBuffer.Stride(), kvBWeights.Data(), nullptr, nullptr, nullptr, 0.0f,
                    kvBuffer.Data(), kvBuffer.Stride());
        } else {
            // Compute KV up projection using cache for all sequences in the batch
            // TODO: optimization by combining into a single MatMul
            TimeLine t("kv_up_projection");
            for (int i = 0; i < batchSize; ++i) {
                auto M = tokenSizes[i] + pastSeqLens[i]; // past data also needs to be re-computed
                auto value = valueCaches[i]->getSequence(0, 0, 0);
#ifdef XFT_DEBUG
                dbg.debugPrint("cached kv_down: M=%d\n", M);
                xft::Matrix<ImT> _cachedKvDown(value.first, M, ctx->kvLoraRank, ctx->kvLoraRank);
                dbg.dumpMatrix(_cachedKvDown, false, ctx->device);
#endif
                ctx->mmHelper->compute(false, M, kvBWeights.Cols(), kvBWeights.Rows(), 1.0f, value.first,
                        ctx->kvLoraRank, kvBWeights.Data(), nullptr, nullptr, nullptr, 0.0f, kvBuffer.Data(),
                        kvBuffer.Stride());
            }
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("qBuffer:\n");
        dbg.dumpMatrix(qBuffer, false, ctx->device);
        dbg.debugPrint("kvBuffer:\n");
        dbg.dumpMatrix(kvBuffer, false, ctx->device);
        dbg.debugPrint("key rope:\n");
        dbg.dumpMatrix(_keyRope, false, ctx->device);
#endif

        // Attention
        {
            TimeLine t("MHA");
            ImT *keyRope = loraBuffer.Data() + ctx->qLoraRank + ctx->kvLoraRank;
            if (seqs[0]->getStep() == 0) { // prefill
                selfAttention16bits(ctx, qBuffer.Data(), qBuffer.Stride(), keyRope, loraBuffer.Stride(),
                        kvBuffer.Data(), kvBuffer.Stride(), mhaOutBuffer.Data(), mhaOutSize, tokenSizes,
                        batchSize);
            } else { // decoding
                const KVCacheT *keyRopes[batchSize];
                for (int i = 0; i < batchSize; ++i) {
                    keyRopes[i] = keyCaches[i]->getSequence(0, 0, 0).first;
                }
                fusedAttention(ctx, qBuffer.Data(), qBuffer.Stride(), keyRopes, ctx->ropeDim, kvBuffer.Data(),
                        kvBuffer.Stride(), mhaOutBuffer.Data(), mhaOutSize, tokenSizes, pastSeqLens, batchSize);
            }
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("MHA output:\n");
        dbg.dumpMatrix(mhaOutBuffer, false, ctx->device);   
#endif

        // Output
        {
            TimeLine t("output");
#ifdef XFT_DEBUG
            dbg.debugPrint("inputBuffer:\n", outBuffer.Data(), inputBuffer.Data());
            dbg.dumpMatrix(inputBuffer, false, ctx->device);
#endif
            if (ctx->splitIdx == 0) {
                ctx->mmHelper->compute_residential(false, qBuffer.Rows(), outWeights.Cols(), outWeights.Rows(), 1.0f,
                        mhaOutBuffer.Data(), mhaOutBuffer.Stride(), outWeights.Data(), nullptr, nullptr, nullptr, 0.0f,
                        outBuffer.Data(), outBuffer.Stride(), nullptr, inputBuffer.Data(), inputBuffer.Stride());
            } else {
                ctx->mmHelper->compute(false, qBuffer.Rows(), outWeights.Cols(), outWeights.Rows(), 1.0f,
                        mhaOutBuffer.Data(), mhaOutBuffer.Stride(), outWeights.Data(), nullptr, nullptr, nullptr, 0.0f,
                        outBuffer.Data(), outBuffer.Stride());
            }
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("attention final output:\n");
        dbg.dumpMatrix(outBuffer, false, ctx->device);
#endif
    }

private:
    void selfAttention16bits(DecoderContext *ctx, ImT *query, int qStride, ImT *keyRope, int krStride, ImT *keyValue,
            int kvStride, OutT *out, int oStride, int *tokenSizes, int batchSize) {
        static_assert(std::is_same_v<ImT, bfloat16_t> || std::is_same_v<ImT, float16_t>,
                "The data type is not supported for DS attention.");

        auto headsOnDuty = endHead - startHead;
        xft::selfAttention_MLA<true>(out, query, keyRope, keyValue, headsOnDuty, ctx->nopeDim, ctx->ropeDim,
                ctx->vHeadDim, oStride, qStride, krStride, kvStride, batchSize, tokenSizes, ctx->attFactor,
                ctx->numThreads);
    }

    void fusedAttention(DecoderContext *ctx, const ImT *query, int qStride, const ImT **keyRopes, int krStride,
            const ImT *keyValue, int kvStride, OutT *out, int oStride, int *tokenSizes, int *pastSeqLens,
            int batchSize) {
        static_assert(std::is_same_v<ImT, bfloat16_t> || std::is_same_v<ImT, float16_t>,
                "The data type is not supported for DS attention.");

        auto headsOnDuty = endHead - startHead;
        xft::crossAttnByHead_DS(out, query, keyRopes, keyValue, headsOnDuty, ctx->nopeDim, ctx->ropeDim, ctx->vHeadDim,
                oStride, qStride, krStride, kvStride, batchSize, tokenSizes, pastSeqLens, ctx->attFactor,
                ctx->numThreads);
    }

    void packDenseWeights(
            DecoderContext *ctx, xft::DenseLayerParams &dense, xft::Matrix<WeiT> &packedW, bool bVerticalSplit) {
        int splitTarget = bVerticalSplit ? dense.output_dim : dense.input_dim;
        if (splitTarget % ctx->attHeadNum != 0) {
            xft::Logger::error("The split dim %d is not divisible by %d.", splitTarget, ctx->attHeadNum);
            exit(-1);
        }

        int dimPerHead = splitTarget / ctx->attHeadNum;
        int splitOffset = startHead * dimPerHead;
        int splitSize = (endHead - startHead) * dimPerHead;

        if (bVerticalSplit) {
            xft::Matrix<WeiT> w(
                    (WeiT *)dense.weight + splitOffset, dense.input_dim, splitSize, dense.output_dim);

            packedW.Resize(dense.input_dim, splitSize);
            ctx->mmHelper->packWeight(dense.wtrans, w, packedW);
        } else {
            xft::Matrix<WeiT> w((WeiT *)dense.weight + splitOffset * dense.output_dim, splitSize, dense.output_dim,
                    dense.output_dim);

            packedW.Resize(splitSize, dense.output_dim);
            ctx->mmHelper->packWeight(dense.wtrans, w, packedW);
        }
    }

    bool isSameWeiType(xft::ParamType type) {
        if constexpr (std::is_same_v<WeiT, int8_t>) {
            return type == xft::ParamType::INT8;
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

    // TODO: write DS rotary embedding
    QKPO_CLS rope;

    int startHead;
    int endHead;

#ifdef XFT_DEBUG
    Debugger dbg;
#endif
};