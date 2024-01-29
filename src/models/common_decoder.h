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

#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "INIReader.h"
#include "abstract_decoder.h"
#include "attention.h"
#include "debugger.h"
#include "decoder_layer.h"
#include "dist_linear.h"
#include "dtype.h"
#include "kvcache_manager.h"
#include "messenger.h"
#include "timeline.h"
#include "transformer_ctx.h"
#include "transpose_util.h"
#include "weight_util.h"

using namespace xft;

struct QKPO_Dummy {
    QKPO_Dummy(int dim, int maxPos) {}
    void forward(float *query, float *key, int qStride, int kStride, const int *qk_shape, const int *position_ids) {}
};

// Template parameters:
// ATTN_CLS - class for attention impl.
// MLP_CLS - MLP implementation
// KVCacheT - data type of the cached keys/values
// ATTN_MLP_PARALLEL - true means attention and MLP are in parallel, using the same initial input
template <typename ATTN_CLS, typename MLP_CLS, typename KVCacheT = float16_t, bool ATTN_MLP_PARALLEL = false>
class CommonDecoder : public AbstractDecoder {
public:
    CommonDecoder(const std::string &modelPath, const std::string &modelType)
        : messenger(Messenger::getInstance())
#ifdef DEBUG
        , dbg("model_decoder.csv")
#endif
    {
        std::string configPath = modelPath + "/config.ini";
        INIReader reader = INIReader(configPath);
        wType = getWeightType(configPath, modelType);

        const int attHeadNum = reader.GetInteger(modelType, "head_num");
        // Use the same head number for the default multi-head attention
        const int kvHeadNum = reader.GetInteger(modelType, "kv_head_num", attHeadNum);
        const int size_per_head = reader.GetInteger(modelType, "size_per_head");
        const int imSize = reader.GetInteger(modelType, "inter_size");
        const int layers = reader.GetInteger(modelType, "num_layer");
        const int vocabSize = reader.GetInteger(modelType, "vocab_size");
        // Max Position Embedding for position embedding functions, with a default value set to 0
        const int maxPosEmbed = reader.GetInteger(modelType, "max_pos_seq_len", 0);
        // Max num of tokens that LLM can process. Also for allocating buffers. Default maxPosEmbed
        const int maxPositions = reader.GetInteger(modelType, "model_max_length", maxPosEmbed);
        // Seq length in Qwen model, if none, please ignore
        const int maxSeqLength = reader.GetInteger(modelType, "seq_length", -1);
        const int hiddenSize = attHeadNum * size_per_head;
        const int embeddingSize = hiddenSize;
        const int multi_query_group_num = reader.GetInteger(modelType, "multi_query_group_num", attHeadNum);
        const float epsilon = reader.GetFloat(modelType, "layernorm_eps", 1e-6);
        const std::string ropeType = reader.Get(modelType, "rope_scaling_type", "");
        const float ropeFactor = reader.GetFloat(modelType, "rope_scaling_factor", 1.0);
        const int ropeOrgMaxPosEmbed
                = reader.GetInteger(modelType, "rope_scaling_original_max_position_embeddings", 2048);
        const float ropeTheta = reader.GetFloat(modelType, "rope_theta", 10000.0);
        RopeParams *ropeParamsPtr = new RopeParams(ropeTheta, ropeType, ropeFactor, ropeOrgMaxPosEmbed);

        std::string act = reader.Get(modelType, "activation_type");
        std::transform(act.begin(), act.end(), act.begin(), ::tolower);

        this->startId = reader.GetInteger(modelType, "start_id", 0);
        this->endId = reader.GetInteger(modelType, "end_id", startId);

        this->initSeqLen = 0;
        this->accSeqLen = 0;

        this->prefixSeqLen = 0;
        this->prefixSharing = false;

        // Quantization config
        const bool quantDecoderWeights = reader.GetBoolean(modelType, "quant_decoder_weights", false);
        const int quantWbits = reader.GetInteger(modelType, "quant_wbits", 8);
        const int quantGroupsize = reader.GetInteger(modelType, "quant_groupsize", -1);

        DataType dt = DataType::fp32;
        if (quantDecoderWeights) {
            REQUIRES(quantWbits == 8, "Only int8 quantization is supported.");
            REQUIRES(quantGroupsize == -1, "Quantization with groupsize is not supported.");
            dt = DataType::int8;
        }

        // Buffer related (not initialized)
        this->inputTokens = nullptr;
        this->maskSize = 0;
        this->attnMask = nullptr;
        embBuf.reset(new hpj::Matrix<float>());
        outBuf.reset(new hpj::Matrix<float>());

        // Context
        DecoderContext *ctx = getDecoderContext(layers, hiddenSize, attHeadNum, kvHeadNum, imSize, act, epsilon,
                vocabSize, embeddingSize, maxPositions, maxPosEmbed, maxSeqLength, ropeParamsPtr);

        // Decoder
        for (int i = 0; i < layers; ++i) {
            auto pdec = new DECODER(ctx, i);
            if (dt == DataType::int8) {
                this->setDecoderWeights<int8_t>(pdec, modelPath, i);
            } else if (dt == DataType::fp32) {
                this->setDecoderWeights<float>(pdec, modelPath, i);
            }
            this->decoders.push_back(pdec);
        }

        // Predictor
        int workers = messenger.getSize();
        int rank = messenger.getRank();
        this->predictor = new DistLinear<float16_t>(hiddenSize, vocabSize, rank, workers);
        this->setPredictorWeight(modelPath);

        // KVCache Manager
        this->kvCacheMgr.reset(new KVCacheManager<KVCacheT>(layers));
    }

    virtual ~CommonDecoder() {
        if (this->inputTokens) free(this->inputTokens);
        if (this->attnMask) free(this->attnMask);

        delete this->predictor;

        for (auto dec : this->decoders) {
            delete dec;
        }
    }

    std::tuple<float *, int, int> forward(int *ids, int64_t *dims, int step, bool logitsAll = false) {
        // Assume input has been synced with master in higher level.
        // Assume the 1st step input's shape is [userSideBS][1][seqLen].
        TimeLine t("Decoder.forward");
        TimeLine t1("Decoder.embedding");

        int userSideBS = dims[0];
        int beamSize = dims[1];
        int batchSize = (step == 0 ? userSideBS : userSideBS * beamSize); // as samples are duplicated at step 0
        int seqLen = dims[2];
        int pastSeqLen = step == 0 ? 0 : this->accSeqLen;
        int inputSeqLen = seqLen;

        // Prepare context
        DecoderContext *ctx = this->getContext();
        ctx->resize(batchSize, seqLen, pastSeqLen);

        if (step == 0) {
            // Reset initial and accumulated sequence length at the first step
            this->initSeqLen = seqLen;
            this->accSeqLen = 0;
            if (this->prefixSharing) {
                pastSeqLen = this->prefixSeqLen;
                inputSeqLen = seqLen - pastSeqLen;

                int *prefixIDs = (int *)malloc(userSideBS * pastSeqLen * sizeof(int));
                int *newIDs = (int *)malloc(userSideBS * inputSeqLen * sizeof(int));
                for (int bs = 0; bs < userSideBS; bs++) {
                    memcpy(prefixIDs + pastSeqLen * bs, ids + seqLen * bs, pastSeqLen * sizeof(int));
                    memcpy(newIDs + inputSeqLen * bs, ids + seqLen * bs + pastSeqLen, inputSeqLen * sizeof(int));
                }

                this->getPositionIds(prefixIDs, batchSize, pastSeqLen, 0);

                free(prefixIDs);
                ids = newIDs;
                ctx->resize(batchSize, inputSeqLen, pastSeqLen);
            }

            // Enlarge buffer if needed
            prepareBuffers(ctx, userSideBS, beamSize, logitsAll);
        }

        // Embedding
        this->embeddingForward(ids, this->embBuf->Data(), batchSize, inputSeqLen);
        this->accSeqLen += seqLen;

#ifdef DEBUG
        dbg.debugPrint("---- embedding.forward ----\n");
        dbg.debugPrint("ids:\n");
        dbg.dumpMatrix(ids, batchSize, inputSeqLen, inputSeqLen);
        dbg.debugPrint("embBuf(rows: %d, cols: %d, stride: %d):\n", this->embBuf->Rows(), this->embBuf->Cols(),
                this->embBuf->Stride());
        dbg.dumpMatrix(*this->embBuf);
#endif

        // Prepare attention mask
        this->prepareAttnMask(ids, step + this->prefixSharing);

        // Token position ids, note: different models may have different impl.
        int *positionIds = this->getPositionIds(ids, batchSize, inputSeqLen, step + this->prefixSharing);
        t1.release();

        // Decoder: forward
        int hiddenSize = ctx->hiddenSize;
        for (int i = 0; i < this->decoders.size(); ++i) {
            int workers = this->messenger.getSize();
            if (step == 0 && this->prefixSharing) {
                // Expand the prefix KV cache for each batch
                this->kvCacheMgr->expandPrefixCache(i, userSideBS, this->prefixSeqLen);
            }
            KVCacheTensor<KVCacheT> &presentKey = this->kvCacheMgr->getKey(i);
            KVCacheTensor<KVCacheT> &presentValue = this->kvCacheMgr->getValue(i);

            // Pls be noted: in attention, 'outBuf' is used as imtermediate buffer, 'tmpBuf' is used as output
            auto &attnOut = this->getContext()->tmpBuf;
            this->decoders[i]->forwardAttention(getContext(), this->embBuf->Data(), this->outBuf->Data(),
                    attnOut.Data(), attnMask,
                    presentKey, // presentKey,
                    presentValue, // presentValue,
                    inputSeqLen, // inputSeqLen,
                    pastSeqLen, // pastSeqLen
                    step == 0, // useSelfAttn,
                    true, // doLnBefore,
                    positionIds);

            // Expand the KV cache as it only has values for beam 0
            if (step == 0 && beamSize > 1) { this->kvCacheMgr->expandCache(i, userSideBS, beamSize, seqLen); }

            // Merge the result of attention
            // When attention and FFN/MLP are in parallel, do not need to reduce after attention
            if constexpr (!ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) {
                    this->messenger.reduceAdd(
                            attnOut.Data(), attnOut.Data(), batchSize * inputSeqLen * attnOut.Stride());
                }
            }

            // When attention and FFN/MLP are in parallel, use the initial embedding as input
            if constexpr (ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) {
                    this->decoders[i]->forwardFFN(
                            getContext(), this->embBuf->Data(), this->outBuf->Data(), hiddenSize, hiddenSize, true);
                    this->messenger.reduceAdd(
                            this->outBuf->Data(), this->embBuf->Data(), batchSize * inputSeqLen * hiddenSize);
                } else {
                    this->decoders[i]->forwardFFN(
                            getContext(), this->embBuf->Data(), this->embBuf->Data(), hiddenSize, hiddenSize, true);
                }
            } else {
                // FFN (for multiple workers, output into outBuf and then reduce add to embBuf)
                if (this->messenger.getSize() > 1) {
                    this->decoders[i]->forwardFFN(
                            getContext(), attnOut.Data(), this->outBuf->Data(), attnOut.Stride(), hiddenSize, true);
                    this->messenger.reduceAdd(
                            this->outBuf->Data(), this->embBuf->Data(), batchSize * inputSeqLen * hiddenSize);
                } else {
                    this->decoders[i]->forwardFFN(
                            getContext(), attnOut.Data(), this->embBuf->Data(), attnOut.Stride(), hiddenSize, true);
                }
            }
        }

        // Prepare input for final Layer Norm (only care about the last row of the result)
        // Shape of embBuf: (bs, seqLen, hiddenSize)
        float *lnIn = this->embBuf->Data();
        if (inputSeqLen > 1 && !logitsAll) { // copy is not needed when seqLen = 1 or logitsAll is true
            lnIn = this->outBuf->Data();
#pragma omp parallel for
            for (int b = 0; b < batchSize; ++b) {
                memcpy(lnIn + b * hiddenSize, this->embBuf->Data() + ((b + 1) * inputSeqLen - 1) * hiddenSize,
                        hiddenSize * sizeof(float));
            }
        }

#ifdef DEBUG
        dbg.debugPrint("LayerNorm In:\n");
        dbg.dumpMatrix(lnIn, batchSize, hiddenSize, hiddenSize);
#endif

        // LN, as it supports inplace computing, input and output can be the same
        float *lnOut = this->embBuf->Data();
        if (!logitsAll)
            lastLayerNormForward(lnIn, lnOut, batchSize);
        else
            lastLayerNormForward(lnIn, lnOut, batchSize * seqLen);

#ifdef DEBUG
        dbg.debugPrint("LayerNorm Out:\n");
        dbg.dumpMatrix(lnOut, batchSize, hiddenSize, hiddenSize);
#endif

        // Predictor
        if (!logitsAll)
            this->predictor->forward(lnOut, this->outBuf->Data(), batchSize);
        else
            this->predictor->forward(lnOut, this->outBuf->Data(), batchSize * seqLen);

#ifdef DEBUG
        auto splitSize = this->predictor->getSplitSize();
        dbg.debugPrint("outBuf:\n");
        dbg.dumpMatrix(outBuf->Data(), batchSize, splitSize, splitSize);
#endif

        // Expand the result to make it cover multiple beams
        if (step == 0 && beamSize > 1) {
            const int splitSize = this->predictor->getSplitSize();
            for (int b = userSideBS - 1; b >= 0; --b) {
                float *src = this->outBuf->Data() + b * splitSize;
#pragma omp parallel for
                for (int idx = b * beamSize; idx < (b + 1) * beamSize; ++idx) {
                    if (idx == b) { continue; }
                    float *dst = this->outBuf->Data() + idx * splitSize;
                    memcpy(dst, src, splitSize * sizeof(float));
                }
            }
        }

        // free temporary new ids for prefix sharing
        if (step == 0 && this->prefixSharing) { free(ids); }

        return std::tuple<float *, int, int>(
                this->outBuf->Data(), this->predictor->getSplitOffset(), this->predictor->getSplitSize());
    }

    void setPrefix(int *ids, int seqLen) {
        this->prefixSharing = true;
        this->prefixSeqLen = seqLen;
        prefixForward(ids, seqLen);
    }

    void unsetPrefix() { this->prefixSharing = false; }

    void prefixForward(int *ids, int seqLen) {
        // Assume input has been synced with master in higher level.
        // Assume the prefix token's shape is [1][1][seqLen].
        TimeLine t("Decoder.prefixForward");
        TimeLine t1("Decoder.prefixEmbedding");

        // Prepare context
        DecoderContext *ctx = this->getContext();
        ctx->resize(1, seqLen, 0);

        prepareBuffers(ctx, 1, 1, false, true);

        // Embedding
        this->embeddingForward(ids, this->embBuf->Data(), 1, seqLen);

        // Prepare attention mask
        this->prepareAttnMask(ids, 0);

        // Token position ids, note: different models may have different impl.
        int *positionIds = this->getPositionIds(ids, 1, seqLen, 0);
        t1.release();

        // Decoder: forward
        int hiddenSize = ctx->hiddenSize;
        for (int i = 0; i < this->decoders.size(); ++i) {
            int workers = this->messenger.getSize();
            KVCacheTensor<KVCacheT> &presentKey = this->kvCacheMgr->getPrefixKey(i);
            KVCacheTensor<KVCacheT> &presentValue = this->kvCacheMgr->getPrefixValue(i);

            // Pls be noted: in attention, 'outBuf' is used as imtermediate buffer, 'tmpBuf' is used as output
            auto &attnOut = this->getContext()->tmpBuf;
            this->decoders[i]->forwardAttention(getContext(), this->embBuf->Data(), this->outBuf->Data(),
                    attnOut.Data(), attnMask,
                    presentKey, // presentKey,
                    presentValue, // presentValue,
                    seqLen, // inputSeqLen,
                    0, // pastSeqLen
                    true, // useSelfAttn,
                    true, // doLnBefore,
                    positionIds);

            // Merge the result of attention
            // When attention and FFN/MLP are in parallel, do not need to reduce after attention
            if constexpr (!ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) {
                    this->messenger.reduceAdd(attnOut.Data(), attnOut.Data(), seqLen * attnOut.Stride());
                }
            }

            // When attention and FFN/MLP are in parallel, use the initial embedding as input
            if constexpr (ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) {
                    this->decoders[i]->forwardFFN(
                            getContext(), this->embBuf->Data(), this->outBuf->Data(), hiddenSize, hiddenSize, true);
                    this->messenger.reduceAdd(this->outBuf->Data(), this->embBuf->Data(), seqLen * hiddenSize);
                } else {
                    this->decoders[i]->forwardFFN(
                            getContext(), this->embBuf->Data(), this->embBuf->Data(), hiddenSize, hiddenSize, true);
                }
            } else {
                // FFN (for multiple workers, output into outBuf and then reduce add to embBuf)
                if (this->messenger.getSize() > 1) {
                    this->decoders[i]->forwardFFN(
                            getContext(), attnOut.Data(), this->outBuf->Data(), attnOut.Stride(), hiddenSize, true);
                    this->messenger.reduceAdd(this->outBuf->Data(), this->embBuf->Data(), seqLen * hiddenSize);
                } else {
                    this->decoders[i]->forwardFFN(
                            getContext(), attnOut.Data(), this->embBuf->Data(), attnOut.Stride(), hiddenSize, true);
                }
            }
        }
    }

    // Reorder cached keys and values, size=batchSize*beamSize
    void reorderCache(int *idx, int size) { kvCacheMgr->reorderCache(idx, size, initSeqLen, accSeqLen); }

    // Get decoder context
    DecoderContext *getContext() { return context.get(); }

    // How many layers
    int getLayers() { return decoders.size(); }

    Messenger &getMessenger() { return messenger; }

    int getRank() { return messenger.getRank(); }

    WDataType getDataType() { return wType; }

    int getEndId() { return endId; }

    int getInitSeqLen() { return initSeqLen; }

    std::tuple<std::shared_ptr<DecoderContext>, std::shared_ptr<KVCacheManager<KVCacheT>>,
            std::shared_ptr<hpj::Matrix<float>>, std::shared_ptr<hpj::Matrix<float>>>
    getSharedResources() {
        return std::make_tuple(context, kvCacheMgr, embBuf, outBuf);
    }

    void setSharedResources(const std::tuple<std::shared_ptr<DecoderContext>, std::shared_ptr<KVCacheManager<KVCacheT>>,
            std::shared_ptr<hpj::Matrix<float>>, std::shared_ptr<hpj::Matrix<float>>> &r) {
        this->context = std::get<0>(r);
        this->kvCacheMgr = std::get<1>(r);
        this->embBuf = std::get<2>(r);
        this->outBuf = std::get<3>(r);
    }

    // When first step is skipped, call this function to make everything aligned
    void skipFirstStep(int initSeqLen) {
        // Reset initial and accumulated sequence length at the first step
        this->initSeqLen = initSeqLen;
        this->accSeqLen = initSeqLen;
    }

protected:
    using DECODER = Decoder<ATTN_CLS, MLP_CLS>;

    static bool fileExists(const std::string &filename) {
        std::ifstream file(filename);
        return file.good();
    }

    DecoderContext *getDecoderContext(int layers, const int hiddenSize, const int attHeadNum, const int kvHeadNum,
            const int imSize, const std::string &act, const float epsilon, int vocabSize, int embeddingSize,
            int maxPositions, int maxPosEmbed, int maxSeqLength, RopeParams *ropeParamsPtr) {
        int splits = messenger.getSize();
        int splitIdx = messenger.getRank();

        if (context != nullptr) {
            if (context->hiddenSize == hiddenSize && context->attHeadNum == attHeadNum
                    && context->kvHeadNum == kvHeadNum && context->intermediateSize == imSize
                    && context->splitIdx == splitIdx) {
                return context.get();
            } else {
                printf("Different context size not unsupported!\n");
                exit(-1);
            }
        } else {
            this->context.reset(new DecoderContext(layers, hiddenSize, attHeadNum, kvHeadNum, imSize, act, epsilon,
                    vocabSize, embeddingSize, maxPositions, maxPosEmbed, maxSeqLength, splitIdx, splits, ropeParamsPtr));
        }

        return this->context.get();
    }

    // SrcT: float or int8_t
    template <typename SrcT>
    void setDecoderWeights(DECODER *pdecoder, const std::string &modelPath, int layerIdx) {
        const int hiddenSize = getContext()->hiddenSize;
        const int imSize = getContext()->intermediateSize;
        const int kvHeadNum = getContext()->kvHeadNum;
        const int attHeadSize = getContext()->attHeadSize;
        const int mlpFactor = (getContext()->actType == DecoderContext::SWIGLU) ? 2 : 1;
        int qSize = hiddenSize;
        int kvSize = attHeadSize * kvHeadNum;
        int qkvSize = qSize + kvSize + kvSize;

#define ALLOC(size, alignment) aligned_alloc((alignment), (size))
        SrcT *qkvWeight = (SrcT *)ALLOC(hiddenSize * qkvSize * sizeof(SrcT), 64);
        float *qkvScales = nullptr;
        float *qkvZeros = nullptr;
        float *qkvBias = (float *)ALLOC(qkvSize * sizeof(float), 64);

        SrcT *attnOutWeight = (SrcT *)ALLOC(hiddenSize * hiddenSize * sizeof(SrcT), 64);
        float *attnOutScales = nullptr;
        float *attnOutZeros = nullptr;
        float *attnOutBias = (float *)ALLOC(hiddenSize * sizeof(float), 64);

        SrcT *fc1Weight = (SrcT *)ALLOC(hiddenSize * imSize * mlpFactor * sizeof(SrcT), 64);
        float *fc1Scales = nullptr;
        float *fc1Zeros = nullptr;
        float *fc1Bias = (float *)ALLOC(imSize * sizeof(float), 64);

        SrcT *fc2Weight = (SrcT *)ALLOC(hiddenSize * imSize * sizeof(SrcT), 64);
        float *fc2Scales = nullptr;
        float *fc2Zeros = nullptr;
        float *fc2Bias = (float *)ALLOC(hiddenSize * sizeof(float), 64);

        float *ln1Gamma = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *ln1Beta = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *ln2Gamma = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *ln2Beta = (float *)ALLOC(hiddenSize * sizeof(float), 64);

        SrcT *fc3Weight = nullptr;
        float *fc3Scales = nullptr;
        float *fc3Zeros = nullptr;

        // INT8 quant, wbits = 8, qweight dtype: int8
        if constexpr (std::is_same_v<SrcT, int8_t>) {
            qkvZeros = (float *)ALLOC(qkvSize * sizeof(float), 64);
            qkvScales = (float *)ALLOC(qkvSize * sizeof(float), 64);
            attnOutZeros = (float *)ALLOC(hiddenSize * sizeof(float), 64);
            attnOutScales = (float *)ALLOC(hiddenSize * sizeof(float), 64);
            fc1Zeros = (float *)ALLOC(imSize * mlpFactor * sizeof(float), 64);
            fc1Scales = (float *)ALLOC(imSize * mlpFactor * sizeof(float), 64);
            fc2Zeros = (float *)ALLOC(imSize * sizeof(float), 64);
            fc2Scales = (float *)ALLOC(imSize * sizeof(float), 64);

            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx)
                            + ".attention.query_key_value.qweight.0.bin",
                    qkvWeight, hiddenSize * qkvSize, WDataType::INT8);
            loadWeight(
                    modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.zeros.0.bin",
                    qkvZeros, qkvSize, WDataType::FP32);
            loadWeight(
                    modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.scales.0.bin",
                    qkvScales, qkvSize, WDataType::FP32);

            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.qweight.0.bin",
                    attnOutWeight, hiddenSize * hiddenSize, WDataType::INT8);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.zeros.0.bin",
                    attnOutZeros, hiddenSize, WDataType::FP32);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.scales.0.bin",
                    attnOutScales, hiddenSize, WDataType::FP32);

            // Stardard 2 layer MLP
            if (fileExists(
                        modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.qweight.0.bin")) {
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.qweight.0.bin",
                        fc1Weight, hiddenSize * imSize * mlpFactor, WDataType::INT8);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.zeros.0.bin",
                        fc1Zeros, imSize * mlpFactor, WDataType::FP32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.scales.0.bin",
                        fc1Scales, imSize * mlpFactor, WDataType::FP32);

                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.qweight.0.bin",
                        fc2Weight, hiddenSize * imSize, WDataType::INT8);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.zeros.0.bin",
                        fc2Zeros, hiddenSize, WDataType::FP32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.scales.0.bin",
                        fc2Scales, hiddenSize, WDataType::FP32);
            }
            // gate, up, down weights for Llama like model
            else {
                fc3Weight = (SrcT *)ALLOC(hiddenSize * imSize * sizeof(SrcT), 64);
                fc3Zeros = (float *)ALLOC(hiddenSize * sizeof(float), 64);
                fc3Scales = (float *)ALLOC(hiddenSize * sizeof(float), 64);

                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.qweight.0.bin",
                        fc1Weight, hiddenSize * imSize * mlpFactor, WDataType::INT8);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.zeros.0.bin",
                        fc1Zeros, imSize * mlpFactor, WDataType::FP32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.scales.0.bin",
                        fc1Scales, imSize * mlpFactor, WDataType::FP32);

                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.qweight.0.bin",
                        fc2Weight, hiddenSize * imSize, WDataType::INT8);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.zeros.0.bin",
                        fc2Zeros, imSize, WDataType::FP32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.scales.0.bin",
                        fc2Scales, imSize, WDataType::FP32);

                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.qweight.0.bin",
                        fc3Weight, hiddenSize * imSize, WDataType::INT8);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.zeros.0.bin",
                        fc3Zeros, hiddenSize, WDataType::FP32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.scales.0.bin",
                        fc3Scales, hiddenSize, WDataType::FP32);
            }

        } else if constexpr (std::is_same_v<SrcT, float>) {
            loadWeight(
                    modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.weight.0.bin",
                    qkvWeight, hiddenSize * qkvSize, getDataType());
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.weight.0.bin",
                    attnOutWeight, hiddenSize * hiddenSize, getDataType());

            // Stardard 2 layer MLP
            if (fileExists(
                        modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.weight.0.bin")) {
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.weight.0.bin",
                        fc1Weight, hiddenSize * imSize * mlpFactor, getDataType());
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.weight.0.bin",
                        fc2Weight, hiddenSize * imSize, getDataType());
            }
            // gate, up, down weights for Llama like model
            else {
                fc3Weight = (SrcT *)ALLOC(hiddenSize * imSize * sizeof(SrcT), 64);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.weight.0.bin",
                        fc1Weight, hiddenSize * imSize * mlpFactor, getDataType());
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.weight.0.bin",
                        fc2Weight, hiddenSize * imSize, getDataType());
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.weight.0.bin",
                        fc3Weight, hiddenSize * imSize, getDataType());
            }
        }

        loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".input_layernorm.weight.bin", ln1Gamma,
                hiddenSize, getDataType());
        loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".post_attention_layernorm.weight.bin",
                ln2Gamma, hiddenSize, getDataType());

#define READ_OPTIONAL(filename, addr, size, errmsg)                             \
    {                                                                           \
        int ret = loadWeight((filename), (addr), (size), getDataType(), false); \
        if (ret == 0) {                                                         \
            free(addr);                                                         \
            addr = nullptr;                                                     \
        } else {                                                                \
            if (ret != (size)) {                                                \
                printf("%s\n", (errmsg));                                       \
                exit(-1);                                                       \
            }                                                                   \
        }                                                                       \
    }

        // The bias is optional
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.bias.0.bin",
                qkvBias, qkvSize, "read QKV bias error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.bias.bin",
                attnOutBias, hiddenSize, "read attn dense bias error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".input_layernorm.bias.bin", ln1Beta,
                hiddenSize, "read LN1 beta error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".post_attention_layernorm.bias.bin",
                ln2Beta, hiddenSize, "read LN2 beta error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.bias.0.bin",
                fc1Bias, imSize, "read FC1 bias error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.bias.bin", fc2Bias,
                hiddenSize, "read FC2 bias error");

        pdecoder->setWeights(getContext(), qkvWeight, qkvScales, qkvZeros, qkvBias, qkvWeight + qSize,
                qkvScales + qSize, qkvZeros + qSize, qkvBias + qSize, qkvWeight + qSize + kvSize,
                qkvScales + qSize + kvSize, qkvZeros + qSize + kvSize, qkvBias + qSize + kvSize, attnOutWeight,
                attnOutScales, attnOutZeros, attnOutBias, ln1Gamma, ln1Beta, fc1Weight, fc1Scales, fc1Zeros, fc1Bias,
                fc2Weight, fc2Scales, fc2Zeros, fc2Bias, ln2Gamma, ln2Beta, fc3Weight, fc3Scales, fc3Zeros, false);

        free(qkvWeight);
        free(attnOutWeight);
        free(fc1Weight);
        free(fc2Weight);
        free(fc3Weight);
        free(qkvZeros);
        free(attnOutZeros);
        free(fc1Zeros);
        free(fc2Zeros);
        free(fc3Zeros);
        free(qkvScales);
        free(attnOutScales);
        free(fc1Scales);
        free(fc2Scales);
        free(fc3Scales);
        free(qkvBias);
        free(attnOutBias);
        free(fc1Bias);
        free(fc2Bias);
        free(ln1Gamma);
        free(ln1Beta);
        free(ln2Gamma);
        free(ln2Beta);
    }

    void setPredictorWeight(const std::string &modelPath) {
        int inputSize = predictor->getInputSize();
        int outputSize = predictor->getOutputSize();

        float *weight = (float *)malloc(inputSize * outputSize * sizeof(float));
        float *bias = nullptr;

        loadWeight(modelPath + "/model.lm_head.weight.bin", weight, inputSize * outputSize, this->getDataType());

        predictor->setWeight(weight, bias);

        free(weight);
    }

    virtual void prepareBuffers(
            DecoderContext *ctx, int userSideBS, int beamSize, bool logitsAll = false, bool prefix = false) {
        int batchSize = ctx->batchSize;
        int hiddenSize = ctx->hiddenSize;
        int seqLen = ctx->inputSeqLen;
        int vocabSize = ctx->vocabSize;
        int maxPositions = ctx->maxPositions;
        int layers = this->decoders.size();
        int workers = this->messenger.getSize();

        // Prepare buffers (embBuf & outBuf), userSideBS * beamSize is the output rows really needed
        int logitsLen = logitsAll ? batchSize * seqLen : userSideBS * beamSize;
        int requiredRows = batchSize * seqLen;

        // The required output buffer size is bigger than the embedding size
        if (logitsLen * vocabSize > batchSize * seqLen * hiddenSize) {
            requiredRows = logitsLen * vocabSize / hiddenSize + 1;
        }
        if (requiredRows > this->embBuf->Rows()) {
            this->embBuf->Resize(requiredRows, hiddenSize);
            this->outBuf->Resize(requiredRows, hiddenSize);
        }

        // Attention mask
        int sizeRequired = batchSize * seqLen * seqLen;
        getAttnMask(sizeRequired);

        // Cached keys/values
        // The maximum sequence length is to be the same as maxPositions, at most
        // And the cache always needs to account for beam size
        int headsPerSplit = (ctx->kvHeadNum + workers - 1) / workers;
        this->kvCacheMgr->resize(prefix ? this->prefixSeqLen : maxPositions, userSideBS * beamSize, headsPerSplit,
                ctx->attHeadSize, prefix);
    }

    float *getAttnMask(int sizeRequired) {
        if (this->maskSize < sizeRequired) {
            if (this->attnMask) free(this->attnMask);
            this->attnMask = (float *)aligned_alloc(64, sizeRequired * sizeof(float));
            this->maskSize = sizeRequired;
        }
        return this->attnMask;
    }

    int getStartId() { return startId; }

    virtual void embeddingForward(int *ids, float *output, int batchSize, int seqLen) = 0;
    virtual void lastLayerNormForward(float *input, float *output, int rows) = 0;
    virtual void prepareAttnMask(int *ids, int step) = 0;

public:
    virtual int *getPositionIds(int *ids, int batchSize, int seqLen, int step) { return nullptr; }

protected:
    // For communication
    Messenger &messenger;

    // Execution context
    std::shared_ptr<DecoderContext> context;

    // The initial input sequence length, which is the prompt token size
    int initSeqLen;
    // Accumulated sequence length, = past_seq_len + current_seq_len
    int accSeqLen;
    // The prefix input  sequence length
    int prefixSeqLen;

    bool prefixSharing;

    // If not the master, need to receive token IDs from the master
    int *inputTokens;

    std::shared_ptr<KVCacheManager<KVCacheT>> kvCacheMgr;

    std::shared_ptr<hpj::Matrix<float>> embBuf; // used to store the embedding result
    std::shared_ptr<hpj::Matrix<float>> outBuf; // output buffer for decoder layers, same size as embBuf

protected:
    // Components most LLMs may use
    std::vector<DECODER *> decoders;
    DistLinear<float16_t> *predictor;

private:
    int maskSize; // size of allocated attnMask
    float *attnMask; // attention mask, set as private as may need to enlarge

    int startId;
    int endId;

    WDataType wType;

#ifdef DEBUG
    Debugger dbg;
#endif
};
