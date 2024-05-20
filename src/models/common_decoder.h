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
#include "datatypes.h"
#include "debugger.h"
#include "decoder_block.h"
#include "decoder_layer.h"
#include "dist_linear.h"
#include "dtype.h"
#include "kvcache_manager.h"
#include "messenger.h"
#include "mlp_chatglm2.h"
#include "mlp_standard.h"
#include "model_factory.h"
#include "sequence.h"
#include "timeline.h"
#include "transformer_ctx.h"
#include "transpose_util.h"
#include "weight_util.h"

using namespace xft;

struct QKPO_Dummy {
    QKPO_Dummy(int dim, int maxPos) {}
    void forward(float *query, float *key, int qStride, int kStride, const int *qk_shape, const int *position_ids) {}
    void forward(float *query, float *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads,
            int *positionIds) {};
};

// To get data types in MLP class
template <typename T>
struct MlpTypeExtractor;
template <template <typename...> class MLP_CLS, typename WeiT, typename InT, typename ImT, typename OutT>
struct MlpTypeExtractor<MLP_CLS<WeiT, InT, ImT, OutT>> {
    using Tin = InT;
    using Tim = ImT;
    using Tout = OutT;
};
template <typename WeiT, typename InT, typename ImT, typename OutT>
struct MlpTypeExtractor<MLP<WeiT, InT, ImT, OutT, true>> {
    using Tin = InT;
    using Tim = ImT;
    using Tout = OutT;
};
template <typename WeiT, typename InT, typename ImT, typename OutT, typename NORM_CLS>
struct MlpTypeExtractor<ChatGLM2MLP<WeiT, InT, ImT, OutT, NORM_CLS, true>> {
    using Tin = InT;
    using Tim = ImT;
    using Tout = OutT;
};

/*
Pipeline parallel and tensor parallel introduction:

  1) MPI_Instances = 16,XFT_PIPELINE_STAGE = 4  =>  ctx->ppSize = 4, ctx->tpSize = 4
  2) TP sync by oneCCL(row_comm) or shared_memory
  3) PP sync by MPI MPI_COMM_WORLD

  World Rank:      => Row Rank:       => Rank:  tp0 tp1 tp2 tp3
  [ 0,  1,  2,  3,    [ 0, 1, 2, 3];      pp0 [  0,  1,  2,  3];
    4,  5,  6,  7,    [ 0, 1, 2, 3];      pp1 [  0,  1,  2,  3];
    8,  9, 10, 11,    [ 0, 1, 2, 3];      pp2 [  0,  1,  2,  3];
   12, 13, 14, 15];   [ 0, 1, 2, 3];      pp3 [  0,  1,  2,  3];

                                      Prompts
                                         │
            ┌──────────────────┬─────────┴────────┬──────────────────┐
            │                  │                  │                  │
            ▼                  ▼                  ▼                  ▼
       Embedding(PP0)     Embedding(PP0)     Embedding(PP0)     Embedding(PP0)
            │                  │                  │                  │
  PP0       │                  │                  │                  │
  ┌─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┐
  │ TP0     │          TP1     │          TP2     │          TP3     │    layer0-7  │
  │ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐     │
  │ │ OMP            │ │ OMP            │ │ OMP            │ │ OMP            │     │
  │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │     │
  │ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│     │
  │ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘     │
  │ ┌───────┼──────────────────┼─────AllReduce────┼──────────────────┼────────┐     │
  │ └───────┼──────────────────┼──────────────────┼──────────────────┼────────┘     │
  └─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┘
  PP1       │ MPI Send/Recv    │                  │                  │
  ┌─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┐
  │ TP0     │          TP1     │           TP2    │            TP3   │   layer8-15  │
  │ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐     │
  │ │ OMP            │ │ OMP            │ │ OMP            │ │ OMP            │     │
  │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │     │
  │ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│     │
  │ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘     │
  │ ┌───────┼──────────────────┼─────AllReduce────┼──────────────────┼────────┐     │
  │ └───────┼──────────────────┼──────────────────┼──────────────────┼────────┘     │
  └─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┘
  PP2       │ MPI Send/Recv    │                  │                  │
  ┌─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┐
  │ TP0     │          TP1     │           TP2    │            TP3   │  layer16-23  │
  │ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐     │
  │ │ OMP            │ │ OMP            │ │ OMP            │ │ OMP            │     │
  │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │     │
  │ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│     │
  │ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘     │
  │ ┌───────┼──────────────────┼─────AllReduce────┼──────────────────┼────────┐     │
  │ └───────┼──────────────────┼──────────────────┼──────────────────┼────────┘     │
  └─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┘
  PP3       │ MPI Send/Recv    │                  │                  │
  ┌─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┐
  │ TP0     │          TP1     │           TP2    │            TP3   │  layer24-31  │
  │ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐     │
  │ │ OMP            │ │ OMP            │ │ OMP            │ │ OMP            │     │
  │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │     │
  │ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│     │
  │ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘     │
  │ ┌───────┼──────────────────┼─────AllReduce────┼──────────────────┼────────┐     │
  │ └───────┼──────────────────┼──────────────────┼──────────────────┼────────┘     │
  └─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┘
            │                  │                  │                  │
            ▼                  ▼                  ▼                  ▼
       Predictor(PP3)     Predictor(PP3)     Predictor(PP3)     Predictor(PP3)
            │ MPI Send/Recv    │                  │                  │
            ▼                  ▼                  ▼                  ▼
       Searchers(PP0)     Searchers(PP0)     Searchers(PP0)     Searchers(PP0)
            │
            ▼
         Output
*/

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
        // Make sure Attention output can be feed to MLP
        static_assert(std::is_same_v<AttnOutT, MlpInT>, "Error: Attention Output and MLP Input are not the same type.");

        // Make sure MLP output can be feed to Attention
        static_assert(std::is_same_v<MlpOutT, AttnInT>, "Error: MLP Output and Attention Input are not the same type.");

        std::string configPath = modelPath + "/config.ini";
        INIReader reader = INIReader(configPath);
        const int attHeadNum = reader.GetInteger(modelType, "head_num");
        // Use the same head number for the default multi-head attention
        const int kvHeadNum = reader.GetInteger(modelType, "kv_head_num", attHeadNum);
        const int headSize = reader.GetInteger(modelType, "size_per_head");
        const int imSize = reader.GetInteger(modelType, "inter_size");
        const int layers = reader.GetInteger(modelType, "num_layer");
        const int vocabSize = reader.GetInteger(modelType, "vocab_size");
        // Max Position Embedding for position embedding functions, with a default value set to 0
        const int maxPosEmbed = reader.GetInteger(modelType, "max_pos_seq_len", 0);
        // Max num of tokens that LLM can process. Also for allocating buffers. Default maxPosEmbed
        const int maxPositions = reader.GetInteger(modelType, "model_max_length", maxPosEmbed);
        // Seq length in Qwen model, if none, please ignore
        const int maxSeqLength = reader.GetInteger(modelType, "seq_length", -1);
        const bool useLogN = reader.GetInteger(modelType, "use_logn_attn", true);
        const bool useNTK = reader.GetInteger(modelType, "use_dynamic_ntk", true);
        const int hiddenSize = reader.GetInteger(modelType, "hidden_size", attHeadNum * headSize);
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
        const std::string quantQweightDataType = reader.Get(modelType, "quant_qweight_data_type", "");
        const std::string quantScalesDataType = reader.Get(modelType, "quant_scales_data_type", "");
        const std::string quantZerosDataType = reader.Get(modelType, "quant_zeros_data_type", "");
        const int quantGroupsize = reader.GetInteger(modelType, "quant_groupsize", -1);

        // DataType dt = getWeightType(configPath, modelType);
        DataType dt = DataType::fp32;
        if (quantQweightDataType == "int8" || quantQweightDataType == "uint4") {
            dt = quantQweightDataType == "int8" ? DataType::int8 : DataType::int4;
            REQUIRES(quantScalesDataType == "fp32", "scales should be fp32 data type.");
            REQUIRES(quantZerosDataType == "fp32", "zeros should be fp32 data type.");
            REQUIRES(quantGroupsize == -1, "Quantization with groupsize is not supported.");
        }

        // Buffer related (not initialized)
        this->inputTokens = nullptr;
        this->maskSize = 0;
        this->attnMask = nullptr;
        actBuffers.reset(new xft::Matrix<float>());

        // Context
        DecoderContext *ctx = getDecoderContext(layers, hiddenSize, headSize, attHeadNum, kvHeadNum, imSize, act,
                epsilon, vocabSize, embeddingSize, maxPositions, maxPosEmbed, maxSeqLength, useLogN, useNTK,
                ropeParamsPtr);

        ctx->ResetConfigReader(configPath);

        // Decoder
        if (layers % ctx->ppSize != 0) {
            std::cerr << "Warning: layers cannot be evenly divided by pipeline parallel stage size(ppSize)."
                      << std::endl;
            std::exit(-1);
        }

        decoderBlock = new DecoderBlock<ATTN_CLS, MLP_CLS, KVCacheT, ATTN_MLP_PARALLEL>(ctx, modelPath, layers, dt);
        auto maxSeqLen = maxSeqLength > 0 ? maxSeqLength : maxPositions;
        KVCacheMgr::instance().configure(maxSeqLen, kvHeadNum, headSize, layers, getDataType<KVCacheT>());

        // Predictor
        int workers = messenger.getSize();
        int rank = messenger.getRank();
        this->predictor = new DistLinear<LinearWeiT>(hiddenSize, vocabSize, rank, workers);
        this->setPredictorWeight(ctx, modelPath);

        // KVCache Manager
        this->kvCacheMgr.reset(new KVCacheManager<KVCacheT>(layers));
    }

    virtual ~CommonDecoder() {
        if (this->inputTokens) free(this->inputTokens);
        if (this->attnMask) free(this->attnMask);

        delete this->decoderBlock;
        delete this->predictor;
    }

    std::tuple<float *, int, int> forward(int *ids, int64_t *dims, int step, bool logitsAll = false) {
        // Assume input has been synced with master in higher level.
        // Assume the 1st step input's shape is [userSideBS][1][seqLen].
        TimeLine t("Decoder.forward");
        TimeLine t1("Decoder.embedding");

        int userSideBS = dims[0];
        int beamSize = dims[1];
        int batchSize = (step == 0 ? userSideBS : userSideBS * beamSize); // as sequence are duplicated at step 0
        int seqLen = dims[2];
        int pastSeqLen = step == 0 ? 0 : this->accSeqLen;
        int inputSeqLen = seqLen;

        // Prepare context
        DecoderContext *ctx = this->getContext();
        ctx->resize(batchSize, seqLen, pastSeqLen);
        int hiddenSize = ctx->hiddenSize;

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

        AttnInT *embBuf = (AttnInT *)actBuffers->Data();
        MlpOutT *outBuf = (MlpOutT *)(embBuf + batchSize * inputSeqLen * hiddenSize);

        // Embedding
        this->embeddingForward(ids, embBuf, batchSize * inputSeqLen);
        this->accSeqLen += seqLen;

#ifdef DEBUG
        dbg.debugPrint("---- embedding.forward ----\n");
        dbg.debugPrint("ids:\n");
        dbg.dumpMatrix(ids, batchSize, inputSeqLen, inputSeqLen);
        dbg.debugPrint("embBuf(rows: %d, cols: %d, stride: %d):\n", batchSize * inputSeqLen, hiddenSize, hiddenSize);
        dbg.dumpMatrix(embBuf, batchSize * inputSeqLen, hiddenSize, hiddenSize);
#endif

        // Prepare attention mask
        this->prepareAttnMask(ids, step + this->prefixSharing);
        // prepareAttnMeta

        // Token position ids, note: different models may have different impl.
        int *positionIds = this->getPositionIds(ids, batchSize, inputSeqLen, step + this->prefixSharing);
        t1.release();

#ifdef PIPELINE_PARALLEL
        // if current pipeline parallel stage rank isn't the first stage, should receive previous stage data
        if (ctx->ppSize > 1 && ctx->ppRank > 0) {
            int curr_world_rank = ctx->ppRank * ctx->tpSize + ctx->tpRank;
            int prev_world_rank = (ctx->ppRank - 1) * ctx->tpSize + ctx->tpRank;
            int count = batchSize * inputSeqLen * hiddenSize;
            int32_t sequenceID;
            MPI_Recv(&sequenceID, 1, MPI_INT32_T, prev_world_rank, curr_world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            TimeLine t("Decoder.Seq" + std::to_string(sequenceID) + ".MPI_Recv");
            MPI_Recv(embBuf, count, MPI_FLOAT, prev_world_rank, curr_world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // TODO: Error: different scope when dynamic loading so file
            // this->messenger.worldRecvFP32(embBuf, count, prev_world_rank, curr_world_rank);
            if (!SequencePool::getInstance().has(sequenceID)) {
                auto *seqs = SequencePool::getInstance().newMeta(sequenceID, seqLen);
                seqs->get(0)->setPastSeqLen(pastSeqLen);
                seqs->get(0)->allocBuffer<AttnInT>(hiddenSize, embBuf);
                SequencePool::getInstance().add(seqs->get(0)->getSequenceID(), seqs);
            }
            TaskWaitingQueue::getInstance().push(SequencePool::getInstance().get(sequenceID));
        }

        if (!InputQueue::getInstance().empty()) {
            if (!TaskWaitingQueue::getInstance().isFull()) {
                auto *seqs = InputQueue::getInstance().pop();
                seqs->get(0)->setPastSeqLen(pastSeqLen);
                seqs->get(0)->allocBuffer<AttnInT>(hiddenSize, embBuf);
                SequencePool::getInstance().add(seqs->get(0)->getSequenceID(), seqs);
                TaskWaitingQueue::getInstance().push(SequencePool::getInstance().get(seqs->get(0)->getSequenceID()));
            }
        }

        while (TaskWaitingQueue::getInstance().empty());

        SequenceGroupMeta *runningTask = nullptr;
        int32_t sequenceID = -1;
        if (!TaskWaitingQueue::getInstance().empty()) {
            runningTask = TaskWaitingQueue::getInstance().pop();
            sequenceID = runningTask->get(0)->getSequenceID();
            TimeLine t("Decoder.Seq" + std::to_string(sequenceID) + ".Step");
#endif

        // Decoder: forward
        int layers_per_pp_stage = decoderBlock->size();
        for (int i = 0; i < layers_per_pp_stage; ++i) {
            int workers = this->messenger.getSize();
            if (step == 0 && this->prefixSharing) {
                // Expand the prefix KV cache for each batch
                this->kvCacheMgr->expandPrefixCache(i, userSideBS, this->prefixSeqLen);
            }
            KVCacheTensor<KVCacheT> &presentKey = this->kvCacheMgr->getKey(i);
            KVCacheTensor<KVCacheT> &presentValue = this->kvCacheMgr->getValue(i);

            // Pls be noted: in attention, 'outBuf' is used as imtermediate buffer, 'tmpBuf' is used as output
            AttnOutT *attnOut = (AttnOutT *)(this->getContext()->tmpBuf.Data());
            // attnMeta (inputSeqLens, pastSeqLens, seqStartLoc, is_prompt(useSelfAttn), causal, attnMask)
            decoderBlock->get(i)->forwardAttention(getContext(), embBuf, outBuf, attnOut, attnMask,
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
                    this->messenger.reduceAdd(attnOut, attnOut, batchSize * inputSeqLen * hiddenSize);
                }
            }

            // When attention and FFN/MLP are in parallel, use the initial embedding as input
            if constexpr (ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) {
                    decoderBlock->get(i)->forwardFFN(getContext(), embBuf, outBuf, hiddenSize, hiddenSize, true);
                    this->messenger.reduceAdd(outBuf, embBuf, batchSize * inputSeqLen * hiddenSize);
                } else {
                    decoderBlock->get(i)->forwardFFN(getContext(), embBuf, embBuf, hiddenSize, hiddenSize, true);
                }
            } else {
                // FFN (for multiple workers, output into outBuf and then reduce add to embBuf)
                if (this->messenger.getSize() > 1) {
                    decoderBlock->get(i)->forwardFFN(getContext(), attnOut, outBuf, hiddenSize, hiddenSize, true);
                    this->messenger.reduceAdd(outBuf, embBuf, batchSize * inputSeqLen * hiddenSize);
                } else {
                    decoderBlock->get(i)->forwardFFN(getContext(), attnOut, embBuf, hiddenSize, hiddenSize, true);
                }
            }
        }

#ifdef PIPELINE_PARALLEL
        }

        // If current pipeline stage isn't the end of stage, should send data to next stage and return nullptr
        if (ctx->ppSize > 1 && ctx->ppRank < ctx->ppSize - 1) {
            TimeLine t("Decoder.Seq" + std::to_string(sequenceID) + ".MPI_Send");
            int next_world_rank = (ctx->ppRank + 1) * ctx->tpSize + ctx->tpRank;
            int count = batchSize * inputSeqLen * hiddenSize;
            MPI_Send(&sequenceID, 1, MPI_INT32_T, next_world_rank, next_world_rank, MPI_COMM_WORLD);
            MPI_Send(embBuf, count, MPI_FLOAT, next_world_rank, next_world_rank, MPI_COMM_WORLD);
            // TODO: Error: different scope when dynamic loading so file
            // this->messenger.worldSendFP32(embBuf, count, next_world_rank, next_world_rank);
            return std::tuple<float *, int, int>(nullptr, 0, 0);
        }
#endif

        // Prepare input for final Layer Norm (only care about the last row of the result)
        // Shape of embBuf: (bs, seqLen, hiddenSize)
        MlpOutT *lnIn = embBuf;
        if (inputSeqLen > 1 && !logitsAll) { // copy is not needed when seqLen = 1 or logitsAll is true
            lnIn = outBuf;
#pragma omp parallel for
            for (int b = 0; b < batchSize; ++b) {
                memcpy(lnIn + b * hiddenSize, embBuf + ((b + 1) * inputSeqLen - 1) * hiddenSize,
                        hiddenSize * sizeof(MlpOutT));
            }
        }

#ifdef DEBUG
        dbg.debugPrint(">>> DecoderLayer Output[%d, %d] (%d):\n", batchSize * inputSeqLen, hiddenSize, hiddenSize);
        dbg.dumpMatrix(embBuf, batchSize * inputSeqLen, hiddenSize, hiddenSize);
        dbg.debugPrint("LayerNorm In:\n");

        if (!logitsAll)
            dbg.dumpMatrix(lnIn, batchSize, hiddenSize, hiddenSize);
        else
            dbg.dumpMatrix(lnIn, batchSize * inputSeqLen, hiddenSize, hiddenSize);
#endif

        // LN, as it supports inplace computing, input and output can be the same
        MlpOutT *lnOut = embBuf;
        if (!logitsAll)
            lastLayerNormForward(lnIn, lnOut, batchSize);
        else
            lastLayerNormForward(lnIn, lnOut, batchSize * seqLen);

#ifdef DEBUG
        dbg.debugPrint("LayerNorm Out:\n");
        if (!logitsAll)
            dbg.dumpMatrix(lnOut, batchSize, hiddenSize, hiddenSize);
        else
            dbg.dumpMatrix(lnOut, batchSize * inputSeqLen, hiddenSize, hiddenSize);
#endif

        // Predictor
        float *finalOut = (float *)outBuf;
        if (!logitsAll)
            this->predictor->forward(ctx, lnOut, finalOut, batchSize);
        else
            this->predictor->forward(ctx, lnOut, finalOut, batchSize * seqLen);

#ifdef DEBUG
        auto splitSize = this->predictor->getSplitSize();
        dbg.debugPrint("finalOut:\n");
        if (!logitsAll)
            dbg.dumpMatrix(finalOut, batchSize, splitSize, splitSize);
        else
            dbg.dumpMatrix(finalOut, batchSize * inputSeqLen, splitSize, splitSize);
#endif

        // Expand the result to make it cover multiple beams
        if (step == 0 && beamSize > 1) {
            const int splitSize = this->predictor->getSplitSize();
            for (int b = userSideBS - 1; b >= 0; --b) {
                float *src = finalOut + b * splitSize;
#pragma omp parallel for
                for (int idx = b * beamSize; idx < (b + 1) * beamSize; ++idx) {
                    if (idx == b) { continue; }
                    float *dst = finalOut + idx * splitSize;
                    memcpy(dst, src, splitSize * sizeof(float));
                }
            }
        }

        // free temporary new ids for prefix sharing
        if (step == 0 && this->prefixSharing) { free(ids); }

        return std::tuple<float *, int, int>(
                finalOut, this->predictor->getSplitOffset(), this->predictor->getSplitSize());
    }

    std::tuple<float *, int, int> forward(std::vector<xft::SequenceMeta *> &seqs, bool logitsAll = false) {
        // Assume all sequences are all prompts(step==0) or all decodes(step>0) 
        // Assume input has been synced with master in higher level.
        TimeLine t("Decoder.forward");
        TimeLine t1("Decoder.embedding");

        if (unlikely(seqs.empty())) { return std::tuple<float *, int, int>(nullptr, 0, 0); }

        DecoderContext *ctx = this->getContext();
        int batchSize = seqs.size();
        int hiddenSize = ctx->hiddenSize;

        // Prepare input
        int totInputSeqLen = 0;
        std::vector<int> allInputIds;
        for (auto seq : seqs) {
            totInputSeqLen += seq->getInputSeqLen();
            auto ids = seq->getInputTokens();
            allInputIds.insert(allInputIds.end(), ids.begin(), ids.end());
        }

        // Prepare context
        ctx->resize(totInputSeqLen);

        // Prepare buffers
        int logitRows = (!logitsAll && seqs[0]->getStep() == 0) ? seqs.size() : totInputSeqLen;
        prepareBuffer(ctx, totInputSeqLen, logitRows);

        AttnInT *embBuf = (AttnInT *)actBuffers->Data();
        MlpOutT *outBuf = (MlpOutT *)(embBuf + totInputSeqLen * hiddenSize);

        // Embedding
        this->embeddingForward(allInputIds.data(), embBuf, totInputSeqLen);

        // Decoder block (all layers)
        decoderBlock->forward(ctx, seqs, embBuf, embBuf);

        // Prepare input for final Layer Norm (only care about the last row of the result)
        // Shape of embBuf: (total_input_seqlen, hiddenSize)
        MlpOutT *lnIn = embBuf;
        if (logitRows != totInputSeqLen) {
            int offset = -1;
            for (int b = 0; b < batchSize; ++b) {
                offset += seqs[b]->getInputSeqLen();
                memcpy(lnIn + b * hiddenSize, embBuf + offset * hiddenSize, hiddenSize * sizeof(MlpOutT));
            }
        }

#ifdef DEBUG
        dbg.debugPrint(">>> DecoderLayer Output[%d, %d] (%d):\n", logitRows, hiddenSize, hiddenSize);
        dbg.dumpMatrix(embBuf, logitRows, hiddenSize, hiddenSize);
        dbg.debugPrint("LayerNorm In:\n");

        dbg.dumpMatrix(lnIn, logitRows, hiddenSize, hiddenSize);
#endif

        // Last normalization layer
        MlpOutT *lnOut = embBuf;
        lastLayerNormForward(lnIn, lnOut, logitRows);

#ifdef DEBUG
        dbg.debugPrint("LayerNorm Out:\n");
        dbg.dumpMatrix(lnOut, logitRows, hiddenSize, hiddenSize);
#endif

        // Predictor
        float *finalOut = (float *)outBuf;
        this->predictor->forward(ctx, lnOut, finalOut, logitRows);

#ifdef DEBUG
        auto splitSize = this->predictor->getSplitSize();
        dbg.debugPrint("finalOut:\n");
        dbg.dumpMatrix(finalOut, logitRows, splitSize, splitSize);
#endif

        return std::tuple<float *, int, int>(
                finalOut, this->predictor->getSplitOffset(), this->predictor->getSplitSize());
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

        AttnInT *embBuf = (AttnInT *)actBuffers->Data();
        MlpOutT *outBuf = (MlpOutT *)(embBuf + 1 * seqLen * ctx->hiddenSize);

        // Embedding
        this->embeddingForward(ids, embBuf, 1 * seqLen);

        // Prepare attention mask
        this->prepareAttnMask(ids, 0);

        // Token position ids, note: different models may have different impl.
        int *positionIds = this->getPositionIds(ids, 1, seqLen, 0);
        t1.release();

        // Decoder: forward
        // TODO: Add PIPELINE_PARALLEL feature
        int hiddenSize = ctx->hiddenSize;
        for (int i = 0; i < this->decoderBlock->size(); ++i) {
            int workers = this->messenger.getSize();
            KVCacheTensor<KVCacheT> &presentKey = this->kvCacheMgr->getPrefixKey(i);
            KVCacheTensor<KVCacheT> &presentValue = this->kvCacheMgr->getPrefixValue(i);

            // Pls be noted: in attention, 'outBuf' is used as imtermediate buffer, 'tmpBuf' is used as output
            AttnOutT *attnOut = (AttnOutT *)(this->getContext()->tmpBuf.Data());
            decoderBlock->get(i)->forwardAttention(getContext(), embBuf, outBuf, attnOut, attnMask,
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
                if (this->messenger.getSize() > 1) { this->messenger.reduceAdd(attnOut, attnOut, seqLen * hiddenSize); }
            }

            // When attention and FFN/MLP are in parallel, use the initial embedding as input
            if constexpr (ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) {
                    decoderBlock->get(i)->forwardFFN(getContext(), embBuf, outBuf, hiddenSize, hiddenSize, true);
                    this->messenger.reduceAdd(outBuf, embBuf, seqLen * hiddenSize);
                } else {
                    decoderBlock->get(i)->forwardFFN(getContext(), embBuf, embBuf, hiddenSize, hiddenSize, true);
                }
            } else {
                // FFN (for multiple workers, output into outBuf and then reduce add to embBuf)
                if (this->messenger.getSize() > 1) {
                    decoderBlock->get(i)->forwardFFN(getContext(), attnOut, outBuf, hiddenSize, hiddenSize, true);
                    this->messenger.reduceAdd(outBuf, embBuf, seqLen * hiddenSize);
                } else {
                    decoderBlock->get(i)->forwardFFN(getContext(), attnOut, embBuf, hiddenSize, hiddenSize, true);
                }
            }
        }
    }

    // Reorder cached keys and values, size=batchSize*beamSize
    void reorderCache(int *idx, int size) { kvCacheMgr->reorderCache(idx, size, initSeqLen, accSeqLen); }

    // Get decoder context
    DecoderContext *getContext() { return context.get(); }

    // How many layers on Duty
    int getLayers() { return decoderBlock->size(); }

    Messenger &getMessenger() { return messenger; }

    bool isMaster() { return messenger.isMaster(); }

    int getRank() { return messenger.getRank(); }

    int getEndId() { return endId; }

    int getInitSeqLen() { return initSeqLen; }

    std::tuple<std::shared_ptr<DecoderContext>, std::shared_ptr<KVCacheManager<KVCacheT>>,
            std::shared_ptr<xft::Matrix<float>>>
    getSharedResources() {
        return std::make_tuple(context, kvCacheMgr, actBuffers);
    }

    void setSharedResources(const std::tuple<std::shared_ptr<DecoderContext>, std::shared_ptr<KVCacheManager<KVCacheT>>,
            std::shared_ptr<xft::Matrix<float>>> &r) {
        this->context = std::get<0>(r);
        this->kvCacheMgr = std::get<1>(r);
        this->actBuffers = std::get<2>(r);
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

    DecoderContext *getDecoderContext(int layers, const int hiddenSize, const int headSize, const int attHeadNum,
            const int kvHeadNum, const int imSize, const std::string &act, const float epsilon, int vocabSize,
            int embeddingSize, int maxPositions, int maxPosEmbed, int maxSeqLength, bool useLogN, bool useNTK,
            RopeParams *ropeParamsPtr) {
        Env &env = Env::getInstance();
        int tpSize = messenger.getSize();
        int tpRank = messenger.getRank();
        int ppSize = env.getPipelineStage();
        int ppRank = messenger.getColor();
        // printf("ppSize: %d, ppRank: %d, tpSize: %d, tpRank: %d\n", ppSize, ppRank, tpSize, tpRank);

        if (context != nullptr) {
            if (context->hiddenSize == hiddenSize && context->attHeadNum == attHeadNum
                    && context->kvHeadNum == kvHeadNum && context->intermediateSize == imSize
                    && context->tpRank == tpRank) {
                return context.get();
            } else {
                printf("Different context size not unsupported!\n");
                exit(-1);
            }
        } else {
            this->context.reset(new DecoderContext(layers, hiddenSize, headSize, attHeadNum, kvHeadNum, imSize, act,
                    epsilon, vocabSize, embeddingSize, maxPositions, maxPosEmbed, maxSeqLength, tpRank, tpSize, ppSize,
                    ppRank, ropeParamsPtr, useLogN, useNTK));

            if (env.getEngineKind() == xft::DeviceKind::iGPU && env.getEngineIndex() < 0) // Sequential assignment
                this->context->mmHelper = new MMHelper(env.getEngineKind(), ppRank * tpSize + tpRank);
            else // assignment through the user
                this->context->mmHelper = new MMHelper(env.getEngineKind(), env.getEngineIndex());
        }

        return this->context.get();
    }

    // OriWeiT: float, int8_t or uint4x2_t
    template <typename OriWeiT>
    void setDecoderWeights(DECODER *pdecoder, const std::string &modelPath, int layerIdx) {
        const int hiddenSize = getContext()->hiddenSize;
        const int imSize = getContext()->intermediateSize;
        const int kvHeadNum = getContext()->kvHeadNum;
        const int attHeadNum = getContext()->attHeadNum;
        const int attHeadSize = getContext()->attHeadSize;
        const int mlpFactor = (getContext()->actType == DecoderContext::SWIGLU) ? 2 : 1;
        int qSize = attHeadSize * attHeadNum;
        int kvSize = attHeadSize * kvHeadNum;
        int qkvSize = qSize + 2 * kvSize;

#define ALLOC(size, alignment) xft::alloc((size), (alignment))
        OriWeiT *qkvWeight = (OriWeiT *)ALLOC(hiddenSize * qkvSize * sizeof(OriWeiT), 64);
        float *qkvScales = nullptr;
        float *qkvZeros = nullptr;
        float *qkvBias = (float *)ALLOC(qkvSize * sizeof(float), 64);

        OriWeiT *attnOutWeight = (OriWeiT *)ALLOC(qSize * hiddenSize * sizeof(OriWeiT), 64);
        float *attnOutScales = nullptr;
        float *attnOutZeros = nullptr;
        float *attnOutBias = (float *)ALLOC(hiddenSize * sizeof(float), 64);

        OriWeiT *fc1Weight = (OriWeiT *)ALLOC(hiddenSize * imSize * mlpFactor * sizeof(OriWeiT), 64);
        float *fc1Scales = nullptr;
        float *fc1Zeros = nullptr;
        float *fc1Bias = (float *)ALLOC(imSize * sizeof(float), 64);

        OriWeiT *fc2Weight = (OriWeiT *)ALLOC(hiddenSize * imSize * sizeof(OriWeiT), 64);
        float *fc2Scales = nullptr;
        float *fc2Zeros = nullptr;
        float *fc2Bias = (float *)ALLOC(hiddenSize * sizeof(float), 64);

        float *ln1Gamma = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *ln1Beta = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *ln2Gamma = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *ln2Beta = (float *)ALLOC(hiddenSize * sizeof(float), 64);

        OriWeiT *fc3Weight = nullptr;
        float *fc3Scales = nullptr;
        float *fc3Zeros = nullptr;

        // INT8/INT4 quant, wbits = 8/4, qweight dtype: int8_t/uint4x2_t
        if constexpr (std::is_same_v<OriWeiT, int8_t> || std::is_same_v<OriWeiT, uint4x2_t>) {
            DataType dt = std::is_same_v<OriWeiT, int8_t> ? DataType::int8 : DataType::int4;

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
                    qkvWeight, hiddenSize * qkvSize, dt);
            loadWeight(
                    modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.zeros.0.bin",
                    qkvZeros, qkvSize, DataType::fp32);
            loadWeight(
                    modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.scales.0.bin",
                    qkvScales, qkvSize, DataType::fp32);

            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.qweight.0.bin",
                    attnOutWeight, qSize * hiddenSize, dt);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.zeros.0.bin",
                    attnOutZeros, hiddenSize, DataType::fp32);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.scales.0.bin",
                    attnOutScales, hiddenSize, DataType::fp32);

            // Stardard 2 layer MLP
            if (fileExists(
                        modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.qweight.0.bin")) {
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.qweight.0.bin",
                        fc1Weight, hiddenSize * imSize * mlpFactor, dt);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.zeros.0.bin",
                        fc1Zeros, imSize * mlpFactor, DataType::fp32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.scales.0.bin",
                        fc1Scales, imSize * mlpFactor, DataType::fp32);

                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.qweight.0.bin",
                        fc2Weight, hiddenSize * imSize, dt);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.zeros.0.bin",
                        fc2Zeros, hiddenSize, DataType::fp32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.scales.0.bin",
                        fc2Scales, hiddenSize, DataType::fp32);
            }
            // gate, up, down weights for Llama like model
            else {
                fc3Weight = (OriWeiT *)ALLOC(hiddenSize * imSize * sizeof(OriWeiT), 64);
                fc3Zeros = (float *)ALLOC(hiddenSize * sizeof(float), 64);
                fc3Scales = (float *)ALLOC(hiddenSize * sizeof(float), 64);

                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.qweight.0.bin",
                        fc1Weight, hiddenSize * imSize * mlpFactor, dt);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.zeros.0.bin",
                        fc1Zeros, imSize * mlpFactor, DataType::fp32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.scales.0.bin",
                        fc1Scales, imSize * mlpFactor, DataType::fp32);

                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.qweight.0.bin",
                        fc2Weight, hiddenSize * imSize, dt);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.zeros.0.bin",
                        fc2Zeros, imSize, DataType::fp32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.scales.0.bin",
                        fc2Scales, imSize, DataType::fp32);

                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.qweight.0.bin",
                        fc3Weight, hiddenSize * imSize, dt);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.zeros.0.bin",
                        fc3Zeros, hiddenSize, DataType::fp32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.scales.0.bin",
                        fc3Scales, hiddenSize, DataType::fp32);
            }

        } else if constexpr (std::is_same_v<OriWeiT, float>) {
            loadWeight(
                    modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.weight.0.bin",
                    qkvWeight, hiddenSize * qkvSize);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.weight.0.bin",
                    attnOutWeight, qSize * hiddenSize);

            // Stardard 2 layer MLP
            if (fileExists(
                        modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.weight.0.bin")) {
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.weight.0.bin",
                        fc1Weight, hiddenSize * imSize * mlpFactor);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.weight.0.bin",
                        fc2Weight, hiddenSize * imSize);
            }
            // gate, up, down weights for Llama like model
            else {
                fc3Weight = (OriWeiT *)ALLOC(hiddenSize * imSize * sizeof(OriWeiT), 64);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.weight.0.bin",
                        fc1Weight, hiddenSize * imSize * mlpFactor);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.weight.0.bin",
                        fc2Weight, hiddenSize * imSize);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.weight.0.bin",
                        fc3Weight, hiddenSize * imSize);
            }
        }

        loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".input_layernorm.weight.bin", ln1Gamma,
                hiddenSize);
        loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".post_attention_layernorm.weight.bin",
                ln2Gamma, hiddenSize);

#define READ_OPTIONAL(filename, addr, size, errmsg)                                 \
    {                                                                               \
        int ret = loadWeight((filename), (addr), (size), DataType::unknown, false); \
        if (ret == 0) {                                                             \
            free(addr);                                                             \
            addr = nullptr;                                                         \
        } else {                                                                    \
            if (ret != (size)) {                                                    \
                printf("%s\n", (errmsg));                                           \
                exit(-1);                                                           \
            }                                                                       \
        }                                                                           \
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

        constexpr int sizeFactor = std::is_same_v<OriWeiT, uint4x2_t> ? 2 : 1;
        pdecoder->setWeights(getContext(), qkvWeight, qkvScales, qkvZeros, qkvBias, qkvWeight + qSize / sizeFactor,
                qkvScales + qSize, qkvZeros + qSize, qkvBias + qSize,
                qkvWeight + qSize / sizeFactor + kvSize / sizeFactor, qkvScales + qSize + kvSize,
                qkvZeros + qSize + kvSize, qkvBias + qSize + kvSize, attnOutWeight, attnOutScales, attnOutZeros,
                attnOutBias, ln1Gamma, ln1Beta, fc1Weight, fc1Scales, fc1Zeros, fc1Bias, fc2Weight, fc2Scales, fc2Zeros,
                fc2Bias, ln2Gamma, ln2Beta, fc3Weight, fc3Scales, fc3Zeros, false);

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

    void setPredictorWeight(DecoderContext *ctx, const std::string &modelPath) {
        int inputSize = predictor->getInputSize();
        int outputSize = predictor->getOutputSize();

        float *weight = (float *)malloc(inputSize * outputSize * sizeof(float));
        float *bias = nullptr;

        loadWeight(modelPath + "/model.lm_head.weight.bin", weight, inputSize * outputSize);

        predictor->setWeight(ctx, weight, bias);

        free(weight);
    }

    virtual void prepareBuffers(
            DecoderContext *ctx, int userSideBS, int beamSize, bool logitsAll = false, bool prefix = false) {
        int batchSize = ctx->batchSize;
        int hiddenSize = ctx->hiddenSize;
        int seqLen = ctx->inputSeqLen;
        int vocabSize = ctx->vocabSize;
        int maxPositions = ctx->maxPositions;
        int layers = this->decoderBlock->size();
        int workers = this->messenger.getSize();

        // Prepare buffers
        int logitsLen = logitsAll ? batchSize * seqLen : userSideBS * beamSize;
        int actRows = batchSize * seqLen; // rows for activation

        // Convert final output buffer size into rows in the units of hiddenSize
        int outRows = actRows;
        if (logitsLen * vocabSize > outRows * hiddenSize) { outRows = logitsLen * vocabSize / hiddenSize + 1; }

        this->actBuffers->Resize(actRows + outRows, hiddenSize);

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

    void prepareBuffer(DecoderContext *ctx, int totInputSeqLen, int logitRows) {
        int hiddenSize = ctx->hiddenSize;
        int vocabSize = ctx->vocabSize;

        // Convert final output buffer size into units of hiddenSize
        int outRows = std::ceil(1.0f * logitRows * vocabSize / hiddenSize);

        this->actBuffers->Resize(totInputSeqLen + outRows, hiddenSize);
    }

    float *getAttnMask(int sizeRequired) {
        if (this->maskSize < sizeRequired) {
            if (this->attnMask) free(this->attnMask);
            this->attnMask = (float *)xft::alloc(sizeRequired * sizeof(float));
            this->maskSize = sizeRequired;
        }
        return this->attnMask;
    }

    int getStartId() { return startId; }

    virtual void embeddingForward(int *ids, float *output, int tokenSize) {
        printf("embeddingForward(float) must be implemented.\n");
        exit(-1);
    }
    virtual void embeddingForward(int *ids, bfloat16_t *output, int tokenSize) {
        printf("embeddingForward(bfloat16_t) must be implemented.\n");
        exit(-1);
    }
    virtual void embeddingForward(int *ids, float16_t *output, int tokenSize) {
        printf("embeddingForward(float16_t) must be implemented.\n");
        exit(-1);
    }

    virtual void lastLayerNormForward(float *input, float *output, int rows) {
        printf("lastLayerNormForward(float) must be implemented.\n");
        exit(-1);
    }
    virtual void lastLayerNormForward(bfloat16_t *input, bfloat16_t *output, int rows) {
        printf("lastLayerNormForward(bfloat16_t) must be implemented.\n");
        exit(-1);
    }
    virtual void lastLayerNormForward(float16_t *input, float16_t *output, int rows) {
        printf("lastLayerNormForward(float16_t) must be implemented.\n");
        exit(-1);
    }

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

    // Embedding output data type = input data type of Attention
    using AttnInT = typename AttnTypeExtractor<ATTN_CLS>::Tin;
    using AttnOutT = typename AttnTypeExtractor<ATTN_CLS>::Tout;
    using MlpInT = typename MlpTypeExtractor<MLP_CLS>::Tin;
    using MlpOutT = typename MlpTypeExtractor<MLP_CLS>::Tout;

    // Activation buffers (declared as float, but the actual data type may be different)
    std::shared_ptr<xft::Matrix<float>> actBuffers;

protected:
    // Decoder block (all decoder layers)
    DecoderBlock<ATTN_CLS, MLP_CLS, KVCacheT, ATTN_MLP_PARALLEL> *decoderBlock;

    using LinearWeiT = typename std::conditional<std::is_same_v<MlpOutT, bfloat16_t>, bfloat16_t, float16_t>::type;
    DistLinear<LinearWeiT> *predictor;

private:
    int maskSize; // size of allocated attnMask
    float *attnMask; // attention mask, set as private as may need to enlarge

    int startId;
    int endId;

#ifdef DEBUG
    Debugger dbg;
#endif
};
