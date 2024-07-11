// Copyright (c) 2024 Intel Corporation
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
#include "decoder_layer.h"
#include "dtype.h"
#include "kvcache_mgr.h"
#include "messenger.h"
#include "weight_util.h"

template <typename ATTN_CLS, typename MLP_CLS, typename KVCacheT, bool ATTN_MLP_PARALLEL>
class DecoderBlock {
public:
    using DECODER = Decoder<ATTN_CLS, MLP_CLS>;

    DecoderBlock(DecoderContext *ctx, const std::string &modelPath, int layers, xft::DataType dt) {
        if (layers % ctx->ppSize != 0) {
            std::cerr << "Warning: layers cannot be evenly divided by pipeline parallel stage size(ppSize)."
                      << std::endl;
            std::exit(-1);
        }

        int layersOnDuty = layers / ctx->ppSize;
        int startLayer = ctx->ppRank * layersOnDuty;
        for (int i = startLayer; i < startLayer + layersOnDuty; ++i) {
            auto pdec = new DECODER(ctx, i);
            if (dt == xft::DataType::int8) {
                this->setDecoderWeights<int8_t>(ctx, pdec, modelPath, i);
            } else if (dt == xft::DataType::int4) {
                this->setDecoderWeights<uint4x2_t>(ctx, pdec, modelPath, i);
            } else if (dt == xft::DataType::fp32) {
                this->setDecoderWeights<float>(ctx, pdec, modelPath, i);
            } else {
                std::cerr << "Error: The data type is NOT supported." << std::endl;
                std::exit(-1);
            }
            this->decoders.push_back(pdec);
        }
    }

    virtual ~DecoderBlock() {
        for (auto dec : this->decoders) {
            delete dec;
        }
    }

    // To make it compatible with the old impl.
    DECODER *get(int layerId) { return this->decoders[layerId]; }

    int size() const { return this->decoders.size(); }

    template <typename T>
    void forward(DecoderContext *ctx, std::vector<xft::SequenceMeta *> &seqs, T *inputBuf, T *outputBuf) {
        using AttnOutT = typename AttnTypeExtractor<ATTN_CLS>::Tout;

        Messenger &messenger = Messenger::getInstance();
        xft::KVCacheMgr &kvCacheMgr = xft::KVCacheMgr::instance();

        // Data preparation
        std::vector<int> seqIDs(seqs.size());
        size_t totInSeqLen = 0;
        for (int i = 0; i < seqs.size(); ++i) {
            seqIDs[i] = seqs[i]->getSequenceID();
            totInSeqLen += seqs[i]->getInputSeqLen();
        }

        // TODO: check and prepare KV cache only needed
        kvCacheMgr.prepareCache(seqIDs);

        // All layers forward
        int layersOnDuty = this->decoders.size();
        auto input = inputBuf;
        auto output = outputBuf;
        AttnOutT *attnOut = (AttnOutT *)(ctx->tmpBuf.Data());

        for (int i = 0; i < layersOnDuty; ++i) {
            int workers = messenger.getSize();

            std::vector<void *> keyCaches = kvCacheMgr.getKey(i);
            std::vector<void *> valueCaches = kvCacheMgr.getValue(i);

            // Reinterpret the keyCaches and valueCaches to the correct type
            this->decoders[i]->forwardAttention(ctx, seqs, input, attnOut, totInSeqLen,
                    *reinterpret_cast<std::vector<KVCacheTensor<KVCacheT> *> *>(&keyCaches),
                    *reinterpret_cast<std::vector<KVCacheTensor<KVCacheT> *> *>(&valueCaches));

            // Merge the result of attention
            // When attention and FFN/MLP are in parallel, do not need to reduce after attention
            if constexpr (!ATTN_MLP_PARALLEL) {
                if (messenger.getSize() > 1) { messenger.reduceAdd(attnOut, attnOut, totInSeqLen * ctx->hiddenSize); }
            }

            // When attention and FFN/MLP are in parallel, use the initial embedding as input
            if constexpr (ATTN_MLP_PARALLEL) {
                std::cerr << "Error: ATTN_MLP_PARALLEL=true is not supported." << std::endl;
                std::exit(-1);
            } else {
                if (messenger.getSize() > 1) {
                    this->decoders[i]->forwardFFN(ctx, attnOut, output, ctx->hiddenSize, ctx->hiddenSize, true, totInSeqLen);
                    messenger.reduceAdd(output, output, totInSeqLen * ctx->hiddenSize);
                } else {
                    this->decoders[i]->forwardFFN(ctx, attnOut, output, ctx->hiddenSize, ctx->hiddenSize, true, totInSeqLen);
                }
            }

            // Update the input/output for the next layer
            std::swap(input, output);
        }

        // Copy final result to the output buffer
        if (inputBuf != outputBuf && layersOnDuty % 2 == 0) {
            xft::memcopy(outputBuf, inputBuf, totInSeqLen * ctx->hiddenSize * sizeof(T), ctx->device);
        }
    }

private:
    // OriWeiT: float, int8_t or uint4x2_t
    template <typename OriWeiT>
    void setDecoderWeights(DecoderContext *ctx, DECODER *pdecoder, const std::string &modelPath, int layerIdx) {
        using xft::DataType;
        using xft::loadWeight;
        using xft::Weight;
        using xft::fileExists;
        using xft::getGroupSize;

        const int hiddenSize = ctx->hiddenSize;
        const int imSize = ctx->intermediateSize;
        const int kvHeadNum = ctx->kvHeadNum;
        const int attHeadNum = ctx->attHeadNum;
        const int attHeadSize = ctx->attHeadSize;
        const int mlpFactor = (ctx->actType == DecoderContext::SWIGLU) ? 2 : 1;
        int qSize = attHeadSize * attHeadNum;
        int kvSize = attHeadSize * kvHeadNum;
        int qkvSize = qSize + 2 * kvSize;
        int groupsize = getGroupSize(modelPath + "config.ini");

        Weight<OriWeiT> qkvWeight, attnOutWeight, fc1Weight, fc2Weight, fc3Weight;
        loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.weight.0.bin",
                        qkvWeight, hiddenSize, qkvSize);
        loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.weight.0.bin",
                        attnOutWeight, qSize, hiddenSize);

        bool standard_mlp = (fileExists( modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.weight.0.bin")
                          || fileExists( modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.qweight.0.bin"));
        // Stardard 2 layer MLP
        if (standard_mlp) {
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.weight.0.bin",
                    fc1Weight, hiddenSize, imSize * mlpFactor);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.weight.0.bin",
                        fc2Weight, imSize, hiddenSize);
        }
        // gate, up, down weights for Llama like model
        else {
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.weight.0.bin",
                   fc1Weight, hiddenSize, imSize * mlpFactor);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.weight.0.bin",
                   fc2Weight, hiddenSize, imSize);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.weight.0.bin",
                   fc3Weight, imSize, hiddenSize);
        }

        float *qkvBias = nullptr, *attnOutBias = nullptr, *fc1Bias = nullptr, *fc2Bias = nullptr;
        float *ln1Gamma = nullptr, *ln1Beta = nullptr, *ln2Gamma = nullptr, *ln2Beta = nullptr;

        loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".input_layernorm.weight.bin",
                ln1Gamma, hiddenSize);
        loadWeight(
                modelPath + "/model.layers." + std::to_string(layerIdx) + ".post_attention_layernorm.weight.bin",
                ln2Gamma, hiddenSize);

#define READ_OPTIONAL(filename, addr, size)                                         \
    {                                                                               \
        if (fileExists(filename)) {                                                 \
            loadWeight((filename), (addr), (size));                                 \
        }                                                                           \
    }

        // The bias is optional
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.bias.0.bin",
                qkvBias, qkvSize);
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.bias.bin",
                attnOutBias, hiddenSize);
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".input_layernorm.bias.bin", ln1Beta,
                hiddenSize);
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".post_attention_layernorm.bias.bin",
                ln2Beta, hiddenSize);
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.bias.0.bin",
                fc1Bias, imSize);
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.bias.bin", fc2Bias,
                hiddenSize);

        constexpr int sizeFactor = std::is_same_v<OriWeiT, uint4x2_t> ? 2 : 1;
        pdecoder->setWeights(ctx, qkvWeight.w, qkvWeight.s, qkvWeight.z, qkvBias, qkvWeight.w + qSize / sizeFactor,
                qkvWeight.s + qSize, qkvWeight.z + qSize, qkvBias + qSize,
                qkvWeight.w + qSize / sizeFactor + kvSize / sizeFactor, qkvWeight.s + qSize + kvSize,
                qkvWeight.z + qSize + kvSize, qkvBias + qSize + kvSize, attnOutWeight.w, attnOutWeight.s, attnOutWeight.z,
                attnOutBias, ln1Gamma, ln1Beta, fc1Weight.w, fc1Weight.s, fc1Weight.z, fc1Bias, fc2Weight.w, fc2Weight.s, fc2Weight.z,
                fc2Bias, ln2Gamma, ln2Beta, fc3Weight.w, fc3Weight.s, fc3Weight.z, false);

        free(qkvBias);
        free(attnOutBias);
        free(fc1Bias);
        free(fc2Bias);
        free(ln1Gamma);
        free(ln1Beta);
        free(ln2Gamma);
        free(ln2Beta);
    }

private:
    std::vector<DECODER *> decoders;
};
