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
#include "llm_params.h"
#include "logger.h"
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

        xft::AttnParams *attnParams = createAttnParams(ctx, modelPath, dt);
        xft::FFNParams *ffnParams = createFFNParams(ctx, modelPath, dt);

        for (int i = startLayer; i < startLayer + layersOnDuty; ++i) {
            auto pdec = new DECODER(ctx, i);
            if (dt == xft::DataType::int8) {
                this->setDecoderWeights<int8_t>(ctx, pdec, modelPath, i);
            } else if (dt == xft::DataType::int4) {
                this->setDecoderWeights<uint4x2_t>(ctx, pdec, modelPath, i);
            } else if (dt == xft::DataType::fp32) {
                this->setDecoderWeights<float>(ctx, pdec, modelPath, i, attnParams, ffnParams);
            } else if (dt == xft::DataType::bf16) {
                this->setDecoderWeights<bfloat16_t>(ctx, pdec, modelPath, i, attnParams, ffnParams);
            } else if (dt == xft::DataType::fp16) {
                this->setDecoderWeights<float16_t>(ctx, pdec, modelPath, i, attnParams, ffnParams);
            } else {
                std::cerr << "Error: The data type is NOT supported." << std::endl;
                std::exit(-1);
            }
            this->decoders.push_back(pdec);
        }

        delete ffnParams;
        delete attnParams;
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
                    this->decoders[i]->forwardFFN(
                            ctx, attnOut, output, ctx->hiddenSize, ctx->hiddenSize, true, totInSeqLen);
                    messenger.reduceAdd(output, output, totInSeqLen * ctx->hiddenSize);
                } else {
                    this->decoders[i]->forwardFFN(
                            ctx, attnOut, output, ctx->hiddenSize, ctx->hiddenSize, true, totInSeqLen);
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
    static bool fileExists(const std::string &filename) {
        std::ifstream file(filename);
        return file.good();
    }

    // OriWeiT: float, int8_t or uint4x2_t
    template <typename OriWeiT>
    void setDecoderWeights(DecoderContext *ctx, DECODER *pdecoder, const std::string &modelPath, int layerIdx) {
        using xft::DataType;
        using xft::loadWeight;

        const int hiddenSize = ctx->hiddenSize;
        const int imSize = ctx->intermediateSize;
        const int kvHeadNum = ctx->kvHeadNum;
        const int attHeadNum = ctx->attHeadNum;
        const int attHeadSize = ctx->attHeadSize;
        const int mlpFactor = (ctx->actType == DecoderContext::SWIGLU) ? 2 : 1;
        int qSize = attHeadSize * attHeadNum;
        int kvSize = attHeadSize * kvHeadNum;
        int qkvSize = qSize + 2 * kvSize;

#define ALLOC(size, alignment) xft::alloc((size), nullptr, (alignment))
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
        float *fc3Bias = nullptr;

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
                if (fileExists(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.bias.0.bin")) {
                    fc3Bias = (float *)ALLOC(hiddenSize * sizeof(float), 64);
                    loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.bias.0.bin",
                            fc3Bias, hiddenSize);
                }
            }
        }

        loadWeight<float>(modelPath + "/model.layers." + std::to_string(layerIdx) + ".input_layernorm.weight.bin",
                ln1Gamma, hiddenSize);
        loadWeight<float>(
                modelPath + "/model.layers." + std::to_string(layerIdx) + ".post_attention_layernorm.weight.bin",
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
        pdecoder->setWeights(ctx, qkvWeight, qkvScales, qkvZeros, qkvBias, qkvWeight + qSize / sizeFactor,
                qkvScales + qSize, qkvZeros + qSize, qkvBias + qSize,
                qkvWeight + qSize / sizeFactor + kvSize / sizeFactor, qkvScales + qSize + kvSize,
                qkvZeros + qSize + kvSize, qkvBias + qSize + kvSize, attnOutWeight, attnOutScales, attnOutZeros,
                attnOutBias, ln1Gamma, ln1Beta, fc1Weight, fc1Scales, fc1Zeros, fc1Bias, fc2Weight, fc2Scales, fc2Zeros,
                fc2Bias, ln2Gamma, ln2Beta, fc3Weight, fc3Scales, fc3Zeros, fc3Bias, false);

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
        free(fc3Bias);
        free(ln1Gamma);
        free(ln1Beta);
        free(ln2Gamma);
        free(ln2Beta);
    }

    xft::ParamType xftDT2PT(xft::DataType dt) {
        switch (dt) {
            case xft::DataType::int8: return xft::ParamType::Int8;
            case xft::DataType::int4: return xft::ParamType::INT4;
            case xft::DataType::fp32: return xft::ParamType::FP32;
            case xft::DataType::bf16: return xft::ParamType::BF16;
            case xft::DataType::fp16: return xft::ParamType::FP16;
            default: return xft::ParamType::None;
        }
    }

    xft::AttnParams *createAttnParams(DecoderContext *ctx, const std::string &modelPath, xft::DataType dt) {
        xft::AttnParams *attnParams = nullptr;

        // Deepseek model
        if (fileExists(modelPath + "/model.layers.0.self_attn.q_a_proj.weight.bin")) {
            attnParams = new xft::MLAttnParams(ctx->hiddenSize, ctx->qLoraRank, ctx->kvLoraRank, ctx->attHeadNum,
                    ctx->nopeDim, ctx->ropeDim, ctx->headDim, xftDT2PT(dt));
        } else {
            attnParams = new xft::GQAttnParams(
                    ctx->hiddenSize, ctx->attHeadNum, ctx->kvHeadNum, ctx->attHeadSize, xftDT2PT(dt));
        }

        return attnParams;
    }

    xft::FFNParams *createFFNParams(DecoderContext *ctx, const std::string &modelPath, xft::DataType dt) {
        xft::FFNParams *ffnParams = nullptr;

        if (fileExists(modelPath + "/model.layers.0.mlp.dense_h_to_4h.weight.0.bin")) {
            ffnParams = new xft::GptFFNParams(ctx->hiddenSize, ctx->intermediateSize, xftDT2PT(dt));
        } else if (fileExists(modelPath + "/model.layers.0.mlp.gate_proj.weight.0.bin")) {
            ffnParams = new xft::LlamaFFNParams(ctx->hiddenSize, ctx->intermediateSize, xftDT2PT(dt));
        } else if (fileExists(modelPath + "/model.layers.0.moe.gate.weight.bin")) {
            ffnParams = new xft::MixtralFFNParams(
                    ctx->sparseExperts, ctx->hiddenSize, ctx->intermediateSize, xftDT2PT(dt));
        } else if (fileExists(modelPath + "/model.layers.0.self_attn.kv_a_layernorm.weight.bin")) {
            ffnParams = new xft::DeepSeekFFNParams(ctx->sparseExperts, ctx->denseExperts, ctx->hiddenSize,
                    ctx->intermediateSize, ctx->moeIntermediateSize, xftDT2PT(dt));
        } else {
            xft::Logger::error("Unable to detect FFN parameters.");
            std::exit(-1);
        }

        return ffnParams;
    }

    // WType: required weight type to save in AttnParams and FFNParams
    template <typename WType>
    void setDecoderWeights(DecoderContext *ctx, DECODER *pdecoder, const std::string &modelPath, int layerIdx,
            xft::AttnParams *attnParams, xft::FFNParams *ffnParams) {
        // INT8/INT4 quant, wbits = 8/4, qweight dtype: int8_t/uint4x2_t
        if constexpr (std::is_same_v<WType, int8_t> || std::is_same_v<WType, uint4x2_t>) {
            xft::Logger::error("Unable to load INT8/INT4 weights yet.");
            exit(-1);
        }
        // FP32/BF16/FP16
        else if constexpr (std::is_same_v<WType, float> || std::is_same_v<WType, bfloat16_t>
                || std::is_same_v<WType, float16_t>) {
            xft::GQAttnParams *gqap = dynamic_cast<xft::GQAttnParams *>(attnParams);
            xft::MLAttnParams *mlap = dynamic_cast<xft::MLAttnParams *>(attnParams);
            if (gqap != NULL) {
                loadGQAttnWeights<WType>(ctx, modelPath, layerIdx, gqap);
            } else if (mlap != NULL) {
                loadMLAttnWeights<WType>(ctx, modelPath, layerIdx, mlap);
            } else {
                xft::Logger::error("Unable to cast AttnParams to GQAttnParams or MLATtnParams.");
                std::exit(-1);
            }

            // Stardard 2 layer MLP
            if (fileExists(modelPath + "/model.layers.0.mlp.dense_h_to_4h.weight.0.bin")) {
                loadGptFFNWeights<WType>(ctx, modelPath, layerIdx, static_cast<xft::GptFFNParams *>(ffnParams));
            }
            // Llama like models
            else if (fileExists(modelPath + "/model.layers.0.mlp.gate_proj.weight.0.bin")) {
                loadLlamaFFNWeights<WType>(ctx, modelPath, layerIdx, static_cast<xft::LlamaFFNParams *>(ffnParams));
            }
            // For models like Mixtral
            else if (fileExists(modelPath + "/model.layers.0.moe.gate.weight.bin")) {
                printf("Loading MixtralFFNWeights\n");
                loadMixtralFFNWeights<WType>(ctx, modelPath, layerIdx, static_cast<xft::MixtralFFNParams *>(ffnParams));
            }
            // For DeepSeekV2+ models
            else if (fileExists(modelPath + "/model.layers.0.self_attn.kv_a_proj_with_mqa.weight.bin")) {
                printf("Loading DeepSeekFFNWeights\n");
                loadDeepSeekFFNWeights<WType>(
                        ctx, modelPath, layerIdx, static_cast<xft::DeepSeekFFNParams *>(ffnParams));
            } else {
                xft::Logger::error("Unable to load FFN weights.");
                std::exit(-1);
            }
        }

        pdecoder->template setWeights<WType>(ctx, attnParams, ffnParams);
    }

    template <typename T>
    void loadGQAttnWeights(DecoderContext *ctx, const std::string &modelPath, int layerIdx, xft::GQAttnParams *attn) {
        int hiddenSize = ctx->hiddenSize;
        int qSize = ctx->attHeadSize * ctx->attHeadNum;
        int kvSize = ctx->attHeadSize * ctx->kvHeadNum;
        int qkvSize = qSize + 2 * kvSize;

        std::string strIdx = std::to_string(layerIdx);
        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".attention.query_key_value.weight.0.bin",
                (T *)attn->qkv.weight, hiddenSize * qkvSize);
        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".attention.dense.weight.0.bin", (T *)attn->out.weight,
                qSize * hiddenSize);
        xft::loadWeight2(
                modelPath + "/model.layers." + strIdx + ".input_layernorm.weight.bin", attn->norm.gamma, hiddenSize);

        // The bias is optional
        loadOptionalBias(modelPath + "/model.layers." + strIdx + ".attention.query_key_value.bias.0.bin", attn->qkv,
                "read QKV bias error");
        loadOptionalBias(modelPath + "/model.layers." + strIdx + ".attention.dense.bias.bin", attn->out,
                "read attn dense bias error");
        loadOptionalBias(modelPath + "/model.layers." + strIdx + ".input_layernorm.bias.bin", attn->norm,
                "read LN1(attention) beta error");
    }

    template <typename T>
    void loadMLAttnWeights(DecoderContext *ctx, const std::string &modelPath, int layerIdx, xft::MLAttnParams *attn) {
        int hiddenSize = ctx->hiddenSize;

        int qLoraRank = ctx->qLoraRank;
        int kvLoraRank = ctx->kvLoraRank;
        int nopeDim = ctx->nopeDim;
        int ropeDim = ctx->ropeDim;
        int vHeadDim = ctx->attHeadSize;

        int qSize = ctx->attHeadNum * (nopeDim + ropeDim);
        int kvSize = ctx->attHeadNum * 2 * (nopeDim + vHeadDim);

        std::string strIdx = std::to_string(layerIdx);

        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".input_layernorm.weight.bin", attn->input_norm.gamma,
                hiddenSize);

        if (qLoraRank > 0) {
            xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".self_attn.q_a_proj.weight.bin",
                    (T *)attn->q_a_proj.weight, hiddenSize * qLoraRank);
            xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".self_attn.q_a_layernorm.weight.bin",
                    attn->q_a_norm.gamma, qLoraRank);
            xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".self_attn.q_b_proj.weight.bin",
                    (T *)attn->q_b_proj.weight, qLoraRank * qSize);
        } else {
            // DeepSeek V2 Lite
            xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".self_attn.q_a_proj.weight.bin",
                    (T *)attn->q_a_proj.weight, hiddenSize * qSize);
        }

        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".self_attn.kv_a_proj_with_mqa.weight.bin",
                (T *)attn->kv_a_proj.weight, hiddenSize * (kvLoraRank + ropeDim));
        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".self_attn.kv_a_layernorm.weight.bin",
                attn->kv_a_norm.gamma, kvLoraRank);
        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".self_attn.kv_b_proj.weight.bin",
                (T *)attn->kv_b_proj.weight, kvLoraRank * kvSize);

        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".attention.dense.weight.bin", (T *)attn->o_proj.weight,
                ctx->attHeadNum * vHeadDim * hiddenSize);
    }

    template <typename T>
    void loadGptFFNWeights(DecoderContext *ctx, const std::string &modelPath, int layerIdx, xft::GptFFNParams *ffn) {
        std::string strIdx = std::to_string(layerIdx);

        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.dense_h_to_4h.weight.0.bin",
                (T *)ffn->fc1.weight, ffn->fc1.input_dim * ffn->fc1.output_dim);
        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.dense_4h_to_h.weight.0.bin",
                (T *)ffn->fc2.weight, ffn->fc2.input_dim * ffn->fc2.output_dim);

        // The bias is optional
        // As of some history reason, the convert script tried to split the first layer's weight and bias, thus contains '0'?
        loadOptionalBias(modelPath + "/model.layers." + strIdx + ".mlp.dense_h_to_4h.bias.0.bin", ffn->fc1,
                "read FC1 bias error");
        loadOptionalBias(
                modelPath + "/model.layers." + strIdx + ".mlp.dense_4h_to_h.bias.bin", ffn->fc2, "read FC2 bias error");

        // Norm params
        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".post_attention_layernorm.weight.bin",
                ffn->norm.gamma, ffn->norm.hidden_size);
        loadOptionalBias(modelPath + "/model.layers." + strIdx + ".post_attention_layernorm.bias.bin", ffn->norm,
                "read LN2(FFN) beta error");
    }

    template <typename T>
    void loadLlamaFFNWeights(
            DecoderContext *ctx, const std::string &modelPath, int layerIdx, xft::LlamaFFNParams *ffn) {
        std::string strIdx = std::to_string(layerIdx);

        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.gate_proj.weight.0.bin", (T *)ffn->gate.weight,
                ffn->gate.input_dim * ffn->gate.output_dim);
        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.up_proj.weight.0.bin", (T *)ffn->up.weight,
                ffn->up.input_dim * ffn->up.output_dim);
        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.down_proj.weight.0.bin", (T *)ffn->down.weight,
                ffn->down.input_dim * ffn->down.output_dim);

        // The bias is optional
        loadOptionalBias(
                modelPath + "/model.layers." + strIdx + ".mlp.gate_proj.bias.0.bin", ffn->gate, "read gate bias error");
        loadOptionalBias(
                modelPath + "/model.layers." + strIdx + ".mlp.up_proj.bias.0.bin", ffn->up, "read up bias error");
        loadOptionalBias(
                modelPath + "/model.layers." + strIdx + ".mlp.down_proj.bias.0.bin", ffn->down, "read down bias error");

        // Norm params
        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".post_attention_layernorm.weight.bin",
                ffn->norm.gamma, ffn->norm.hidden_size);
        loadOptionalBias(modelPath + "/model.layers." + strIdx + ".post_attention_layernorm.bias.bin", ffn->norm,
                "read LN2(FFN) beta error");
    }

    template <typename T>
    void loadMixtralFFNWeights(
            DecoderContext *ctx, const std::string &modelPath, int layerIdx, xft::MixtralFFNParams *ffn) {
        std::string strIdx = std::to_string(layerIdx);

        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".moe.gate.weight.bin", (T *)ffn->gating.weight,
                ffn->gating.input_dim * ffn->gating.output_dim);

        // Load expert weights
        if (ffn->experts.empty()) {
            for (int i = 0; i < ctx->sparseExperts; ++i) {
                xft::ExpertParams expert(ctx->hiddenSize, ctx->intermediateSize, xft::ParamType::FP32);
                xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".moe.sparse_experts." + std::to_string(i)
                                + ".w1.0.bin",
                        (T *)expert.gate.weight, ctx->hiddenSize * ctx->intermediateSize);
                xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".moe.sparse_experts." + std::to_string(i)
                                + ".w2.0.bin",
                        (T *)expert.up.weight, ctx->hiddenSize * ctx->intermediateSize);
                xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".moe.sparse_experts." + std::to_string(i)
                                + ".w3.0.bin",
                        (T *)expert.down.weight, ctx->hiddenSize * ctx->intermediateSize);
                ffn->experts.push_back(std::move(expert));
            }
        } else {
            for (int i = 0; i < ctx->sparseExperts; ++i) {
                xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".moe.sparse_experts." + std::to_string(i)
                                + ".w1.0.bin",
                        (T *)ffn->experts[i].gate.weight, ctx->hiddenSize * ctx->intermediateSize);
                xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".moe.sparse_experts." + std::to_string(i)
                                + ".w2.0.bin",
                        (T *)ffn->experts[i].up.weight, ctx->hiddenSize * ctx->intermediateSize);
                xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".moe.sparse_experts." + std::to_string(i)
                                + ".w3.0.bin",
                        (T *)ffn->experts[i].down.weight, ctx->hiddenSize * ctx->intermediateSize);
            }
        }

        // Norm params
        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".post_attention_layernorm.weight.bin",
                ffn->norm.gamma, ffn->norm.hidden_size);
        loadOptionalBias(modelPath + "/model.layers." + strIdx + ".post_attention_layernorm.bias.bin", ffn->norm,
                "read LN2(FFN) beta error");
    }

    template <typename T>
    void loadDeepSeekFFNWeights(
            DecoderContext *ctx, const std::string &modelPath, int layerIdx, xft::DeepSeekFFNParams *ffn) {
        std::string strIdx = std::to_string(layerIdx);

        // Norm params
        xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".post_attention_layernorm.weight.bin",
                ffn->norm.gamma, ffn->norm.hidden_size);

        if (layerIdx < ctx->firstKDenseReplace) {
            // Load MLP
            xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.down_proj.weight.bin",
                    (T *)ffn->mlp.down.weight, ctx->hiddenSize * ctx->intermediateSize);
            xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.up_proj.weight.bin", (T *)ffn->mlp.up.weight,
                    ctx->hiddenSize * ctx->intermediateSize);
            xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.gate_proj.weight.bin",
                    (T *)ffn->mlp.gate.weight, ctx->hiddenSize * ctx->intermediateSize);
        } else {
            // Load experts weights
            // Load shared expert weights
            xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.shared_experts.down_proj.weight.bin",
                    (T *)ffn->sharedExpert.down.weight, ctx->hiddenSize * ctx->moeIntermediateSize * ctx->denseExperts);
            xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.shared_experts.up_proj.weight.bin",
                    (T *)ffn->sharedExpert.up.weight, ctx->hiddenSize * ctx->moeIntermediateSize * ctx->denseExperts);
            xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.shared_experts.gate_proj.weight.bin",
                    (T *)ffn->sharedExpert.gate.weight, ctx->hiddenSize * ctx->moeIntermediateSize * ctx->denseExperts);

            // Load routed expert weights
            if (ffn->routedExperts.empty()) {
                for (int i = 0; i < ctx->denseExperts; ++i) {
                        ffn->routedExperts.emplace_back(
                            ctx->hiddenSize, ctx->moeIntermediateSize, ffn->mlp.gate.wtype, ffn->mlp.gate.wtrans);
                }
            }

            for (int i = 0; i < ctx->sparseExperts; ++i) {
                xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.experts." + std::to_string(i)
                                + ".down_proj.weight.bin",
                        (T *)ffn->routedExperts[i].down.weight, ctx->hiddenSize * ctx->moeIntermediateSize);
                xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.experts." + std::to_string(i)
                                + ".up_proj.weight.bin",
                        (T *)ffn->routedExperts[i].up.weight, ctx->hiddenSize * ctx->moeIntermediateSize);
                xft::loadWeight2(modelPath + "/model.layers." + strIdx + ".mlp.experts." + std::to_string(i)
                                + ".gate_proj.weight.bin",
                        (T *)ffn->routedExperts[i].gate.weight, ctx->hiddenSize * ctx->moeIntermediateSize);
            }
        }
    }

    void loadOptionalBias(const std::string &filename, xft::DenseLayerParams &dense, const char *errmsg) {
        if (!fileExists(filename)) {
            dense.removeBias();
            return;
        }

        // Load the bias as float
        float *addr = (float *)xft::alloc(dense.output_dim * sizeof(float));
        int size = dense.output_dim;
        int ret = xft::loadWeight2<float>(filename, addr, size);

        // No bias?
        if (ret == 0) {
            dense.removeBias();
        }
        // Success
        else if (ret == size) {
            dense.setBiasValue(addr);
        } else {
            printf("Error loading the bias: %s\n", errmsg);
            exit(-1);
        }

        xft::dealloc(addr);
    }

    void loadOptionalBias(const std::string &filename, xft::NormParams &norm, const char *errmsg) {
        if (!fileExists(filename)) {
            norm.emptyBeta();
            return;
        }

        // Load the beta value as float
        int ret = xft::loadWeight2<float>(filename, norm.beta, norm.hidden_size);

        // No bias?
        if (ret == 0) {
            norm.emptyBeta();
        }
        // Failed
        else if (ret != norm.hidden_size) {
            xft::Logger::error(errmsg);
            exit(-1);
        }
    }

private:
    std::vector<DECODER *> decoders;
};
