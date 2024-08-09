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
#include "decoder_layer.h"
#include "attention.h"
#include "kvcache_manager.h"
#include "layer_norm.h"
#include "layers_attention.h"
#include "layers_decoder.h"
#include "layers_mlp.h"
#include "mlp_llama.h"
#include "rms_norm.h"
#include "numa_allocator.h"

#include <unordered_map>

namespace xft {

template <typename DataT, typename KVCacheT, typename RopeT, typename NormT>
void LayerLLaMAImpl(DataType dt, ActivationType at, NormType nt, int batchSize, int inputSeqLen, int attHeadDim,
        int attHeadNum, int kvHeadNum, int maxPositions, int maxPosEmbed, int pastSeqLen, int currentSeqLen, int step,
        int hiddenSize, int intermediateSize, void *output, int outputStride, const void *input, int inputStride,
        const float *ln1Gamma, const float *ln1Beta, const void *queryWeight, const void *keyWeight,
        const void *valueWeight, const void *attnOutWeight, const float *ln2Gamma, const float *ln2Beta,
        const void *gateWeight, const void *upWeight, const void *downWeight, const float *queryBias,
        const float *keyBias, const float *valueBias, const float *attnOutBias, 
        MMHelper *mmHelper, DecoderContext *ctx, KVCacheManager<KVCacheT> *kvCacheMgr) {

    // TODO: will deprecate attention mask in future, so need to change this
    auto prepareAttnMask = [&](DecoderContext *ctx, int step) {
        int seqLen = ctx->inputSeqLen;
        int accSeqLen = pastSeqLen + currentSeqLen;
        float *mask = nullptr;

        auto getAttnMask = [](int sizeRequired) {
            static float *attnMask;
            static int maskSize = 0;
            if (maskSize < sizeRequired) {
                if (attnMask) free(attnMask);
                attnMask = (float *)xft::alloc(sizeRequired * sizeof(float));
                maskSize = sizeRequired;
            }
            return attnMask;
        };

        if (step == 0) {
            int sizeRequired = ctx->batchSize * seqLen * seqLen;
            mask = getAttnMask(sizeRequired);
            for (int b = 0; b < ctx->batchSize; ++b) {
                auto pmask = mask + b * seqLen * seqLen;
                for (int i = 0; i < seqLen; ++i) {
                    memset(pmask + i * seqLen, 0, (i + 1) * sizeof(float)); // bottom left are 0
                    std::fill_n(pmask + i * seqLen + i + 1, seqLen - i - 1, std::numeric_limits<float>::lowest());
                }
            }
        } else if (seqLen > 1) {
            int sizeRequired = ctx->batchSize * accSeqLen * seqLen;
            mask = getAttnMask(sizeRequired);
            for (int b = 0; b < ctx->batchSize; ++b) {
                auto pmask = mask + b * accSeqLen * seqLen;
                int pastLen = accSeqLen - seqLen;
                for (int i = 0; i < seqLen; ++i) {
                    memset(pmask + i * accSeqLen, 0, (pastLen + i + 1) * sizeof(float));
                    std::fill_n(pmask + i * accSeqLen + pastLen + i + 1, seqLen - i - 1,
                            std::numeric_limits<float>::lowest());
                }
            }
        } else {
            int sizeRequired = ctx->batchSize * accSeqLen;
            mask = getAttnMask(sizeRequired);
            memset(mask, 0, ctx->batchSize * accSeqLen * sizeof(float)); // all elements are 0
        }

        return mask;
    };

    //using DECODER = Decoder<Attention<DataT, LlamaRotaryEmbedding, NormT>, LlamaMLP<DataT>>;
    using DECODER = Decoder<Attention<DataT, RopeT, NormT>, LlamaMLP<DataT>>;
    static std::unordered_map<std::string, DECODER *> llama_layer_hub;

    // create hash key and value: if hidden and intermediateSize is changed , then memory pointer is also changed.
    std::stringstream weights_addr;
    weights_addr << queryWeight << "_" << keyWeight << "_" << valueWeight << "_" << attnOutWeight << "_" << gateWeight
                 << "_" << upWeight << "_" << downWeight << "_" << dt << "_" << at << "_" << nt << "_" << attHeadDim
                 << "_" << attHeadNum << "_" << kvHeadNum;
    std::string llama_layer_key = weights_addr.str();
    DECODER *llama_layer;

    auto it_created = llama_layer_hub.find(llama_layer_key);
    if (it_created == llama_layer_hub.end()) {
        int firstNode = getenv("FIRST_TOKEN_WEIGHT_LOCATION") ? atoi(getenv("FIRST_TOKEN_WEIGHT_LOCATION")) : -1;
        xft_set_preferred_node(firstNode);
        llama_layer = new DECODER(ctx, 0);
        llama_layer->setWeights(ctx, (const float *)queryWeight, nullptr, nullptr, queryBias, (const float *)keyWeight,
                nullptr, nullptr, keyBias, (const float *)valueWeight, nullptr, nullptr, valueBias,
                (const float *)attnOutWeight, nullptr, nullptr, attnOutBias, ln1Gamma, ln1Beta,
                (const float *)gateWeight, nullptr, nullptr, nullptr, (const float *)upWeight, nullptr, nullptr,
                nullptr, ln2Gamma, ln2Beta, (const float *)downWeight, nullptr, nullptr, false);
        llama_layer_hub[llama_layer_key] = llama_layer;
        printf(">> create llama_layer_key: %s\n", llama_layer_key.c_str());
        xft_set_preferred_node(-1);
    } else {
        llama_layer = it_created->second;
    }

    ctx->resize(batchSize, inputSeqLen, pastSeqLen);
    xft::Matrix<float> actBuffers;
    actBuffers.Resize(batchSize * inputSeqLen * 2, hiddenSize);
    float *attnMask = prepareAttnMask(ctx, step);

    int workers = 1;
    int headsPerSplit = (ctx->kvHeadNum + workers - 1) / workers;
    kvCacheMgr->resize(maxPositions, batchSize, headsPerSplit, attHeadDim);
    KVCacheTensor<KVCacheT> &presentKey = kvCacheMgr->getKey(0);
    KVCacheTensor<KVCacheT> &presentValue = kvCacheMgr->getValue(0);

    float *attnOut = (float *)(ctx->tmpBuf.Data());

    llama_layer->forwardAttention(ctx, (float *)input, actBuffers.Data(), attnOut, attnMask,
            presentKey, // presentKey,
            presentValue, // presentValue,
            inputSeqLen, // inputSeqLen,
            pastSeqLen, // pastSeqLen
            step == 0, // useSelfAttn,
            true, // doLnBefore,
            nullptr);

    llama_layer->forwardFFN(ctx, attnOut, (float *)output, inputStride, outputStride, true);
}

template <typename KVCacheT, typename RopeT>
void LayerLLaMAWrapper(DataType dt, ActivationType at, NormType nt, int batchSize, int inputSeqLen, int attHeadDim,
        int attHeadNum, int kvHeadNum, int maxPositions, int maxPosEmbed, int pastSeqLen, int currentSeqLen, int step,
        int hiddenSize, int intermediateSize, void *output, int outputStride, const void *input, int inputStride,
        const float *ln1Gamma, const float *ln1Beta, const void *queryWeight, const void *keyWeight,
        const void *valueWeight, const void *attnOutWeight, const float *ln2Gamma, const float *ln2Beta,
        const void *gateWeight, const void *upWeight, const void *downWeight, const float *queryBias,
        const float *keyBias, const float *valueBias, const float *attnOutBias) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    std::string actType;
    if (at == ActivationType::SILU)
        actType = "silu";
    else if (at == ActivationType::RELU)
        actType = "relu";
    else if (at == ActivationType::GELU)
        actType = "gelu";
    else if (at == ActivationType::SWIGLU)
        actType = "swiglu";
    else {
        printf(">> unsupported activation type\n");
        return;
    }

    static MMHelper *mmHelper;
    static DecoderContext *ctx;
    if (ctx == nullptr
            || (ctx != nullptr && (ctx->hiddenSize != hiddenSize || ctx->intermediateSize != intermediateSize))) {
        if (ctx != nullptr) delete ctx;
        printf(">> create context: %d %d\n", hiddenSize, intermediateSize);
        mmHelper = new MMHelper(Env::getInstance().getEngineKind(), Env::getInstance().getEngineIndex());
        ctx = new DecoderContext(1, hiddenSize, attHeadDim, attHeadNum, kvHeadNum, intermediateSize, actType, 1e-6, 0,
                0, maxPositions, maxPosEmbed, -1, 0, 1, mmHelper);
    }

    KVCacheManager<KVCacheT> *kvCacheMgr;
    static std::unordered_map<std::string, KVCacheManager<KVCacheT> *> kv_hub;

    // create hash key and value: if hidden and intermediateSize is changed , then memory pointer is also changed.
    std::stringstream layer_key;
    layer_key << queryWeight << "_" << keyWeight << "_" << valueWeight << "_" << attnOutWeight << "_" << gateWeight
                 << "_" << upWeight << "_" << downWeight << "_" << dt << "_" << at << "_" << nt << "_" << attHeadDim
                 << "_" << attHeadNum << "_" << kvHeadNum;
    std::string kv_hub_key = layer_key.str();

    auto it_created = kv_hub.find(kv_hub_key);
    if (it_created == kv_hub.end()) {
        int nextNode = getenv("NEXT_TOKEN_WEIGHT_LOCATION") ? atoi(getenv("NEXT_TOKEN_WEIGHT_LOCATION")) : -1;
        xft_set_preferred_node(nextNode);
        kvCacheMgr = new KVCacheManager<KVCacheT>(1);
        kv_hub[kv_hub_key] = kvCacheMgr;
        printf(">> create kv_hub_key: %s\n", kv_hub_key.c_str());
        xft_set_preferred_node(-1);
    } else {
        kvCacheMgr = it_created->second;
    }

    if (dt == DataType::bf16) {
        if (nt == NormType::RMS) {
            LayerLLaMAImpl<bfloat16_t, KVCacheT, RopeT, RmsNorm>(dt, at, nt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum,
                    maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, intermediateSize, output,
                    outputStride, input, inputStride, ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight,
                    attnOutWeight, ln2Gamma, ln2Beta, gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias,
                    attnOutBias, mmHelper, ctx, kvCacheMgr);
        } else if (nt == NormType::LN) {
            LayerLLaMAImpl<bfloat16_t, KVCacheT, RopeT, LayerNorm>(dt, at, nt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum,
                    maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, intermediateSize, output,
                    outputStride, input, inputStride, ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight,
                    attnOutWeight, ln2Gamma, ln2Beta, gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias,
                    attnOutBias, mmHelper, ctx, kvCacheMgr);
        } else {
            printf(">> unsupported norm type\n");
        }
    } else if (dt == DataType::fp16) {
        if (nt == NormType::RMS) {
            LayerLLaMAImpl<float16_t, KVCacheT, RopeT, RmsNorm>(dt, at, nt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum,
                    maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, intermediateSize, output,
                    outputStride, input, inputStride, ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight,
                    attnOutWeight, ln2Gamma, ln2Beta, gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias,
                    attnOutBias, mmHelper, ctx, kvCacheMgr);
        } else if (nt == NormType::LN) {
            LayerLLaMAImpl<float16_t, KVCacheT, RopeT, LayerNorm>(dt, at, nt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum,
                    maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, intermediateSize, output,
                    outputStride, input, inputStride, ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight,
                    attnOutWeight, ln2Gamma, ln2Beta, gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias,
                    attnOutBias, mmHelper, ctx, kvCacheMgr);
        } else {
            printf(">> unsupported norm type\n");
        }
    } else if (dt == DataType::bf16_int8) {
        if (nt == NormType::RMS) {
            auto firstTokenFunc = LayerLLaMAImpl<bfloat16_t, KVCacheT, RopeT, RmsNorm>;
            auto nextTokenFunc = LayerLLaMAImpl<int8_t, KVCacheT, RopeT, RmsNorm>;
            if (step == 0) {
                    firstTokenFunc(dt, at, nt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum,
                        maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, intermediateSize, output,
                        outputStride, input, inputStride, ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight,
                        attnOutWeight, ln2Gamma, ln2Beta, gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias,
                        attnOutBias, mmHelper, ctx, kvCacheMgr);

            } else {
                    nextTokenFunc(dt, at, nt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum,
                        maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, intermediateSize, output,
                        outputStride, input, inputStride, ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight,
                        attnOutWeight, ln2Gamma, ln2Beta, gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias,
                        attnOutBias, mmHelper, ctx, kvCacheMgr);
            }
        } else if (nt == NormType::LN) {
            auto firstTokenFunc = LayerLLaMAImpl<bfloat16_t, KVCacheT, RopeT, LayerNorm>;
            auto nextTokenFunc = LayerLLaMAImpl<int8_t, KVCacheT, RopeT, LayerNorm>;
            if (step == 0)
                    firstTokenFunc(dt, at, nt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum,
                        maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, intermediateSize, output,
                        outputStride, input, inputStride, ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight,
                        attnOutWeight, ln2Gamma, ln2Beta, gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias,
                        attnOutBias, mmHelper, ctx, kvCacheMgr);
            else
                    nextTokenFunc(dt, at, nt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum,
                        maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, intermediateSize, output,
                        outputStride, input, inputStride, ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight,
                        attnOutWeight, ln2Gamma, ln2Beta, gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias,
                        attnOutBias, mmHelper, ctx, kvCacheMgr);
        } else {
            printf(">> unsupported norm type\n");
        }
    } else {
        printf(">> unsupported data type\n");
    }
}

void invokeLayerLLaMA(DataType dt, DataType kvcdt, RopeType rt, ActivationType at, NormType nt, int batchSize, int inputSeqLen, int attHeadDim,
        int attHeadNum, int kvHeadNum, int maxPositions, int maxPosEmbed, int pastSeqLen, int currentSeqLen, int step,
        int hiddenSize, int intermediateSize, void *output, int outputStride, const void *input, int inputStride,
        const float *ln1Gamma, const float *ln1Beta, const void *queryWeight, const void *keyWeight,
        const void *valueWeight, const void *attnOutWeight, const float *ln2Gamma, const float *ln2Beta,
        const void *gateWeight, const void *upWeight, const void *downWeight, const float *queryBias,
        const float *keyBias, const float *valueBias, const float *attnOutBias) {

    if (kvcdt == DataType::fp16) {
        if (rt == RopeType::LLAMA_ROPE)
            return LayerLLaMAWrapper<float16_t, LlamaRotaryEmbedding>(dt, at, nt, batchSize, inputSeqLen, attHeadDim,
                attHeadNum, kvHeadNum, maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step,
                hiddenSize, intermediateSize, output, outputStride, input, inputStride,
                ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight, attnOutWeight, ln2Gamma, ln2Beta,
                gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias, attnOutBias) ;
        else {
            printf(">> unsupported Rope type: %d\n", rt);
        }
    } else if (kvcdt == DataType::int8) {
        if (rt == RopeType::LLAMA_ROPE)
            return LayerLLaMAWrapper<int8_t, LlamaRotaryEmbedding>(dt, at, nt, batchSize, inputSeqLen, attHeadDim,
                attHeadNum, kvHeadNum, maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step,
                hiddenSize, intermediateSize, output, outputStride, input, inputStride,
                ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight, attnOutWeight, ln2Gamma, ln2Beta,
                gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias, attnOutBias) ;
        else {
            printf(">> unsupported Rope type: %d\n", rt);
        }
    } else {
        printf(">> unsupported KVcache data type: %d\n", kvcdt);
        return;
    }

}

} // namespace xft
