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

#include <unordered_map>

namespace xft {

template <typename DataT, typename NormT>
void LayerLLaMAImpl(DataType dt, ActivationType at, NormType nt, int batchSize, int inputSeqLen, int attHeadDim,
        int attHeadNum, int kvHeadNum, int maxPositions, int maxPosEmbed, int pastSeqLen, int currentSeqLen, int step,
        int hiddenSize, int intermediateSize, void *output, int outputStride, const void *input, int inputStride,
        const float *ln1Gamma, const float *ln1Beta, const void *queryWeight, const void *keyWeight,
        const void *valueWeight, const void *attnOutWeight, const float *ln2Gamma, const float *ln2Beta,
        const void *gateWeight, const void *upWeight, const void *downWeight, const float *queryBias,
        const float *keyBias, const float *valueBias, const float *attnOutBias) {

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

    using DECODER = Decoder<Attention<DataT, LlamaRotaryEmbedding, NormT>, LlamaMLP<DataT>>;
    static std::unordered_map<std::string, DECODER *> llama_layer_hub;
    static DecoderContext *ctx;
    static KVCacheManager<float16_t> *kvCacheMgr;

    std::string actType;
    if (at == ActivationType::SILU)
        actType = "silu";
    else if (at == ActivationType::RELU)
        actType = "relu";
    else if (at == ActivationType::GELU)
        actType = "gelu";
    else if (at == ActivationType::SWIGLU)
        actType = "swiglu";
    else
        printf(">> unsupported activation type\n");

    if (ctx == nullptr
            || (ctx != nullptr && (ctx->hiddenSize != hiddenSize || ctx->intermediateSize != intermediateSize))) {
        if (ctx != nullptr) delete ctx;
        printf(">> create context: %d %d\n", hiddenSize, intermediateSize);
        ctx = new DecoderContext(1, hiddenSize, attHeadDim, attHeadNum, kvHeadNum, intermediateSize, actType, 1e-6, 0,
                0, maxPositions, maxPosEmbed, -1, 0, 1);
        ctx->mmHelper = new MMHelper(Env::getInstance().getEngineKind(), Env::getInstance().getEngineIndex());
        if (kvCacheMgr != nullptr) delete kvCacheMgr;
        kvCacheMgr = new KVCacheManager<float16_t>(1);
    }

    // create hash key and value: if hidden and intermediateSize is changed , then memory pointer is also changed.
    std::stringstream weights_addr;
    weights_addr << queryWeight << "_" << keyWeight << "_" << valueWeight << "_" << attnOutWeight << "_" << gateWeight
                 << "_" << upWeight << "_" << downWeight << "_" << dt << "_" << at << "_" << nt << "_" << attHeadDim
                 << "_" << attHeadNum << "_" << kvHeadNum;
    std::string llama_layer_key = weights_addr.str();
    DECODER *llama_layer;

    auto it_created = llama_layer_hub.find(llama_layer_key);
    if (it_created == llama_layer_hub.end()) {
        llama_layer = new DECODER(ctx, 0);
        llama_layer->setWeights(ctx, (const float *)queryWeight, nullptr, nullptr, queryBias, (const float *)keyWeight,
                nullptr, nullptr, keyBias, (const float *)valueWeight, nullptr, nullptr, valueBias,
                (const float *)attnOutWeight, nullptr, nullptr, attnOutBias, ln1Gamma, ln1Beta,
                (const float *)gateWeight, nullptr, nullptr, nullptr, (const float *)upWeight, nullptr, nullptr,
                nullptr, ln2Gamma, ln2Beta, (const float *)downWeight, nullptr, nullptr, false);
        llama_layer_hub[llama_layer_key] = llama_layer;
        printf(">> create llama_layer_key: %s\n", llama_layer_key.c_str());
    } else {
        llama_layer = it_created->second;
    }

    ctx->resize(batchSize, inputSeqLen, pastSeqLen);
    hpj::Matrix<float> actBuffers;
    actBuffers.Resize(batchSize * inputSeqLen * 2, hiddenSize);
    float *attnMask = prepareAttnMask(ctx, step);

    int workers = 1;
    int headsPerSplit = (ctx->kvHeadNum + workers - 1) / workers;
    kvCacheMgr->resize(maxPositions, batchSize, headsPerSplit, attHeadDim);
    KVCacheTensor<float16_t> &presentKey = kvCacheMgr->getKey(0);
    KVCacheTensor<float16_t> &presentValue = kvCacheMgr->getValue(0);

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

void invokeLayerLLaMA(DataType dt, ActivationType at, NormType nt, int batchSize, int inputSeqLen, int attHeadDim,
        int attHeadNum, int kvHeadNum, int maxPositions, int maxPosEmbed, int pastSeqLen, int currentSeqLen, int step,
        int hiddenSize, int intermediateSize, void *output, int outputStride, const void *input, int inputStride,
        const float *ln1Gamma, const float *ln1Beta, const void *queryWeight, const void *keyWeight,
        const void *valueWeight, const void *attnOutWeight, const float *ln2Gamma, const float *ln2Beta,
        const void *gateWeight, const void *upWeight, const void *downWeight, const float *queryBias,
        const float *keyBias, const float *valueBias, const float *attnOutBias) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (dt == DataType::bf16) {
        if (nt == NormType::RMS)
            LayerLLaMAImpl<bfloat16_t, RmsNorm>(dt, at, nt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum,
                    maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, intermediateSize, output,
                    outputStride, input, inputStride, ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight,
                    attnOutWeight, ln2Gamma, ln2Beta, gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias,
                    attnOutBias);
        else if (nt == NormType::LN) {
            LayerLLaMAImpl<bfloat16_t, LayerNorm>(dt, at, nt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum,
                    maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, intermediateSize, output,
                    outputStride, input, inputStride, ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight,
                    attnOutWeight, ln2Gamma, ln2Beta, gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias,
                    attnOutBias);
        } else {
            printf(">> unsupported norm type\n");
        }
    } else if (dt == DataType::fp16) {
        if (nt == NormType::RMS)
            LayerLLaMAImpl<float16_t, RmsNorm>(dt, at, nt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum,
                    maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, intermediateSize, output,
                    outputStride, input, inputStride, ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight,
                    attnOutWeight, ln2Gamma, ln2Beta, gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias,
                    attnOutBias);
        else if (nt == NormType::LN) {
            LayerLLaMAImpl<float16_t, LayerNorm>(dt, at, nt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum,
                    maxPositions, maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, intermediateSize, output,
                    outputStride, input, inputStride, ln1Gamma, ln1Beta, queryWeight, keyWeight, valueWeight,
                    attnOutWeight, ln2Gamma, ln2Beta, gateWeight, upWeight, downWeight, queryBias, keyBias, valueBias,
                    attnOutBias);
        } else {
            printf(">> unsupported norm type\n");
        }
    } else {
        printf(">> unsupported data type\n");
    }
}

} // namespace xft
