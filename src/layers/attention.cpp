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
#include "attention.h"
#include "kvcache_manager.h"
#include "layers_attention.h"
#include "rms_norm.h"

#include <unordered_map>

namespace xft {

template <typename DataT>
void AttentionLLaMAImpl(DataType dt, int batchSize, int inputSeqLen, int attHeadDim, int attHeadNum, int kvHeadNum,
        int maxPositions, int maxPosEmbed, int pastSeqLen, int currentSeqLen, int step, int hiddenSize, void *output,
        int outputStride, const void *input, int inputStride, const void *queryWeight, const void *keyWeight,
        const void *valueWeight, const void *attnOutWeight, const float *queryBias, const float *keyBias,
        const float *valueBias, const float *attnOutBias) {

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

    using ATTENTION = Attention<DataT, LlamaRotaryEmbedding, RmsNorm>;
    static std::unordered_map<std::string, ATTENTION *> llama_attention_hub;
    static DecoderContext *ctx;
    static KVCacheManager<float16_t> *kvCacheMgr;

    if (ctx == nullptr || (ctx != nullptr && (ctx->hiddenSize != hiddenSize || ctx->attHeadSize != attHeadDim))) {
        if (ctx != nullptr) delete ctx;
        printf(">> create context: %d %d\n", hiddenSize, attHeadDim);
        ctx = new DecoderContext(1, hiddenSize, attHeadDim, attHeadNum, kvHeadNum, 1, "silu", 1e-6, 0, 0, maxPositions,
                maxPosEmbed, -1, 0, 1);
        ctx->mmHelper = new MMHelper(Env::getInstance().getEngineKind(), Env::getInstance().getEngineIndex());
        if (kvCacheMgr != nullptr) delete kvCacheMgr;
        kvCacheMgr = new KVCacheManager<float16_t>(1);
    }

    // create hash key and value: if hidden and intermediateSize is changed , then memory pointer is also changed.
    std::stringstream weights_addr;
    weights_addr << queryWeight << "_" << keyWeight << "_" << valueWeight << "_" << attnOutWeight << "_" << dt << "_"
                 << attHeadDim << "_" << attHeadNum << "_" << kvHeadNum;
    std::string llama_attention_key = weights_addr.str();
    ATTENTION *llama_attention;

    auto it_created = llama_attention_hub.find(llama_attention_key);
    if (it_created == llama_attention_hub.end()) {
        llama_attention = new ATTENTION(0, ctx);
        llama_attention->setWeights(ctx, (const float *)queryWeight, nullptr, nullptr, queryBias,
                (const float *)keyWeight, nullptr, nullptr, keyBias, (const float *)valueWeight, nullptr, nullptr,
                valueBias, (const float *)attnOutWeight, nullptr, nullptr, attnOutBias, false, nullptr, nullptr, false);
        llama_attention_hub[llama_attention_key] = llama_attention;
        printf(">> create llama_attention_key: %s\n", llama_attention_key.c_str());
    } else {
        llama_attention = it_created->second;
    }

    ctx->resize(batchSize, inputSeqLen, pastSeqLen);
    hpj::Matrix<float> actBuffers;
    actBuffers.Resize(batchSize * inputSeqLen, hiddenSize);
    float *attnMask = prepareAttnMask(ctx, step);

    int workers = 1;
    int headsPerSplit = (ctx->kvHeadNum + workers - 1) / workers;
    kvCacheMgr->resize(maxPositions, batchSize, headsPerSplit, attHeadDim);
    KVCacheTensor<float16_t> &presentKey = kvCacheMgr->getKey(0);
    KVCacheTensor<float16_t> &presentValue = kvCacheMgr->getValue(0);

    llama_attention->forward(ctx, (float *)input, actBuffers.Data(), (float *)output, attnMask, presentKey,
            presentValue, inputSeqLen, pastSeqLen, step == 0, false, false, nullptr);
}

void invokeAttentionLLaMA(DataType dt, int batchSize, int inputSeqLen, int attHeadDim, int attHeadNum, int kvHeadNum,
        int maxPositions, int maxPosEmbed, int pastSeqLen, int currentSeqLen, int step, int hiddenSize, void *output,
        int outputStride, const void *input, int inputStride, const void *queryWeight, const void *keyWeight,
        const void *valueWeight, const void *attnOutWeight, const float *queryBias, const float *keyBias,
        const float *valueBias, const float *attnOutBias) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (dt == DataType::bf16) {
        AttentionLLaMAImpl<bfloat16_t>(dt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum, maxPositions,
                maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, output, outputStride, input, inputStride,
                queryWeight, keyWeight, valueWeight, attnOutWeight, queryBias, keyBias, valueBias, attnOutBias);
    } else if (dt == DataType::fp16) {
        AttentionLLaMAImpl<float16_t>(dt, batchSize, inputSeqLen, attHeadDim, attHeadNum, kvHeadNum, maxPositions,
                maxPosEmbed, pastSeqLen, currentSeqLen, step, hiddenSize, output, outputStride, input, inputStride,
                queryWeight, keyWeight, valueWeight, attnOutWeight, queryBias, keyBias, valueBias, attnOutBias);
    }
}

} // namespace xft