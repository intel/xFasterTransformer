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
#include "moe_deepseek.h"
#include "layers_mlp.h"

#include <unordered_map>

namespace xft {
    
template <typename DataT>
void MoEDeepSeekImpl(DataType dt, ActivationType at, int numTokens, int hiddenSize, int intermediateSize, int moeIntermediateSize,
        int numSharedExperts, int numRoutedExperts, void *output, int outputStride, const void *input, int inputStride, const void *gatingWeight,
        const void *gatingBias, const void *gateWeight, const void *upWeight, const void *downWeight) {

    using MLP = DeepSeekMoE<DataT>;
    static std::unordered_map<std::string, MLP *> deepseek_moe_hub;
    static MMHelper *mmHelper;
    static DecoderContext *ctx;

    std::string actType;
    if (at == ActivationType::SILU)
        actType = "silu";
    else
        printf(">> unsupported activation type\n");

    if (ctx == nullptr
            || (ctx != nullptr && (ctx->hiddenSize != hiddenSize || ctx->intermediateSize != intermediateSize))) {
        if (ctx != nullptr) delete ctx;
        printf(">> create context: %d %d\n", hiddenSize, intermediateSize);
        mmHelper = new MMHelper(Env::getInstance().getEngineKind(), Env::getInstance().getEngineIndex());
        ctx = new DecoderContext(1, hiddenSize, 1, 1, 1, intermediateSize, actType, 1e-6, 0, 0, 0, 0, 0, 0, 1, mmHelper);
        // For DeepSeek MoE
        ctx->denseExperts = numSharedExperts;
	    ctx->sparseExperts = numRoutedExperts;
        ctx->moeIntermediateSize = moeIntermediateSize;
        ctx->normTopKProb = true;
        // ctx->firstKDenseReplace = 3;
        ctx->firstKDenseReplace = 0;
        ctx->numExpertsPerTok = 8;
        ctx->topkGroup = 4;
        ctx->nGroup = 8;
        ctx->topkMethod = "noaux_tc";
        ctx->scoringFunc = "sigmoid";
        ctx->routedScalingFac = 2.5;
    }

    // create hash key and value: if hidden and intermediateSize is changed , then memory pointer is also changed.
    std::stringstream weights_addr;
    weights_addr << gateWeight << "_" << upWeight << "_" << downWeight << "_" << dt << "_" << at;

    std::string deepseek_moe_key = weights_addr.str();
    MLP *deepseek_moe;

    auto it_created = deepseek_moe_hub.find(deepseek_moe_key);
    if (it_created == deepseek_moe_hub.end()) {
        // MLP &deepseek_moe = MLP::getInstance();
        deepseek_moe = new MLP(0, ctx);
        xft::DeepSeekFFNParams *ffnParams = new xft::DeepSeekFFNParams(ctx->sparseExperts, ctx->denseExperts, ctx->hiddenSize,
            ctx->intermediateSize, ctx->moeIntermediateSize, xft::ParamType::BF16);
        // load weights to deepseek_moe
        if (numRoutedExperts > 0) {
            printf(">> loadWeights for moe (%d)\n", numRoutedExperts);
            ffnParams->gating.weight = (float *)const_cast<void *>(gatingWeight);
            ffnParams->gating.bias = (float *)const_cast<void *>(gatingBias);

            for (int i = 0; i < numRoutedExperts; i++) {
                ffnParams->routedExperts.emplace_back(ctx->hiddenSize, ctx->moeIntermediateSize, xft::ParamType::BF16, false);
                ffnParams->routedExperts[i].gate.weight = (float *)const_cast<void *>(gateWeight);
                ffnParams->routedExperts[i].up.weight = (float *)const_cast<void *>(upWeight);
                ffnParams->routedExperts[i].down.weight = (float *)const_cast<void *>(downWeight);
            }
            ffnParams->sharedExpert.gate.weight = (float *)const_cast<void *>(gateWeight);
            ffnParams->sharedExpert.up.weight = (float *)const_cast<void *>(upWeight);
            ffnParams->sharedExpert.down.weight = (float *)const_cast<void *>(downWeight);
        } else {
            ffnParams->mlp.gate.weight = (float *)const_cast<void *>(gateWeight);
            ffnParams->mlp.up.weight = (float *)const_cast<void *>(upWeight);
            ffnParams->mlp.down.weight = (float *)const_cast<void *>(downWeight);
        }

        deepseek_moe->template setWeights<float>(ctx, ffnParams);
        deepseek_moe_hub[deepseek_moe_key] = deepseek_moe;
        printf(">> create deepseek_moe_key: %s\n", deepseek_moe_key.c_str());
    } else {
        deepseek_moe = it_created->second;
    }

    ctx->resize(1, numTokens, 0);
    deepseek_moe->forward(ctx, (bfloat16_t *)const_cast<void *>(input), (bfloat16_t *)output, inputStride, outputStride, false);
}

void invokeMoEDeepSeek(DataType dt, ActivationType at, int numTokens, int hiddenSize, int intermediateSize, int moeIntermediateSize,
        int numSharedExperts, int numRoutedExperts, void *output, int outputStride, const void *input, int inputStride, const void *gatingWeight,
        const void *gatingBias, const void *gateWeight, const void *upWeight, const void *downWeight) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (dt == DataType::bf16) {
        MoEDeepSeekImpl<bfloat16_t>(dt, at, numTokens, hiddenSize, intermediateSize, moeIntermediateSize, numSharedExperts, numRoutedExperts,
		output, outputStride, input, inputStride, gatingWeight, gatingBias, gateWeight, upWeight, downWeight);
    } else if (dt == DataType::fp16) {
        MoEDeepSeekImpl<float16_t>(dt, at, numTokens, hiddenSize, intermediateSize, moeIntermediateSize, numSharedExperts, numRoutedExperts,
	        output, outputStride, input, inputStride, gatingWeight, gatingBias, gateWeight, upWeight, downWeight);
    }
}

} // namespace xft
