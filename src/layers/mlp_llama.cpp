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
#include "mlp_llama.h"
#include "layers_mlp.h"

#include <unordered_map>

namespace xft {

template <typename DataT>
void MLPLLaMAImpl(DataType dt, ActivationType at, int numTokens, int hiddenSize, int intermediateSize, void *output, int outputStride,
        const void *input, int inputStride, const void *gateWeight, const void *upWeight, const void *downWeight) {

    using MLP = LlamaMLP<DataT>;
    static std::unordered_map<std::string, MLP *> llama_mlp_hub;
    static DecoderContext *ctx;

    if (ctx == nullptr
            || (ctx != nullptr && (ctx->hiddenSize != hiddenSize || ctx->intermediateSize != intermediateSize))) {
        if (ctx != nullptr) delete ctx;
        printf(">> create context: %d %d\n", hiddenSize, intermediateSize);
        ctx = new DecoderContext(1, hiddenSize, 1, 1, 1, intermediateSize, "silu", 1e-6, 0, 0, 0, 0, 0, 0, 1);
        ctx->mmHelper = new MMHelper(Env::getInstance().getEngineKind(), Env::getInstance().getEngineIndex());
    }

    // create hash key and value: if hidden and intermediateSize is changed , then memory pointer is also changed.
    std::stringstream weights_addr;
    weights_addr << gateWeight << "_" << upWeight << "_" << downWeight;
    std::string llama_mlp_key = weights_addr.str();
    MLP *llama_mlp;

    auto it_created = llama_mlp_hub.find(llama_mlp_key);
    if (it_created == llama_mlp_hub.end()) {
        // MLP &llama_mlp = MLP::getInstance();
        llama_mlp = new MLP();
        llama_mlp->setWeights(ctx, (float *)gateWeight, nullptr, nullptr, nullptr, (float *)upWeight, nullptr, nullptr,
                nullptr, nullptr, nullptr, (float *)downWeight, nullptr, nullptr, false);
        llama_mlp_hub[llama_mlp_key] = llama_mlp;
        printf(">> create llama_mlp_key: %s\n", llama_mlp_key.c_str());
    } else {
        llama_mlp = it_created->second;
    }

    ctx->resize(1, numTokens, 0);
    llama_mlp->forward(ctx, (float *)const_cast<void *>(input), (float *)output, inputStride, outputStride, false);
}

void invokeMLPLLaMA(DataType dt, ActivationType at, int numTokens, int hiddenSize, int intermediateSize, void *output,
        int outputStride, const void *input, int inputStride, const void *gateWeight, const void *upWeight,
        const void *downWeight) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (dt == DataType::bf16) {
        MLPLLaMAImpl<bfloat16_t>(dt, at, numTokens, hiddenSize, intermediateSize, output, outputStride, input,
                inputStride, gateWeight, upWeight, downWeight);
    } else if (dt == DataType::fp16) {
        MLPLLaMAImpl<float16_t>(dt, at, numTokens, hiddenSize, intermediateSize, output, outputStride, input,
                inputStride, gateWeight, upWeight, downWeight);
    }
}

} // namespace xft
