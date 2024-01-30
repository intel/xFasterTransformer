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

bool enableCATMLP;
bool enableCBLASMLP;

void setMLPOPTConfig() {
    enableCATMLP = (getenv("ENABLE_CAT_MLP") ? atoi(getenv("ENABLE_CAT_MLP")) : 1);
    enableCBLASMLP = (getenv("ENABLE_CBLAS_MLP") ? atoi(getenv("ENABLE_CBLAS_MLP")) : 0);
}

namespace xft {

void invokeMLPLLaMA(DataType dt, int numTokens, int hiddenSize, int intermediateSize, void *output, int outputStride,
        const void *input, int inputStride, const void *gateWeight, const void *upWeight, const void *downWeight) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (dt == DataType::bf16) {
        static std::unordered_map<std::string, LlamaMLP<bfloat16_t> *> llama_mlp_hub;

        static DecoderContext *ctx;
        if (ctx == nullptr || (ctx != nullptr && (ctx->hiddenSize != hiddenSize || ctx->intermediateSize != intermediateSize))) {
            delete ctx;
            printf(">> create context: %d %d\n", hiddenSize, intermediateSize);
            ctx = new DecoderContext(1, hiddenSize, 1, 1, intermediateSize, "silu", 1e-6, 0, 0, 0, 0, 0, 0, 1);
        }

        // create hash key and value: if hidden and intermediateSize is changed , then memory pointer is also changed.
        std::stringstream weights_addr;
        weights_addr << gateWeight << "_" << upWeight << "_" << downWeight;
        std::string llama_mlp_key = weights_addr.str();
        LlamaMLP<bfloat16_t> *llama_mlp;

        auto it_created = llama_mlp_hub.find(llama_mlp_key);
        if (it_created == llama_mlp_hub.end()) {
            // LlamaMLP<bfloat16_t> &llama_mlp = LlamaMLP<bfloat16_t>::getInstance();
            std::vector<float *> params {(float *)gateWeight, (float *)nullptr, (float *)upWeight, (float *)nullptr,
                    (float *)nullptr, (float *)nullptr, (float *)downWeight};

            llama_mlp = new LlamaMLP<bfloat16_t>;
            llama_mlp->setWeights(ctx, params, false);
            llama_mlp_hub[llama_mlp_key] = llama_mlp;
            printf(">> create llama_mlp_key: %s\n", llama_mlp_key.c_str());
        } else {
            llama_mlp = it_created->second;
        }

        ctx->resize(1, numTokens, 0);
        llama_mlp->forward(ctx, (float *)const_cast<void *>(input), (float *)output, inputStride, outputStride, false);
    }
}

} // namespace xft
