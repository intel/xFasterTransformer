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

void invokeMLPLLaMA(DataType dt, int numTokens, int hiddenSize, int intermediateSize, void *output, int outputStride,
        const void *input, int inputStride, const void *gateWeight, const void *upWeight, const void *downWeight) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (dt == DataType::bf16) {
        static std::unordered_map<std::string, std::tuple<DecoderContext *, LlamaMLP<bfloat16_t> *>> llama_mlp_hub;

        // create hash key
        std::stringstream weights_addr;
        weights_addr << gateWeight << "_" << upWeight << "_" << downWeight;
        std::string llama_mlp_key
                = std::to_string(hiddenSize) + "_" + std::to_string(intermediateSize) + "_" + weights_addr.str();

        DecoderContext *ctx;
        LlamaMLP<bfloat16_t> *llama_mlp;

        auto it_created = llama_mlp_hub.find(llama_mlp_key);
        if (it_created == llama_mlp_hub.end()) {
            // LlamaMLP<bfloat16_t> &llama_mlp = LlamaMLP<bfloat16_t>::getInstance();
            ctx = new DecoderContext(1, hiddenSize, 1, 1, intermediateSize, "silu", 1e-6, 0, 0, 0, 0, 0, 1);
            std::vector<float *> params {(float *)gateWeight, (float *)nullptr, (float *)upWeight, (float *)nullptr,
                    (float *)nullptr, (float *)nullptr, (float *)downWeight};

            llama_mlp = new LlamaMLP<bfloat16_t>;
            llama_mlp->setWeights(ctx, params, false);

            std::tuple<DecoderContext *, LlamaMLP<bfloat16_t> *> value(ctx, llama_mlp);
            llama_mlp_hub[llama_mlp_key] = value;
            printf("create llama_mlp_key: %s\n", llama_mlp_key.c_str());
        } else {
            ctx = std::get<0>(it_created->second);
            llama_mlp = std::get<1>(it_created->second);
        }

        ctx->resize(1, numTokens, 0);
        llama_mlp->forward(ctx, (float *)const_cast<void *>(input), (float *)output, inputStride, outputStride, false);
    }
}

} // namespace xft