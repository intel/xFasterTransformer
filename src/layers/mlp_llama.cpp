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
#include "layers_mlp.h"

#include "mlp_llama.h"

namespace xft {

void invokeMLPLLaMA(DataType dt, int batchSize, int inputSeqLen, int hiddenSize, int intermediateSize, void *output,
        int outputStride, const void *input, int inputStride, const void *gateWeight, const void *upWeight,
        const void *downWeight) {
    if (dt == DataType::bf16) {
        LlamaMLP<bfloat16_t> &llama_mlp = LlamaMLP<bfloat16_t>::getInstance();
        DecoderContext ctx(1, hiddenSize, 1, 1, intermediateSize, "silu", 1e-6, 0, 0, 0, 0, 0, 1);
        ctx.resize(batchSize, inputSeqLen, 0);
        std::vector<float *> params {(float *)gateWeight, (float *)nullptr, (float *)upWeight, (float *)nullptr,
                (float *)nullptr, (float *)nullptr, (float *)downWeight};
        llama_mlp.setWeights(&ctx, params);
        llama_mlp.forward(&ctx, (float *)const_cast<void *>(input), (float *)output, inputStride, outputStride, false);
    }
}

} // namespace xft