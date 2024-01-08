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
#pragma once
#include <cmath>

#include "attention.h"

template <typename WeiT, typename QKPO_CLS, typename NORM_CLS>
class ChatGlmAttention : public Attention<WeiT, QKPO_CLS, NORM_CLS, float, float, float, false> {
public:
    ChatGlmAttention(int layerId, DecoderContext *ctx)
        : Attention<WeiT, QKPO_CLS, NORM_CLS, float, float, float, false>(layerId, ctx) {
        residScale = std::sqrt(2 * ctx->layers);
        scalingCoeff = 1.0f / (std::sqrt(ctx->attHeadSize) * (layerId + 1));
    }

protected:
    float getResidentialScale() override { return residScale; }

    // Do NOT needed, in ChatGLM, query_layer is divided by 'query_key_layer_scaling_coeff',
    // but 'attention_scores' is multiplied by the same factor before softmax
    // float getScalingCoeff() override {
    //     return scalingCoeff;
    // }

private:
    // Residential scale
    float residScale;
    // query_key_layer_scaling_coeff
    float scalingCoeff;
};