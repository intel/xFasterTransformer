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
#include "decoder_layer.h"

template <typename WeiT, typename QKPO_CLS, typename NORM_CLS, typename InT = float, typename ImT = float,
        typename OutT = float, bool INPUT_AS_RESID = true>
class RopeScalingAttention : public Attention<WeiT, QKPO_CLS, NORM_CLS, InT, ImT, OutT, INPUT_AS_RESID> {
public:
    RopeScalingAttention(int layerId, DecoderContext *ctx)
        : Attention<WeiT, QKPO_CLS, NORM_CLS, InT, ImT, OutT, INPUT_AS_RESID>(layerId, ctx) {
        this->qkpo = QKPO_CLS(ctx->attHeadSize, ctx->maxPosEmbed, ctx->ropeParamsPtr);
    }

    virtual ~RopeScalingAttention() {}
};

template <typename WeiT, typename QKPO_CLS, typename NORM_CLS, typename InT, typename ImT, typename OutT>
struct AttnTypeExtractor<RopeScalingAttention<WeiT, QKPO_CLS, NORM_CLS, InT, ImT, OutT, true>> {
    using Tin = InT;
    using Tim = ImT;
    using Tout = OutT;
};
