// Copyright (c) 2023-2024 Intel Corporation
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
#include "common_decoder.h"
#include "rms_norm.h"
#include "type_selector.h"

template <typename WeiT, typename QKPO_CLS, typename NORM_CLS = RmsNorm>
class QwenAttention : public Attention<WeiT, QKPO_CLS, NORM_CLS, typename TypeSelector<WeiT>::InType,
                              typename TypeSelector<WeiT>::ImType, typename TypeSelector<WeiT>::OutType, true> {
public:
    QwenAttention(int layerId, DecoderContext *ctx)
        : Attention<WeiT, QKPO_CLS, NORM_CLS, typename TypeSelector<WeiT>::InType, typename TypeSelector<WeiT>::ImType,
                typename TypeSelector<WeiT>::OutType, true>(layerId, ctx) {
        this->qkpo.init_logn(ctx->maxSeqLength, ctx->useLogN, ctx->useNTK);
    }
};

template <typename WeiT, typename QKPO_CLS, typename NORM_CLS>
struct AttnTypeExtractor<QwenAttention<WeiT, QKPO_CLS, NORM_CLS>> {
    using Tin = typename TypeSelector<WeiT>::InType;
    using Tim = typename TypeSelector<WeiT>::ImType;
    using Tout = typename TypeSelector<WeiT>::OutType;
};