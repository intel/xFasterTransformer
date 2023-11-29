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

template <typename WeiT, typename QKPO_CLS, typename NORM_CLS, bool INPUT_AS_RESID>
class ChatGLM2Attention : public Attention<WeiT, QKPO_CLS, NORM_CLS, INPUT_AS_RESID> {
public:
    ChatGLM2Attention(int layerId, DecoderContext *ctx)
        : Attention<WeiT, QKPO_CLS, NORM_CLS, INPUT_AS_RESID>(layerId, ctx) {}
    virtual ~ChatGLM2Attention() {}

};
