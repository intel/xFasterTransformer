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

#include "chatglm2.h"

// ChatGLM3 and ChatGLM2 have the same structure, so ChatGLM3 utilizes the implementation of ChatGLM2.
template <typename WeiT>
class ChatGLM3 : public ChatGLM2<WeiT> {
public:
    ChatGLM3(const std::string &modelPath) : ChatGLM2<WeiT>(modelPath, "chatglm3") {}
};

REGISTER_DECODER(ChatGLM3, chatglm3, float)
REGISTER_DECODER(ChatGLM3, chatglm3, float16_t)
REGISTER_DECODER(ChatGLM3, chatglm3, bfloat16_t)
REGISTER_DECODER(ChatGLM3, chatglm3, int8_t)
REGISTER_DECODER(ChatGLM3, chatglm3, w8a8_t)
REGISTER_DECODER(ChatGLM3, chatglm3, uint4x2_t)
REGISTER_DECODER(ChatGLM3, chatglm3, nf4x2_t)
REGISTER_HYBRID_MODEL(ChatGLM3, chatglm3, bfloat16_t, float16_t)
REGISTER_HYBRID_MODEL(ChatGLM3, chatglm3, bfloat16_t, int8_t)
REGISTER_HYBRID_MODEL(ChatGLM3, chatglm3, bfloat16_t, w8a8_t)
REGISTER_HYBRID_MODEL(ChatGLM3, chatglm3, bfloat16_t, uint4x2_t)
REGISTER_HYBRID_MODEL(ChatGLM3, chatglm3, bfloat16_t, nf4x2_t)
REGISTER_HYBRID_MODEL(ChatGLM3, chatglm3, w8a8_t, int8_t)
REGISTER_HYBRID_MODEL(ChatGLM3, chatglm3, w8a8_t, uint4x2_t)
REGISTER_HYBRID_MODEL(ChatGLM3, chatglm3, w8a8_t, nf4x2_t)

