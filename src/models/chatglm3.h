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
template <typename WeiT, typename KVCacheT>
class ChatGLM3 : public ChatGLM2<WeiT, KVCacheT> {
public:
    ChatGLM3(const std::string &modelPath) : ChatGLM2<WeiT, KVCacheT>(modelPath, "chatglm3") {}
};

REGISTER_MODEL(ChatGLM3, chatglm3)
