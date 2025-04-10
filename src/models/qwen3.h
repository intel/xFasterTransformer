// Copyright (c) 2025 Intel Corporation
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

#include "qwen2.h"

template <typename WeiT, typename KVCacheT>
class Qwen3LLM : public Qwen2LLM<WeiT, KVCacheT> {
public:
Qwen3LLM(const std::string &modelPath) : Qwen2LLM<WeiT, KVCacheT>(modelPath, "qwen3") {}
};

REGISTER_MODEL(Qwen3LLM, qwen3)