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

#include "moe_qwen3.h"
#include "qwen2.h"

template <typename WeiT, typename KVCacheT>
class Qwen3MOELLM : public Qwen2LLM<WeiT, KVCacheT,
                            Qwen3MOE<WeiT, typename TypeSelector<WeiT>::InType, typename TypeSelector<WeiT>::ImType,
                                    typename TypeSelector<WeiT>::OutType>> {
public:
    Qwen3MOELLM(const std::string &modelPath)
        : Qwen2LLM<WeiT, KVCacheT,
                  Qwen3MOE<WeiT, typename TypeSelector<WeiT>::InType, typename TypeSelector<WeiT>::ImType,
                          typename TypeSelector<WeiT>::OutType>>(modelPath, "qwen3_moe") {}
};

REGISTER_MODEL(Qwen3MOELLM, qwen3_moe)