// Copyright (c) 2024 Intel Corporation
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
#include "bfloat16.h"

// Selected data types according to weight data type
template <typename WeiT>
struct TypeSelector {
    using InType = float;
    using ImType = float; // intermediate data type, default in float
    using OutType = float;
    using KVCacheType = float16_t;
};

// Specialization for bfloat16_t
template <>
struct TypeSelector<bfloat16_t> {
    //todo(marvin):  Switch temporarily back to the original version
    //to bypass the issues with Multi Query Attention and Group Query Attention.
    // using InType = bfloat16_t;
    // using ImType = bfloat16_t;
    // using OutType = bfloat16_t;
    using InType = float;
    using ImType = float; // intermediate data type, default in float
    using OutType = float;
    using KVCacheType = float16_t;
};