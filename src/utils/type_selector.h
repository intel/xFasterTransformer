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
};

// Specialization for bfloat16_t
template <>
struct TypeSelector<bfloat16_t> {
    using InType = bfloat16_t;
    using ImType = bfloat16_t;
    using OutType = bfloat16_t;
};

#ifdef XFT_GPU
template <>
struct TypeSelector<float16_t> {
    using InType = float16_t;
    using ImType = float16_t;
    using OutType = float16_t;
};
#endif

#ifdef AVX512_FP16_WEIGHT_ONLY_FP16
template <>
struct TypeSelector<float16_t> {
    using InType = float16_t;
    using ImType = float16_t;
    using OutType = float16_t;
};
#endif

template <typename T>
struct AttnTypeSelector;

template <>
struct AttnTypeSelector<float> {
#if defined(AVX512_BF16_WEIGHT_ONLY_BF16)
    using type = bfloat16_t;
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
    using type = float16_t;
#else
    using type = float;
#endif
};

template <>
struct AttnTypeSelector<bfloat16_t> {
#if defined(AVX512_BF16_WEIGHT_ONLY_BF16)
    using type = bfloat16_t;
#else
    using type = float;
#endif
};

template <>
struct AttnTypeSelector<float16_t> {
#if defined(AVX512_FP16_WEIGHT_ONLY_FP16)
    using type = float16_t;
#else
    using type = float;
#endif
};
