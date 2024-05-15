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
#include <tuple>
#include "bfloat16.h"
#include "dtype.h"
#include "float16.h"
#include "my_types.h"
#include "normal_float4x2.h"
#include "uint4x2.h"

namespace xft {
inline std::string getTypeIdName(xft::DataType dtype) {
    switch (dtype) {
        case xft::DataType::fp32: return "float";
        case xft::DataType::bf16: return "bfloat16_t";
        case xft::DataType::fp16: return "float16_t";
        case xft::DataType::int8: return "int8_t";
        case xft::DataType::w8a8: return "w8a8_t";
        case xft::DataType::int4: return "uint4x2_t";
        case xft::DataType::nf4: return "nf4x2_t";
        case xft::DataType::bf16_fp16: return "bfloat16_t-float16_t";
        case xft::DataType::bf16_int8: return "bfloat16_t-int8_t";
        case xft::DataType::bf16_w8a8: return "bfloat16_t-w8a8_t";
        case xft::DataType::bf16_int4: return "bfloat16_t-uint4x2_t";
        case xft::DataType::bf16_nf4: return "bfloat16_t-nf4x2_t";
        case xft::DataType::w8a8_int8: return "w8a8_t-int8_t";
        case xft::DataType::w8a8_int4: return "w8a8_t-uint4x2_t";
        case xft::DataType::w8a8_nf4: return "w8a8_t-nf4x2_t";
        case xft::DataType::unknown: return "unknown";
    }
    return std::string("unknown");
}

// Get DataType according to c++ types
template <typename T>
inline DataType getDataType() {
    static_assert(sizeof(T) == 0, "Unsupported type");
    return DataType::unknown;
}

template <>
inline DataType getDataType<float>() {
    return DataType::fp32;
}

template <>
inline DataType getDataType<bfloat16_t>() {
    return DataType::bf16;
}

template <>
inline DataType getDataType<float16_t>() {
    return DataType::fp16;
}

template <>
inline DataType getDataType<int8_t>() {
    return int8;
}
} // namespace xft
