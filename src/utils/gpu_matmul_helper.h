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
#include <immintrin.h>
#include "bfloat16.h"
#include "float16.h"
#include "my_types.h"
#include "normal_float4x2.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "gpu_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "split_util.h"
#include "timeline.h"
#include "transformer_ctx.h"
#include "uint4x2.h"
#include "verbose.h"
#include "xdnn.h"

#include <cstring>
#include <map>
#include <tuple>

class GPUMatMulHelper {
public:
    static dnnl::engine &get_dnnl_engine(dnnl::engine::kind engine_kind, int engine_index) {
        static dnnl::engine engine(engine_kind, engine_index);
        return engine;
    }

    static dnnl::stream &get_dnnl_stream(dnnl::engine::kind engine_kind, int engine_index) {
        static dnnl::stream engine_stream(get_dnnl_engine(engine_kind, engine_index));
        return engine_stream;
    }

    static std::unordered_map<std::string, std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *>> &
    get_dnnl_matmul() {
        static std::unordered_map<std::string, std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *>> matmul;
        return matmul;
    }

    static std::string create_key(bool transA, int M, int N, int K, int matmul_kind) {
        std::string key = std::to_string(transA) + "_" + std::to_string(M) + "_" + std::to_string(N) + "_"
                + std::to_string(K) + "_" + std::to_string(matmul_kind);
        return key;
    }

    static dnnl::memory::format_tag get_amx_f32f16f32_input_layout() {
        // return dnnl::memory::format_tag::AB32a16b;
        return dnnl::memory::format_tag::any;
    }

    static dnnl::memory::format_tag get_amx_f32f16f32_weight_layout() {
        // return dnnl::memory::format_tag::BA4b8a8b2a;
        return dnnl::memory::format_tag::any;
    }

    static dnnl::memory::format_tag get_amx_f32f16f32_output_layout() {
        // return dnnl::memory::format_tag::AB32a16b;
        return dnnl::memory::format_tag::any;
    }

    enum matmul_kinds {
        Basic = 0,
        BiasAdd = 1,
        BiasAdd_Relu = 2,
        Silu = 3,
        Resmul = 4,
        Residential = 5,
        Resext = 6,
    };

    static void onednn_sgemm_f32f32f32_compute(dnnl::engine::kind engine_kind, int engine_index, bool transA, int M, int N, int K, float alpha, const float *A,
            int lda, const float *B, int ldb, float beta, float *C, int ldc) {
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        auto &engine = get_dnnl_engine(engine_kind, engine_index);
        auto &stream = get_dnnl_stream(engine_kind, engine_index);
        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Basic);
        auto it = get_dnnl_matmul().find(key);
        if (it != get_dnnl_matmul().end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B) and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims output_dims = {M, N};

            // Create memory descriptors and memory objects for src, weights, bias, and dst.
            auto input_md = memory::desc(input_dims, dt::f16, get_amx_f32f16f32_input_layout());
            auto weight_md = memory::desc(weight_dims, dt::f16, get_amx_f32f16f32_weight_layout());
            auto output_md = memory::desc(output_dims, dt::f16, get_amx_f32f16f32_output_layout());

            // Create primitive descriptor and primitive.
            matmul_pd = new matmul::primitive_desc(engine, input_md, weight_md, output_md);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            std::string key = create_key(transA, M, N, K, matmul_kinds::Basic);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            get_dnnl_matmul()[key] = value;
        }

        // Repack and convert input data.
        auto packed_input_mem = memory(matmul_pd->src_desc(), engine);
        auto packed_weight_mem = memory(matmul_pd->weights_desc(), engine);
        auto packed_output_mem = memory(matmul_pd->dst_desc(), engine);

        // Reorder input
        auto input_md = memory::desc({M, K}, dt::f32, tag::ab);
        auto input_mem = memory(input_md, engine);
        write_to_dnnl_memory(const_cast<float *>(A), input_mem);
        dnnl::reorder(input_mem, packed_input_mem).execute(stream, input_mem, packed_input_mem);

        // Reorder weight
        auto weight_md = memory::desc({K, N}, dt::f32, tag::ab);
        auto weight_mem = memory(weight_md, engine);
        write_to_dnnl_memory(const_cast<float *>(B), weight_mem);
        dnnl::reorder(weight_mem, packed_weight_mem).execute(stream, weight_mem, packed_weight_mem);

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, packed_input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, packed_weight_mem});
        matmul_args.insert({DNNL_ARG_DST, packed_output_mem});

        // Executions
        matmul_prim->execute(stream, matmul_args);

        // Reorder output
        auto output_md = memory::desc({M, N}, dt::f32, tag::ab);
        auto output_mem = memory(output_md, engine);
        dnnl::reorder(packed_output_mem, output_mem).execute(stream, packed_output_mem, output_mem);
        stream.wait();

        read_from_dnnl_memory(C, output_mem);
    }
};