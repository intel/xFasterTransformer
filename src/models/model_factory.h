
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
#include <functional>
#include <map>
#include <string>
#include "abstract_decoder.h"
#include "hybrid_model.h"

class DecoderFactory {
public:
    using CreateFunc = std::function<AbstractDecoder *(const std::string &)>;

    static void Register(const std::string &key, CreateFunc createFunc) { GetRegistry()[key] = createFunc; }

    static AbstractDecoder *Create(const std::string &key, const std::string &modelPath) {
        auto it = GetRegistry().find(key);
        if (it != GetRegistry().end()) {
            return it->second(modelPath);
        } else {
            printf("Unsupported model type, data type or KV cache data type.\n");
            exit(-1);
        }
    }

    // private:
    static std::map<std::string, CreateFunc> &GetRegistry() {
        static std::map<std::string, CreateFunc> registry;
        return registry;
    }
};

class DecoderRegister {
public:
    DecoderRegister(const std::string &key, DecoderFactory::CreateFunc createFunc) {
        DecoderFactory::Register(key, createFunc);
    }
};

#define IMPLEMENT_DECODER(CLASS, NAME, T, CacheT) \
    template class CLASS<T, CacheT>;

#define IMPLEMENT_HYBRID_MODEL(CLASS, NAME, T1, T2, CacheT) \
    template class HybridModel<CLASS, T1, T2, CacheT>;

#define REGISTER_DECODER(CLASS, NAME, T, CacheT)             \
    static DecoderRegister decoder_##CLASS##_##T##_##CacheT( \
            #NAME "-" #T "-" #CacheT, [](const std::string &modelPath) { return new CLASS<T, CacheT>(modelPath); });

#define REGISTER_HYBRID_MODEL(CLASS, NAME, T1, T2, CacheT)                                                  \
    static DecoderRegister hybridModel_##CLASS##_##T1##_##T2##_##CacheT(#NAME "-" #T1 "-" #T2 "-" #CacheT,  \
            [](const std::string &modelPath) { return new HybridModel<CLASS, T1, T2, CacheT>(modelPath); });

#define DECODER_ALL_CACHETYPE(KIND, CLASS, NAME, T) \
    KIND##_DECODER(CLASS, NAME, T, float16_t)       \
    KIND##_DECODER(CLASS, NAME, T, int8_t)

#define HYBRID_MODEL_ALL_CACHETYPE(KIND, CLASS, NAME, T1, T2) \
    KIND##_HYBRID_MODEL(CLASS, NAME, T1, T2, float16_t)       \
    KIND##_HYBRID_MODEL(CLASS, NAME, T1, T2, int8_t)

// Kernels in BF16 PATH not support FP32 KVCache
#define DECODER_ALL_TYPE(KIND, CLASS, NAME)             \
    DECODER_ALL_CACHETYPE(KIND, CLASS, NAME, bfloat16_t)\
    DECODER_ALL_CACHETYPE(KIND, CLASS, NAME, float16_t) \
    DECODER_ALL_CACHETYPE(KIND, CLASS, NAME, int8_t)    \
    DECODER_ALL_CACHETYPE(KIND, CLASS, NAME, w8a8_t)    \
    DECODER_ALL_CACHETYPE(KIND, CLASS, NAME, uint4x2_t) \
    DECODER_ALL_CACHETYPE(KIND, CLASS, NAME, nf4x2_t)

#define HYBRID_MODEL_ALL_TYPE(KIND, CLASS, NAME)                         \
    HYBRID_MODEL_ALL_CACHETYPE(KIND, CLASS, NAME, bfloat16_t, float16_t) \
    HYBRID_MODEL_ALL_CACHETYPE(KIND, CLASS, NAME, bfloat16_t, int8_t)    \
    HYBRID_MODEL_ALL_CACHETYPE(KIND, CLASS, NAME, bfloat16_t, w8a8_t)    \
    HYBRID_MODEL_ALL_CACHETYPE(KIND, CLASS, NAME, bfloat16_t, uint4x2_t) \
    HYBRID_MODEL_ALL_CACHETYPE(KIND, CLASS, NAME, bfloat16_t, nf4x2_t)   \
    HYBRID_MODEL_ALL_CACHETYPE(KIND, CLASS, NAME, w8a8_t, int8_t)        \
    HYBRID_MODEL_ALL_CACHETYPE(KIND, CLASS, NAME, w8a8_t, uint4x2_t)     \
    HYBRID_MODEL_ALL_CACHETYPE(KIND, CLASS, NAME, w8a8_t, nf4x2_t)

// Please implement the model in your header file;
// implementing it in the .cpp file will not take effect.
#define MODEL(KIND, CLASS, NAME)        \
    DECODER_ALL_TYPE(KIND, CLASS, NAME) \
    HYBRID_MODEL_ALL_TYPE(KIND, CLASS, NAME)

#define IMPLEMENT_MODEL(CLASS, NAME) \
    MODEL(IMPLEMENT, CLASS, NAME)

#define REGISTER_MODEL(CLASS, NAME) \
    MODEL(REGISTER, CLASS, NAME)
