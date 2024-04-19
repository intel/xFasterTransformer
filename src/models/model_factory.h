
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
            printf("Unsupported model type or data type.\n");
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


// When implementing a new model, you need to add the corresponding model header file in `models.cpp`.
// Otherwise, the model registration mechanism won't be able to find the corresponding model.
#define REGISTER_DECODER(CLASS, NAME, T)          \
    template class CLASS<T>;                      \
    static DecoderRegister decoder_##CLASS##_##T( \
            #NAME "-" #T, [](const std::string &modelPath) { return new CLASS<T>(modelPath); });

#define REGISTER_HYBRID_MODEL(CLASS, NAME, T1, T2)                                  \
    template class HybridModel<CLASS, T1, T2>;                                      \
    static DecoderRegister hybridModel_##CLASS##_##T1##_##T2(#NAME "-" #T1 "-" #T2, \
            [](const std::string &modelPath) { return new HybridModel<CLASS, T1, T2>(modelPath); });
