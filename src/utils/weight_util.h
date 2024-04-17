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
#include <filesystem>
#include <fstream>
#include <omp.h>
#include <string>

#include "INIReader.h"
#include "allocator.h"
#include "bfloat16.h"
#include "compile_util.h"
#include "copy_util.h"
#include "dtype.h"
#include "float16.h"
#include "my_types.h"
#include "normal_float4x2.h"
#include "uint4x2.h"
#include "environment.h"

namespace xft {

inline DataType getWeightType(const std::string &ini_file, std::string section_name = "") {
    DataType w_type;
    INIReader reader = INIReader(ini_file);
    if (reader.ParseError() < 0) {
        printf("Can't load %s. Use FP32 as default", ini_file.c_str());
        w_type = DataType::fp32;
    } else {
        if (section_name == "") {
            if (!reader.Sections().empty()) {
                section_name = *(reader.Sections().begin());
            } else {
                printf("Can't load %s. Use FP32 as default", ini_file.c_str());
                return DataType::fp32;
            }
        }
        std::string weight_data_type_str = std::string(reader.Get(section_name, "weight_data_type"));
        if (weight_data_type_str.find("fp32") != std::string::npos) {
            w_type = DataType::fp32;
        } else if (weight_data_type_str.find("fp16") != std::string::npos) {
            w_type = DataType::fp16;
        } else if (weight_data_type_str.find("bf16") != std::string::npos) {
            w_type = DataType::bf16;
        } else {
            printf("Invalid type %s. Use FP32 as default", weight_data_type_str.c_str());
            w_type = DataType::fp32;
        }
    }
    return w_type;
}

// Read weights from file
template <typename T>
int readFile(const std::string &path, T *values, int size) {
    int count = 0;
    int nthreads = std::min(omp_get_max_threads(), 16);
    int chunk_size = (size + nthreads - 1) / nthreads;
    Env &env = Env::getInstance();
    if (env.getFakeModelEnabled()) {
        if (env.getFakeLoadInfoEnabled()) {
            printf("Loading fake model file %s.\n", path.c_str());
        }
        memset(values, 0, sizeof(T) * size);
        return size;
    }

    {
        std::ifstream file(path, std::ios::binary);
        if (!file) return 0;
        if (file.is_open()) file.close();
    }

#pragma omp parallel num_threads(nthreads) reduction(+ : count)
    {
        std::ifstream file(path, std::ios::binary);
        int tid = omp_get_thread_num();
        int start_idx = tid * chunk_size;
        int end_idx = std::min(start_idx + chunk_size, size);
        if (file.is_open()) {
            file.seekg(start_idx * sizeof(T), std::ios::beg);
            file.read(reinterpret_cast<char *>(values + start_idx), (end_idx - start_idx) * sizeof(T));
            count += end_idx - start_idx;
            file.close();
        }
    }
    return count;
}

// Function to load weights with optional dynamic type conversion
// T: The computation type
// WT: The model file storage type
template <typename T, typename WT>
int loadWeightWithConvert(T *ptr, int size, const std::string &filename, bool required = true) {
    int file_size = 0;
    if constexpr (std::is_same_v<T, WT> == true) {
        // If T and WT are the same, directly read the file
        file_size = readFile(filename, ptr, size);
        if (required) REQUIRES(file_size == size, "read %s failed!", filename.c_str());
    } else {
        // If T and WT are different types, perform dynamic type conversion
        WT *w_ptr = nullptr;
        w_ptr = (WT *)xft::alloc(sizeof(WT) * size);
        file_size = readFile(filename, w_ptr, size);
        if (required) REQUIRES(file_size == size, "read %s failed!", filename.c_str());

        if constexpr ((std::is_same_v<T, float16_t> && std::is_same_v<WT, float>)
                || (std::is_same_v<T, bfloat16_t> && std::is_same_v<WT, float>)
                || (std::is_same_v<T, float> && std::is_same_v<WT, float16_t>)
                || (std::is_same_v<T, bfloat16_t> && std::is_same_v<WT, float16_t>)) {
            xft::copy_MT(ptr, w_ptr, size);
        } else if constexpr (std::is_same_v<T, int8_t> && std::is_same_v<WT, float>) {
            printf("Not support float to int8_t\n");
            exit(-1);
        } else if constexpr (std::is_same_v<T, uint4x2_t> && std::is_same_v<WT, float>) {
            printf("Not support float to uint4x2_t\n");
            exit(-1);
        } else if constexpr (std::is_same_v<T, nf4x2_t> && std::is_same_v<WT, float>) {
            printf("Not support float to nf4x2_t\n");
            exit(-1);
        } else if constexpr (std::is_same_v<T, int8_t> && std::is_same_v<WT, float16_t>) {
            printf("Not support float16_t to int8_t\n");
            exit(-1);
        } else if constexpr (std::is_same_v<T, uint4x2_t> && std::is_same_v<WT, float16_t>) {
            printf("Not support float16_t to uint4x2_t\n");
            exit(-1);
        } else if constexpr (std::is_same_v<T, nf4x2_t> && std::is_same_v<WT, float16_t>) {
            printf("Not support float16_t to nf4x2_t\n");
            exit(-1);
        } else {
            printf("Not support data loading with unknown type!\n");
            exit(-1);
        }

        if (w_ptr) {
            free(w_ptr);
            w_ptr = nullptr;
        }
    }
    return file_size;
}

template <typename T>
int loadWeight(std::string filename, T *&ptr, int size, DataType w_type = DataType::unknown, bool required = true) {

    // By default, read the config.ini configuration file
    // in the same directory as the model file to determine the data type of the file.
    if (w_type == DataType::unknown) {
        std::size_t pos = filename.find_last_of("/\\");
        std::string dirPath = filename.substr(0, pos);
        std::string configFilePath = dirPath + "/config.ini";
        w_type = getWeightType(configFilePath);
    }
    //1 uint4x2 stores 2 uint4 value, so load size is halfed.
    if constexpr (std::is_same_v<T, uint4x2_t>) { size = size / 2; }
    if (!ptr) { ptr = (T *)xft::alloc(size * sizeof(T)); }
    int file_size = 0;
    switch (w_type) {
        case DataType::fp32: file_size = loadWeightWithConvert<T, float>(ptr, size, filename, required); break;
        case DataType::fp16: file_size = loadWeightWithConvert<T, float16_t>(ptr, size, filename, required); break;
        case DataType::bf16: file_size = loadWeightWithConvert<T, bfloat16_t>(ptr, size, filename, required); break;
        case DataType::int8: file_size = loadWeightWithConvert<T, int8_t>(ptr, size, filename, required); break;
        case DataType::int4: file_size = loadWeightWithConvert<T, uint4x2_t>(ptr, size, filename, required); break;
        default: printf("Not support loading %s with DataType=%d", filename.c_str(), w_type);
    }
    return file_size;
}

template int loadWeightWithConvert<float, float>(float *, int, const std::string &, bool);
template int loadWeightWithConvert<float16_t, float>(float16_t *, int, const std::string &, bool);
template int loadWeightWithConvert<bfloat16_t, float>(bfloat16_t *, int, const std::string &, bool);
template int loadWeightWithConvert<int8_t, float>(int8_t *, int, const std::string &, bool);
template int loadWeightWithConvert<uint4x2_t, float>(uint4x2_t *, int, const std::string &, bool);
template int loadWeightWithConvert<nf4x2_t, float>(nf4x2_t *, int, const std::string &, bool);

template int loadWeightWithConvert<float, float16_t>(float *, int, const std::string &, bool);
template int loadWeightWithConvert<float16_t, float16_t>(float16_t *, int, const std::string &, bool);
template int loadWeightWithConvert<bfloat16_t, float16_t>(bfloat16_t *, int, const std::string &, bool);
template int loadWeightWithConvert<int8_t, float16_t>(int8_t *, int, const std::string &, bool);
template int loadWeightWithConvert<uint4x2_t, float16_t>(uint4x2_t *, int, const std::string &, bool);
template int loadWeightWithConvert<nf4x2_t, float16_t>(nf4x2_t *, int, const std::string &, bool);

template int loadWeightWithConvert<int8_t, int8_t>(int8_t *, int, const std::string &, bool);
template int loadWeightWithConvert<uint4x2_t, uint4x2_t>(uint4x2_t *, int, const std::string &, bool);
} // namespace xft
