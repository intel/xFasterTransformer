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
#include <immintrin.h>

#include <cstdlib>
#include <cstring>

#include "allocator.h"
#include "rms_norm.h"
#include "rmsnorm_kernels.h"
#include "timeline.h"

namespace xft {

template <typename T>
RmsNormImp<T>::RmsNormImp() {
    weight = nullptr;
    normSize = 0;
}

template <typename T>
RmsNormImp<T>::RmsNormImp(DecoderContext *ctx) {
    device = ctx->device;
    weight = nullptr;
    normSize = 0;
}

template <typename T>
RmsNormImp<T>::~RmsNormImp() {
    if (weight) { free(weight); }
}

template <typename T>
void RmsNormImp<T>::setWeight(const float *w, const float *, int cols) {
    T weightBuf[cols];
    if constexpr (std::is_same_v<T, float>) {
        memcpy(weightBuf, w, cols * sizeof(float));
    } else if constexpr (std::is_same_v<T, float16_t>) {
        float16_t::cvt_float_to_float16(w, weightBuf, cols);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        bfloat16_t::cvt_float_to_bfloat16(w, weightBuf, cols);
    } else {
        printf("%s:%d: Could not setWeight in RmsNorm with undefined data type.\n", __FILE__, __LINE__);
        exit(-1);
    }

    this->normSize = cols;
    this->weight = (T *)xft::alloc(cols * sizeof(T));
    memcpy(weight, weightBuf, cols * sizeof(T));
}

template <typename T>
void RmsNormImp<T>::setWeight(const std::string &modelPath, const std::string &, int cols) {
    this->normSize = cols;
    loadWeight(modelPath, weight, cols);
}

// input and output are in shape of (rows, normSize)
template <typename T>
void RmsNormImp<T>::forward(const float *input, float *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("RmsNorm.forward");
    if constexpr (std::is_same_v<T, float>) {
        rmsNorm(output, input, weight, rows, normSize, iStride, oStride, epsilon);
    } else {
        printf("%s:%d: Could not forward in RmsNorm with undefined data type.\n", __FILE__, __LINE__);
        exit(-1);
    }
}

template <typename T>
void RmsNormImp<T>::forward(const float *input, bfloat16_t *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("RmsNorm.forward");
    if constexpr (std::is_same_v<T, float>) {
        rmsNorm(output, input, weight, rows, normSize, iStride, oStride, epsilon);
    } else {
        printf("%s:%d: Could not forward in RmsNorm with undefined data type.\n", __FILE__, __LINE__);
        exit(-1);
    }
}

template <typename T>
void RmsNormImp<T>::forward(
        const bfloat16_t *input, bfloat16_t *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("RmsNorm.forward");
    if constexpr (std::is_same_v<T, float>) {
        rmsNorm(output, input, weight, rows, normSize, iStride, oStride, epsilon);
    } else {
        printf("%s:%d: Could not forward in RmsNorm with undefined data type.\n", __FILE__, __LINE__);
        exit(-1);
    }
}

template <typename T>
void RmsNormImp<T>::forward(const float *input, float16_t *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("RmsNorm.forward");
    if constexpr (std::is_same_v<T, float>) {
        rmsNorm(output, input, weight, rows, normSize, iStride, oStride, epsilon);
    } else {
        printf("%s:%d: Could not forward in RmsNorm with undefined data type.\n", __FILE__, __LINE__);
        exit(-1);
    }
}

template <typename T>
void RmsNormImp<T>::forward(
        const float16_t *input, float16_t *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("RmsNorm.forward");
    if constexpr (std::is_same_v<T, float>) {
        rmsNorm(output, input, weight, rows, normSize, iStride, oStride, epsilon);
    } else {
        printf("%s:%d: Could not forward in RmsNorm with undefined data type.\n", __FILE__, __LINE__);
        exit(-1);
    }
}

template class RmsNormImp<float>;
template class RmsNormImp<float16_t>;
template class RmsNormImp<bfloat16_t>;

} // namespace xft