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
#include "dtype.h"
#include "float16.h"
#include "my_types.h"

namespace xft {

template <typename T>
struct LayerNormWeight {
    const T *gamma = nullptr;
    const T *beta = nullptr;
};

void invokeLayerNorm(float *output, const float *input, const float *gamma, const float *beta, const int rows,
        const int size, int iStride = -1, int oStride = -1, const float epsilon = 1e-5);

void invokeLayerNorm(bfloat16_t *output, const bfloat16_t *input, const bfloat16_t *gamma, const bfloat16_t *beta,
        const int rows, const int size, int iStride = -1, int oStride = -1, const float epsilon = 1e-5);

} // namespace xft