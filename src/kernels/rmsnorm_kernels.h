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

namespace xft {

void rmsNorm(float *output, const float *input, const float *weight, int rows, int cols, int iStride = -1,
        int oStride = -1, float epsilon = 1e-6);

void rmsNorm(bfloat16_t *output, const float *input, const float *weight, int rows, int cols, int iStride = -1,
        int oStride = -1, float epsilon = 1e-6);

void rmsNorm(bfloat16_t *output, const bfloat16_t *input, const float *weight, int rows, int cols, int iStride = -1,
        int oStride = -1, float epsilon = 1e-6);

void rmsNorm(bfloat16_t *output, const bfloat16_t *input, const bfloat16_t *weight, int rows, int cols,
        int iStride = -1, int oStride = -1, float epsilon = 1e-6);

void rmsNorm(float16_t *output, const float *input, const float *weight, int rows, int cols,
        int iStride = -1, int oStride = -1, float epsilon = 1e-6);

void rmsNorm(float16_t *output, const float16_t *input, const float *weight, int rows, int cols,
        int iStride = -1, int oStride = -1, float epsilon = 1e-6);

void rmsNorm(float16_t *output, const float16_t *input, const float16_t *weight, int rows, int cols,
        int iStride = -1, int oStride = -1, float epsilon = 1e-6);

} // namespace xft