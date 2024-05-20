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

#include "bfloat16.h"
#include "weight_util.h"

namespace xft {

// RMS normalization: only support the norm along last dimension
class RmsNorm {
public:
    RmsNorm();
    ~RmsNorm();

    void setWeight(const float *w, const float *, int cols);
    void setWeight(const std::string &modelPath, const std::string &, int cols);

    // Input and output are in shape of (rows, normSize)
    void forward(const float *input, float *output, int rows, int iStride = -1, int oStride = -1, float epsilon = 1e-6);

    // Input = float, output = bfloat16_t
    void forward(
            const float *input, bfloat16_t *output, int rows, int iStride = -1, int oStride = -1, float epsilon = 1e-6);

    // Input = bfloat16_t, output = bfloat16_t
    void forward(const bfloat16_t *input, bfloat16_t *output, int rows, int iStride = -1, int oStride = -1,
            float epsilon = 1e-6);

    void forward(const float *input, float16_t *output, int rows, int iStride = -1, int oStride = -1,
            float epsilon = 1e-6);

    void forward(const float16_t *input, float16_t *output, int rows, int iStride = -1, int oStride = -1,
            float epsilon = 1e-6);

private:
    int normSize;

    // the scale weight
    float *weight = nullptr;
};

} // namespace xft