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

#include "layers_norm.h"
#include "rmsnorm_kernels.h"
#include "timeline.h"

namespace xft {

RmsNorm::RmsNorm() {
    weight = nullptr;
    normSize = 0;
}

RmsNorm::~RmsNorm() {
    if (weight) { free(weight); }
}

void RmsNorm::setWeight(const float *w, const float *, int size) {
    this->normSize = size;
    this->weight = (float *)aligned_alloc(64, size * sizeof(float));
    memcpy(weight, w, size * sizeof(float));
}

// input and output are in shape of (rows, normSize)
void RmsNorm::forward(const float *input, float *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("RmsNorm.forward");
    invokeRmsNorm(output, input, weight, rows, normSize, iStride, oStride, epsilon);
}

} // namespace xft