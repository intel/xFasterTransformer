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

#include "layernorm_kernels.h"
#include "layers_norm.h"
#include "timeline.h"

namespace xft {

// Layer normalization: only support the norm along last dimension
LayerNorm::LayerNorm() {
    weights = nullptr;
    normSize = 0;
}

LayerNorm::~LayerNorm() {
    if (weights) { free(weights); }
}

void LayerNorm::setWeight(const float *gamma, const float *beta, int cols) {
    this->normSize = cols;
    this->weights = (float *)aligned_alloc(64, 2 * cols * sizeof(float));
    memcpy(weights, gamma, cols * sizeof(float));
    memcpy(weights + cols, beta, cols * sizeof(float));
}

// input and output are in shape of (rows, normSize)
// TODO: column-wise parallel
void LayerNorm::forward(const float *input, float *output, int rows, int iStride, int oStride) {
    TimeLine t("LayerNorm.forward");
    const float *pgamma = weights;
    const float *pbeta = weights + normSize;
    invokeLayerNorm(output, input, pgamma, pbeta, rows, normSize, iStride, oStride);
}

} // namespace xft