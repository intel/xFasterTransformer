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
#include "layer_norm.h"
#include "layernorm_kernels.h"
#include "timeline.h"

namespace xft {

// Layer normalization: only support the norm along last dimension
LayerNorm::LayerNorm() {
    gamma = nullptr;
    beta = nullptr;
    normSize = 0;
}

LayerNorm::LayerNorm(DecoderContext *ctx) {
    device = ctx->device;
    gamma = nullptr;
    beta = nullptr;
    normSize = 0;
}

LayerNorm::~LayerNorm() {
    if (gamma) { xft::dealloc(gamma, device); }
    if (beta) { xft::dealloc(beta, device); }
}

void LayerNorm::setWeight(const float *gamma, const float *beta, int cols) {
    this->normSize = cols;
    this->gamma = (float *)xft::alloc(cols * sizeof(float), device);
    this->beta = (float *)xft::alloc(cols * sizeof(float), device);
    xft::memcopy(this->gamma, gamma, cols * sizeof(float), device);
    xft::memcopy(this->beta, beta, cols * sizeof(float), device);
}

void LayerNorm::setWeight(const std::string &gammaPath, const std::string &betaPath, int cols) {
    this->normSize = cols;
    loadWeight(gammaPath, this->gamma, cols);
    if (betaPath != "") loadWeight(betaPath, this->beta, cols);
}

// input and output are in shape of (rows, normSize)
// TODO: column-wise parallel
#ifdef XFT_GPU
void LayerNorm::forward(const float *input, float *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("LayerNorm.forward");
    const float *pgamma = gamma;
    const float *pbeta = beta;
    // TODO: Add LayerNorm Impl
    printf("%s:%d: Could not forward in LayerNorm with undefined data type.\n", __FILE__, __LINE__);
    exit(-1);
}
#else
void LayerNorm::forward(const float *input, float *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("LayerNorm.forward");
    const float *pgamma = gamma;
    const float *pbeta = beta;
    invokeLayerNorm(output, input, pgamma, pbeta, rows, normSize, iStride, oStride);
}
#endif
} // namespace xft