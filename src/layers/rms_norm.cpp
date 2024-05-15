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
#include "transformer_ctx.h"

#ifdef GPU
#include "gpudnn/gpu_layernorm_kernels.h"
#endif

namespace xft {

RmsNorm::RmsNorm() {
    weight = nullptr;
    normSize = 0;
}

RmsNorm::RmsNorm(DecoderContext *ctx) {
    device = ctx->device;
    weight = nullptr;
    normSize = 0;
}

RmsNorm::~RmsNorm() {
    if (weight) { xft::dealloc(weight); }
}

void RmsNorm::setWeight(const float *w, const float *, int cols) {
    this->normSize = cols;
#ifdef GPU
    sycl::queue *gpu_queue = static_cast<sycl::queue *>(device);
    this->weight = sycl::malloc_device<float>(cols, *gpu_queue);
    gpu_queue->memcpy(this->weight, w, cols * sizeof(float)).wait();
#else
    this->weight = (float *)xft::alloc(cols * sizeof(float));
    memcpy(weight, w, cols * sizeof(float));
#endif
}

void RmsNorm::setWeight(const std::string &modelPath, const std::string &, int cols) {
    this->normSize = cols;
    loadWeight(modelPath, weight, cols);
}

#ifdef GPU
void RmsNorm::forward(const float *input, float *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("RmsNorm.forward");
    sycl::queue *gpu_queue = static_cast<sycl::queue *>(device);
    fastertransformer::invokeGeneralT5LayerNorm(
            output, input, weight, (const float *)nullptr, epsilon, rows, iStride, gpu_queue);
}

void RmsNorm::forward(const float *input, bfloat16_t *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("RmsNorm.forward");
}

void RmsNorm::forward(const bfloat16_t *input, bfloat16_t *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("RmsNorm.forward");
}
#else
// input and output are in shape of (rows, normSize)
void RmsNorm::forward(const float *input, float *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("RmsNorm.forward");
    rmsNorm(output, input, weight, rows, normSize, iStride, oStride, epsilon);
}

void RmsNorm::forward(const float *input, bfloat16_t *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("RmsNorm.forward");
    rmsNorm(output, input, weight, rows, normSize, iStride, oStride, epsilon);
}

void RmsNorm::forward(const bfloat16_t *input, bfloat16_t *output, int rows, int iStride, int oStride, float epsilon) {
    TimeLine t("RmsNorm.forward");
    rmsNorm(output, input, weight, rows, normSize, iStride, oStride, epsilon);
}
#endif
} // namespace xft