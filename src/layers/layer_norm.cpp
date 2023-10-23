#pragma once
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

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

void LayerNorm::setWeight(const float *gamma, const float *beta, int size) {
    this->normSize = size;
    this->weights = (float *)aligned_alloc(64, 2 * size * sizeof(float));
    memcpy(weights, gamma, size * sizeof(float));
    memcpy(weights + size, beta, size * sizeof(float));
}

// input and output are in shape of (rows, normSize)
// TODO: column-wise parallel
void LayerNorm::forward(const float *input, float *output, int rows, int iStride, int oStride) {
    TimeLine t("LayerNorm.forward");
    const float *pgamma = weights;
    const float *pbeta = weights + normSize;
    xft::invokeLayerNorm(output, input, pgamma, pbeta, rows, normSize, iStride, oStride);
}

} // namespace xft