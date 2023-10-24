#pragma once
#include <immintrin.h>

#include <cstdlib>
#include <cstring>

#include "layers_norm.h"
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
    xft::invokeRmsNorm(output, input, weight, rows, normSize, iStride, oStride, epsilon);
}

} // namespace xft