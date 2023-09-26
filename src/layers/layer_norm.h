#pragma once
#include <cstdlib>
#include <immintrin.h>

#include "layernorm_kernels.h"

// Layer normalization: only support the norm along last dimension
class LayerNorm {
public:
    LayerNorm() {
        weights = nullptr;
        normSize = 0;
    }

    ~LayerNorm() {
        if (weights) { free(weights); }
    }

    void setWeight(const float *gamma, const float *beta, int size) {
        this->normSize = size;
        this->weights = (float *)aligned_alloc(64, 2 * size * sizeof(float));
        memcpy(weights, gamma, size * sizeof(float));
        memcpy(weights + size, beta, size * sizeof(float));
    }

    // input and output are in shape of (rows, normSize)
    // TODO: column-wise parallel
    void forward(const float *input, float *output, int rows, int iStride = -1, int oStride = -1) {
        const float *pgamma = weights;
        const float *pbeta = weights + normSize;
        xft::invokeLayerNorm(output, input, pgamma, pbeta, rows, normSize, iStride, oStride);
    }

private:
    int normSize;

    // the weights contains gamma and beta concated together
    float *weights;
};