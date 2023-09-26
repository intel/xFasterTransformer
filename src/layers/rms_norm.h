#pragma once
#include <immintrin.h>

#include <cstdlib>
#include <cstring>

#include "timeline.h"
#include "transformer_util.h"

// Layer normalization: only support the norm along last dimension
class RmsNorm {
public:
    RmsNorm() {
        weight = nullptr;
        normSize = 0;
    }

    ~RmsNorm() {
        if (weight) { free(weight); }
    }

    void setWeight(const float *w, const float *, int size) {
        this->normSize = size;
        this->weight = (float *)aligned_alloc(64, size * sizeof(float));
        memcpy(weight, w, size * sizeof(float));
    }

    // input and output are in shape of (rows, normSize)
    void forward(
            const float *input, float *output, int rows, int iStride = -1, int oStride = -1, float epsilon = 1e-6) {
        TimeLine t("RmsNorm.forward");
        int size = normSize;

        if (iStride == -1) iStride = normSize;
        if (oStride == -1) oStride = normSize;

#pragma omp parallel for
        for (int r = 0; r < rows; ++r) {
            const float *px = input + r * iStride;
            float *py = output + r * oStride;

            float squareSum = 0;

            __m512 vsqare = _mm512_set1_ps(0);

            int col = 0;
            for (; col + 15 < size; col += 16) {
                // SUM(x*x)
                __m512 vx = _mm512_loadu_ps(px + col);
                __m512 tmp = _mm512_mul_ps(vx, vx);
                vsqare = _mm512_add_ps(vsqare, tmp);
            }
            if (col < size) {
                __mmask16 mask = (1 << (size - col)) - 1;
                __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
                __m512 tmp = _mm512_mul_ps(vx, vx);
                vsqare = _mm512_add_ps(vsqare, tmp);
            }

            squareSum = _mm512_reduce_add_ps(vsqare);

            // Variance
            float var = 1 / sqrt(squareSum / size + epsilon);
            __m512 vvar = _mm512_set1_ps(var);

            for (col = 0; col + 15 < size; col += 16) {
                __m512 vx = _mm512_loadu_ps(px + col);
                __m512 vw = _mm512_loadu_ps(weight + col);
                __m512 vy = vx * vvar * vw;
                _mm512_storeu_ps(py + col, vy);
            }
            if (col < size) {
                __mmask16 mask = (1 << (size - col)) - 1;
                __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
                __m512 vw = _mm512_maskz_loadu_ps(mask, weight + col);
                __m512 vy = vx * vvar * vw;
                _mm512_mask_storeu_ps(py + col, mask, vy);
            }
        } // end for rows
    }

private:
    int normSize;

    // the scale weight
    float *weight;
};