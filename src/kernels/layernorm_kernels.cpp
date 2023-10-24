#include <immintrin.h>

#include "bfloat16.h"
#include "float16.h"
#include "my_types.h"

namespace xft {

template <typename T>
struct LayerNormWeight {
    const T *gamma = nullptr;
    const T *beta = nullptr;
};

void invokeLayerNorm(float *output, const float *input, const float *gamma, const float *beta, const int rows,
        const int size, int iStride, int oStride, const float epsilon) {

    if (iStride == -1) iStride = size;
    if (oStride == -1) oStride = size;

#pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        const float *px = input + r * iStride;
        float *py = output + r * oStride;

        float sum = 0;
        float squareSum = 0;

        __m512 vsum = _mm512_set1_ps(0);
        __m512 vsqare = _mm512_set1_ps(0);

        for (int col = 0; col < size; col += 16) {
            int remain = size - col;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            // SUM(x)
            __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
            vsum = _mm512_add_ps(vsum, vx);

            // SUM(x*x)
            __m512 tmp = _mm512_mul_ps(vx, vx);
            vsqare = _mm512_add_ps(vsqare, tmp);
        }

        sum = _mm512_reduce_add_ps(vsum);
        squareSum = _mm512_reduce_add_ps(vsqare);

        // Mean
        float mean = sum / size;
        __m512 vmean = _mm512_set1_ps(mean);

        // Variance
        float var = 1 / sqrt(squareSum / size - mean * mean + epsilon);
        __m512 vvar = _mm512_set1_ps(var);

        for (int col = 0; col < size; col += 16) {
            int remain = size - col;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

            __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
            __m512 vgamma = _mm512_maskz_loadu_ps(mask, gamma + col);
            __m512 vbeta = _mm512_maskz_loadu_ps(mask, beta + col);
            __m512 vy = (vx - vmean) * vgamma * vvar + vbeta;
            _mm512_mask_storeu_ps(py + col, mask, vy);
        }
    }
}
} // namespace xft