#include <immintrin.h>

#include "bfloat16.h"
#include "float16.h"
#include "my_types.h"

namespace xft {

void invokeRmsNorm(float *output, const float *input, const float *weight, int rows, int cols, int iStride, int oStride,
        float epsilon) {
    int size = cols;

    if (iStride == -1) iStride = cols;
    if (oStride == -1) oStride = cols;

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
} // namespace xft