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
#ifndef _TRANSPOSE_H
#define _TRANSPOSE_H

#include <immintrin.h>
#include <stdio.h>

class TransposeUtil {
public:
    // Referred code from https://github.com/oneapi-src/oneDNN/blob/master/src/cpu/x64/matmul/brgemm_matmul_copy_utils.cpp
    static void transpose16x16(int *src, int *dst, int src_stride, int dst_stride) {
        __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
        __m512i t0, t1;

        // First 8 lines
        // swap 1
        r0 = _mm512_mask_loadu_epi32(r0, 0xffff, src);
        src += src_stride;
        r1 = _mm512_mask_loadu_epi32(r0, 0xffff, src);
        src += src_stride;
        t0 = _mm512_alignr_epi32(r0, r0, 0x1); // 15  0  1  2 ...
        t1 = _mm512_alignr_epi32(r1, r1, 0xf); // 31 16 17 18 ...
        r0 = _mm512_mask_mov_epi32(r0, 0xAAAA, t1);
        r1 = _mm512_mask_mov_epi32(r1, 0x5555, t0);

        r2 = _mm512_mask_loadu_epi32(r0, 0xffff, src);
        src += src_stride;
        r3 = _mm512_mask_loadu_epi32(r0, 0xffff, src);
        src += src_stride;
        t0 = _mm512_alignr_epi32(r2, r2, 0x1);
        t1 = _mm512_alignr_epi32(r3, r3, 0xf);
        r2 = _mm512_mask_mov_epi32(r2, 0xAAAA, t1);
        r3 = _mm512_mask_mov_epi32(r3, 0x5555, t0);

        r4 = _mm512_mask_loadu_epi32(r0, 0xffff, src);
        src += src_stride;
        r5 = _mm512_mask_loadu_epi32(r0, 0xffff, src);
        src += src_stride;
        t0 = _mm512_alignr_epi32(r4, r4, 0x1);
        t1 = _mm512_alignr_epi32(r5, r5, 0xf);
        r4 = _mm512_mask_mov_epi32(r4, 0xAAAA, t1);
        r5 = _mm512_mask_mov_epi32(r5, 0x5555, t0);

        r6 = _mm512_mask_loadu_epi32(r0, 0xffff, src);
        src += src_stride;
        r7 = _mm512_mask_loadu_epi32(r0, 0xffff, src);
        src += src_stride;
        t0 = _mm512_alignr_epi32(r6, r6, 0x1);
        t1 = _mm512_alignr_epi32(r7, r7, 0xf);
        r6 = _mm512_mask_mov_epi32(r6, 0xAAAA, t1);
        r7 = _mm512_mask_mov_epi32(r7, 0x5555, t0);

        // swap 2
        t0 = _mm512_alignr_epi32(r0, r0, 0x2);
        t1 = _mm512_alignr_epi32(r2, r2, 0xe);
        r2 = _mm512_mask_mov_epi32(r2, 0x3333, t0);
        r0 = _mm512_mask_mov_epi32(r0, 0xCCCC, t1);

        t0 = _mm512_alignr_epi32(r1, r1, 0x2);
        t1 = _mm512_alignr_epi32(r3, r3, 0xe);
        r3 = _mm512_mask_mov_epi32(r3, 0x3333, t0);
        r1 = _mm512_mask_mov_epi32(r1, 0xCCCC, t1);

        t0 = _mm512_alignr_epi32(r4, r4, 0x2);
        t1 = _mm512_alignr_epi32(r6, r6, 0xe);
        r6 = _mm512_mask_mov_epi32(r6, 0x3333, t0);
        r4 = _mm512_mask_mov_epi32(r4, 0xCCCC, t1);

        t0 = _mm512_alignr_epi32(r5, r5, 0x2);
        t1 = _mm512_alignr_epi32(r7, r7, 0xe);
        r7 = _mm512_mask_mov_epi32(r7, 0x3333, t0);
        r5 = _mm512_mask_mov_epi32(r5, 0xCCCC, t1);

        // swap 4
        t0 = r0;
        r0 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r0), 0xF0F0, _mm512_castsi512_ps(r4), _mm512_castsi512_ps(r4), 0xb1));
        r4 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r4), 0x0F0F, _mm512_castsi512_ps(t0), _mm512_castsi512_ps(t0), 0xb1));
        t0 = r1;
        r1 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r1), 0xF0F0, _mm512_castsi512_ps(r5), _mm512_castsi512_ps(r5), 0xb1));
        r5 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r5), 0x0F0F, _mm512_castsi512_ps(t0), _mm512_castsi512_ps(t0), 0xb1));
        t0 = r2;
        r2 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r2), 0xF0F0, _mm512_castsi512_ps(r6), _mm512_castsi512_ps(r6), 0xb1));
        r6 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r6), 0x0F0F, _mm512_castsi512_ps(t0), _mm512_castsi512_ps(t0), 0xb1));
        t0 = r3;
        r3 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r3), 0xF0F0, _mm512_castsi512_ps(r7), _mm512_castsi512_ps(r7), 0xb1));
        r7 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r7), 0x0F0F, _mm512_castsi512_ps(t0), _mm512_castsi512_ps(t0), 0xb1));

        // Next 8 lines
        // swap 1
        r8 = _mm512_mask_loadu_epi32(r8, 0xffff, src);
        src += src_stride;
        r9 = _mm512_mask_loadu_epi32(r8, 0xffff, src);
        src += src_stride;
        t0 = _mm512_alignr_epi32(r8, r8, 0x1); // 15  0  1  2 ...
        t1 = _mm512_alignr_epi32(r9, r9, 0xf); // 31 16 17 18 ...
        r8 = _mm512_mask_mov_epi32(r8, 0xAAAA, t1);
        r9 = _mm512_mask_mov_epi32(r9, 0x5555, t0);

        r10 = _mm512_mask_loadu_epi32(r8, 0xffff, src);
        src += src_stride;
        r11 = _mm512_mask_loadu_epi32(r8, 0xffff, src);
        src += src_stride;
        t0 = _mm512_alignr_epi32(r10, r10, 0x1);
        t1 = _mm512_alignr_epi32(r11, r11, 0xf);
        r10 = _mm512_mask_mov_epi32(r10, 0xAAAA, t1);
        r11 = _mm512_mask_mov_epi32(r11, 0x5555, t0);

        r12 = _mm512_mask_loadu_epi32(r8, 0xffff, src);
        src += src_stride;
        r13 = _mm512_mask_loadu_epi32(r8, 0xffff, src);
        src += src_stride;
        t0 = _mm512_alignr_epi32(r12, r12, 0x1);
        t1 = _mm512_alignr_epi32(r13, r13, 0xf);
        r12 = _mm512_mask_mov_epi32(r12, 0xAAAA, t1);
        r13 = _mm512_mask_mov_epi32(r13, 0x5555, t0);

        r14 = _mm512_mask_loadu_epi32(r8, 0xffff, src);
        src += src_stride;
        r15 = _mm512_mask_loadu_epi32(r8, 0xffff, src);
        src += src_stride;
        t0 = _mm512_alignr_epi32(r14, r14, 0x1);
        t1 = _mm512_alignr_epi32(r15, r15, 0xf);
        r14 = _mm512_mask_mov_epi32(r14, 0xAAAA, t1);
        r15 = _mm512_mask_mov_epi32(r15, 0x5555, t0);

        // swap 2
        t0 = _mm512_alignr_epi32(r8, r8, 0x2);
        t1 = _mm512_alignr_epi32(r10, r10, 0xe);
        r10 = _mm512_mask_mov_epi32(r10, 0x3333, t0);
        r8 = _mm512_mask_mov_epi32(r8, 0xCCCC, t1);

        t0 = _mm512_alignr_epi32(r9, r9, 0x2);
        t1 = _mm512_alignr_epi32(r11, r11, 0xe);
        r11 = _mm512_mask_mov_epi32(r11, 0x3333, t0);
        r9 = _mm512_mask_mov_epi32(r9, 0xCCCC, t1);

        t0 = _mm512_alignr_epi32(r12, r12, 0x2);
        t1 = _mm512_alignr_epi32(r14, r14, 0xe);
        r14 = _mm512_mask_mov_epi32(r14, 0x3333, t0);
        r12 = _mm512_mask_mov_epi32(r12, 0xCCCC, t1);

        t0 = _mm512_alignr_epi32(r13, r13, 0x2);
        t1 = _mm512_alignr_epi32(r15, r15, 0xe);
        r15 = _mm512_mask_mov_epi32(r15, 0x3333, t0);
        r13 = _mm512_mask_mov_epi32(r13, 0xCCCC, t1);

        // swap 4
        t0 = r8;
        r8 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r8), 0xF0F0, _mm512_castsi512_ps(r12), _mm512_castsi512_ps(r12), 0xb1));
        r12 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r12), 0x0F0F, _mm512_castsi512_ps(t0), _mm512_castsi512_ps(t0), 0xb1));
        t0 = r9;
        r9 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r9), 0xF0F0, _mm512_castsi512_ps(r13), _mm512_castsi512_ps(r13), 0xb1));
        r13 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r13), 0x0F0F, _mm512_castsi512_ps(t0), _mm512_castsi512_ps(t0), 0xb1));
        t0 = r10;
        r10 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r10), 0xF0F0, _mm512_castsi512_ps(r14), _mm512_castsi512_ps(r14), 0xb1));
        r14 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r14), 0x0F0F, _mm512_castsi512_ps(t0), _mm512_castsi512_ps(t0), 0xb1));
        t0 = r11;
        r11 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r11), 0xF0F0, _mm512_castsi512_ps(r15), _mm512_castsi512_ps(r15), 0xb1));
        r15 = _mm512_castps_si512(_mm512_mask_shuffle_f32x4(
                _mm512_castsi512_ps(r15), 0x0F0F, _mm512_castsi512_ps(t0), _mm512_castsi512_ps(t0), 0xb1));

        // At last, shuffle and save
        t0 = _mm512_shuffle_i64x2(r0, r8, 0x44);
        _mm512_mask_storeu_epi32(dst + 0 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r1, r9, 0x44);
        _mm512_mask_storeu_epi32(dst + 1 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r2, r10, 0x44);
        _mm512_mask_storeu_epi32(dst + 2 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r3, r11, 0x44);
        _mm512_mask_storeu_epi32(dst + 3 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r4, r12, 0x44);
        _mm512_mask_storeu_epi32(dst + 4 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r5, r13, 0x44);
        _mm512_mask_storeu_epi32(dst + 5 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r6, r14, 0x44);
        _mm512_mask_storeu_epi32(dst + 6 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r7, r15, 0x44);
        _mm512_mask_storeu_epi32(dst + 7 * dst_stride, 0xffff, t0);

        t0 = _mm512_shuffle_i64x2(r0, r8, 0xee);
        _mm512_mask_storeu_epi32(dst + 8 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r1, r9, 0xee);
        _mm512_mask_storeu_epi32(dst + 9 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r2, r10, 0xee);
        _mm512_mask_storeu_epi32(dst + 10 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r3, r11, 0xee);
        _mm512_mask_storeu_epi32(dst + 11 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r4, r12, 0xee);
        _mm512_mask_storeu_epi32(dst + 12 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r5, r13, 0xee);
        _mm512_mask_storeu_epi32(dst + 13 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r6, r14, 0xee);
        _mm512_mask_storeu_epi32(dst + 14 * dst_stride, 0xffff, t0);
        t0 = _mm512_shuffle_i64x2(r7, r15, 0xee);
        _mm512_mask_storeu_epi32(dst + 15 * dst_stride, 0xffff, t0);
    }

    // 16*16 block, row and col is start index, rows is total row num, cols is total col num
    static void transpose16(float *mat, int row, int col, int rows, int cols, float *matT) {
        __m512 r[16], t[16];
        __m512 zeros = _mm512_setzero_ps();

        int start = row * cols + col;

        if (col + 16 <= cols && row + 16 <= rows) {
            for (int i = 0; i < 16; ++i)
                r[i] = _mm512_loadu_ps(&mat[start + cols * i]);
        } else if (col + 16 > cols && row + 16 <= rows) {
            int edge = cols - col;
            __mmask16 mask = (edge >= 16) ? 0xffff : 0xffff >> (16 - edge);
            for (int i = 0; i < 16; ++i)
                r[i] = _mm512_mask_loadu_ps(zeros, mask, &mat[start + cols * i]);
        } else if (col + 16 <= cols && row + 16 > rows) {
            int edge = rows - row;
            for (int i = 0; i < 16; ++i)
                if (--edge >= 0)
                    r[i] = _mm512_loadu_ps(&mat[start + cols * i]);
                else
                    r[i] = zeros;
        } else {
            int edge_row = rows - row;
            int edge_col = cols - col;
            __mmask16 mask = (edge_col >= 16) ? 0xffff : 0xffff >> (16 - edge_col);
            for (int i = 0; i < 16; ++i)
                if (--edge_row >= 0)
                    r[i] = _mm512_mask_loadu_ps(zeros, mask, &mat[start + cols * i]);
                else
                    r[i] = zeros;
        }

        for (int i = 0; i < 16; i += 2) {
            t[i + 0] = _mm512_unpacklo_ps(r[i + 0], r[i + 1]);
            t[i + 1] = _mm512_unpackhi_ps(r[i + 0], r[i + 1]);
        }

        for (int i = 0; i < 16; i += 4) {
            r[i + 0] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[i + 0]), _mm512_castps_pd(t[i + 2])));
            r[i + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[i + 0]), _mm512_castps_pd(t[i + 2])));
            r[i + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[i + 1]), _mm512_castps_pd(t[i + 3])));
            r[i + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[i + 1]), _mm512_castps_pd(t[i + 3])));
        }

        for (int i = 0; i < 4; i++) {
            t[i + 0] = _mm512_shuffle_f32x4(r[i + 0], r[i + 4], 0x88);
            t[i + 4] = _mm512_shuffle_f32x4(r[i + 0], r[i + 4], 0xdd);
            t[i + 8] = _mm512_shuffle_f32x4(r[i + 8], r[i + 12], 0x88);
            t[i + 12] = _mm512_shuffle_f32x4(r[i + 8], r[i + 12], 0xdd);
        }

        for (int i = 0; i < 8; i++) {
            r[i + 0] = _mm512_shuffle_f32x4(t[i + 0], t[i + 8], 0x88);
            r[i + 8] = _mm512_shuffle_f32x4(t[i + 0], t[i + 8], 0xdd);
        }

        int rowsT = cols;
        int colsT = rows;
        int rowT = col;
        int colT = row;
        int startT = rowT * colsT + colT;

        if (col + 16 <= cols && row + 16 <= rows) {
            for (int i = 0; i < 16; i++)
                _mm512_storeu_ps(&matT[startT + colsT * i], r[i]);
        } else if (col + 16 > cols && row + 16 <= rows) {
            int edge = rowsT - rowT;
            for (int i = 0; i < 16; i++)
                if (--edge >= 0) _mm512_storeu_ps(&matT[startT + colsT * i], r[i]);
        } else if (col + 16 <= cols && row + 16 > rows) {
            int edge = colsT - colT;
            __mmask16 mask = (edge >= 16) ? 0xffff : 0xffff >> (16 - edge);
            for (int i = 0; i < 16; i++)
                _mm512_mask_storeu_ps(&matT[startT + colsT * i], mask, r[i]);
        } else {
            int edge_colT = colsT - colT;
            int edge_rowT = rowsT - rowT;
            __mmask16 mask = (edge_colT >= 16) ? 0xffff : 0xffff >> (16 - edge_colT);
            for (int i = 0; i < 16; i++)
                if (--edge_rowT >= 0) _mm512_mask_storeu_ps(&matT[startT + colsT * i], mask, r[i]);
        }
    }

    // Does not support in-place transpose
    static void transpose(float *A, float *B, int rows, int cols) {

#pragma omp parallel for collapse(2)
        for (int i = 0; i < rows; i += 16) {
            for (int j = 0; j <= cols; j += 16) {
                float *pA = A + i * cols + j;
                float *pB = B + j * rows + i;
                transpose16(A, i, j, rows, cols, B);
            }
        }

        // Remaining rows
        if (rows % 16 != 0) {
            int startRow = rows - (rows % 16);
            for (int i = startRow; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    float *pA = A + i * cols + j;
                    float *pB = B + j * rows + i;
                    *pB = *pA;
                }
            }
        }

        // Remaining cols
        if (cols % 16 != 0) {
            int startCol = cols - (cols % 16);
            for (int i = 0; i < rows; ++i) {
                for (int j = startCol; j < cols; ++j) {
                    float *pA = A + i * cols + j;
                    float *pB = B + j * rows + i;
                    *pB = *pA;
                }
            }
        }
    }

private:
    static void print_int(__m512i v) {
        int data[16];

        _mm512_mask_storeu_epi32(data, 0xffff, v);

        for (int i = 0; i < 16; ++i) {
            printf("%d ", data[i]);
        }

        printf("\n");
    }
};

#endif