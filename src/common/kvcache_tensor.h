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
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <utility>

#include "allocator.h"
#include "bfloat16.h"
#include "float16.h"

extern bool kvTrans();

/**
 * Tensor specially designed for KV Cache
 * Naturaly, it could be represented in the shape of [seq_length][batch_size][head_num][head_size]
 * 
 *        |________ bs[0] ________|_______|_______ bs[N-1] _______|
 * seq=0  |       |       |       |  ...  |       |       |       |
 * seq=1  |       |       |       |  ...  |       |       |       |
 * seq=2  | head0 | head1 | head2 |  ...  | head0 | head1 | head2 |
 *  ...   |       |       |       |  ...  |       |       |       |
 *        |       |       |       |  ...  |       |       |       |
 *        `````````````````````````````````````````````````````````
 * For better performance (export ENABLE_KV_TRANS=1), it can be represented as [batch_size][head_num][seq_length][head_size]
 *        __________________
 *        |       |       ^
 *        |       |       |
 *        | head0 |       |
 *        |       |       |
 *        |       |       |
 *        |_______|       |
 *        |       |       |
 *        |       |       |
 *        | head1 |     bs[0]
 *        |       |       |
 *        |       |       |
 *        |_______|       |
 *        |       |       |
 *        |       |       |
 *        | head2 |       |
 *        |       |       |
 *        |       |       v
 *        ``````````````````
 *              ....
 * Note: The batch size in KVCache can be larger than the batch size in model inference (when beam size > 1)
 * The batch size of model inference is smaller to save the computing
 * The batch size of KV Cache is larger to make the KV cache expanding easier
*/
template <typename T>
class KVCacheTensor {
public:
    KVCacheTensor()
        : maxSeqLen(0), batchSize(0), headNum(0), headSize(0), data(nullptr), allocSize(0), scales(nullptr) {}

    ~KVCacheTensor() {
        if (this->data) { free(this->data); }
        if (this->scales) { free(this->scales); }
    }

    void resize(int maxSeqLen, int batchSize, int headNum, int headSize) {
        this->maxSeqLen = maxSeqLen;
        this->batchSize = batchSize;
        this->headNum = headNum;
        this->headSize = headSize;

        uint64_t requiredSize = (uint64_t)maxSeqLen * batchSize * headNum * headSize;
        if (requiredSize > allocSize) {
            if (this->data) { free(this->data); }
            this->data = (T *)xft::alloc(requiredSize * sizeof(T));
            if (!this->data) {
                printf("Failed to alloc mem for KV Cache [%d][%d][%d][%d].\n", maxSeqLen, batchSize, headNum, headSize);
                exit(-1);
            }
            allocSize = requiredSize;
        }

        if (this->scales) { free(this->scales); }
        this->scales = (float *)xft::alloc((uint64_t)maxSeqLen * batchSize * headNum * sizeof(float));
    }

    int getBatchSize() const { return batchSize; }
    int getHeadNum() const { return headNum; }
    int getHeadSize() const { return headSize; }

    // Get a vector for a specified sequence, return the start address, and the scale factor
    std::pair<T *, float *> getSequence(int seqIdx, int batchIdx, int headIdx) {
        // scale is in [batchSize, headNum, seq]
        float *scale = scales + (uint64_t)batchIdx * headNum * maxSeqLen + (uint64_t)headIdx * maxSeqLen + seqIdx;

        if (kvTrans()) {
            // [batchSize, headNum, seq, headSize] but also need to modify expand and reorder function
            T *addr = data + (uint64_t)batchIdx * headNum * maxSeqLen * headSize
                    + (uint64_t)headIdx * maxSeqLen * headSize + (uint64_t)seqIdx * headSize;
            return std::make_pair(addr, scale);
        } else {
            // [seqLen, batchSize, headNum, headSize] but also need to modify expand and reorder function
            T *addr = data + (uint64_t)seqIdx * batchSize * headNum * headSize + (uint64_t)batchIdx * headNum * headSize
                    + (uint64_t)headIdx * headSize;
            return std::make_pair(addr, scale);
        }
    }

    // Get a head matrix, return the start address, stride and the scale factor
    std::tuple<T *, int, float *> getHead(int batchIdx, int headIdx) {
        // scale is in [batchSize, headNum, seq]
        float *scale = scales + (uint64_t)batchIdx * headNum * maxSeqLen + (uint64_t)headIdx * maxSeqLen;

        if (kvTrans()) {
            // [batchSize, headNum, seq, headSize] but also need to modify expand and reorder function
            T *addr = data + (uint64_t)batchIdx * headNum * maxSeqLen * headSize + (uint64_t)headIdx * maxSeqLen * headSize;
            return std::make_tuple(addr, headSize, scale);
        } else {
            // [seqLen, batchSize, headNum, headSize] but also need to modify expand and reorder function
            T *addr = data + (uint64_t)batchIdx * headNum * headSize + (uint64_t)headIdx * headSize;
            return std::make_tuple(addr, batchSize * headNum * headSize, scale);
        }
    }

    /**
     * Expand the tensor by broadcasting each sample to multiple beams.
     * It is needed when beam_size > 1 by just passing the unique user side samples to do the inference.
     * For example, when user_side_bs=2, beam_size=3, it will expand:
     *  _______________________________________________
     * |  bs0  |  bs1  |       |       |       |       |
     *  ```````````````````````````````````````````````
     * to
     *  _______________________________________________
     * |  bs0  |  bs0  |  bs0  |  bs1  |  bs1  |  bs1  |
     *  ```````````````````````````````````````````````
    */
    void expandAllSequence(int userSideBS, int beamSize, int seqLen) {
        if (userSideBS * beamSize != batchSize) {
            printf("Cannot expand the KV Cache as userSideBS(%d) * beamSize(%d) != batchSize(%d)\n", userSideBS,
                    beamSize, batchSize);
            return;
        }

        if (!kvTrans()) {
#pragma omp parallel for
            for (int seq = 0; seq < seqLen; ++seq) {
                for (int b = batchSize - 1; b > 0; --b) {
                    T *dst = getSequence(seq, b, 0);
                    T *src = getSequence(seq, b / beamSize, 0);
                    memcpy(dst, src, sizeof(T) * headNum * headSize);
                }
            }
        } else {
            printf("Unsupported kv tensor optimization [ENABLE_KV_TRANS] in beam search for now.\n");
            exit(-1);
        }
    }

    void expandOneSequence(int userSideBS, int beamSize, int seq) {
        if (!kvTrans()) {
            for (int b = batchSize - 1; b > 0; --b) {
                auto dst = getSequence(seq, b, 0);
                auto src = getSequence(seq, b / beamSize, 0);
                memcpy(dst.first, src.first, sizeof(T) * headNum * headSize);
                if constexpr (std::is_same_v<T, int8_t>) { memcpy(dst.second, src.second, sizeof(float) * headNum); }
            }
        } else {
            printf("Unsupported kv tensor optimization [ENABLE_KV_TRANS] in beam search for now.\n");
            exit(-1);
        }
    }

    /**
     * Reorder the tensor, needed by beam search
     * idx: reorder index which has 'size' elements
     * size: user_side_bs * beamSize
     * initSeqLen: initial sequence length, which is the prompt token size
     * accSeqLen: accumulated sequence length
    */
    void reorder(int *idx, int size, int initSeqLen, int accSeqLen) {
        const int cols = this->getHeadNum() * this->getHeadSize();
        const int batchSize = this->getBatchSize();

        T *pdata = this->data + initSeqLen * batchSize * cols;

        // Temporary buffer used for reorder
        T *extraKeyBuf = (T *)xft::alloc((batchSize - 1) * cols * sizeof(T));

        for (int seq = initSeqLen; seq < accSeqLen; ++seq) { // Reorder is not needed for the first few lines
            int extraBufIdx = 0;
            int remapped[batchSize];
            memcpy(remapped, idx, batchSize * sizeof(int));

            for (int i = 0; i < batchSize; ++i) {
                int from = remapped[i];
                if (from < i) { // The source line already reordered
                    // Current line will be used in future, thus save to extra buffer
                    if (valueExist(remapped + i + 1, batchSize - i - 1, i)) {
                        memcpy(extraKeyBuf + extraBufIdx * cols, pdata + i * cols, cols * sizeof(T));

                        // When need line i, should look into temporary buffer, (extraBufIdx - batchSize) < 0, always
                        std::replace(remapped + i + 1, remapped + batchSize, i, extraBufIdx - batchSize);
                        extraBufIdx += 1;
                    }

                    if (from < 0) { // copy from extraBuf
                        skippableCopy(pdata + i * cols, extraKeyBuf + (from + batchSize) * cols, cols);
                    } else {
                        skippableCopy(pdata + i * cols, pdata + from * cols, cols);
                    }
                } else if (from > i) {
                    // Just need to swap
                    if (remapped[from] == i) {
                        swapValues(pdata + i * cols, pdata + from * cols, cols);

                        // Update the map information
                        std::transform(remapped + i + 1, remapped + batchSize, remapped + i + 1, [&](int num) {
                            if (num == i) {
                                return from;
                            } else if (num == from) {
                                return i;
                            }
                            return num;
                        });
                    }
                    // Current line will be used in future, thus save to extra buffer
                    else if (valueExist(remapped + i + 1, batchSize - i - 1, i)) {
                        memcpy(extraKeyBuf + extraBufIdx * cols, pdata + i * cols, cols * sizeof(T));

                        // When need line i, should look into temporary buffer, (extraBufIdx - batchSize) < 0, always
                        std::replace(remapped + i + 1, remapped + batchSize, i, extraBufIdx - batchSize);
                        extraBufIdx += 1;

                        skippableCopy(pdata + i * cols, pdata + from * cols, cols);

                        // When need line 'from', should look into line i
                        std::replace(remapped + i + 1, remapped + batchSize, from, i);
                    }
                    // Current line will never be used in futre, just overwrite it
                    else {
                        skippableCopy(pdata + i * cols, pdata + from * cols, cols);

                        // When need line 'from', should look into line i
                        std::replace(remapped + i + 1, remapped + batchSize, from, i);
                    }
                }
            }

            pdata += batchSize * cols;
        }

        free(extraKeyBuf);
    }

private:
    /******************** Start functions used by reorder *******************/
    static void swapValues8(int8_t *p1, int8_t *p2, size_t size) {
        size_t i = 0;
        for (; i + 63 < size; i += 64) {
            __m512i v1 = _mm512_loadu_si512((__m512i *)(p1 + i));
            __m512i v2 = _mm512_loadu_si512((__m512i *)(p2 + i));
            _mm512_storeu_si512(p1 + i, v2);
            _mm512_storeu_si512(p2 + i, v1);
        }

        if (i < size) {
            int remain = size - i;
            __mmask64 mask = (1ULL << remain) - 1;

            __m512i v1 = _mm512_maskz_loadu_epi8(mask, (__m512i *)(p1 + i));
            __m512i v2 = _mm512_maskz_loadu_epi8(mask, (__m512i *)(p2 + i));
            _mm512_mask_storeu_epi8(p1 + i, mask, v2);
            _mm512_mask_storeu_epi8(p2 + i, mask, v1);
        }
    }

    static void swapValues(float *p1, float *p2, size_t size) {
        swapValues8((int8_t *)p1, (int8_t *)p2, size * sizeof(float));
    }

    static void swapValues(float16_t *p1, float16_t *p2, size_t size) {
        swapValues8((int8_t *)p1, (int8_t *)p2, size * sizeof(float16_t));
    }

    static void swapValues(bfloat16_t *p1, bfloat16_t *p2, size_t size) {
        swapValues8((int8_t *)p1, (int8_t *)p2, size * sizeof(bfloat16_t));
    }

    static void swapValues(int8_t *p1, int8_t *p2, size_t size) { swapValues8(p1, p2, size); }

    template <typename DT>
    static void skippableCopy(DT *dst, DT *src, size_t size) {
        // Copy only when different
        // TODO: check if there are any risks
        if (*(uint64_t *)dst != *(uint64_t *)src) { memcpy(dst, src, size * sizeof(DT)); }
    }

    template <typename DT>
    static bool valueExist(DT *arr, size_t size, DT val) {
        for (size_t i = 0; i < size; ++i) {
            if (arr[i] == val) { return true; }
        }
        return false;
    }

    /******************** end functions used by reorder *******************/

private:
    int maxSeqLen;
    int batchSize;
    int headNum;
    int headSize;

    T *data;
    uint64_t allocSize;

    // The scale factor for each head (if T is int8)
    float *scales;
};
