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
#include "kvcache_manager.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "bfloat16.h"
#include "float16.h"

/******************** Start functions used by reorderCache *******************/
template <typename T>
void swapValues32(T *p1, T *p2, int size) {
    static_assert(sizeof(T) == 4, "swapValues32 is designed for data types with 4 bytes.");

    int i = 0;
    for (; i + 15 < size; i += 16) {
        __m512 v1 = _mm512_loadu_ps(p1 + i);
        __m512 v2 = _mm512_loadu_ps(p2 + i);
        _mm512_storeu_ps(p1 + i, v2);
        _mm512_storeu_ps(p2 + i, v1);
    }

    if (i < size) {
        int remain = size - i;
        __mmask16 mask = (1 << remain) - 1;

        __m512 v1 = _mm512_maskz_loadu_ps(mask, p1 + i);
        __m512 v2 = _mm512_maskz_loadu_ps(mask, p2 + i);
        _mm512_mask_storeu_ps(p1 + i, mask, v2);
        _mm512_mask_storeu_ps(p2 + i, mask, v1);
    }
}

template <typename T>
void swapValues16(T *p1, T *p2, int size) {
    static_assert(sizeof(T) == 2, "swapValues16 is designed for data types with 2 bytes.");

    int i = 0;
    for (; i + 31 < size; i += 32) {
        __m512i v1 = _mm512_loadu_si512((__m512i *)(p1 + i));
        __m512i v2 = _mm512_loadu_si512((__m512i *)(p2 + i));
        _mm512_storeu_si512(p1 + i, v2);
        _mm512_storeu_si512(p2 + i, v1);
    }

    if (i < size) {
        int remain = size - i;
        __mmask32 mask = (1 << remain) - 1;

        __m512i v1 = _mm512_maskz_loadu_epi16(mask, (__m512i *)(p1 + i));
        __m512i v2 = _mm512_maskz_loadu_epi16(mask, (__m512i *)(p2 + i));
        _mm512_mask_storeu_epi16(p1 + i, mask, v2);
        _mm512_mask_storeu_epi16(p2 + i, mask, v1);
    }
}

static void swapValues(float *p1, float *p2, int size) {
    swapValues32(p1, p2, size);
}

static void swapValues(float16_t *p1, float16_t *p2, int size) {
    swapValues16(p1, p2, size);
}

static void swapValues(bfloat16_t *p1, bfloat16_t *p2, int size) {
    swapValues16(p1, p2, size);
}

template <typename T>
static void skippableCopy(T *dst, T *src, int size) {
    // Copy only when different
    // TODO: check if there are any risks
    if (*(uint64_t *)dst != *(uint64_t *)src) { memcpy(dst, src, size * sizeof(T)); }
}

template <typename T>
static bool valueExist(T *arr, int size, T val) {
    for (int i = 0; i < size; ++i) {
        if (arr[i] == val) { return true; }
    }
    return false;
}

/******************** end functions used by reorderCache *******************/

template <typename KVCacheT>
void KVCacheManager<KVCacheT>::resize(int maxSeqLen, int batchSize, int headsPerSplit, int headSize, bool prefix) {
    if (prefix && this->cachedPrefixKeys == nullptr) {
        this->cachedPrefixKeys = new KVCacheTensor<KVCacheT>[layers];
        this->cachedPrefixValues = new KVCacheTensor<KVCacheT>[layers];
    }
    for (int i = 0; i < this->layers; ++i) {
        if (prefix) {
            this->cachedPrefixKeys[i].resize(maxSeqLen, 1, headsPerSplit, headSize);
            this->cachedPrefixValues[i].resize(maxSeqLen, 1, headsPerSplit, headSize);
        } else {
            this->cachedKeys[i].resize(maxSeqLen, batchSize, headsPerSplit, headSize);
            this->cachedValues[i].resize(maxSeqLen, batchSize, headsPerSplit, headSize);
        }
    }
}

template <typename KVCacheT>
void KVCacheManager<KVCacheT>::expandCache(int layerId, int userSideBS, int beamSize, int seqLen) {
    KVCacheTensor<KVCacheT> *pTensors[2];
    pTensors[0] = &this->cachedKeys[layerId];
    pTensors[1] = &this->cachedValues[layerId];

#pragma omp parallel for collapse(2)
    for (int i = 0; i < 2; ++i) {
        for (int seq = 0; seq < seqLen; ++seq) {
            pTensors[i]->expandOneSequence(userSideBS, beamSize, seq);
        }
    }
}

template <typename KVCacheT>
void KVCacheManager<KVCacheT>::expandPrefixCache(int layerId, int userSideBS, int seqLen) {
    KVCacheTensor<KVCacheT> *dstTensors[2];
    dstTensors[0] = &this->cachedKeys[layerId];
    dstTensors[1] = &this->cachedValues[layerId];

    KVCacheTensor<KVCacheT> *srcTensors[2];
    srcTensors[0] = &this->cachedPrefixKeys[layerId];
    srcTensors[1] = &this->cachedPrefixValues[layerId];

    int headNum = dstTensors[0]->getHeadNum();
    int headSize = dstTensors[0]->getHeadSize();

#pragma omp parallel for collapse(2)
    for (int i = 0; i < 2; ++i) {
        for (int seq = 0; seq < seqLen; ++seq) {
            auto *src = srcTensors[i]->getSequence(seq, 0, 0);
            for (int b = userSideBS - 1; b >= 0; --b) {
                auto *dst = dstTensors[i]->getSequence(seq, b, 0);
                memcpy(dst, src, headNum * headSize * sizeof(KVCacheT));
            }
        }
    }
}

// Reorder cached keys and values
// TODO: move to KVCacheTensor is better
template <typename KVCacheT>
void KVCacheManager<KVCacheT>::reorderCache(int *idx, int size, int initSeqLen, int accSeqLen) {
    // Reorder for all the layers
#pragma omp parallel for
    for (int layer = 0; layer < this->layers; ++layer) {
        KVCacheTensor<KVCacheT> &keyTensor = this->getKey(layer);
        KVCacheTensor<KVCacheT> &valueTensor = this->getValue(layer);

        const int cols = keyTensor.getHeadNum() * keyTensor.getHeadSize();
        const int batchSize = keyTensor.getBatchSize();

        KVCacheT *keys = keyTensor.getData() + initSeqLen * batchSize * cols;
        KVCacheT *values = valueTensor.getData() + initSeqLen * batchSize * cols;

        // Temporary buffer used for reorder
        KVCacheT *extraKeyBuf = (KVCacheT *)aligned_alloc(64, 2 * (batchSize - 1) * cols * sizeof(KVCacheT));
        KVCacheT *extraValBuf = extraKeyBuf + (batchSize - 1) * cols;

        for (int seq = initSeqLen; seq < accSeqLen; ++seq) { // Reorder is not needed for the first few lines
            int extraBufIdx = 0;
            int remapped[batchSize];
            memcpy(remapped, idx, batchSize * sizeof(int));

            for (int i = 0; i < batchSize; ++i) {
                int from = remapped[i];
                if (from < i) { // The source line already reordered
                    // Current line will be used in future, thus save to extra buffer
                    if (valueExist(remapped + i + 1, batchSize - i - 1, i)) {
                        memcpy(extraKeyBuf + extraBufIdx * cols, keys + i * cols, cols * sizeof(KVCacheT));
                        memcpy(extraValBuf + extraBufIdx * cols, values + i * cols, cols * sizeof(KVCacheT));

                        // When need line i, should look into temporary buffer, (extraBufIdx - batchSize) < 0, always
                        std::replace(remapped + i + 1, remapped + batchSize, i, extraBufIdx - batchSize);
                        extraBufIdx += 1;
                    }

                    if (from < 0) { // copy from extraBuf
                        skippableCopy(keys + i * cols, extraKeyBuf + (from + batchSize) * cols, cols);
                        skippableCopy(values + i * cols, extraValBuf + (from + batchSize) * cols, cols);
                    } else {
                        skippableCopy(keys + i * cols, keys + from * cols, cols);
                        skippableCopy(values + i * cols, values + from * cols, cols);
                    }
                } else if (from > i) {
                    // Just need to swap
                    if (remapped[from] == i) {
                        swapValues(keys + i * cols, keys + from * cols, cols);
                        swapValues(values + i * cols, values + from * cols, cols);

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
                        memcpy(extraKeyBuf + extraBufIdx * cols, keys + i * cols, cols * sizeof(KVCacheT));
                        memcpy(extraValBuf + extraBufIdx * cols, values + i * cols, cols * sizeof(KVCacheT));

                        // When need line i, should look into temporary buffer, (extraBufIdx - batchSize) < 0, always
                        std::replace(remapped + i + 1, remapped + batchSize, i, extraBufIdx - batchSize);
                        extraBufIdx += 1;

                        skippableCopy(keys + i * cols, keys + from * cols, cols);
                        skippableCopy(values + i * cols, values + from * cols, cols);

                        // When need line 'from', should look into line i
                        std::replace(remapped + i + 1, remapped + batchSize, from, i);
                    }
                    // Current line will never be used in futre, just overwrite it
                    else {
                        skippableCopy(keys + i * cols, keys + from * cols, cols);
                        skippableCopy(values + i * cols, values + from * cols, cols);

                        // When need line 'from', should look into line i
                        std::replace(remapped + i + 1, remapped + batchSize, from, i);
                    }
                }
            }

            keys += batchSize * cols;
            values += batchSize * cols;
        }

        // Clean up
        free(extraKeyBuf);
    }
}

template class KVCacheManager<float16_t>;
template class KVCacheManager<bfloat16_t>;
template class KVCacheManager<float>;