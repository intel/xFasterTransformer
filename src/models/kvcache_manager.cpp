#include "kvcache_manager.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "float16.h"


/******************** Start functions used by reorderCache *******************/

static void swapValues(float *p1, float *p2, int size) {
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

static void swapValues(float16_t *p1, float16_t *p2, int size) {
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
void KVCacheManager<KVCacheT>::resize(int maxSeqLen, int batchSize, int headsPerSplit, int headSize) {
    for (int i = 0; i < this->layers; ++i) {
        this->cachedKeys[i].resize(maxSeqLen, batchSize, headsPerSplit, headSize);
        this->cachedValues[i].resize(maxSeqLen, batchSize, headsPerSplit, headSize);
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
template class KVCacheManager<float>;