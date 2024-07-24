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
#include "allocator.h"
#include "bfloat16.h"
#include "float16.h"

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

    if (!Env::getInstance().getKVTransEnabled()) {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < 2; ++i) {
            for (int seq = 0; seq < seqLen; ++seq) {
                auto src = srcTensors[i]->getSequence(seq, 0, 0);
                for (int b = userSideBS - 1; b >= 0; --b) {
                    auto dst = dstTensors[i]->getSequence(seq, b, 0);
                    memcpy(dst.first, src.first, sizeof(KVCacheT) * headNum * headSize);
                    if constexpr (std::is_same_v<KVCacheT, int8_t>) {
                        memcpy(dst.second, src.second, sizeof(float) * headNum);
                    }
                }
            }
        }
    } else {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < 2; ++i) {
            for (int b = userSideBS - 1; b >= 0; --b) {
                auto srcInfo = srcTensors[i]->getHead(b, 0);
                auto src = std::get<0>(srcInfo);
                auto srcStride = std::get<1>(srcInfo);
                auto srcScales = std::get<2>(srcInfo);
                for (int h = 0; h < headNum; ++h) {
                    auto dstInfo = dstTensors[i]->getHead(b, h);
                    auto dst = std::get<0>(srcInfo);
                    auto dstStride = std::get<1>(srcInfo);
                    auto dstScales = std::get<2>(srcInfo);
                    memcpy(dst, src, sizeof(KVCacheT) * seqLen * headSize);
                    if constexpr (std::is_same_v<KVCacheT, int8_t>) {
                        memcpy(dstScales, srcScales, sizeof(float) * seqLen);
                    }
                }
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
    for (int i = 0; i < 2 * this->layers; ++i) {
        int layer = i / 2;
        if (i % 2 == 0) {
            KVCacheTensor<KVCacheT> &keyTensor = this->getKey(layer);
            keyTensor.reorder(idx, size, initSeqLen, accSeqLen);
        } else {
            KVCacheTensor<KVCacheT> &valueTensor = this->getValue(layer);
            valueTensor.reorder(idx, size, initSeqLen, accSeqLen);
        }
    }
}

template class KVCacheManager<float16_t>;
template class KVCacheManager<bfloat16_t>;
template class KVCacheManager<float>;
template class KVCacheManager<int8_t>;
