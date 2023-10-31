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
#include <cstdlib>
#include "kvcache_tensor.h"

// KVCacheT: data type of the key/value buffer
template <typename KVCacheT>
class KVCacheManager {
public:
    KVCacheManager(int layers) {
        this->layers = layers;
        this->cachedKeys = new KVCacheTensor<KVCacheT>[layers];
        this->cachedValues = new KVCacheTensor<KVCacheT>[layers];
    }

    ~KVCacheManager() {
        delete[] this->cachedKeys;
        delete[] this->cachedValues;
    }

    // Resize, enlarge key/value buffers if not big enough
    void resize(int maxSeqLen, int batchSize, int headsPerSplit, int headSize);

    KVCacheTensor<KVCacheT> &getKey(int layerId) { return cachedKeys[layerId]; }

    KVCacheTensor<KVCacheT> &getValue(int layerId) { return cachedValues[layerId]; }

    /**
     * Expand both key and value cache for a specified layer
     * Needed when beam size > 1, while only unique samples are sent to do inference
     * See more in KVCacheTensor::expandAllSequence
    */
    void expandCache(int layerId, int userSideBS, int beamSize, int seqLen);

    /**
     * Reorder cached keys/values, needed by beam search
     * idx: reorder index which has 'size' elements
     * size: user_side_bs * beamSize
     * initSeqLen: initial sequence length, which is the prompt token size
     * accSeqLen: accumulated sequence length
    */
    void reorderCache(int *idx, int size, int initSeqLen, int accSeqLen);

private:
    int layers; // how many layers
    KVCacheTensor<KVCacheT> *cachedKeys; // all accumulated keys
    KVCacheTensor<KVCacheT> *cachedValues; // all accumulated values
};