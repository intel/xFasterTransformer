// Copyright (c) 2024 Intel Corporation
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

#include <vector>
#include "kvcache_tensor.h"
#include <unordered_map>

namespace xft {

class KVCacheMgrImplBase {
public:
    virtual ~KVCacheMgrImplBase() = default;
    virtual bool delSequence(int seqID) = 0;
    virtual bool addSequence(int seqID, int maxSeqLen = -1, int prefixId = -1) = 0;
    virtual bool reorderCache(const std::vector<int> &seqIDs, const std::vector<int> &prevSeqIDs) = 0;
    virtual bool addPrefix(int prefixId, int seqID) = 0;
    virtual bool prepareCache(const std::vector<int> &seqIDs) = 0;
    virtual bool exist(int seqID) const = 0;
    virtual std::vector<void *> getKey(int layerId) = 0;
    virtual std::vector<void *> getValue(int layerId) = 0;
};

template <typename T>
class KVCacheMgrImpl : public KVCacheMgrImplBase {
public:
    KVCacheMgrImpl(int maxSeqLen, int headNum, int headSize, int layers) {
        this->maxSeqLen_ = maxSeqLen;
        this->headNum_ = headNum;
        this->headSize_ = headSize;
        this->layers_ = layers;
    }

    ~KVCacheMgrImpl() {
        // Free resource in cachePool (readyCaches are in cachePool too)
        for (auto &it : sequenceCaches) {
            delete[] it.second;
        }
        // Free resource in prefixCaches
        for (auto &it : prefixCaches) {
            delete[] it.second;
        }
        // Free resource in freeCaches
        for (auto &it : freeCaches) {
            delete[] it;
        }
    }

    // Free KVCache by sample ID.
    bool delSequence(int seqID) override {
        auto it = sequenceCaches.find(seqID);

        // Fail if not exist
        if (it == sequenceCaches.end()) { return false; }

        // Move from sequenceCaches to freeCaches
        freeCaches.push_back(it->second);

        sequenceCaches.erase(it);

        return true;
    }

    bool addSequence(int seqID, int maxSeqLen = -1, int prefixId = -1) override {
        // Fail if already exist
        if (sequenceCaches.find(seqID) != sequenceCaches.end()) { return false; }

        // Get a free cache or create a new one
        KVCacheTensor<T> *cache = nullptr;
        if (!freeCaches.empty()) {
            cache = freeCaches.back();
            freeCaches.pop_back();
        } else {
            cache = new KVCacheTensor<T>[2 * layers_];
        }

        // User specified maxSeqLen needs to be <= model's configured maxSeqLen
        auto maxLen = maxSeqLen > 0 ? std::min(maxSeqLen, maxSeqLen_) : maxSeqLen_;
        for (int i = 0; i < 2 * layers_; ++i) {
            cache[i].resize(maxLen, 1, headNum_, headSize_);
        }

        sequenceCaches.insert({seqID, cache});

        return true;
    }

    // Reorder cache based on prevSeqIDs for beam search (caches reordered from prevSeqIDs to seqIDs)
    // For example, if seqIDs = {1, 2, 3, 4} and prevSeqIDs = {1, 1, 1, 1}, then means to expand cache for sample 1
    bool reorderCache(const std::vector<int> &seqIDs, const std::vector<int> &prevSeqIDs) override {
        // TODO: implement reorderCache
        return false;
    }

    // Create KVCache for prefix sharing
    bool addPrefix(int prefixId, int seqID) override {
        // Fail if already exist
        if (prefixCaches.find(prefixId) != prefixCaches.end()) { return false; }

        // Cannot find the sample cache
        if (sequenceCaches.find(seqID) == sequenceCaches.end()) { return false; }

        // Create a new one
        KVCacheTensor<T> *cache = new KVCacheTensor<T>[2 * layers_];

        for (int i = 0; i < 2 * layers_; i++) {
            // TODO: add from method in KVCacheTensor
            //cache[i].from(sequenceCaches[seqID][i]);
        }

        prefixCaches.insert({prefixId, cache});

        return true;
    }

    // Set cache to be ready for this order of sampleIds
    bool prepareCache(const std::vector<int> &seqIDs) override {
        std::vector<KVCacheTensor<T> *> readyList;
        readyList.reserve(seqIDs.size());

        for (auto seqID : seqIDs) {
            auto it = sequenceCaches.find(seqID);
            if (it == sequenceCaches.end()) { return false; }
            readyList.push_back(it->second);
        }

        readyCaches = std::move(readyList);

        return true;
    }

    // Get key caches for a layer
    std::vector<void *> getKey(int layerId) override {
        std::vector<void *> keyCaches;
        keyCaches.reserve(readyCaches.size());
        for (auto cache : readyCaches) {
            keyCaches.push_back(&cache[2 * layerId]);
        }
        return keyCaches;
    }

    // Get value caches for a layer
    std::vector<void *> getValue(int layerId) override {
        std::vector<void *> valueCaches;
        valueCaches.reserve(readyCaches.size());
        for (auto cache : readyCaches) {
            valueCaches.push_back(&cache[2 * layerId + 1]);
        }
        return valueCaches;
    }

    bool exist(int seqID) const override { return sequenceCaches.find(seqID) != sequenceCaches.end(); }

private:
    // seqID -> pointer to an array of caches (each element is a KVCacheTensor, size=2*layers)
    // Layout of each array is:
    //     <Key cache for layer 0>
    //     <Value cache for layer 0>
    //     <Key cache for layer 1>
    //     <Value cache for layer 1>
    //     ...
    std::unordered_map<int, KVCacheTensor<T> *> sequenceCaches;

    // prefixID -> pointer to an array of caches (each element is a KVCacheTensor, size=2*layers)
    std::unordered_map<int, KVCacheTensor<T> *> prefixCaches;

    // List of ready caches, each element is for a sample; subset of sequenceCaches
    std::vector<KVCacheTensor<T> *> readyCaches;

    // List of pending free caches, each element is for a sample
    std::vector<KVCacheTensor<T> *> freeCaches;

    int maxSeqLen_;
    int headNum_;
    int headSize_;
    int layers_;
};

class KVCacheMgr {
public:
    static KVCacheMgr &instance() {
        static KVCacheMgr inst;
        return inst;
    }

    void configure(int maxSeqLen, int headNum, int headSize, int layers, DataType dataType) {
        switch (dataType) {
            case DataType::int8: cacheMgrImpl = new KVCacheMgrImpl<int8_t>(maxSeqLen, headNum, headSize, layers); break;
            case DataType::fp16:
                cacheMgrImpl = new KVCacheMgrImpl<float16_t>(maxSeqLen, headNum, headSize, layers);
                break;
            default: cacheMgrImpl = new KVCacheMgrImpl<float16_t>(maxSeqLen, headNum, headSize, layers); break;
        }
    }

    bool delSequence(int seqID) { return cacheMgrImpl->delSequence(seqID); }

    bool addSequence(int seqID, int maxSeqLen = -1, int prefixId = -1) {
        return cacheMgrImpl->addSequence(seqID, maxSeqLen, prefixId);
    }

    bool reorderCache(const std::vector<int> &seqIDs, const std::vector<int> &prevSeqIDs) {
        return cacheMgrImpl->reorderCache(seqIDs, prevSeqIDs);
    }

    bool addPrefix(int prefixId, int seqID) { return cacheMgrImpl->addPrefix(prefixId, seqID); }

    bool prepareCache(const std::vector<int> &seqIDs) { return cacheMgrImpl->prepareCache(seqIDs); }

    std::vector<void *> getKey(int layerId) { return cacheMgrImpl->getKey(layerId); }

    std::vector<void *> getValue(int layerId) { return cacheMgrImpl->getValue(layerId); }

    bool exist(int seqID) const { return cacheMgrImpl->exist(seqID); }

private:
    KVCacheMgrImplBase *cacheMgrImpl;

    KVCacheMgr() : cacheMgrImpl(nullptr) {}

    ~KVCacheMgr() { delete cacheMgrImpl; }

    KVCacheMgr(const KVCacheMgr &) = delete;
    KVCacheMgr &operator=(const KVCacheMgr &) = delete;
};

} // namespace xft