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

class KVCacheMgrImplBase {
public:
    virtual ~KVCacheMgrImplBase() = default;
    virtual bool deleteSample(int sampleID) = 0;
    virtual bool addSample(int sampleID, int prefixId = 0) = 0;
    virtual bool reorderCache(const std::vector<int> &sampleIDs, const std::vector<int> &prevSampleIDs) = 0;
    virtual bool addPrefix(int prefixId, int sampleID) = 0;
    virtual bool prepareCache(const std::vector<int> &sampleIDs) = 0;
    virtual std::vector<void *> getKey(int layerId) = 0;
    virtual std::vector<void *> getValue(int layerId) = 0;
};

template <typename T>
class KVCacheMgrImpl : public KVCacheMgrImplBase {
public:
    KVCacheMgrImpl(int layers) { this->layers = layers; }

    ~KVCacheMgrImpl() {
        // Free resource in cachePool (readyCaches are in cachePool too)
        for (auto &it : sampleCaches) {
            delete it.second;
        }
        // Free resource in freeCaches
        for (auto &it : freeCaches) {
            delete it;
        }
    }

    // Free KVCache by sample ID.
    bool deleteSample(int sampleID) override {
        // Fail if not exist
        if (sampleCaches.find(sampleID) == sampleCaches.end()) { return false; }

        // Move from sampleCaches to freeCaches
        auto it = sampleCaches.erase(sampleID);
        freeCaches.push_back(it->second);

        return true;
    }

    bool addSample(int sampleID, int prefixId = 0) override {
        // Fail if already exist
        if (sampleCaches.find(sampleID) != sampleCaches.end()) { return false; }

        // Get a free cache or create a new one
        KVCacheTensor<T> *cache = nullptr;
        if (!freeCaches.empty()) {
            cache = freeCaches.back();
            freeCaches.pop_back();
        } else {
            cache = new KVCacheTensor<T>[2 * layers];
        }

        sampleCaches.insert({sampleID, cache});

        return true;
    }

    // Reorder cache based on prevSampleIDs for beam search (caches reordered from prevSampleIDs to sampleIDs)
    // For example, if sampleIDs = {1, 2, 3, 4} and prevSampleIDs = {1, 1, 1, 1}, then means to expand cache for sample 1
    bool reorderCache(const std::vector<int> &sampleIDs, const std::vector<int> &prevSampleIDs) override {
        return false;
    }

    // Create KVCache for prefix sharing
    bool addPrefix(int prefixId, int sampleID) override { return false; }

    // Set cache to be ready for this order of sampleIds
    bool prepareCache(const std::vector<int> &sampleIDs) override { return false; }

    // Get key caches for a layer
    std::vector<void *> getKey(int layerId) override {
        std::vector<void *> keyCaches;
        keyCaches.reserve(readyCaches.size());
        for (auto cache : readyCaches) {
            keyCaches.push_back(cache[2 * layerId]);
        }
    }

    // Get value caches for a layer
    std::vector<void *> getValue(int layerId) override {
        std::vector<void *> valueCaches;
        valueCaches.reserve(readyCaches.size());
        for (auto cache : readyCaches) {
            valueCaches.push_back(cache[2 * layerId + 1]);
        }
    }

private:
    // sampleID -> pointer to an array (each element is a KVCacheTensor, size=2*layers)
    // Layout of each array is:
    //     Key cache for layer 0
    //     Value cache for layer 0
    //     Key cache for layer 1
    //     Value cache for layer 1
    //     ...
    std::unordered_map<int, KVCacheTensor<T> *> sampleCaches;

    // List of ready caches, each element is for a sample
    std::vector<KVCacheTensor<T> *> readyCaches;

    // List of pending free caches, each element is for a sample
    std::vector<KVCacheTensor<T> *> freeCaches;

    int layers;
};

class KVCacheMgr {
public:
    static KVCacheMgr &instance() {
        static KVCacheMgr inst;
        return inst;
    }

    void configure(int layers, bool useINT8) {
        if (useINT8) {
            cacheMgrImpl = new KVCacheMgrImpl<int8_t>(layers);
        } else {
            cacheMgrImpl = new KVCacheMgrImpl<float16_t>(layers);
        }
    }

    bool deleteSample(int sampleID) { return cacheMgrImpl->deleteSample(sampleID); }

    bool addSample(int sampleID, int prefixId = 0) { return cacheMgrImpl->addSample(sampleID, prefixId); }

    bool reorderCache(const std::vector<int> &sampleIDs, const std::vector<int> &prevSampleIDs) {
        return cacheMgrImpl->reorderCache(sampleIDs, prevSampleIDs);
    }

    bool addPrefix(int prefixId, int sampleID) { return cacheMgrImpl->addPrefix(prefixId, sampleID); }

    bool prepareCache(const std::vector<int> &sampleIDs) { return cacheMgrImpl->prepareCache(sampleIDs); }

    void getKey(int layerId) { cacheMgrImpl->getKey(layerId); }

    void getValue(int layerId) { cacheMgrImpl->getValue(layerId); }

private:
    KVCacheMgrImplBase *cacheMgrImpl;

    KVCacheMgr() : cacheMgrImpl(nullptr) {}

    ~KVCacheMgr() { delete cacheMgrImpl; }

    KVCacheMgr(const KVCacheMgr &) = delete;
    KVCacheMgr &operator=(const KVCacheMgr &) = delete;
};