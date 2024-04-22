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

#include <cstdint>
#include <queue>
#include <unordered_map>

/*
                           SamplePool
                          ┌──────┬──────┬──────┐
                          │      │      │  ◄───┼──┬─ SampleMeta
                          ├──────┼──────┼──────┤  │
    BatchInputs           │      │      │  ◄───┼──┘
      │                   └▲─┬─▲─┴──────┴──────┘
      │                    │ │ └───────────────────────────────────┐
      ▼     ┌──┬──┬──┬──┐  │ │      ┌──┬──┬──┬──┬──┬──┬──┬──┬──┐   │
    Input ─►│  │  │  │  ├──┘ └─────►│  │  │  │  │  │  │  │  │  ├─┐ │
            └──┴──┴──┴──┘           └──┴──┴──┴──┴──┴──┴──┴──┴──┘ │ │
            InputQueue              TaskWaitingQueue0            │ │
                                 ┌───────────────────────────────┘ │
                                 │  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┐   │
                                 └─►│  │  │  │  │  │  │  │  │  ├───┘
                                    └──┴──┴──┴──┴──┴──┴──┴──┴──┘
                                    TaskWaitingQueue1
*/

namespace xft {

// The Sample is one of batch inputs
template <typename T>
class SampleMeta {
public:
    SampleMeta(int32_t _sampleID, int32_t _inputSeqLen, std::vector<int32_t> _inputTokens)
        : sampleID(_sampleID), inputSeqLen(_inputSeqLen), bePrefill(true) {
        inputTokens.resize(_inputSeqLen);
        inputTokens.assign(_inputTokens.begin(), _inputTokens.end());
        pastTokens.resize(_inputSeqLen);
    }

    SampleMeta(int32_t _sampleID, int32_t _inputSeqLen, int32_t _hiddenSize)
        : sampleID(_sampleID), inputSeqLen(_inputSeqLen), hiddenSize(_hiddenSize), bePrefill(true) {
        inputTokens.resize(_inputSeqLen);
        pastTokens.resize(_inputSeqLen);
    }

    void ResetKVCache(int32_t _hiddenSize, int32_t _pastSeqLen, int32_t _layerIdx, void *_hiddenStates, void *_kvm) {
        hiddenSize = _hiddenSize;
        pastSeqLen = _pastSeqLen;
        layerIdx = _layerIdx;
        hiddenStates.Resize(inputSeqLen, hiddenSize, hiddenSize);
        memcpy(hiddenStates.Data(), _hiddenStates, sizeof(T) * inputSeqLen * hiddenSize);
        kvm = _kvm;
    }

    int32_t getSampleID() const { return sampleID; }

    // Get the input tokens in sample
    int32_t *getInputTokens() const { return inputTokens.data(); }

    // For generated tokens
    void addGeneratedToken(int32_t token) { pastTokens.push_back(token); }

    int32_t getLatestToken() const { return pastTokens.back(); }

    int32_t *getTotalTokens() const { return pastTokens.data(); }

    bool isPrefill() const { return bePrefill; }

    void setPrefill(bool _bePrefill) { bePrefill = _bePrefill; }

private:
    int32_t sampleID;
    int32_t inputSeqLen;
    int32_t hiddenSize;
    int32_t pastSeqLen;
    std::vector<int32_t> inputTokens;
    std::vector<int32_t> pastTokens; // generated tokens

    // Indicates whether the sample is in the prefill phase
    bool bePrefill;

    int32_t layerIdx;
    hpj::Matrix<T> hiddenStates;
    void *kvm; // KVCacheManager<KVCacheT>
};

template <typename T>
class InputQueue {
public:
    static InputQueue &getInstance() {
        static InputQueue instance;
        return instance;
    }

    int32_t createSampleID() {
        int32_t id = sampleID++;
        if (id >= 10 * 1024) {
            sampleID = 0;
            id = sampleID++;
        }
        return id;
    }

    bool empty() { return queue.empty(); }

    SampleMeta<T> *pop() {
        auto buffer = queue.front();
        queue.pop();
        return buffer;
    }

    void push(SampleMeta<T> *buffer) { queue.push(buffer); }

private:
    InputQueue() {}

    static int32_t sampleID;
    static int32_t tokenID;
    std::queue<SampleMeta<T> *> queue;
};

template <typename T>
int32_t InputQueue<T>::sampleID = 0;

template <typename T>
int32_t InputQueue<T>::tokenID = 0;

template <typename T>
class TaskWaitingQueue {
public:
    static TaskWaitingQueue &getInstance() {
        static TaskWaitingQueue instance;
        return instance;
    }

    bool empty() { return queue.empty(); }

    int32_t size() { return queue.size(); }

    bool isFull() {
        bool full = false;
        if (this->size() >= 4) full = true;
        return full;
    }

    SampleMeta<T> *pop() {
        auto buffer = queue.front();
        queue.pop();
        return buffer;
    }

    void push(SampleMeta<T> *buffer) { queue.push(buffer); }

private:
    TaskWaitingQueue() {}

    std::queue<SampleMeta<T> *> queue;
};

template <typename T>
class SamplePool {
public:
    static SamplePool &getInstance() {
        static SamplePool instance;
        return instance;
    }

    void insert(int32_t key, SampleMeta<T> *sample) { hub[key] = sample; }

    bool has(int32_t key) const { return hub.find(key) != hub.end(); }

    SampleMeta<T> *get(int32_t key) const {
        auto it = hub.find(key);
        if (it != hub.end()) {
            return it->second;
        } else {
            return nullptr;
        }
    }

    void remove(int32_t key) { hub.erase(key); }

    void modify(int32_t oldKey, SampleMeta<T> *newSample) {
        auto it = hub.find(oldKey);
        if (it != hub.end()) {
            delete it->second;
            it->second = newSample;
        }
    }

private:
    SamplePool() {}

    std::unordered_map<int32_t, SampleMeta<T> *> hub;
};

} // namespace xft