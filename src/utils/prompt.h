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
                  PromptPool
                 ┌──────┬──────┬──────┐
                 │      │      │  ◄───┼──┬─ PromptMeta
                 ├──────┼──────┼──────┤  │
                 │      │      │  ◄───┼──┘
                 └▲─┬─▲─┴──────┴──────┘
                  │ │ └───────────────────────────────────┐
   ┌──┬──┬──┬──┐  │ │      ┌──┬──┬──┬──┬──┬──┬──┬──┬──┐   │
   │  │  │  │  ├──┘ └─────►│  │  │  │  │  │  │  │  │  ├─┐ │
   └──┴──┴──┴──┘           └──┴──┴──┴──┴──┴──┴──┴──┴──┘ │ │
    InputQueue              TaskWaitingQueue0           │ │
                        ┌───────────────────────────────┘ │
                        │  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┐   │
                        └─►│  │  │  │  │  │  │  │  │  ├───┘
                           └──┴──┴──┴──┴──┴──┴──┴──┴──┘
                            TaskWaitingQueue1
*/

namespace xft {

template <typename AttnInT>
class PromptMeta {
public:
    PromptMeta(int32_t _promptID, int32_t _tokenID, int32_t _batchSize, int32_t _inputSeqLen,
            std::vector<int32_t> _inputs) {
        promptID = _promptID;
        tokenID = _tokenID;
        batchSize = _batchSize;
        inputSeqLen = _inputSeqLen;
        inputs = _inputs;
        hiddenStatesReceived = false;
    }

    PromptMeta(int32_t _promptID, int32_t _tokenID, int32_t _batchSize, int32_t _inputSeqLen, int32_t _hiddenSize) {
        promptID = _promptID;
        tokenID = _tokenID;
        batchSize = _batchSize;
        inputSeqLen = _inputSeqLen;
        hiddenSize = _hiddenSize;
        hiddenStatesReceived = false;
        hiddenStates.Resize(batchSize * inputSeqLen, hiddenSize, hiddenSize);
    }

    void ResetKVCache(int32_t _hiddenSize, int32_t _pastSeqLen, int32_t _layerIdx, void *_hiddenStates, void *_kvm) {
        hiddenSize = _hiddenSize;
        pastSeqLen = _pastSeqLen;
        layerIdx = _layerIdx;
        hiddenStates.Resize(batchSize * inputSeqLen, hiddenSize, hiddenSize);
        memcpy(hiddenStates.Data(), _hiddenStates, sizeof(AttnInT) * batchSize * inputSeqLen * hiddenSize);
        kvm = _kvm;
    }

    int32_t promptID;
    int32_t tokenID;
    bool hiddenStatesReceived;

private:
    int32_t batchSize;
    int32_t inputSeqLen;
    int32_t hiddenSize;
    int32_t pastSeqLen;
    std::vector<int32_t> inputs;
    std::vector<int32_t> outputs;
    int32_t layerIdx;
    hpj::Matrix<AttnInT> hiddenStates;
    void *kvm; //KVCacheManager<KVCacheT>
};

template <typename T>
class InputQueue {
public:
    static InputQueue &getInstance() {
        static InputQueue instance;
        return instance;
    }

    int32_t createPromptID() {
        int32_t id = promptID++;
        if (id > 1000) {
            promptID = 0;
            id = promptID++;
        }
        return id;
    }

    int32_t createTokenID() {
        int32_t id = tokenID++;
        if (id > 1000) {
            tokenID = 0;
            id = tokenID++;
        }
        return id;
    }

    bool empty() { return queue.empty(); }

    PromptMeta<T> *pop() {
        auto buffer = queue.front();
        queue.pop();
        return buffer;
    }

    void push(PromptMeta<T> *buffer) { queue.push(buffer); }

private:
    InputQueue() {}

    static int32_t promptID;
    static int32_t tokenID;
    std::queue<PromptMeta<T> *> queue;
};

template <typename T>
int32_t InputQueue<T>::promptID = 0;

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

    PromptMeta<T> *pop() {
        auto buffer = queue.front();
        queue.pop();
        return buffer;
    }

    void push(PromptMeta<T> *buffer) { queue.push(buffer); }

private:
    TaskWaitingQueue() {}

    std::queue<PromptMeta<T> *> queue;
};

template <typename T>
class PromptPool {
public:
    static PromptPool &getInstance() {
        static PromptPool instance;
        return instance;
    }

    void insert(int32_t key, PromptMeta<T> *prompt) { hub[key] = prompt; }

    bool has(int32_t key) const { return hub.find(key) != hub.end(); }

    PromptMeta<T> *get(int32_t key) const {
        auto it = hub.find(key);
        if (it != hub.end()) {
            return it->second;
        } else {
            return nullptr;
        }
    }

    std::vector<PromptMeta<T> *> getAll() {
        std::vector<PromptMeta<T> *> metas;
        for (const auto &pair : hub) {
            metas.push_back(pair.second);
        }
        return metas;
    }

    void remove(int32_t key) { hub.erase(key); }

    void modify(int32_t oldKey, PromptMeta<T> *newPrompt) {
        auto it = hub.find(oldKey);
        if (it != hub.end()) { it->second = newPrompt; }
    }

    bool isUpdated(int32_t key) const {
        auto it = hub.find(key);
        if (it != hub.end()) {
            return it->second.hiddenStatesReceived;
        } else {
            printf("error: key not found\n");
            return false;
        }
    }

    bool setOld(int32_t key) {
        auto it = hub.find(key);
        if (it != hub.end()) {
            it->second.hiddenStatesReceived = false;
        } else {
            printf("error: key not found\n");
            return false;
        }
    }

private:
    PromptPool() {}

    std::unordered_map<int32_t, PromptMeta<T> *> hub;
};

} // namespace xft