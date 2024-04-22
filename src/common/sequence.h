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
                           SequencePool
                          ┌──────┬──────┬──────┐
                          │      │      │  ◄───┼──┬─ SequenceMeta
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

// The Sequence is one sequence of batch inputs and includes the generated tokens.
class SequenceMeta {
public:
    SequenceMeta(int32_t _sequenceID, int32_t _inputSeqLen, std::vector<int32_t> _inputTokens)
        : sequenceID(_sequenceID), inputSeqLen(_inputSeqLen), step(0) {
        inputTokens.resize(_inputSeqLen);
        inputTokens.assign(_inputTokens.begin(), _inputTokens.end());
        nextTokens.resize(_inputSeqLen);
    }

    SequenceMeta(int32_t _sequenceID, int32_t _inputSeqLen)
        : sequenceID(_sequenceID), inputSeqLen(_inputSeqLen), inputTokens(_inputSeqLen, 0), step(0) {
        nextTokens.resize(_inputSeqLen);
    }

    ~SequenceMeta() {}

    int32_t getSequenceID() const { return sequenceID; }

    // Get the input tokens in sequence
    int32_t getInputSeqLen() const { return inputSeqLen; }

    const int32_t *getInputTokens() const { return inputTokens.data(); }

    int32_t getPastSeqLen() const { return pastSeqLen; }

    void setPastSeqLen(int32_t _pastSeqLen) { pastSeqLen = _pastSeqLen; }

    // For next tokens
    void addNextToken(int32_t token) { nextTokens.push_back(token); }

    int32_t getLatestToken() const { return nextTokens.back(); }

    const int32_t *getTotalTokens() const { return nextTokens.data(); }

    int32_t getStep() const { return step; }

    void setStep(int32_t _step) { step = _step; }

private:
    int32_t sequenceID;
    int32_t inputSeqLen;
    int32_t pastSeqLen;
    std::vector<int32_t> inputTokens; // input tokens + next tokens
    std::vector<int32_t> nextTokens; // next tokens

    // Indicates whether the sequence is in the prefill phase
    int32_t step;

#ifdef PIPELINE_PARALLEL
public:
    template <typename T>
    void allocBuffer(int32_t hiddenSize, void *_hiddenStates) {
        hiddenStates = xft::alloc(sizeof(T) * getInputSeqLen() * hiddenSize);
        memcpy(hiddenStates, _hiddenStates, sizeof(T) * getInputSeqLen() * hiddenSize);
    }

private:
    int32_t hiddenSize;
    void* hiddenStates;
#endif
};

// For beam searcher
// class SequenceGroupMeta {
// public:
//     SequenceGroupMeta(int32_t num_beams) { sequence = new SequenceMeta[num_beams]; }

//     SequenceMeta *sequence;
// };

class InputQueue {
public:
    static InputQueue &getInstance() {
        static InputQueue instance;
        return instance;
    }

    int32_t createSequenceID() {
        int32_t id = sequenceID++;
        if (id >= 10 * 1024) {
            sequenceID = 0;
            id = sequenceID++;
        }
        return id;
    }

    bool empty() { return queue.empty(); }

    SequenceMeta *pop() {
        auto buffer = queue.front();
        queue.pop();
        return buffer;
    }

    void push(SequenceMeta *buffer) { queue.push(buffer); }

private:
    InputQueue() {}

    int32_t sequenceID = 0;
    std::queue<SequenceMeta *> queue;
};


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
        if (this->size() >= Env::getInstance().getMaxRequestNum()) { full = true; }
        return full;
    }

    SequenceMeta *pop() {
        auto buffer = queue.front();
        queue.pop();
        return buffer;
    }

    void push(SequenceMeta *buffer) { queue.push(buffer); }

private:
    TaskWaitingQueue() {}

    std::queue<SequenceMeta *> queue;
};


class SequencePool {
public:
    static SequencePool &getInstance() {
        static SequencePool instance;
        return instance;
    }

    bool add(int32_t key, SequenceMeta *sequence, bool force = false) {
        bool isSuccess = false;
        if (force) {
            auto it = hub.find(key);
            if (it != hub.end()) { delete it->second; }

            hub[key] = sequence;
            isSuccess = true;
        } else {
            bool exist = has(key);
            if (!exist) {
                hub[key] = sequence;
                isSuccess = true;
            }
        }

        return isSuccess;
    }

    bool has(int32_t key) const { return hub.find(key) != hub.end(); }

    SequenceMeta *get(int32_t key) const {
        auto it = hub.find(key);
        if (it != hub.end()) {
            return it->second;
        } else {
            return nullptr;
        }
    }

    void remove(int32_t key) {
        if (has(key)) { hub.erase(key); }
    }

    bool replace(int32_t oldKey, SequenceMeta *newSequence) {
        bool ret = false;
        auto it = hub.find(oldKey);
        if (it != hub.end()) {
            delete it->second;
            it->second = newSequence;
            ret = true;
        }

        return ret;
    }

private:
    SequencePool() {}

    std::unordered_map<int32_t, SequenceMeta *> hub;

    //mgr
};

} // namespace xft