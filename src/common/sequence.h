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

// The SequenceMeta is one sequence of batch inputs and includes the generated tokens.
class SequenceMeta {
public:
    SequenceMeta(int32_t _sequenceID, int32_t _inputSeqLen, std::vector<int32_t> &_inputTokens)
        : sequenceID(_sequenceID), inputSeqLen(_inputSeqLen), pastSeqLen(0), step(0) {
        inputTokens.resize(_inputSeqLen);
        inputTokens.assign(_inputTokens.begin(), _inputTokens.end());
        nextTokens.resize(_inputSeqLen);
        setPastSeqLen(getPastSeqLen());
    }

    SequenceMeta(int32_t _sequenceID, int32_t _inputSeqLen)
        : sequenceID(_sequenceID), inputSeqLen(_inputSeqLen), inputTokens(_inputSeqLen, 0), pastSeqLen(0), step(0) {
        nextTokens.resize(_inputSeqLen);
    }

    ~SequenceMeta() {}

    int32_t getSequenceID() const { return sequenceID; }

    // For first tokens
    void stepForward() {
        if (getStep() == 0) {
            setPastSeqLen(inputTokens.size());
            setStep(getStep() + 1);
        }
    }

    // For next token
    void stepForward(int32_t token) {
        // addNextToken(token);
        setPastSeqLen(getPastSeqLen() + 1);
        setStep(getStep() + 1);
    }

    // Get the input tokens in sequence
    int32_t getInputSeqLen() const { return inputSeqLen; }

    const int32_t *getInputTokens() const { return inputTokens.data(); }

    int32_t getPastSeqLen() const { return pastSeqLen; }

    void setPastSeqLen(int32_t _pastSeqLen) { pastSeqLen = _pastSeqLen; }

    // For next tokens
    void addNextToken(int32_t token) {
        nextTokens.clear();
        nextTokens.push_back(token);
        inputTokens.push_back(token);
    }

    int32_t getLatestToken() const { return nextTokens.back(); }

    const int32_t *getTotalTokens() const { return getInputTokens(); }

    int32_t getStep() const { return step; }

    void setStep(int32_t _step) { step = _step; }

private:
    int32_t sequenceID;
    int32_t inputSeqLen;
    int32_t pastSeqLen;
    std::vector<int32_t> inputTokens; // input tokens + next tokens
    std::vector<int32_t> nextTokens; // next tokens
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
    void *hiddenStates;
#endif
};

// For beam searcher
class SequenceGroupMeta {
public:
    SequenceGroupMeta(int32_t _num_beams, std::vector<SequenceMeta *> &seq) {
        num_beams = _num_beams;
        sequences = seq;
    }

private:
    int32_t num_beams;
    std::vector<SequenceMeta *> sequences;
};

//    SequencePool
//    ┌──────┬──────┬──────┐
//    │      │      │  ◄───┼──┬─ SequenceMeta
//    ├──────┼──────┼──────┤  │
//    │      │      │  ◄───┼──┘
//    └──────┴──────┴──────┘
class SequencePool {
public:
    static SequencePool &getInstance() {
        static SequencePool instance;
        return instance;
    }

    int32_t createSequenceID() {
        int32_t id = globalSequenceID++;
        if (id >= 10 * 1024) {
            globalSequenceID = 0;
            id = globalSequenceID++;
        }
        return id;
    }

    SequenceMeta *createMeta(int32_t sequenceID, int32_t inputSeqLen, std::vector<int32_t> &inputTokens) {
        auto *sequenceMeta = new SequenceMeta(sequenceID, inputSeqLen, inputTokens);
        return sequenceMeta;
    }

    SequenceMeta *createMeta(int32_t sequenceID, int32_t inputSeqLen) {
        auto *sequenceMeta = new SequenceMeta(sequenceID, inputSeqLen);
        return sequenceMeta;
    }

    bool add(int32_t sequenceID, SequenceMeta *sequence, bool force = false) {
        bool isSuccess = false;
        if (force) {
            auto it = hub.find(sequenceID);
            if (it != hub.end()) { remove(it->first, true); }

            hub[sequenceID] = sequence;
            isSuccess = true;
        } else {
            bool exist = has(sequenceID);
            if (!exist) {
                hub[sequenceID] = sequence;
                isSuccess = true;
            }
        }

        return isSuccess;
    }

    bool has(int32_t sequenceID) const { return hub.find(sequenceID) != hub.end(); }

    SequenceMeta *get(int32_t sequenceID) const {
        auto it = hub.find(sequenceID);
        if (it != hub.end()) {
            return it->second;
        } else {
            return nullptr;
        }
    }

    bool remove(int32_t sequenceID, bool deep = false) {
        bool isSuccess = false;
        if (has(sequenceID)) {
            if (deep == true) {
                auto it = hub.find(sequenceID);
                if (it != hub.end()) { delete it->second; }
            }
            isSuccess = hub.erase(sequenceID);
        }

        return isSuccess;
    }

    bool replace(int32_t sequenceID, SequenceMeta *newSequenceMeta) {
        bool isSuccess = false;
        auto it = hub.find(sequenceID);
        if (it != hub.end()) {
            remove(it->first, true);
            hub[sequenceID] = newSequenceMeta;
            isSuccess = true;
        }

        return isSuccess;
    }

private:
    SequencePool() {}

    int32_t globalSequenceID = 0;
    std::unordered_map<int32_t, SequenceMeta *> hub;
};

// Manage input sequenceMeta
class InputQueue {
public:
    static InputQueue &getInstance() {
        static InputQueue instance;
        return instance;
    }

    bool empty() { return queue.empty(); }

    SequenceMeta *pop() {
        auto seq = queue.front();
        queue.pop();
        return seq;
    }

    void push(SequenceMeta *seq) { queue.push(seq); }

private:
    InputQueue() {}

    std::queue<SequenceMeta *> queue;
};

// Manage executive sequenceMeta
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

    SequenceMeta *front() { return queue.front(); }

    SequenceMeta *pop() {
        auto seq = queue.front();
        queue.pop();
        return seq;
    }

    void push(SequenceMeta *seq) { queue.push(seq); }

private:
    TaskWaitingQueue() {}

    std::queue<SequenceMeta *> queue;
};

} // namespace xft