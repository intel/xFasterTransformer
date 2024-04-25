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
                          │      │      │  ◄───┼──┬─ SequenceGroupMeta
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
        inputTokens.reserve(_inputSeqLen);
        inputTokens.assign(_inputTokens.begin(), _inputTokens.end());
        nextTokens.reserve(_inputSeqLen);
        setPastSeqLen(getPastSeqLen());
    }

    SequenceMeta(int32_t _sequenceID, int32_t _inputSeqLen)
        : sequenceID(_sequenceID), inputSeqLen(_inputSeqLen), inputTokens(_inputSeqLen, 0), pastSeqLen(0), step(0) {
        nextTokens.reserve(_inputSeqLen);
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
        addNextToken(token);
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
    SequenceGroupMeta(std::vector<SequenceMeta> &seq) {
        size_per_group = seq.size();
        sequences.reserve(size_per_group);
        sequences.assign(seq.begin(), seq.end());
    }

    int32_t getGroupSize() { return size_per_group; }

    SequenceMeta *get() { return sequences.data(); }

    SequenceMeta *get(int index) { return &sequences[index]; }

    SequenceMeta &operator[](int index) {
        return sequences[index];
    }

private:
    int32_t size_per_group;
    std::vector<SequenceMeta> sequences;
};

//    SequencePool
//    ┌──────┬──────┬──────┐
//    │      │      │  ◄───┼──┬─ SequenceGroupMeta
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

    SequenceGroupMeta *newMeta(int32_t sequenceID, int32_t inputSeqLen, std::vector<int32_t> &inputTokens) {
        std::vector<SequenceMeta> sequence;
        sequence.emplace_back(SequenceMeta(sequenceID, inputSeqLen, inputTokens));

        auto *group = new SequenceGroupMeta(sequence);
        return group;
    }

    SequenceGroupMeta *newMeta(int32_t sequenceID, int32_t inputSeqLen) {
        std::vector<SequenceMeta> sequence;
        sequence.emplace_back(SequenceMeta(sequenceID, inputSeqLen));

        auto *group = new SequenceGroupMeta(sequence);
        return group;
    }

    SequenceGroupMeta *newGroupMeta(std::vector<int32_t> &sequenceIDs, std::vector<int32_t> &inputSeqLens,
            std::vector<std::vector<int32_t>> &inputTokens) {
        assert(sequenceIDs.size() == inputSeqLens.size());
        assert(sequenceIDs.size() == inputTokens.size());

        std::vector<SequenceMeta> sequences;
        for (int i = 0; i < sequenceIDs.size(); ++i) {
            sequences.emplace_back(SequenceMeta(sequenceIDs[i], inputSeqLens[i], inputTokens[i]));
        }

        auto *group = new SequenceGroupMeta(sequences);
        return group;
    }

    SequenceGroupMeta *newGroupMeta(std::vector<int32_t> &sequenceIDs, std::vector<int32_t> &inputSeqLens) {
        assert(sequenceIDs.size() == inputSeqLens.size());

        std::vector<SequenceMeta> sequences;
        for (int i = 0; i < sequenceIDs.size(); ++i) {
            sequences.emplace_back(SequenceMeta(sequenceIDs[i], inputSeqLens[i]));
        }

        auto *group = new SequenceGroupMeta(sequences);
        return group;
    }

    // Use first sequenceID if num_beam = 4
    bool add(int32_t sequenceID, SequenceGroupMeta *sequence, bool force = false) {
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

    SequenceGroupMeta *get(int32_t sequenceID) const {
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

    bool replace(int32_t sequenceID, SequenceGroupMeta *sequences) {
        bool isSuccess = false;
        auto it = hub.find(sequenceID);
        if (it != hub.end()) {
            remove(it->first, true);
            hub[sequenceID] = sequences;
            isSuccess = true;
        }

        return isSuccess;
    }

private:
    SequencePool() {}

    int32_t globalSequenceID = 0;
    std::unordered_map<int32_t, SequenceGroupMeta *> hub;
};

// Manage input sequenceMeta
class InputQueue {
public:
    static InputQueue &getInstance() {
        static InputQueue instance;
        return instance;
    }

    bool empty() { return queue.empty(); }

    SequenceGroupMeta *pop() {
        auto seq = queue.front();
        queue.pop();
        return seq;
    }

    void push(SequenceGroupMeta *seq) { queue.push(seq); }

private:
    InputQueue() {}

    std::queue<SequenceGroupMeta *> queue;
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

    SequenceGroupMeta *pop() {
        auto seq = queue.front();
        queue.pop();
        return seq;
    }

    void push(SequenceGroupMeta *seq) { queue.push(seq); }

private:
    TaskWaitingQueue() {}

    std::queue<SequenceGroupMeta *> queue;
};

} // namespace xft