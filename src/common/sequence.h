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
#include "sampling_params.h"
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
    SequenceMeta(std::vector<int32_t> &_inputTokens)
        : sequenceID(SequencePool::getInstance().createSequenceID())
        , inputSeqLen(_inputTokens.size())
        , inputTokens(_inputTokens)
        , pastSeqLen(0)
        , step(0) {
        nextTokens.reserve(inputSeqLen);
        setPastSeqLen(getPastSeqLen());
    }

    SequenceMeta(int32_t _inputSeqLen)
        : sequenceID(SequencePool::getInstance().createSequenceID())
        , inputSeqLen(_inputSeqLen)
        , inputTokens(_inputSeqLen, 0)
        , pastSeqLen(0)
        , step(0) {
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
    SequenceGroupMeta(std::vector<SequenceMeta> &seq, SamplingMeta samplingMeta_) : samplingMeta(samplingMeta_) {
        assert(samplingMeta.config.numBeams == seq.size());
        sequences.reserve(samplingMeta.config.numBeams);
        sequences.assign(seq.begin(), seq.end());
        groupID = sequences[0].getSequenceID();
    }

    SequenceGroupMeta(std::vector<int32_t> &_inputTokens, SamplingMeta samplingMeta_) : samplingMeta(samplingMeta_) {
        sequences.reserve(samplingMeta.config.numBeams);
        for (int i = 0; i < samplingMeta.config.numBeams; ++i) {
            sequences.emplace_back(SequenceMeta(_inputTokens));
        }
        groupID = sequences[0].getSequenceID();
    }

    SequenceGroupMeta(int32_t _inputSeqLen, SamplingMeta samplingMeta_) : samplingMeta(samplingMeta_) {
        sequences.reserve(samplingMeta.config.numBeams);
        for (int i = 0; i < samplingMeta.config.numBeams; ++i) {
            sequences.emplace_back(SequenceMeta(_inputSeqLen));
        }
        groupID = sequences[0].getSequenceID();
    }

    int32_t getGroupID() { return groupID; }

    int32_t getGroupSize() { return samplingMeta.config.numBeams; }

    SequenceMeta *get() { return sequences.data(); }

    SequenceMeta *get(int index) { return &sequences[index]; }

    SequenceMeta &operator[](int index) { return sequences[index]; }

private:
    // using 1st sequence ID as group ID.
    int32_t groupID;

    // The number of sequences in the group, equal to num beams
    int32_t size;
    std::vector<SequenceMeta> sequences;
    SamplingMeta samplingMeta;
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

    SequenceGroupMeta *newMeta(std::vector<int32_t> &inputTokens, SamplingMeta samplingMeta_) {
        std::vector<SequenceMeta> sequences;
        sequences.emplace_back(SequenceMeta(inputTokens));

        auto *group = new SequenceGroupMeta(sequences, samplingMeta_);
        return group;
    }

    SequenceGroupMeta *newMeta(int32_t inputSeqLen, SamplingMeta samplingMeta_) {
        std::vector<SequenceMeta> sequences;
        sequences.emplace_back(SequenceMeta(inputSeqLen));

        auto *group = new SequenceGroupMeta(sequences, samplingMeta_);
        return group;
    }

    SequenceGroupMeta *newGroupMeta(std::vector<std::vector<int32_t>> &inputTokens, SamplingMeta samplingMeta_) {
        std::vector<SequenceMeta> sequences;
        for (int i = 0; i < inputTokens.size(); ++i) {
            sequences.emplace_back(SequenceMeta(inputTokens[i]));
        }

        auto *group = new SequenceGroupMeta(sequences, samplingMeta_);
        return group;
    }

    SequenceGroupMeta *newGroupMeta(std::vector<int32_t> &inputSeqLens, SamplingMeta samplingMeta_) {

        std::vector<SequenceMeta> sequences;
        for (int i = 0; i < inputSeqLens.size(); ++i) {
            sequences.emplace_back(SequenceMeta(inputSeqLens[i]));
        }

        auto *group = new SequenceGroupMeta(sequences, samplingMeta_);
        return group;
    }

    bool add(SequenceGroupMeta *sequenceGroup, bool force = false) {
        int32_t groupID = sequenceGroup->getGroupID();
        bool isSuccess = false;
        if (force) {
            auto it = hub.find(groupID);
            if (it != hub.end()) { remove(it->first, true); }

            hub[groupID] = sequenceGroup;
            isSuccess = true;
        } else {
            bool exist = has(groupID);
            if (!exist) {
                hub[groupID] = sequenceGroup;
                isSuccess = true;
            }
        }

        return isSuccess;
    }

    bool has(int32_t groupID) const { return hub.find(groupID) != hub.end(); }

    SequenceGroupMeta *get(int32_t groupID) const {
        auto it = hub.find(groupID);
        if (it != hub.end()) {
            return it->second;
        } else {
            return nullptr;
        }
    }

    bool remove(int32_t groupID, bool deep = false) {
        bool isSuccess = false;
        if (has(groupID)) {
            if (deep == true) {
                auto it = hub.find(groupID);
                if (it != hub.end()) { delete it->second; }
            }
            isSuccess = hub.erase(groupID);
        }

        return isSuccess;
    }

    bool replace(int32_t groupID, SequenceGroupMeta *sequenceGroup) {
        bool isSuccess = false;
        auto it = hub.find(groupID);
        if (it != hub.end()) {
            remove(it->first, true);
            hub[groupID] = sequenceGroup;
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