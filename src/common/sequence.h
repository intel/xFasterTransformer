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

#include <cassert>
#include <cstdint>
#include <queue>
#include <unordered_map>

#include "environment.h"
#include "sampling_params.h"

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
// Global sequence ID manager
class SequenceIDManager {
public:
    static SequenceIDManager &getInstance() {
        static SequenceIDManager instance;
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

private:
    SequenceIDManager() {}
    int32_t globalSequenceID = 0;
};

// The SequenceMeta is one sequence of batch inputs and includes the generated tokens.
class SequenceMeta {
public:
    SequenceMeta(std::vector<int32_t> &_promptTokens)
        : sequenceID(SequenceIDManager::getInstance().createSequenceID())
        , inputSeqLen(_promptTokens.size())
        , pastSeqLen(0)
        , promptTokens(_promptTokens)
        , step(0) {}

    SequenceMeta(int32_t _inputSeqLen)
        : sequenceID(SequenceIDManager::getInstance().createSequenceID())
        , inputSeqLen(_inputSeqLen)
        , pastSeqLen(0)
        , promptTokens(_inputSeqLen, 0)
        , step(0) {}

    ~SequenceMeta() {}

    int32_t getSequenceID() const { return sequenceID; }

    std::vector<int32_t> getPromptTokens() const { return promptTokens; }

    std::vector<int32_t> getGeneratedTokens() const { return generatedTokens; }

    // Step forward given the generated token ID
    void stepForward(int32_t genToken) {
        inputSeqLen = 1;
        if (getStep() == 0) {
            setPastSeqLen(promptTokens.size());
        } else {
            setPastSeqLen(getPastSeqLen() + 1);
        }
        addNextToken(genToken);
        setStep(getStep() + 1);
    }

    // Step forward given the candidate token IDs (for verification)
    void stepForward(const std::vector<int32_t> &candidateIDs) {
        inputSeqLen = candidateIDs.size();
        if (getStep() == 0) {
            setPastSeqLen(promptTokens.size());
        } else {
            setPastSeqLen(getPastSeqLen() + 1);
        }
        generatedTokens.insert(generatedTokens.end(), candidateIDs.begin(), candidateIDs.end());
        setStep(getStep() + 1);
    }

    // Get current input sequence length
    int32_t getInputSeqLen() const { return inputSeqLen; }

    std::vector<int32_t> getInputTokens() const {
        if (getStep() == 0) {
            return promptTokens;
        } else {
            return std::vector<int32_t>(generatedTokens.end() - inputSeqLen, generatedTokens.end());
        }
    }

    int getTotalLen() const { return promptTokens.size() + generatedTokens.size(); }

    int32_t getPastSeqLen() const { return pastSeqLen; }

    void setPastSeqLen(int32_t _pastSeqLen) { pastSeqLen = _pastSeqLen; }

    // For next tokens
    void addNextToken(int32_t token) { generatedTokens.push_back(token); }

    std::vector<int32_t> getTotalTokens() const {
        std::vector<int32_t> totalTokens = promptTokens;
        totalTokens.insert(totalTokens.end(), generatedTokens.begin(), generatedTokens.end());
        return totalTokens;
    }

    int32_t getStep() const { return step; }

    void setStep(int32_t _step) { step = _step; }

private:
    int32_t sequenceID;
    int32_t inputSeqLen;
    int32_t pastSeqLen;
    std::vector<int32_t> promptTokens; // prompt tokens (user's input)
    std::vector<int32_t> generatedTokens; // all generated tokens
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
    SequenceGroupMeta(std::vector<SequenceMeta> &seq, SamplingMeta &samplingMeta_) : samplingMeta(samplingMeta_) {
        assert(samplingMeta.config.numBeams == seq.size());
        sequences.reserve(samplingMeta.config.numBeams);
        sequences.assign(seq.begin(), seq.end());
        groupID = sequences[0].getSequenceID();
    }

    SequenceGroupMeta(std::vector<int32_t> &_inputTokens, SamplingMeta &samplingMeta_) : samplingMeta(samplingMeta_) {
        sequences.reserve(samplingMeta.config.numBeams);
        for (int i = 0; i < samplingMeta.config.numBeams; ++i) {
            sequences.emplace_back(SequenceMeta(_inputTokens));
        }
        groupID = sequences[0].getSequenceID();
    }

    SequenceGroupMeta(int32_t _inputSeqLen, SamplingMeta &samplingMeta_) : samplingMeta(samplingMeta_) {
        sequences.reserve(samplingMeta.config.numBeams);
        for (int i = 0; i < samplingMeta.config.numBeams; ++i) {
            sequences.emplace_back(SequenceMeta(_inputSeqLen));
        }
        groupID = sequences[0].getSequenceID();
    }

    SequenceGroupMeta(std::vector<int32_t> &_inputTokens) {
        sequences.reserve(samplingMeta.config.numBeams);
        for (int i = 0; i < samplingMeta.config.numBeams; ++i) {
            sequences.emplace_back(SequenceMeta(_inputTokens));
        }
        groupID = sequences[0].getSequenceID();
    }

    SequenceGroupMeta(int32_t _inputSeqLen) {
        sequences.reserve(samplingMeta.config.numBeams);
        for (int i = 0; i < samplingMeta.config.numBeams; ++i) {
            sequences.emplace_back(SequenceMeta(_inputSeqLen));
        }
        groupID = sequences[0].getSequenceID();
    }

    int32_t getGroupID() { return groupID; }

    int32_t getGroupSize() { return samplingMeta.config.numBeams; }

    // using 1st sequence'step as group step.
    int32_t getStep() { return sequences[0].getStep(); }

    SequenceMeta *get() { return sequences.data(); }

    SequenceMeta *get(int index) { return &sequences[index]; }

    SequenceMeta &operator[](int index) { return sequences[index]; }

    bool isDone() { return samplingMeta.done; }

    SamplingMeta *getSamplingMeta() { return &samplingMeta; }

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

    // New sequenceGroupMeta will be added into pool.
    SequenceGroupMeta *newGroupMeta(std::vector<int32_t> &inputTokens, SamplingMeta &samplingMeta_) {
        auto *group = new SequenceGroupMeta(inputTokens, samplingMeta_);
        this->add(group);
        return group;
    }

    SequenceGroupMeta *newGroupMeta(int32_t inputSeqLen, SamplingMeta &samplingMeta_) {
        auto *group = new SequenceGroupMeta(inputSeqLen, samplingMeta_);
        this->add(group);
        return group;
    }

    SequenceGroupMeta *newGroupMeta(std::vector<int32_t> &inputTokens) {
        auto *group = new SequenceGroupMeta(inputTokens);
        this->add(group);
        return group;
    }

    SequenceGroupMeta *newGroupMeta(int32_t inputSeqLen) {
        auto *group = new SequenceGroupMeta(inputSeqLen);
        this->add(group);
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
        if (this->size() >= MaxRequestNum) { full = true; }
        return full;
    }

    SequenceGroupMeta *pop() {
        auto seq = queue.front();
        queue.pop();
        return seq;
    }

    void push(SequenceGroupMeta *seq) { queue.push(seq); }

private:
    TaskWaitingQueue() : MaxRequestNum(Env::getInstance().getMaxRequestNum()) {}

    std::queue<SequenceGroupMeta *> queue;

    int32_t MaxRequestNum;
};

} // namespace xft