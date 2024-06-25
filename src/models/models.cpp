// Copyright (c) 2023-2024 Intel Corporation
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
#include "models.h"

#include <string.h>

#include <stdexcept>

#include "INIReader.h"
#include "baichuan.h"
#include "chatglm.h"
#include "chatglm2.h"
#include "chatglm3.h"
#include "datatypes.h"
#include "gemma.h"
#include "hybrid_model.h"
#include "kvcache_mgr.h"
#include "llama.h"
#include "opt_decoder.h"
#include "qwen.h"
#include "qwen2.h"
#include "sampling.h"
#include "search_utils.h"
#include "searcher.h"
#include "sequence.h"
#include "timeline.h"
#include "yarn_llama.h"

namespace xft {
enum class GenerationMode { GREEDY_SEARCH, BEAM_SEARCH, SAMPLE };

GenerationMode getGenerationMode(SearcherConfig &config_) {
    if (config_.numBeams == 1) {
        if (config_.doSample) {
            return GenerationMode::SAMPLE;
        } else {
            return GenerationMode::GREEDY_SEARCH;
        }
    } else if (config_.numBeams > 1) {
        return GenerationMode::BEAM_SEARCH;
    } else {
        printf("numBeams should greater than or equal to 1.\n");
        exit(-1);
    }
}

Model::Model() : decoder(nullptr), searcher(nullptr), isNewInput(true) {
    TimeLine::init();
}

Model::~Model() {
    exitSlaves();
    if (decoder != nullptr) { delete decoder; }
    if (searcher != nullptr) { delete searcher; }
}

void Model::initMaxSeqLen() {
    DecoderContext *ctx = decoder->getContext();
    this->maxSeqLen = ctx->maxSeqLength > 0 ? ctx->maxSeqLength : ctx->maxPositions;
}

void Model::exitSlaves() {
    if (decoder->getRank() == 0) {
        if (searcher != nullptr) {
            configuration.numBeams = 0;
            Messenger &messenger = decoder->getMessenger();
            messenger.broadcast((int *)&configuration, sizeof(SearcherConfig) / sizeof(int));
            return;
        } else {
            // Only work for Model::set_input(std::vector<int32_t> &inputIds_, std::vector<int32_t> &seqLens_,
            // std::vector<int> seqIDs, std::vector<int> &maxLen)
            // TODO: Add support for other continuous batching interface
            Messenger &messenger = decoder->getMessenger();
            int dim[4] = {-1, -1, -1, -1};
            messenger.broadcast(dim, 4);
        }
    }
}

// TODO: deprecate the following function
void Model::input(std::vector<int32_t> &inputIds_, int batchSize_) {
    // TODO: remove new_input flag
    isNewInput = true;
    Messenger &messenger = decoder->getMessenger();
    int dims[2];
    if (decoder->getRank() == 0) {
        dims[0] = batchSize_;
        dims[1] = inputIds_.size();
    }
    messenger.broadcast(dims, 2);
    batchSize = dims[0];
    seqLen = dims[1] / batchSize;

    inputIds.resize(dims[1]);
    if (decoder->getRank() == 0) { inputIds = inputIds_; }
    messenger.broadcast(inputIds.data(), dims[1]);
}

// TODO: deprecate the following function
void Model::config(int maxLen_, int numBeams_, int numBeamHypsToKeep_, float lenPenalty_, bool doEarlyStopping_,
        int eosTokenId_, int padTokenId_, bool doSample_, float temperature_, int topK_, float topP_,
        float repetitionPenalty_, const std::vector<std::vector<int>> &stopWordsList_) {
    configuration.maxLen = maxLen_;
    configuration.numBeams = numBeams_;
    configuration.numBeamHypsToKeep = numBeamHypsToKeep_;
    configuration.lenPenalty = lenPenalty_;
    configuration.doEarlyStopping = doEarlyStopping_;
    configuration.eosTokenId = eosTokenId_;
    configuration.padTokenId = padTokenId_;
    configuration.doSample = doSample_;
    configuration.temperature = temperature_;
    configuration.topK = topK_;
    configuration.topP = topP_;
    configuration.repetitionPenalty = repetitionPenalty_;

    this->config(configuration, stopWordsList_);
}

void syncStopWordsList(std::vector<std::vector<int>> &stopWordsList) {
    Messenger &messenger = Messenger::getInstance();

    int listSize = stopWordsList.size();
    messenger.broadcast(&listSize, 1);
    // If stopWordsList is empty, stop broadcasting and return.
    if (listSize == 0) { return; }

    vector<int> wordsSize(listSize);
    if (messenger.getRank() == 0) {
        for (int i = 0; i < listSize; i++) {
            wordsSize[i] = stopWordsList[i].size();
        }
    }
    messenger.broadcast(wordsSize.data(), listSize);

    int wordsDataLen = 0;
    for (auto x : wordsSize) {
        wordsDataLen += x;
    }

    // flatten to 1-D vector
    vector<int> wordsData(wordsDataLen);
    if (messenger.getRank() == 0) {
        int currentIndex = 0;
        for (const auto &words : stopWordsList) {
            std::copy(words.begin(), words.end(), wordsData.begin() + currentIndex);
            currentIndex += words.size();
        }
    }
    messenger.broadcast(wordsData.data(), wordsDataLen);

    if (messenger.getRank() != 0) {
        // restore stop words list to 2-D vector
        std::vector<std::vector<int>> restoredList;
        int currentIndex = 0;
        for (int i = 0; i < wordsSize.size(); ++i) {
            int size = wordsSize[i];
            std::vector<int> subVector(wordsData.begin() + currentIndex, wordsData.begin() + currentIndex + size);
            currentIndex += size;
            restoredList.emplace_back(subVector);
        }
    }
}

std::vector<int> Model::set_input(std::vector<int32_t> &inputIds_, int batchSize_, int maxLen_, int numBeams_,
        int numBeamHypsToKeep_, float lenPenalty_, bool doEarlyStopping_, int eosTokenId_, int padTokenId_,
        bool doSample_, float temperature_, int topK_, float topP_, float repetitionPenalty_,
        const std::vector<std::vector<int>> &stopWordsList_) {
    configuration.maxLen = maxLen_;
    configuration.numBeams = numBeams_;
    configuration.numBeamHypsToKeep = numBeamHypsToKeep_;
    configuration.lenPenalty = lenPenalty_;
    configuration.doEarlyStopping = doEarlyStopping_;
    configuration.eosTokenId = eosTokenId_;
    configuration.padTokenId = padTokenId_;
    configuration.doSample = doSample_;
    configuration.temperature = temperature_;
    configuration.topK = topK_;
    configuration.topP = topP_;
    configuration.repetitionPenalty = repetitionPenalty_;

    return this->set_input(inputIds_, batchSize_, configuration, stopWordsList_);
}

std::vector<int> Model::set_input(std::vector<int32_t> &inputIds_, int batchSize_, SearcherConfig &config_,
        const std::vector<std::vector<int>> &stopWordsList_) {
    if (config_.eosTokenId == -1) { config_.eosTokenId = decoder->getEndId(); }
    if (config_.padTokenId == -1) { config_.padTokenId = config_.eosTokenId; }
    if (config_.maxLen < 0) { config_.maxLen = this->maxSeqLen; }
    SamplingMeta samplingMeta(config_, stopWordsList_);

    Messenger &messenger = Messenger::getInstance();
    if (isMaster()) { inputIds = inputIds_; }

    // Sync input and sampling param in distributed mode.
    if (messenger.getSize() > 1) {
        // [batch size, inputIds size]
        int dims[2];
        if (isMaster()) {
            dims[0] = batchSize_;
            dims[1] = inputIds_.size();
        }
        messenger.broadcast(dims, 2);
        batchSize = dims[0];
        seqLen = dims[1] / batchSize;

        inputIds.resize(dims[1]);
        messenger.broadcast(inputIds.data(), dims[1]);

        messenger.broadcast((int *)&samplingMeta.config, sizeof(SearcherConfig) / sizeof(int));

        syncStopWordsList(samplingMeta.stopWordsList);
    } else {
        batchSize = batchSize_;
        seqLen = inputIds_.size() / batchSize_;
    }

    samplingMeta.config.maxLen = std::max(samplingMeta.config.maxLen, seqLen);
    std::vector<int> seqIDs;

    SequencePool &seqPool = SequencePool::getInstance();
    KVCacheMgr &kvCacheMgr = KVCacheMgr::instance();
    workingGroup.clear();
    for (int i = 0; i < batchSize; i++) {
        auto group = seqPool.newGroupMeta(inputIds, samplingMeta);
        workingGroup.push_back(group);
        seqIDs.push_back(group->getGroupID());
        // TODO: inin KVCache for beamsearch
        kvCacheMgr.addSequence(group->getGroupID(), samplingMeta.config.maxLen);
    }

    return seqIDs;
}

std::vector<int> Model::set_input(std::vector<std::vector<int32_t>> &inputIds_, int maxLen_, int numBeams_,
        int numBeamHypsToKeep_, float lenPenalty_, bool doEarlyStopping_, int eosTokenId_, int padTokenId_,
        bool doSample_, float temperature_, int topK_, float topP_, float repetitionPenalty_,
        const std::vector<std::vector<int>> &stopWordsList_) {
    configuration.maxLen = maxLen_;
    configuration.numBeams = numBeams_;
    configuration.numBeamHypsToKeep = numBeamHypsToKeep_;
    configuration.lenPenalty = lenPenalty_;
    configuration.doEarlyStopping = doEarlyStopping_;
    configuration.eosTokenId = eosTokenId_;
    configuration.padTokenId = padTokenId_;
    configuration.doSample = doSample_;
    configuration.temperature = temperature_;
    configuration.topK = topK_;
    configuration.topP = topP_;
    configuration.repetitionPenalty = repetitionPenalty_;

    return this->set_input(inputIds_, configuration, stopWordsList_);
}

std::vector<int> Model::set_input(std::vector<std::vector<int32_t>> &inputIds_, SearcherConfig &config_,
        const std::vector<std::vector<int>> &stopWordsList_) {
    if (config_.eosTokenId == -1) { config_.eosTokenId = decoder->getEndId(); }
    if (config_.padTokenId == -1) { config_.padTokenId = config_.eosTokenId; }
    if (config_.maxLen < 0) { config_.maxLen = this->maxSeqLen; }
    SamplingMeta samplingMeta(config_, stopWordsList_);

    Messenger &messenger = Messenger::getInstance();

    batchSize = inputIds_.size();

    std::vector<int> seqIDs;
    SequencePool &seqPool = SequencePool::getInstance();
    KVCacheMgr &kvCacheMgr = KVCacheMgr::instance();
    workingGroup.clear();
    std::vector<int> seqLens;
    if (isMaster()) {
        for (auto &ids : inputIds_) {
            seqLens.push_back(ids.size());
            samplingMeta.config.maxLen = std::max(samplingMeta.config.maxLen, (int)ids.size());
        }
    }

    // Sync input and sampling param in distributed mode.
    if (messenger.getSize() > 1) {
        // [batch size, inputIds size]
        int dims[2];
        if (isMaster()) {
            inputIds.clear();
            for (auto &ids : inputIds_) {
                inputIds.insert(inputIds.end(), ids.begin(), ids.end());
            }
            dims[0] = batchSize;
            dims[1] = inputIds.size();
        }

        messenger.broadcast(dims, 2);
        batchSize = dims[0];

        inputIds.resize(dims[1]);

        messenger.broadcast(seqLens.data(), batchSize);
        messenger.broadcast(inputIds.data(), dims[1]);

        messenger.broadcast((int *)&samplingMeta.config, sizeof(SearcherConfig) / sizeof(int));

        syncStopWordsList(samplingMeta.stopWordsList);

        if (!isMaster()) {
            auto it = inputIds.begin();
            for (int i = 0; i < batchSize; i++) {
                std::vector<int32_t> input_(it, it + seqLens[i]);
                auto group = seqPool.newGroupMeta(input_, samplingMeta);
                workingGroup.push_back(group);
                seqIDs.push_back(group->getGroupID());
                // TODO: inin KVCache for beamsearch
                kvCacheMgr.addSequence(group->getGroupID(), samplingMeta.config.maxLen);

                it += seqLens[i];
            }

            return seqIDs;
        }
    }

    for (int i = 0; i < batchSize; i++) {
        auto group = seqPool.newGroupMeta(inputIds, samplingMeta);
        workingGroup.push_back(group);
        seqIDs.push_back(group->getGroupID());
        // TODO: inin KVCache for beamsearch
        kvCacheMgr.addSequence(group->getGroupID(), samplingMeta.config.maxLen);
    }

    return seqIDs;
}

std::vector<int> Model::set_input(std::vector<std::vector<int32_t>> &inputIds_, std::vector<int> seqIDs,
        SearcherConfig &config_, const std::vector<std::vector<int>> &stopWordsList_) {
    if (config_.eosTokenId == -1) { config_.eosTokenId = decoder->getEndId(); }
    if (config_.padTokenId == -1) { config_.padTokenId = config_.eosTokenId; }
    if (config_.maxLen < 0) { config_.maxLen = this->maxSeqLen; }
    config_.maxLen = std::min(config_.maxLen, this->maxSeqLen);

    SamplingMeta samplingMeta(config_, stopWordsList_);

    Messenger &messenger = Messenger::getInstance();

    batchSize = inputIds_.size();

    SequencePool &seqPool = SequencePool::getInstance();
    KVCacheMgr &kvCacheMgr = KVCacheMgr::instance();
    workingGroup.clear();

    // Sync input and sampling param in distributed mode.
    if (messenger.getSize() > 1) {
        // TODO: Sync
    }

    if (seqIDs.empty()) {
        // Prompt(1st token)
        // Create seq meta for inputs and return seq IDs
        for (auto &ids : inputIds_) {
            samplingMeta.config.maxLen = std::max(samplingMeta.config.maxLen, (int)ids.size());
        }

        for (int i = 0; i < batchSize; i++) {
            auto group = seqPool.newGroupMeta(inputIds_[i], samplingMeta);
            workingGroup.push_back(group);
            seqIDs.push_back(group->getGroupID());
            kvCacheMgr.addSequence(group->getGroupID(), samplingMeta.config.maxLen);
        }
    } else {
        // Decode(next token)
        // Update seq meta with inputs and return seq IDs
        if (inputIds_.size() != seqIDs.size()) {
            printf("[ERROR] Input size and seqIDs size mismatch.\n");
            exit(-1);
        }
        for (int i = 0; i < batchSize; i++) {
            auto group = seqPool.get(seqIDs[i]);
            if (group == nullptr) {
                // TODO: Address beam search case.
                printf("[ERROR] Sequence ID %d not found.\n", seqIDs[i]);
                exit(-1);
            }
            workingGroup.push_back(group);
            if (!kvCacheMgr.exist(seqIDs[i])) {
                printf("[ERROR] Sequence ID %d not found in KVCache.\n", seqIDs[i]);
                exit(-1);
            }
        }
    }
    return seqIDs;
}

std::vector<int> Model::set_input(std::vector<std::vector<int32_t>> &inputIds_, std::vector<int> seqIDs, int maxLen) {
    Messenger &messenger = Messenger::getInstance();
    SequencePool &seqPool = SequencePool::getInstance();
    KVCacheMgr &kvCacheMgr = KVCacheMgr::instance();
    workingGroup.clear();
    batchSize = inputIds_.size();

    maxLen = maxLen < 0 ? this->maxSeqLen : std::min(maxLen, this->maxSeqLen);

    if (messenger.getSize() > 1) {
        // TODO: Sync input and sampling param in distributed mode.
        // [batch_size, total_length, seqID_size, maxLen]
    }
    if (seqIDs.empty()) {
        // Prompt(1st token)
        // Create seq meta for inputs and return seq IDs
        for (auto &ids : inputIds_) {
            maxLen = std::max(maxLen, (int)ids.size());
        }

        for (int i = 0; i < batchSize; i++) {
            auto group = seqPool.newGroupMeta(inputIds_[i]);
            workingGroup.push_back(group);
            seqIDs.push_back(group->getGroupID());
            kvCacheMgr.addSequence(group->getGroupID(), maxLen);
        }
    } else {
        // Decode(next token)
        // Update seq meta with inputs and return seq IDs
        if (inputIds_.size() != seqIDs.size()) {
            printf("[ERROR] Input size and seqIDs size mismatch.\n");
            exit(-1);
        }
        for (int i = 0; i < batchSize; i++) {
            auto group = seqPool.get(seqIDs[i]);
            if (group == nullptr) {
                // TODO: Address beam search case.
                printf("[ERROR] Sequence ID %d not found.\n", seqIDs[i]);
                exit(-1);
            }
            group->get(0)->stepForward(inputIds_[i][0]);
            workingGroup.push_back(group);
            if (!kvCacheMgr.exist(seqIDs[i])) {
                printf("[ERROR] Sequence ID %d not found in KVCache.\n", seqIDs[i]);
                exit(-1);
            }
        }
    }
    return seqIDs;
}

std::vector<int> Model::set_input(
        std::vector<int32_t> &inputIds_, int batchSize_, std::vector<int> seqIDs, int maxLen) {
    Messenger &messenger = Messenger::getInstance();
    SequencePool &seqPool = SequencePool::getInstance();
    KVCacheMgr &kvCacheMgr = KVCacheMgr::instance();
    workingGroup.clear();
    batchSize = batchSize_;
    seqLen = inputIds_.size() / batchSize;

    maxLen = maxLen < 0 ? this->maxSeqLen : std::min(maxLen, this->maxSeqLen);
    maxLen = std::max(maxLen, seqLen);

    if (messenger.getSize() > 1) {
        // TODO: Sync input and sampling param in distributed mode.
        // [batch_size, total_length, seqID_size, maxLen]
    }
    if (seqIDs.empty()) {
        // Prompt(1st token)
        // Create seq meta for inputs and return seq IDs
        for (int i = 0; i < batchSize; i++) {
            std::vector<int32_t> inputTokens(inputIds_.begin() + i * seqLen, inputIds_.begin() + (i + 1) * seqLen);
            auto group = seqPool.newGroupMeta(inputTokens);
            workingGroup.push_back(group);
            seqIDs.push_back(group->getGroupID());
            kvCacheMgr.addSequence(group->getGroupID(), maxLen);
        }
    } else {
        // Decode(next token)
        // Update seq meta with inputs and return seq IDs
        if (inputIds_.size() != seqIDs.size()) {
            printf("[ERROR] Input size and seqIDs size mismatch.\n");
            exit(-1);
        }
        if (inputIds_.size() != batchSize_) {
            printf("[ERROR] Input size and batch size mismatch.\n");
            exit(-1);
        }
        for (int i = 0; i < batchSize; i++) {
            auto group = seqPool.get(seqIDs[i]);
            if (group == nullptr) {
                // TODO: Address beam search case.
                printf("[ERROR] Sequence ID %d not found.\n", seqIDs[i]);
                exit(-1);
            }
            group->get(0)->stepForward(inputIds_[i]);
            workingGroup.push_back(group);
            if (!kvCacheMgr.exist(seqIDs[i])) {
                printf("[ERROR] Sequence ID %d not found in KVCache.\n", seqIDs[i]);
                exit(-1);
            }
        }
    }

    return seqIDs;
}

std::vector<int> Model::set_input(
        std::vector<int32_t> &inputIds_, int batchSize_, std::vector<int> seqIDs, const std::vector<int> &maxLen) {
    Messenger &messenger = Messenger::getInstance();
    SequencePool &seqPool = SequencePool::getInstance();
    KVCacheMgr &kvCacheMgr = KVCacheMgr::instance();
    workingGroup.clear();
    batchSize = batchSize_;
    seqLen = inputIds_.size() / batchSize;

    if (!(maxLen.size() == batchSize || maxLen.size() == 1 || maxLen.empty())) {
        printf("[ERROR] maxLen size and batch size mismatch.\n");
        exit(-1);
    }

    if (messenger.getSize() > 1) {
        // TODO: Sync input and sampling param in distributed mode.
        // [batch_size, total_length, seqID_size, maxLen]
    }
    if (seqIDs.empty()) {
        // Prompt(1st token)
        // Create seq meta for inputs and return seq IDs
        for (int i = 0; i < batchSize; i++) {
            std::vector<int32_t> inputTokens(inputIds_.begin() + i * seqLen, inputIds_.begin() + (i + 1) * seqLen);
            auto group = seqPool.newGroupMeta(inputTokens);
            workingGroup.push_back(group);
            seqIDs.push_back(group->getGroupID());

            int maxLength = this->maxSeqLen;
            if (maxLen.size() == batchSize) {
                maxLength = maxLen[i] < 0 ? this->maxSeqLen : std::min(maxLen[i], this->maxSeqLen);
                maxLength = std::max(maxLength, seqLen);
            } else if (maxLen.size() == 1) {
                maxLength = maxLen[0] < 0 ? this->maxSeqLen : std::min(maxLen[0], this->maxSeqLen);
                maxLength = std::max(maxLength, seqLen);
            }

            kvCacheMgr.addSequence(group->getGroupID(), maxLength);
        }
    } else {
        // Decode(next token)
        // Update seq meta with inputs and return seq IDs
        if (inputIds_.size() != seqIDs.size()) {
            printf("[ERROR] Input size and seqIDs size mismatch.\n");
            exit(-1);
        }
        if (inputIds_.size() != batchSize_) {
            printf("[ERROR] Input size and batch size mismatch.\n");
            exit(-1);
        }
        for (int i = 0; i < batchSize; i++) {
            auto group = seqPool.get(seqIDs[i]);
            if (group == nullptr) {
                // TODO: Address beam search case.
                printf("[ERROR] Sequence ID %d not found.\n", seqIDs[i]);
                exit(-1);
            }
            group->get(0)->stepForward(inputIds_[i]);
            workingGroup.push_back(group);
            if (!kvCacheMgr.exist(seqIDs[i])) {
                printf("[ERROR] Sequence ID %d not found in KVCache.\n", seqIDs[i]);
                exit(-1);
            }
        }
    }

    return seqIDs;
}

std::vector<int> Model::set_input(std::vector<int32_t> &inputIds_, std::vector<int32_t> &seqLens_,
        std::vector<int> seqIDs, std::vector<int> &maxLen) {
    // inputIds_: for prompt(1st token), contains all tokens of the batch
    //              for decode(next token), contains the next token of the batch, size equal to batchSize
    // seqLens_: used for prompt(1st token), the length of each sequence in the batch
    // seqIDs: used for decode(next token), the sequence IDs of the batch
    // maxLen: used for prompt(1st token), the max length of each sequence in the batch,
    //          empty means the maxSeqLen, size = 1 means all seq use the same maxLen,
    //          size = batchSize means each seq has its own maxLen.
    Messenger &messenger = Messenger::getInstance();
    SequencePool &seqPool = SequencePool::getInstance();
    KVCacheMgr &kvCacheMgr = KVCacheMgr::instance();
    workingGroup.clear();

    if (messenger.getSize() > 1) {
        // [total_length, batch_size, seqID_size, maxLen_size]
        int dim[4] = {static_cast<int>(inputIds_.size()), static_cast<int>(seqLens_.size()),
                static_cast<int>(seqIDs.size()), static_cast<int>(maxLen.size())};
        messenger.broadcast(dim, 4);

        if (messenger.getRank() != 0) {
            if (dim[0] < 0) { exit(0); }
            inputIds_.resize(dim[0]);
            seqLens_.resize(dim[1]);
            seqIDs.resize(dim[2]);
            maxLen.resize(dim[3]);
        }

        messenger.broadcast(inputIds_.data(), dim[0]);
        if (dim[1] != 0) { messenger.broadcast(seqLens_.data(), dim[1]); }
        if (dim[2] != 0) { messenger.broadcast(seqIDs.data(), dim[2]); }
        if (dim[3] != 0) { messenger.broadcast(maxLen.data(), dim[3]); }
    }

    if (seqIDs.empty()) {
        batchSize = seqLens_.size();
        if (!(maxLen.size() == batchSize || maxLen.size() == 1 || maxLen.empty())) {
            printf("[ERROR] maxLen size and batch size mismatch.\n");
            exit(-1);
        }
        // Prompt(1st token)
        // Create seq meta for inputs and return seq IDs
        auto startIt = inputIds_.begin();
        auto endIt = inputIds_.begin();
        for (int i = 0; i < batchSize; i++) {
            endIt += seqLens_[i];
            std::vector<int32_t> inputTokens(startIt, endIt);
            startIt += seqLens_[i];
            auto group = seqPool.newGroupMeta(inputTokens);
            workingGroup.push_back(group);
            seqIDs.push_back(group->getGroupID());

            int maxLength = this->maxSeqLen;
            if (maxLen.size() == batchSize) {
                maxLength = maxLen[i] < 0 ? this->maxSeqLen : std::min(maxLen[i], this->maxSeqLen);
                maxLength = std::max(maxLength, seqLens_[i]);
            } else if (maxLen.size() == 1) {
                maxLength = maxLen[0] < 0 ? this->maxSeqLen : std::min(maxLen[0], this->maxSeqLen);
                maxLength = std::max(maxLength, seqLens_[i]);
            }

            kvCacheMgr.addSequence(group->getGroupID(), maxLength);
        }
    } else {
        // Decode(next token)
        // Update seq meta with inputs and return seq IDs
        if (inputIds_.size() != seqIDs.size()) {
            printf("[ERROR] Input size and seqIDs size mismatch.\n");
            exit(-1);
        }
        batchSize = seqIDs.size();
        for (int i = 0; i < batchSize; i++) {
            auto group = seqPool.get(seqIDs[i]);
            if (group == nullptr) {
                // TODO: Address beam search case.
                printf("[ERROR] Sequence ID %d not found.\n", seqIDs[i]);
                exit(-1);
            }
            group->get(0)->stepForward(inputIds_[i]);
            workingGroup.push_back(group);
            if (!kvCacheMgr.exist(seqIDs[i])) {
                printf("[ERROR] Sequence ID %d not found in KVCache.\n", seqIDs[i]);
                exit(-1);
            }
        }
    }

    return seqIDs;
}

bool Model::freeSeqs(std::vector<int> &seqIDs) {
    Messenger &messenger = Messenger::getInstance();
    // Sync
    if (messenger.getSize() > 1) {
        // Get correct size
        int size = seqIDs.size();
        messenger.broadcast(&size, 1);

        // Broadcast seqIDs
        if (messenger.getRank() != 0) { seqIDs.resize(size); }
        if (seqIDs.size() != 0) { messenger.broadcast(seqIDs.data(), size); }
    }

    // If size is empty(), return true
    if (seqIDs.size() == 0) { return true; }

    KVCacheMgr &kvCacheMgr = KVCacheMgr::instance();
    SequencePool &seqPool = SequencePool::getInstance();
    bool ret = true;
    for (auto &id : seqIDs) {
        ret = ret && kvCacheMgr.delSequence(id);
        ret = ret && seqPool.remove(id);
    }
    return ret;
}

// TODO: Deprecate the following function
void Model::config(SearcherConfig &config_, const std::vector<std::vector<int>> &stopWordsList_) {
    isNewInput = true;
    if (decoder->getRank() == 0) { configuration = config_; }
    Messenger &messenger = decoder->getMessenger();
    messenger.broadcast((int *)&configuration, sizeof(SearcherConfig) / sizeof(int));

    // Slaves get exit flags and exit directly
    if (decoder->getRank() > 0 && configuration.numBeams == 0) { exit(0); }

    createSearcher(configuration);
    setStopWords(stopWordsList_);
}

bool Model::isDone() {
    // TODO: Deprecate the following Path
    if (searcher != nullptr) {
        if (inputIds.empty()) {
            printf("Please set input and config first.\n");
            exit(-1);
        }
        return !isNewInput && searcher->isDone();
    }
    for (auto x : workingGroup) {
        if (!x->isDone()) { return false; }
    }
    return true;
}

std::vector<int32_t> Model::finalize() {
    // TODO: Deprecate the following Path
    if (searcher != nullptr) {
        return searcher->finalize();
    } else {
        std::vector<int32_t> result;
        // TODO: Unequal-length input & output
        for (auto x : workingGroup) {
            std::vector<int32_t> seq = x->get(0)->getTotalTokens();
            result.insert(result.end(), seq.begin(), seq.end());
        }
        // Clear KVCache
        KVCacheMgr &kvCacheMgr = KVCacheMgr::instance();
        SequencePool &seqPool = SequencePool::getInstance();
        for (auto x : workingGroup) {
            kvCacheMgr.delSequence(x->getGroupID());
            seqPool.remove(x->getGroupID());
        }
        workingGroup.clear();

        return result;
    }
}

std::tuple<float *, int, int> Model::forward(bool logitsAll) {
    // This forward will sync and gather all logits.
    // Return is a tuple of (logits, totalSeqSize, VocabSize)
    // TODO: Deprecate the following Path
    // Old path reture is (logits, offset, size)
    if (searcher != nullptr) {
        int64_t dims[3] = {batchSize, 1, seqLen};
        return decoder->forward(inputIds.data(), dims, 0, logitsAll);
    }
    // TODO: checking waiting queue
    if (workingGroup.empty()) {
        printf("Please input prompt first.\n");
        exit(-1);
    }
    // Assume that all sequences in the group are all prompts or all decodes.
    // Prepare input data for the decoder.
    std::vector<SequenceMeta *> workingSeqs;
    for (auto x : workingGroup) {
        workingSeqs.push_back(x->get(0));
        if (x->getGroupSize() > 1 && x->getStep() > 1) {
            for (int32_t i = 1; i < x->getGroupSize(); i++) {
                workingSeqs.push_back(x->get(i));
            }
        }
    }

    std::tuple<float *, int, int> result = decoder->forward(workingSeqs, logitsAll);

    int totalSeqSize = workingSeqs.size();
    if (logitsAll && workingSeqs[0]->getStep() == 0) {
        totalSeqSize = 0;
        for (auto x : workingSeqs) {
            totalSeqSize += x->getInputSeqLen();
        }
    }

    Messenger &messenger = decoder->getMessenger();
    if (messenger.getSize() > 1) {
        // Sync and gather all logits
        float *outBuf = std::get<0>(result);

        int workers = messenger.getSize();
        int splitSize = vocabSize / workers;
        std::vector<long unsigned int> recvCount(workers);
        std::vector<long unsigned int> splitSizes(workers);
        for (int i = 0; i < workers; i++) {
            splitSizes[i] = splitSize;
            if (i < vocabSize % workers) { splitSizes[i]++; }
            recvCount[i] = splitSizes[i] * totalSeqSize;
        }
        // warning: vocabSize * totalSeqSize may exceed the range of int when seq and batch size is large.
        logits.resize(vocabSize * totalSeqSize);
        logitsRecvBuf.resize(vocabSize * totalSeqSize);
        messenger.allgatherv(outBuf, recvCount[messenger.getRank()], logitsRecvBuf.data(), recvCount);

        // Reorder
        int offset = 0;
        for (int i = 0; i < workers; ++i) {
            for (int j = 0; j < totalSeqSize; ++j) {
                memcpy(logits.data() + (offset + j * vocabSize),
                        logitsRecvBuf.data() + offset * totalSeqSize + j * splitSizes[i],
                        splitSizes[i] * sizeof(float));
            }
            offset += splitSizes[i];
        }

        return std::tuple<float *, int, int>(logits.data(), totalSeqSize, vocabSize);
    } else {
        return std::tuple<float *, int, int>(std::get<0>(result), totalSeqSize, vocabSize);
    }
}

// We assume all gen kwargs in the batch are the same
// and all sequences are all prompts(step==0) or all decodes(step>0)
std::vector<int32_t> Model::generate() {
    // TODO: Deprecate the following Path
    if (searcher != nullptr) {
        if (inputIds.empty()) {
            printf("Please set input tokens by model.input().\n");
            exit(-1);
        }
        if (isNewInput) {
            isNewInput = false;
            return searcher->getNextToken(inputIds.data(), batchSize, inputIds.size() / batchSize);
        } else {
            return searcher->getNextToken();
        }
    } else {
        // TODO
        // Assume that all sequences in the group are all prompts or all decodes.
        // Prepare input data for the decoder.
        std::vector<SequenceMeta *> workingSeqs;
        for (auto x : workingGroup) {
            workingSeqs.push_back(x->get(0));
            if (x->getGroupSize() > 1 && x->getStep() > 1) {
                for (int32_t i = 1; i < x->getGroupSize(); i++) {
                    workingSeqs.push_back(x->get(i));
                }
            }
        }
        std::tuple<float *, int, int> result = decoder->forward(workingSeqs, false);
        float *outBuf = std::get<0>(result);
        int sampleOffset = std::get<1>(result);
        int sampleSize = std::get<2>(result);

        // Assume all gen kwargs in the batch are the same
        auto &config = workingGroup[0]->getSamplingMeta()->config;

        if (config.numBeams != 1) {
            // TODO: BeamSearch
            throw std::logic_error("Beam Search Method not implemented");
        } else {

            // Logits processor
            // Repetition penalty
            if (config.repetitionPenalty != 1.0) {
                repetitionPenaltyLogitsProcess(outBuf, sampleOffset, sampleSize, workingGroup);
            }

            std::vector<int> result;

            if (config.doSample) {
                //TODO: samling
                throw std::logic_error("Sampling Method not implemented");
            } else {
                // Greedy search
                result = greedySearch(outBuf, sampleOffset, sampleSize, batchSize);
            }

            // Check stop status
            stopCheck(result, workingGroup);

            // Step forward on all seqs
            for (int i = 0; i < workingGroup.size(); i++) {
                workingGroup[i]->get(0)->stepForward(result[i]);
            }

            return result;
        }
        throw std::logic_error("Method not implemented");
        return {};
    }
}

// TODO: Deprecate the following function
void Model::createSearcher(SearcherConfig &config_) {
    if (searcher != nullptr) { delete searcher; }

    GenerationMode genMode = getGenerationMode(config_);
    if (genMode == GenerationMode::GREEDY_SEARCH) {
        searcher = new GreedySearch(*decoder, config_);
    } else if (genMode == GenerationMode::BEAM_SEARCH) {
        searcher = new BeamSearch(*decoder, config_);
    } else if (genMode == GenerationMode::SAMPLE) {
        searcher = new SampleSearch(*decoder, config_);
    }
}

bool Model::isMaster() {
    return decoder->isMaster();
}

int Model::getRank() {
    return decoder->getRank();
}

void Model::setDecoder(AbstractDecoder *dec) {
    decoder = dec;
}

void Model::setPrefix(std::vector<int32_t> &prefixIDs_) {
    Messenger &messenger = decoder->getMessenger();
    int prefixSeqLen = prefixIDs_.size();

    messenger.broadcast(&prefixSeqLen, 1);

    std::vector<int32_t> perfixIDs;
    perfixIDs.resize(prefixSeqLen);
    if (decoder->getRank() == 0) { perfixIDs = prefixIDs_; }
    messenger.broadcast(perfixIDs.data(), prefixSeqLen);

    decoder->setPrefix(perfixIDs.data(), prefixSeqLen);
}

void Model::unsetPrefix() {
    decoder->unsetPrefix();
}

bool Model::setStopWords(std::vector<std::vector<int>> stopWordsList) {
    if (searcher == nullptr) {
        printf("[Warning] Fails to set stop words. Please config model first.");
        return false;
    }
    Messenger &messenger = decoder->getMessenger();

    // Remove empty words and words containing non-positive elements.
    if (decoder->getRank() == 0) {
        for (auto it = stopWordsList.rbegin(); it != stopWordsList.rend(); ++it) {
            if ((*it).empty()) {
                stopWordsList.erase(std::next(it).base());
                continue;
            }
            for (auto x : *it) {
                if (x <= 0) { stopWordsList.erase(std::next(it).base()); }
            }
        }
    }

    int listSize = stopWordsList.size();
    messenger.broadcast(&listSize, 1);
    // If stopWordsList is empty, stop broadcasting and return.
    if (listSize == 0) { return false; }

    vector<int> wordsSize(listSize);
    if (decoder->getRank() == 0) {
        for (int i = 0; i < listSize; i++) {
            wordsSize[i] = stopWordsList[i].size();
        }
    }
    messenger.broadcast(wordsSize.data(), listSize);

    int wordsDataLen = 0;
    for (auto x : wordsSize) {
        wordsDataLen += x;
    }

    // flatten to 1-D vector
    vector<int> wordsData(wordsDataLen);
    if (decoder->getRank() == 0) {
        int currentIndex = 0;
        for (const auto &words : stopWordsList) {
            std::copy(words.begin(), words.end(), wordsData.begin() + currentIndex);
            currentIndex += words.size();
        }
    }
    messenger.broadcast(wordsData.data(), wordsDataLen);

    if (decoder->getRank() == 0) {
        return searcher->setStopWords(stopWordsList);
    } else {
        // restore stop words list to 2-D vector
        std::vector<std::vector<int>> restoredList;
        int currentIndex = 0;
        for (int i = 0; i < wordsSize.size(); ++i) {
            int size = wordsSize[i];
            std::vector<int> subVector(wordsData.begin() + currentIndex, wordsData.begin() + currentIndex + size);
            currentIndex += size;
            restoredList.emplace_back(subVector);
        }

        return searcher->setStopWords(restoredList);
    }
}

AutoModel::AutoModel(std::string modelPath, xft::DataType dataType, xft::DataType KVCacheDataType) : Model() {
    std::string configPath = modelPath + "/config.ini";
    INIReader reader = INIReader(configPath);

    if (reader.ParseError() < 0) {
        printf("Could not load model config.ini.\n");
        exit(-1);
    }
    std::string modeltype = *reader.Sections().begin();
    setVocabSize(reader.GetInteger(modeltype, "vocab_size"));

    if (dataType != xft::DataType::unknown) {
        setDecoder(DecoderFactory::Create(
                modeltype + "-" + xft::getTypeIdName(dataType) + "-" + xft::getTypeIdName(KVCacheDataType), modelPath));
    } else {
        printf("Unsupported data type or KV cache data type.\n");
        exit(-1);
    }

    initMaxSeqLen();
}
} // namespace xft
