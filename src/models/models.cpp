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
#include "llama.h"
#include "opt_decoder.h"
#include "qwen.h"
#include "qwen2.h"
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

void Model::exitSlaves() {
    if (decoder->getRank() == 0) {
        configuration.numBeams = 0;
        Messenger &messenger = decoder->getMessenger();
        messenger.broadcast((int *)&configuration, sizeof(SearcherConfig) / sizeof(int));
    }
}

// TODO: deprecate the following function
void Model::input(std::vector<int32_t> &inputIds_, int batchSize_) {
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

void Model::set_input(std::vector<int32_t> &inputIds_, int batchSize_, int maxLen_, int numBeams_,
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

    this->set_input(inputIds_, batchSize_, configuration, stopWordsList_);
}

void Model::set_input(std::vector<int32_t> &inputIds_, int batchSize_, SearcherConfig &config_,
        const std::vector<std::vector<int>> &stopWordsList_) {
    // TODO: remove new_input flag
    if (config_.eosTokenId == -1) { config_.eosTokenId = decoder->getEndId(); }
    if (config_.padTokenId == -1) { config_.padTokenId = config_.eosTokenId; }
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

    SequencePool &seqPool = SequencePool::getInstance();
    InputQueue &inputQueue = InputQueue::getInstance();
    for (int i = 0; i < batchSize; i++) {
        auto group = seqPool.newGroupMeta(inputIds, samplingMeta);
        inputQueue.push(group);
    }

    workingGroup.clear();
    while (!inputQueue.empty()) {
        workingGroup.push_back(inputQueue.pop());
    }
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

std::tuple<float *, int, int> Model::forward(bool logits_all) {
    // TODO: Deprecate the following Path
    if (searcher != nullptr) {
        int64_t dims[3] = {batchSize, 1, seqLen};
        return decoder->forward(inputIds.data(), dims, 0, logits_all);
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

    return decoder->forward(workingSeqs, logits_all);
}

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
        std::tuple<float *, int, int> result = forward(false);
        float *outBuf = std::get<0>(result);
        int sampleOffset = std::get<1>(result);
        int sampleSize = std::get<2>(result);
        throw std::logic_error("Method not implemented");
        return {};
    }
}

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
}
} // namespace xft
