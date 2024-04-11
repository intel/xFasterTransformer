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
#include "sampling.h"
#include "searcher.h"
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

Model::Model() : decoder(nullptr), searcher(nullptr) {
    TimeLine::init();
}

Model::~Model() {
    exitSlaves();
    if (decoder != nullptr) { delete decoder; }
    if (searcher != nullptr) { delete searcher; }
}

void Model::exitSlaves() {
    if (decoder->getRank() == 0) {
        searchCtx.config.numBeams = 0;
        Messenger &messenger = decoder->getMessenger();
        messenger.broadcast((int *)&searchCtx.config, sizeof(SearcherConfig) / sizeof(int));
    }
}

void Model::input(std::vector<int32_t> &inputIds_, int batchSize_) {
    searchCtx.step = 0;
    Messenger &messenger = decoder->getMessenger();
    int dims[2];
    if (decoder->getRank() == 0) {
        dims[0] = batchSize_;
        dims[1] = inputIds_.size();
    }
    messenger.broadcast(dims, 2);

    searchCtx.promptIds.resize(dims[1]);
    if (decoder->getRank() == 0) { searchCtx.promptIds = inputIds_; }
    messenger.broadcast(searchCtx.promptIds.data(), dims[1]);

    searchCtx.batchSize = dims[0];
    searchCtx.seqLen = dims[1] / dims[0];

    searchCtx.doneBatch = std::vector<int>(dims[0], 0);
}

void Model::config(int maxLen_, int numBeams_, int numBeamHypsToKeep_, float lenPenalty_, bool doEarlyStopping_,
        int eosTokenId_, int padTokenId_, bool doSample_, float temperature_, int topK_, float topP_,
        float repetitionPenalty_, const std::vector<std::vector<int>> &stopWordsList_) {
    SearcherConfig configuration(maxLen_, numBeams_, numBeamHypsToKeep_, lenPenalty_, doEarlyStopping_, eosTokenId_,
            padTokenId_, doSample_, temperature_, topK_, topP_, repetitionPenalty_);
    this->config(configuration, stopWordsList_);
}

void Model::config(SearcherConfig &config_, const std::vector<std::vector<int>> &stopWordsList_) {
    searchCtx.step = 0;

    if (decoder->getRank() == 0) {
        if (config_.eosTokenId == -1) { config_.eosTokenId = decoder->getEndId(); }
        if (config_.padTokenId == -1) { config_.padTokenId = config_.eosTokenId; }

        if (config_.repetitionPenalty <= 0) {
            printf("`repetitionPenalty` has to be a strictly positive float, but is %f.\n", config_.repetitionPenalty);
            exit(-1);
        }
        searchCtx.config = config_;
    }
    Messenger &messenger = decoder->getMessenger();
    messenger.broadcast((int *)&searchCtx.config, sizeof(SearcherConfig) / sizeof(int));

    // Slaves get exit flags and exit directly
    if (decoder->getRank() > 0 && searchCtx.config.numBeams == 0) { exit(0); }

    createSearcher(searchCtx.config);
    setStopWords(stopWordsList_);
}

bool Model::isDone() {
    if (searchCtx.promptIds.empty()) {
        printf("Please set input and config first.\n");
        exit(-1);
    } else if (searcher == nullptr) {
        return xft::isDone(searchCtx);
    }
    return searchCtx.step && searcher->isDone();
}

std::vector<int32_t> Model::finalize() {
    if (searcher == nullptr) { return xft::finalize(searchCtx); }
    return searcher->finalize();
}

std::tuple<float *, int, int> Model::forward() {
    int64_t dims[3] = {searchCtx.batchSize, 1, searchCtx.seqLen};
    return decoder->forward(searchCtx.promptIds.data(), dims, 0, true);
}

std::vector<int32_t> Model::generate() {
    if (searchCtx.promptIds.empty()) {
        printf("Please set input tokens by model.input().\n");
        exit(-1);
    }
    // if (searcher == nullptr) {
    //     printf("Please set generation config by model.config().\n");
    //     exit(-1);
    // }

    if (!searchCtx.step) {
        if (searcher == nullptr) {
            TimeLine t("1st Token");
            if (!searchCtx.stopWordsList.empty()) {
                searchCtx.stopWordsIndex = std::vector<std::vector<int>>(
                        searchCtx.stopWordsList.size(), std::vector<int>(searchCtx.batchSize, 0));
            }
            int64_t dims[3] = {searchCtx.batchSize, 1, searchCtx.seqLen};
            std::tuple<float *, int, int> result = decoder->forward(searchCtx.promptIds.data(), dims, searchCtx.step++);

            return greedySearch(std::get<0>(result), std::get<1>(result), std::get<2>(result), searchCtx,
                    *decoder->getContext(), decoder->getMessenger());
        } else {
            searchCtx.step++;
            return searcher->getNextToken(searchCtx.promptIds.data(), searchCtx.batchSize, searchCtx.seqLen);
        }
    } else {
        if (searcher == nullptr) {
            TimeLine t("Next Token");
            int64_t dims[3] = {searchCtx.batchSize, 1, 1};
            std::tuple<float *, int, int> result
                    = decoder->forward(searchCtx.nextTokens.data(), dims, searchCtx.step++);
            return greedySearch(std::get<0>(result), std::get<1>(result), std::get<2>(result), searchCtx,
                    *decoder->getContext(), decoder->getMessenger());
        } else {
            return searcher->getNextToken();
        }
    }
}

void Model::createSearcher(SearcherConfig &config_) {
    if (searcher != nullptr) { delete searcher; }

    GenerationMode genMode = getGenerationMode(config_);
    if (genMode == GenerationMode::GREEDY_SEARCH) {
        if (Env::getInstance().getSearcherEnabled()) {
            searcher = new GreedySearch(*decoder, config_);
        } else {
            searcher = nullptr;
        }
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
    Messenger &messenger = decoder->getMessenger();

    // Remove empty words and words containing non-positive elements.
    // Remove words containing only eosTokenId.
    if (decoder->getRank() == 0) {
        for (auto it = stopWordsList.rbegin(); it != stopWordsList.rend(); ++it) {
            if ((*it).empty()) {
                stopWordsList.erase(std::next(it).base());
                continue;
            } else if ((*it).size() == 1 && (*it)[0] == searchCtx.config.eosTokenId) {
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
        searchCtx.stopWordsList = stopWordsList;
    } else {
        // restore stop words list to 2-D vector
        searchCtx.stopWordsList.clear();
        int currentIndex = 0;
        for (int i = 0; i < wordsSize.size(); ++i) {
            int size = wordsSize[i];
            std::vector<int> subVector(wordsData.begin() + currentIndex, wordsData.begin() + currentIndex + size);
            currentIndex += size;
            searchCtx.stopWordsList.emplace_back(subVector);
        }
    }
    return !searchCtx.stopWordsList.empty();
}

AutoModel::AutoModel(std::string modelPath, xft::DataType datatype) : Model() {
    std::string configPath = modelPath + "/config.ini";
    INIReader reader = INIReader(configPath);

    if (reader.ParseError() < 0) {
        printf("Could not load model config.ini.\n");
        exit(-1);
    }
    std::string modeltype = *reader.Sections().begin();
    setVocabSize(reader.GetInteger(modeltype, "vocab_size"));

    if (datatype != xft::DataType::unknown) {
        setDecoder(DecoderFactory::Create(modeltype + "-" + xft::getTypeIdName(datatype), modelPath));
    } else {
        printf("Unsupported data type.\n");
        exit(-1);
    }
}
} // namespace xft
