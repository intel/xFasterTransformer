// Copyright (c) 2023 Intel Corporation
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

#include <iostream>
#include <vector>

#include "abstract_decoder.h"
#include "abstract_searcher.h"
#include "dtype.h"

namespace xft {
class Model {
public:
    Model();
    ~Model();

    void input(std::vector<int32_t> &inputIds_, int batchSize_);

    void config(int maxLen_ = -1, int numBeams_ = 1, int numBeamHypsToKeep_ = 1, float lenPenalty_ = 1.0,
            bool doEarlyStopping_ = false, int eosTokenId_ = -1, int padTokenId_ = -1, bool doSample_ = false,
            float temperature_ = 1.0, int topK_ = 50, float topP_ = 1.0, float repetitionPenalty_ = 1.0,
            const std::vector<std::vector<int>> &stopWordsList_ = {});

    void config(SearcherConfig &config_, const std::vector<std::vector<int>> &stopWordsList_ = {});

    // Return the sequences' IDs in the order of the input batch
    std::vector<int> set_input(std::vector<int32_t> &inputIds_, int batchSize_, int maxLen_ = -1, int numBeams_ = 1,
            int numBeamHypsToKeep_ = 1, float lenPenalty_ = 1.0, bool doEarlyStopping_ = false, int eosTokenId_ = -1,
            int padTokenId_ = -1, bool doSample_ = false, float temperature_ = 1.0, int topK_ = 50, float topP_ = 1.0,
            float repetitionPenalty_ = 1.0, const std::vector<std::vector<int>> &stopWordsList_ = {});

    std::vector<int> set_input(std::vector<int32_t> &inputIds_, int batchSize_, SearcherConfig &config_,
            const std::vector<std::vector<int>> &stopWordsList_ = {});

    std::vector<int> set_input(std::vector<std::vector<int32_t>> &inputIds_, SearcherConfig &config_,
            const std::vector<std::vector<int>> &stopWordsList_ = {});

    std::vector<int> set_input(std::vector<std::vector<int32_t>> &inputIds_, int maxLen_ = -1, int numBeams_ = 1,
            int numBeamHypsToKeep_ = 1, float lenPenalty_ = 1.0, bool doEarlyStopping_ = false, int eosTokenId_ = -1,
            int padTokenId_ = -1, bool doSample_ = false, float temperature_ = 1.0, int topK_ = 50, float topP_ = 1.0,
            float repetitionPenalty_ = 1.0, const std::vector<std::vector<int>> &stopWordsList_ = {});

    std::vector<int> set_input(std::vector<std::vector<int32_t>> &inputIds_, std::vector<int> seqIDs,
            SearcherConfig &config_, const std::vector<std::vector<int>> &stopWordsList_ = {});

    // Only used for model.forward()
    std::vector<int> set_input(
            std::vector<std::vector<int32_t>> &inputIds_, std::vector<int> seqIDs = {}, int maxLen = -1);

    // Only used for model.forward()
    std::vector<int> set_input(
            std::vector<int32_t> &inputIds_, int batchSize_, std::vector<int> seqIDs = {}, int maxLen = -1);

    // Only used for model.forward()
    std::vector<int> set_input(
            std::vector<int32_t> &inputIds_, int batchSize_, std::vector<int> seqIDs, const std::vector<int> &maxLen);

    // Only used for model.forward()
    std::vector<int> set_input(std::vector<int32_t> &inputIds_, std::vector<int32_t> &seqLens_, std::vector<int> seqIDs,
            std::vector<int> &maxLen);

    bool isDone();

    std::tuple<float *, int, int> forward(bool logitsAll = true);

    std::vector<int32_t> generate();

    void createSearcher(SearcherConfig &config_);

    bool isMaster();

    int getRank();

    int getBatchSize() { return batchSize; }

    int getSeqLen() { return seqLen; }

    void setVocabSize(int vocabSize) { this->vocabSize = vocabSize; }

    int getVocabSize() { return this->vocabSize; }

    void initMaxSeqLen();

    int getMaxSeqLen() { return maxSeqLen; }

    SearcherConfig getConfig() { return configuration; }

    void setDecoder(AbstractDecoder *dec);

    std::vector<int32_t> finalize();

    void exitSlaves();

    void setPrefix(std::vector<int32_t> &prefixIDs);

    void unsetPrefix();

    bool setStopWords(std::vector<std::vector<int>> stopWordsList);

    bool freeSeqs(std::vector<int> &seqIDs);

private:
    AbstractDecoder *decoder;
    AbstractSearcher *searcher;
    std::vector<int32_t> inputIds;
    int batchSize;
    int seqLen;
    int vocabSize;
    int maxSeqLen;
    SearcherConfig configuration;
    bool isNewInput;
    std::vector<SequenceGroupMeta *> workingGroup;
    std::vector<float> logits;
    std::vector<float> logitsRecvBuf;
};

class AutoModel : public Model {
public:
    AutoModel(std::string modelPath, xft::DataType dataType, xft::DataType KVCacheDataType = xft::DataType::fp16);
};
} // namespace xft