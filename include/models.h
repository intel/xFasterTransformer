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
    Model() : decoder(nullptr), searcher(nullptr), isNewInput(true) {}
    ~Model();

    void input(std::vector<int32_t> &inputIds_, int batchSize_);

    void config(int maxLen_ = -1, int numBeams_ = 1, int numBeamHypsToKeep_ = 1, float lenPenalty_ = 1.0,
            bool doEarlyStopping_ = false, int eosTokenId_ = -1, int padTokenId_ = -1);

    bool isDone();

    std::vector<int32_t> generate();

    void createSearcher(SearcherConfig &config_);

    int getRank();

    int getBatchSize() { return batchSize; }

    int getSeqLen() { return seqLen; }

    SearcherConfig getConfig() { return configuration; }

    void setDecoder(AbstractDecoder *dec);

    std::vector<int32_t> finalize() { return searcher->finalize(); }

    void exitSlaves();

private:
    AbstractDecoder *decoder;
    AbstractSearcher *searcher;
    std::vector<int32_t> inputIds;
    int batchSize;
    int seqLen;
    SearcherConfig configuration;
    bool isNewInput;
};

class AutoModel : public Model {
public:
    AutoModel(std::string modelPath, xft::DataType datatype);
};
} // namespace xft