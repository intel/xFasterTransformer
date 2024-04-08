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

#include <cmath>
#include <random>
#include "abstract_decoder.h"
#include "abstract_searcher.h"
#include "messenger.h"
#include "timeline.h"
#include "transformer_ctx.h"

class SampleSearch : public AbstractSearcher {
public:
    SampleSearch(AbstractDecoder &dec, const SearcherConfig &config);

    // Get next tokens accoring to the prompt IDs
    std::vector<int> getNextToken(int *ids, int batchSize, int seqLen);

    // Get next tokens according to previous predicted ID
    std::vector<int> getNextToken();

    bool isDone();

    std::vector<int32_t> finalize();

    bool setStopWords(std::vector<std::vector<int>> stopWordsList);

private:
    void sample(std::tuple<float *, int, int> &result);

    AbstractDecoder &decoder;

    // Predicted token IDs
    std::vector<int> nextTokens;
    std::vector<int> output;
    std::vector<std::vector<int>> cachedRepetVec;
    std::vector<int> doneBatch;

    int batchSize;
    int step;
    int curLen;
    int maxLen;
    int vocabSize;
    int eosTokenId;
    int padTokenId;
    int topK;
    float topP;
    float temperatureInv;
    float repetitionPenalty;
    std::vector<std::vector<int>> stopWordsList;
    std::vector<std::vector<int>> stopWordsIndex;
};