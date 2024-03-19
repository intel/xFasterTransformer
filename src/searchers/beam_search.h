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
/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#pragma once

#include "abstract_decoder.h"
#include "abstract_searcher.h"
#include "messenger.h"
#include "timeline.h"
#include "transformer_ctx.h"

class BeamHypotheses {
public:
    BeamHypotheses(int numBeams, int maxLen, float lenPenalty, bool earlyStopping);
    BeamHypotheses() {};

    void add(std::vector<int32_t> hyp, float sumLogProbs);
    bool isDone(float bestSumLogProbs, int curLen);
    int size() { return beams.size(); }

public:
    int maxLen;
    float lenPenalty;
    bool earlyStopping;
    int numBeams;
    float worstScore;
    std::vector<std::pair<float, std::vector<int32_t>>> beams;
};

/*******************************************************************************
Reference for the beam search algorithm and implementation in
[huggingface/transformers
implementation](https://github.com/huggingface/transformers/blob/v4.24-release/src/transformers/generation_beam_search.py)
*******************************************************************************/
class BeamSearchScorer {
public:
    int maxLen;
    int batchSize;
    int numBeams;
    float lenPenalty;
    bool doEarlyStopping;
    int numBeamHypsToKeep;
    std::vector<BeamHypotheses> beamHyps;
    std::vector<bool> doneBatch;

    BeamSearchScorer(int batchSize, int maxLen, int numBeams, float lenPenalty = 1.0, bool doEarlyStopping = false,
            int numBeamHypsToKeep = 1);
    BeamSearchScorer() {}

    bool isDone() const;

    std::tuple<std::vector<float>, std::vector<int32_t>, std::vector<int32_t>> process(std::vector<int32_t> &inputIds,
            std::vector<float> &nextScores, std::vector<int32_t> &nextTokens, std::vector<int32_t> &nextIndices,
            int padTokenId = -1, int eosTokenId = -1);

    std::vector<int32_t> finalize(std::vector<int32_t> &inputIds, std::vector<float> &finalBeamScores,
            std::vector<int32_t> &finalBeamTokens, std::vector<int32_t> &finalBeamIndices, int padTokenId = -1,
            int eosTokenId = -1);
};

class BeamSearch : public AbstractSearcher {
public:
    BeamSearch(AbstractDecoder &dec, const SearcherConfig &config);

    // The first setp to get next tokens accoring to the prompt IDs
    std::vector<int> getNextToken(int *ids, int batchSize, int seqLen);

    // Get next tokens according to previous predicted ID
    std::vector<int> getNextToken();

    std::vector<int> getNextTokens() { return beamNextTokens; }
    std::vector<float> getNextScores() { return beamNextScores; }
    std::vector<int> getNextIndices() { return beamNextIndices; }

    // Get current output sequence.
    std::vector<int32_t> finalize();

    bool isDone();

    bool setStopWords(std::vector<std::vector<int>> stopWordsList);

private:
    void searchTopK(std::tuple<float *, int, int> &result);

    void beam_search(std::tuple<float *, int, int> &result);

    AbstractDecoder &decoder;
    BeamSearchScorer beamScorer;

    std::vector<int32_t> inputIds;
    std::vector<float> beamNextScores;
    std::vector<float> nextScores;
    std::vector<int32_t> nextTokens;
    std::vector<int32_t> nextIndices;
    std::vector<int32_t> beamNextTokens;
    std::vector<int32_t> beamNextIndices;

    bool doEarlyStopping;
    int batchSize;
    int numBeams;
    int numBeamHypsToKeep;
    int kVal;
    int step;
    int curLen;
    int maxLen;
    int vocabSize;
    int padTokenId;
    int eosTokenId;
    float lenPenalty;
};