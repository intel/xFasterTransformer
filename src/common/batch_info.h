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
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace xft {

class BatchInfo {

public:
    // Constructor for first token
    BatchInfo(size_t *inputTokens, size_t batchSize) : batchSize(batchSize), allGenPhase(false) {
        endTokenIdxs.resize(batchSize);
        pastTokens.resize(batchSize);

        size_t endIdx = 0;
        for (size_t i = 0; i < batchSize; ++i) {
            endIdx += inputTokens[i];
            endTokenIdxs[i] = endIdx;
        }

        std::fill(pastTokens.begin(), pastTokens.end(), 0);
    }

    // Constructor for next tokens
    BatchInfo(size_t *inputTokens, size_t *pastSeqs, size_t batchSize) : batchSize(batchSize) {
        endTokenIdxs.resize(batchSize);
        pastTokens.resize(batchSize);

        if (inputTokens == nullptr) {
            std::iota(endTokenIdxs.begin(), endTokenIdxs.end(), 1);
            allGenPhase = true;
        } else {
            bool allGenPhase = true;
            size_t endIdx = 0;
            for (size_t i = 0; i < batchSize; ++i) {
                endIdx += inputTokens[i];
                endTokenIdxs[i] = endIdx;
                if (inputTokens[i] != 1) { allGenPhase = false; }
            }
        }

        // Assign pastSeqs to pastTokens
        for (size_t i = 0; i < batchSize; ++i) {
            pastTokens[i] = pastSeqs[i];
        }
    }

    // Get the batch size
    size_t getBatchSize() const { return batchSize; }

    // Get the input token size for a given sample index
    size_t getInputTokenSize(size_t sampleIdx) const {
        if (allGenPhase) { return 1; }
        return endTokenIdxs[sampleIdx] - (sampleIdx == 0 ? 0 : endTokenIdxs[sampleIdx - 1]);
    }

    // Get the past token size for a given sample index
    size_t getPastTokenSize(size_t sampleIdx) const { return pastTokens[sampleIdx]; }

    // Get the total number of tokens inside the batch
    size_t getTotalTokens() const {
        return endTokenIdxs[batchSize - 1];
    }

    // Get the sample index and token index inside the sample for a given token index
    // The input is not checked; the caller needs to ensure its correctness
    std::pair<size_t, size_t> getInfo(size_t tokenIdx) const {
        // If all samples are in the generation phase, the token index is the sample index
        if (allGenPhase) { return {tokenIdx, 0}; }

        size_t sampleIdx = -1;
        for (size_t i = 0; i < batchSize; ++i) {
            if (tokenIdx < endTokenIdxs[i]) {
                sampleIdx = i;
                break;
            }
        }
        if (sampleIdx >= 0) {
            size_t startIdx = sampleIdx == 0 ? 0 : endTokenIdxs[sampleIdx - 1];
            return {sampleIdx, tokenIdx - startIdx};
        }
        return {-1, -1};
    }

private:
    size_t batchSize;

    // End token offset for each sample inside the batch
    std::vector<size_t> endTokenIdxs;

    // Past token size for each sample inside the batch
    std::vector<size_t> pastTokens;

    // Indicates whether all samples inside the batch are in the generation phase
    // This means that all samples have an input token size of 1
    bool allGenPhase;
};

} // namespace xft