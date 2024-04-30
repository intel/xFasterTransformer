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

class AttnMetaData {

public:
    AttnMetaData () : batchSize(0), attnMask(nullptr) {}

    AttnMetaData (int batchSize, int *inputTokenSizes, int *pastTokenSizes, bool isPrompt, bool isCausal, float *attnMask = nullptr)
        : batchSize(batchSize), isPrompt(isPrompt), isCausal(isCausal), attnMask(attnMask) {
        // causal=True, no need mask
        assert(isCausal && attnMask == nullptr)
        // causal=False, need mask
        assert(!isCausal && attnMask)

        // fill inputSeqLens, pastSeqLens, seqStartLoc
        inputSeqLens.resize(batchSize);
        pastSeqLens.resize(batchSize);
        seqStartLoc.resize(batchSize + 1);

        seqStartLoc[0] = 0;
        for (int i = 0; i < batchSize; i++) {
            inputSeqLens[i] = inputTokenSizes[i];
            pastSeqLens[i] = pastTokenSizes[i];
            seqStartLoc[i + 1] = seqStartLoc[i] + inputSeqLens[i];
        }

    AttnMetaData (vector<int> &inputTokens, vector<int> &pastTokens, bool isPrompt, bool isCausal, float *attnMask = nullptr)
        : batchSize(inputTokenSizes.size()), isPrompt(isPrompt), isCausal(isCausal), attnMask(attnMask), 
            inputSeqLens(inputTokenSizes), pastSeqLens(pastTokenSizes){
        // causal=True, no need mask
        assert(isCausal && attnMask == nullptr)
        // causal=False, need mask
        assert(!isCausal && attnMask)

        // fill seqStartLoc
        seqStartLoc.resize(batchSize + 1);

        seqStartLoc[0] = 0;
        for (int i = 0; i < batchSize; i++) {
            seqStartLoc[i + 1] = seqStartLoc[i] + inputSeqLens[i];
        }

    }

private:
    bool isPrompt;
    bool isCausal;

    int batchSize;
    std::vector<int> inputSeqLens;
    std::vector<int> pastSeqLens;
    std::vector<int> seqStartLoc;

    float *attnMask;

};
