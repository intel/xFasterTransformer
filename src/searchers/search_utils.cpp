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
#include <algorithm>
#include <cassert>
#include <vector>
#include "search_utils.h"

// Insert an element into a sorted vector while maintaining the order
void insertAndSort(std::vector<int> &targetVector, int num) {
    auto it = std::lower_bound(targetVector.begin(), targetVector.end(), num);
    if (it == targetVector.end() || *it != num) { targetVector.insert(it, num); }
}

void repetitionPenaltyLogitsProcess(float penalty, float *logits, int sampleOffset, int sampleSize,
        std::vector<int> &inputIds, int batchSize, std::vector<std::vector<int>> &cachedVec, int step, bool multiRank) {
    int seqLen = inputIds.size() / batchSize;
    if (step != 1) { assert(seqLen == 1); }

    if (step == 1) {
#pragma omp parallel for
        for (int b = 0; b < batchSize; b++) {
            cachedVec[b].clear();
            cachedVec[b].insert(cachedVec[b].end(), inputIds.begin() + b * seqLen, inputIds.begin() + (b + 1) * seqLen);
            std::sort(cachedVec[b].begin(), cachedVec[b].end());
            cachedVec[b].erase(std::unique(cachedVec[b].begin(), cachedVec[b].end()), cachedVec[b].end());

            if (multiRank) {
                // Get (sampleOffset, sampleOffset + sampleSize)
                auto boundBegin = std::upper_bound(cachedVec[b].begin(), cachedVec[b].end(), sampleOffset);
                auto boundEnd = std::lower_bound(cachedVec[b].begin(), cachedVec[b].end(), sampleOffset + sampleSize);

                cachedVec[b].erase(boundEnd, cachedVec[b].end());
                cachedVec[b].erase(cachedVec[b].begin(), boundBegin);

                std::transform(cachedVec[b].begin(), cachedVec[b].end(), cachedVec[b].begin(),
                        [sampleOffset](int num) { return num - sampleOffset; });
            }
        }
    } else {
        if (multiRank) {
            for (int b = 0; b < batchSize; b++) {
                if (inputIds[b] >= sampleOffset && inputIds[b] < sampleOffset + sampleSize) {
                    insertAndSort(cachedVec[b], inputIds[b] - sampleOffset);
                }
            }
        } else {
            for (int b = 0; b < batchSize; b++) {
                insertAndSort(cachedVec[b], inputIds[b]);
            }
        }
    }

#pragma omp parallel for
    for (int b = 0; b < batchSize; b++) {
        for (int index : cachedVec[b]) {
            logits[index] = logits[index] < 0 ? logits[index] * penalty : logits[index] / penalty;
        }
    }
}

void stopWordsCheck(std::vector<int> &nextTokenIds, std::vector<std::vector<int>> &stopWordsList,
        std::vector<std::vector<int>> &stopWordsIndex, std::vector<int> &doneBatch) {
    for (int batchId = 0; batchId < nextTokenIds.size(); batchId++) {
        if (doneBatch[batchId] == 0) {
            for (int i = 0; i < stopWordsList.size(); i++) {
                auto &stopWords = stopWordsList[i];
                auto &wordsIndex = stopWordsIndex[i];
                auto stopWordsLen = stopWords.size();
                if (wordsIndex[batchId] < stopWordsLen) {
                    if (stopWords[wordsIndex[batchId]] == nextTokenIds[batchId]) {
                        wordsIndex[batchId]++;
                    } else {
                        wordsIndex[batchId] = 0;
                    }
                } else {
                    doneBatch[batchId] = -1;
                }
            }
        }
    }
}