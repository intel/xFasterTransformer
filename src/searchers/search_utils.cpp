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
#include "messenger.h"
#include "search_utils.h"
#include "timeline.h"

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
        int startLogits = b * sampleSize;
        for (int index : cachedVec[b]) {
            float &logit = logits[startLogits + index];
            logit = logit < 0 ? logit * penalty : logit / penalty;
        }
    }
}

void stopWordsCheck(std::vector<int> &nextTokenIds, std::vector<std::vector<int>> &stopWordsList,
        std::vector<std::vector<int>> &stopWordsIndex, std::vector<int> &doneBatch) {
    //TODO: Enable OMP for large batch sizes or long word lists.
    for (int batchId = 0; batchId < nextTokenIds.size(); batchId++) {
        if (doneBatch[batchId] == 0) {
            for (int i = 0; i < stopWordsList.size(); i++) {
                auto &stopWords = stopWordsList[i];
                auto &wordsIndex = stopWordsIndex[i];
                auto stopWordsLen = stopWords.size();
                if (wordsIndex[batchId] < stopWordsLen) {
                    if (stopWords[wordsIndex[batchId]] == nextTokenIds[batchId]) {
                        wordsIndex[batchId]++;
                        if (wordsIndex[batchId] == stopWordsLen) { doneBatch[batchId] = -1; }
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

namespace xft {
// Assume all sequences are all prompts or decodes.
// TODO: support num_beams > 1 (beam search)
void repetitionPenaltyLogitsProcess(
        float *logits, int sampleOffset, int sampleSize, std::vector<SequenceGroupMeta *> &seqGroups) {
    TimeLine t("RepetitionPenaltyLogitsProcess");
    bool multiRank = Messenger::getInstance().getSize() > 1;

    std::vector<int> groupIndex;
    // TODO: Num_beam > 1 (beam search)
    int batchSize = seqGroups.size();

    // Assume all seqences are all prompts or decodes.
    int step = seqGroups[0]->getStep();

    // For prompts
    if (step == 0) {
#pragma omp parallel for
        for (int b = 0; b < batchSize; b++) {
            if (seqGroups[b]->getSamplingMeta()->config.repetitionPenalty == 1.0) { continue; }
            SequenceMeta *seqMeta = seqGroups[b]->get(0);
            std::vector<int> &cachedVec = seqGroups[b]->getSamplingMeta()->cachedRepetVec;
            cachedVec = seqMeta->getPromptTokens();
            std::sort(cachedVec.begin(), cachedVec.end());
            cachedVec.erase(std::unique(cachedVec.begin(), cachedVec.end()), cachedVec.end());

            if (multiRank) {
                // Get (sampleOffset, sampleOffset + sampleSize)
                auto boundBegin = std::upper_bound(cachedVec.begin(), cachedVec.end(), sampleOffset);
                auto boundEnd = std::lower_bound(cachedVec.begin(), cachedVec.end(), sampleOffset + sampleSize);

                cachedVec.erase(boundEnd, cachedVec.end());
                cachedVec.erase(cachedVec.begin(), boundBegin);

                std::transform(cachedVec.begin(), cachedVec.end(), cachedVec.begin(),
                        [sampleOffset](int num) { return num - sampleOffset; });
            }
        }
    } else {
        if (multiRank) {
#pragma omp parallel for
            for (int b = 0; b < batchSize; b++) {
                if (seqGroups[b]->getSamplingMeta()->config.repetitionPenalty == 1.0) { continue; }
                std::vector<int> inputIds = seqGroups[b]->get(0)->getInputTokens();
                for (auto x : inputIds) {
                    if (x >= sampleOffset && x < sampleOffset + sampleSize) {
                        insertAndSort(seqGroups[b]->getSamplingMeta()->cachedRepetVec, x - sampleOffset);
                    }
                }
            }
        } else {
#pragma omp parallel for
            for (int b = 0; b < batchSize; b++) {
                if (seqGroups[b]->getSamplingMeta()->config.repetitionPenalty == 1.0) { continue; }
                std::vector<int> inputIds = seqGroups[b]->get(0)->getInputTokens();
                for (auto x : inputIds) {
                    insertAndSort(seqGroups[b]->getSamplingMeta()->cachedRepetVec, x);
                }
            }
        }
    }

#pragma omp parallel for
    for (int b = 0; b < batchSize; b++) {
        if (seqGroups[b]->getSamplingMeta()->config.repetitionPenalty == 1.0) { continue; }
        int startLogits = b * sampleSize;
        auto &penalty = seqGroups[b]->getSamplingMeta()->config.repetitionPenalty;
        for (int index : seqGroups[b]->getSamplingMeta()->cachedRepetVec) {
            float &logit = logits[startLogits + index];
            logit = logit < 0 ? logit * penalty : logit / penalty;
        }
    }
}
} // namespace xft