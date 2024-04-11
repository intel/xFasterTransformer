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
#include "sampling.h"

namespace xft {
bool isDone(SearchContext &ctx) {
    if (ctx.step == 0) {
        return false;
    } else if (ctx.seqLen >= ctx.config.maxLen) {
        return true;
    } else {
        for (auto flag : ctx.doneBatch) {
            if (flag <= 0) { return false; }
        }
    }
    return true;
}

std::vector<int> finalize(SearchContext &ctx) {
    TimeLine t("dumpFile");
    t.dumpFile("timeline.json");
    return ctx.promptIds;
}

std::vector<int> greedySearch(float *logits, int sampleOffset, int sampleSize, SearchContext &searchCtx,
        DecoderContext &decoderCtx, Messenger &messenger) {
    TimeLine t("GreedySearch");

    auto msgerSize = messenger.getSize();
    int &batchSize = searchCtx.batchSize;
    int &step = searchCtx.step;

    // Repetition penalty logits processor
    if (searchCtx.config.repetitionPenalty != 1.0) {
        TimeLine t("GreedySearch.repetitionPenalty");
        // step has already been incremented by 1
        if (searchCtx.step == 1) {
            searchCtx.cachedRepetVec.clear();
            searchCtx.cachedRepetVec.resize(batchSize, std::vector<int>());

            repetitionPenaltyLogitsProcess(searchCtx.config.repetitionPenalty, logits, sampleOffset, sampleSize,
                    searchCtx.promptIds, batchSize, searchCtx.cachedRepetVec, step, msgerSize > 1);
        } else {
            repetitionPenaltyLogitsProcess(searchCtx.config.repetitionPenalty, logits, sampleOffset, sampleSize,
                    searchCtx.nextTokens, batchSize, searchCtx.cachedRepetVec, step, msgerSize > 1);
        }
    }

    // Max ID and value for each sample
    int maxIds[batchSize];
    float maxVals[batchSize];


    // Small batch size (each sample can have at least 2 threads)
    if (decoderCtx.numThreads / batchSize >= 2) {
        int thrPerSample = decoderCtx.numThreads / batchSize;
        int sizePerThr = (sampleSize + thrPerSample - 1) / thrPerSample;
        int maxIndices[batchSize * thrPerSample];
        float maxValues[batchSize * thrPerSample];

        // TODO: if size is small, possible to cause out of boundary
#pragma omp parallel for collapse(2)
        for (int b = 0; b < batchSize; ++b) {
            for (int t = 0; t < thrPerSample; ++t) { // thread index inside the sample
                int start = t * sizePerThr;
                int end = (start + sizePerThr) > sampleSize ? sampleSize : (start + sizePerThr);
                float *p = logits + b * sampleSize;

                int maxIdx = start;
                float maxVal = p[start];
                for (int off = start + 1; off < end; ++off) {
                    if (p[off] > maxVal) {
                        maxVal = p[off];
                        maxIdx = off;
                    }
                }

                // False sharing happens, but since only one time, not avoided
                maxIndices[b * thrPerSample + t] = maxIdx;
                maxValues[b * thrPerSample + t] = maxVal;
            }
        }

        // Local reduction
        for (int i = 0; i < batchSize; ++i) {
            int *pIndices = maxIndices + i * thrPerSample;
            float *pValues = maxValues + i * thrPerSample;
            int maxIdx = pIndices[0];
            float maxVal = pValues[0];
            for (int j = 1; j < thrPerSample; ++j) {
                if (pValues[j] > maxVal) {
                    maxVal = pValues[j];
                    maxIdx = pIndices[j];
                }
            }
            maxIds[i] = maxIdx;
            maxVals[i] = maxVal;
        }
    }

    // Each thread handle one sample (one row)
    else {
#pragma omp parallel for
        for (int i = 0; i < batchSize; ++i) {
            int maxId = 0;
            float *p = logits + i * sampleSize;
            float maxVal = p[0];
            for (int j = 1; j < sampleSize; ++j) {
                if (p[j] > maxVal) {
                    maxVal = p[j];
                    maxId = j;
                }
            }
            maxIds[i] = maxId;
            maxVals[i] = maxVal;
        }
    }

    // Reduce to get the max index (any better method??)
    if (msgerSize > 1) {
        float sendBuf[2 * batchSize];
        float recvBuf[2 * batchSize * msgerSize];

        for (int i = 0; i < batchSize; ++i) {
            sendBuf[2 * i] = (float)(maxIds[i] + sampleOffset);
            sendBuf[2 * i + 1] = maxVals[i];
        }

        std::vector<long unsigned int> recvCount(msgerSize, static_cast<long unsigned int>(2 * batchSize));
        messenger.allgatherv(sendBuf, 2 * batchSize, recvBuf, recvCount);

        for (int i = 0; i < batchSize; ++i) {
            int maxId = (int)(recvBuf[2 * i] + 0.5f);
            float maxVal = recvBuf[2 * i + 1];
            for (int j = 1; j < msgerSize; ++j) {
                if (recvBuf[2 * j * batchSize + 2 * i + 1] > maxVal) {
                    maxVal = recvBuf[2 * j * batchSize + 2 * i + 1];
                    maxId = (int)(recvBuf[2 * j * batchSize + 2 * i] + 0.5f);
                }
            }
            maxIds[i] = maxId;
        }
    }

    int &eosTokenId = searchCtx.config.eosTokenId;
    int &padTokenId = searchCtx.config.padTokenId;
    std::vector<int> &doneBatch = searchCtx.doneBatch;
    std::vector<std::vector<int>> &stopWordsList = searchCtx.stopWordsList;
    std::vector<std::vector<int>> &stopWordsIndex = searchCtx.stopWordsIndex;

    if (eosTokenId != -1) {
        for (int batchId = 0; batchId < batchSize; ++batchId) {
            if (doneBatch[batchId] == 0) {
                if (maxIds[batchId] == eosTokenId) { doneBatch[batchId] = 1; }
            } else if (doneBatch[batchId] > 0) {
                // Padding finished seq with padTokenId;
                maxIds[batchId] = padTokenId;
            } else if (doneBatch[batchId] < 0) {
                // Set to eosTokenId as really done;
                maxIds[batchId] = eosTokenId;
                doneBatch[batchId] = 1;
            }
        }
    }

    searchCtx.nextTokens = std::vector<int>(maxIds, maxIds + batchSize);
    if (!stopWordsList.empty() && !stopWordsIndex.empty()) {
        stopWordsCheck(searchCtx.nextTokens, stopWordsList, stopWordsIndex, doneBatch);
    }

    searchCtx.seqLen++;
    for (int batchId = 0; batchId < batchSize; ++batchId) {
        searchCtx.promptIds.insert(
                searchCtx.promptIds.begin() + (batchId + 1) * searchCtx.seqLen++ - 1, searchCtx.nextTokens[batchId]);
    }
    
    return searchCtx.nextTokens;
}
} // namespace xft