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
#include "sample_search.h"
#include "decoder_util.h"
#include "search_utils.h"

SampleSearch::SampleSearch(AbstractDecoder &dec, const SearcherConfig &config)
    : decoder(dec)
    , maxLen(config.maxLen)
    , topK(config.topK)
    , topP(config.topP)
    , repetitionPenalty(config.repetitionPenalty) {
    vocabSize = decoder.getContext()->vocabSize;
    eosTokenId = config.eosTokenId == -1 ? decoder.getEndId() : config.eosTokenId;
    padTokenId = config.padTokenId == -1 ? eosTokenId : config.padTokenId;
    if (config.temperature <= 0) {
        printf("Temperature should greater than 0.\n");
        exit(-1);
    }
    temperatureInv = 1 / config.temperature;
    if (topK < 2) { topK = 2; }

    if (repetitionPenalty <= 0) {
        printf("`repetitionPenalty` has to be a strictly positive float, but is %f.\n", repetitionPenalty);
        exit(-1);
    }
    stopWordsList = {};
    stopWordsIndex = {};
}

// Get next tokens accoring to the prompt IDs
std::vector<int> SampleSearch::getNextToken(int *ids, int batchSize, int seqLen) {
    TimeLine t("1st Token");
    this->step = 0;
    this->batchSize = batchSize;
    this->curLen = seqLen;
    this->doneBatch = std::vector<int>(batchSize, 0);

    if (!this->stopWordsList.empty()) {
        stopWordsIndex = std::vector<std::vector<int>>(stopWordsList.size(), std::vector<int>(batchSize, 0));
    }

    this->output.resize(batchSize * seqLen);
    std::copy(ids, ids + batchSize * seqLen, output.begin());

    int64_t dims[3] = {batchSize, 1, seqLen};

    std::tuple<float *, int, int> result = decoder.forward(ids, dims, this->step++);

    nextTokens.resize(batchSize);
    // TODO: Add PIPELINE_PARALLEL feature
    sample(result);

    this->curLen++;
    for (int batchId = 0; batchId < batchSize; ++batchId) {
        output.insert(output.begin() + (batchId + 1) * curLen - 1, nextTokens[batchId]);
    }

    return this->nextTokens;
}

// Get next tokens according to previous predicted ID
std::vector<int> SampleSearch::getNextToken() {
    TimeLine t("Next Token");
    int64_t dims[3] = {batchSize, 1, 1};
    std::tuple<float *, int, int> result = decoder.forward(nextTokens.data(), dims, this->step++);

    // TODO: Add PIPELINE_PARALLEL feature
    sample(result);

    this->curLen++;
    for (int batchId = 0; batchId < batchSize; ++batchId) {
        output.insert(output.begin() + (batchId + 1) * curLen - 1, nextTokens[batchId]);
    }

    return this->nextTokens;
}

bool SampleSearch::isDone() {
    if (step == 0) {
        return false;
    } else if (curLen >= maxLen) {
        return true;
    } else {
        for (auto flag : doneBatch) {
            if (flag <= 0) { return false; }
        }
    }
    return true;
}

std::vector<int32_t> SampleSearch::finalize() {
    TimeLine t("dumpFile");
    t.dumpFile("timeline.json");
    return output;
}

bool SampleSearch::setStopWords(std::vector<std::vector<int>> stopWordsList) {
    this->stopWordsList = stopWordsList;
    for (auto it = this->stopWordsList.rbegin(); it != this->stopWordsList.rend(); ++it) {
        if ((*it).size() == 1 && (*it)[0] == this->eosTokenId) { this->stopWordsList.erase(std::next(it).base()); }
    }
    return !this->stopWordsList.empty();
}

void SampleSearch::sample(std::tuple<float *, int, int> &result) {
    TimeLine t("Sample.searchTop");
    float *outBuf = std::get<0>(result);
    int sampleOffset = std::get<1>(result);
    int sampleSize = std::get<2>(result);

    Messenger &messenger = decoder.getMessenger();
    auto msgerSize = messenger.getSize();

    // Repetition penalty logits processor
    if (this->repetitionPenalty != 1.0) {
        TimeLine t("GreedySearch.repetitionPenalty");
        // step has already been incremented by 1
        if (this->step == 1) {
            this->cachedRepetVec.clear();
            this->cachedRepetVec.resize(batchSize, std::vector<int>());

            repetitionPenaltyLogitsProcess(this->repetitionPenalty, outBuf, sampleOffset, sampleSize, this->output,
                    batchSize, this->cachedRepetVec, this->step, msgerSize > 1);
        } else {
            repetitionPenaltyLogitsProcess(this->repetitionPenalty, outBuf, sampleOffset, sampleSize, this->nextTokens,
                    batchSize, this->cachedRepetVec, this->step, msgerSize > 1);
        }
    }

    // 1. Get top K candidates for each sample, inculde topK ids and vals
    int topKIds[batchSize * topK];
    float topKVals[batchSize * topK];

    // Do topK during splited outputs on each sample
    // Each thread handle one sample (one row)
#pragma omp parallel for
    for (int i = 0; i < batchSize; ++i) {
        std::vector<std::pair<float, int>> elements(sampleSize);
        for (int j = 0; j < sampleSize; j++) {
            elements[j] = std::make_pair(*(outBuf + i * sampleSize + j), j);
        }
        std::partial_sort(
                elements.begin(), elements.begin() + topK, elements.end(), std::greater<std::pair<float, int>>());

        for (int j = 0; j < topK; ++j) {
            topKVals[i * topK + j] = elements[j].first;
            topKIds[i * topK + j] = elements[j].second + sampleOffset;
        }
    }

    // Reduce to get the topK index during all batch's sample
    if (msgerSize > 1) {
        float sendBuf[2 * batchSize * topK];
        float recvBuf[2 * batchSize * topK * msgerSize];

        for (int i = 0; i < batchSize * topK; ++i) {
            sendBuf[2 * i] = (float)topKIds[i];
            sendBuf[2 * i + 1] = topKVals[i];
        }

        size_t sendSize = 2 * batchSize * topK;
        std::vector<long unsigned int> recvCount(msgerSize, static_cast<long unsigned int>(sendSize));
        messenger.allgatherv(sendBuf, sendSize, recvBuf, recvCount);

#pragma omp parallel for
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            std::vector<std::pair<float, int>> elements(msgerSize * topK);
            for (int i = 0; i < msgerSize; ++i) {
                int idx = (i * batchSize + batchIdx) * topK;
                for (int k = 0; k < topK; ++k) {
                    elements[i * topK + k]
                            = std::make_pair(recvBuf[2 * (idx + k) + 1], (int)(recvBuf[2 * (idx + k)] + 0.5f));
                }
            }
            // Take the top K
            std::sort(elements.begin(), elements.end(), std::greater<std::pair<float, int>>());

            for (int i = 0; i < topK; i++) {
                topKVals[batchIdx * topK + i] = elements[i].first;
                topKIds[batchIdx * topK + i] = elements[i].second;
            }
        }
    }

    // 2. Divided by temperature.
    if (temperatureInv != 1.0) {
        TimeLine t("Sample.temperature");
        for (int i = 0; i < batchSize * topK; i++) {
            topKVals[i] *= temperatureInv;
        }
    }
    // 3. Get topP candidates, at least 2.
    std::vector<int> topPNums(batchSize, topK);
    if (topP < 1.0) {
#pragma omp parallel for
        for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            float probs[topK];
            float probs_sum = 0;
            float cursum = 0;
            for (int i = 0; i < topK; i++) {
                probs[i] = exp(topKVals[batchIdx * topK + i]);
                probs_sum += probs[i];
            }
            float probs_sum_inv = 1 / probs_sum;
            for (int i = 0; i < topK; i++) {
                cursum += probs[i] * probs_sum_inv;
                if (cursum > topP) {
                    topPNums[batchIdx] = i > 2 ? i : 2;
                    break;
                }
            }
        }
    }

    // 4. Sample.
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
#pragma omp parallel for
    for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        float cursum = 0;
        float randomValue = distribution(generator);

        // Calculate softmax
        DecoderUtil::computeSoftmax(topKVals + batchIdx * topK, topPNums[batchIdx]);

        for (int i = 0; i < topPNums[batchIdx]; i++) {
            cursum += topKVals[batchIdx * topK + i];
            if (cursum >= randomValue) {
                nextTokens[batchIdx] = topKIds[batchIdx * topK + i];
                break;
            }
        }
    }

    if (msgerSize > 1) { messenger.broadcast(nextTokens.data(), nextTokens.size()); }

    if (eosTokenId != -1) {
        for (int batchId = 0; batchId < batchSize; ++batchId) {
            if (doneBatch[batchId] == 0) {
                if (nextTokens[batchId] == eosTokenId) { doneBatch[batchId] = 1; }
            } else if (doneBatch[batchId] > 0) {
                // Padding finished seq with padTokenId;
                nextTokens[batchId] = padTokenId;
            } else if (doneBatch[batchId] < 0) {
                // Set to eosTokenId as really done;
                nextTokens[batchId] = eosTokenId;
                doneBatch[batchId] = 1;
            }
        }
    }

    if (!this->stopWordsList.empty() && !this->stopWordsIndex.empty()) {
        stopWordsCheck(nextTokens, this->stopWordsList, this->stopWordsIndex, this->doneBatch);
    }
};
