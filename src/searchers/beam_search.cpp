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
#include "beam_search.h"

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "bert_util.h"

static void computeLogSoftmax(float *input, int size) {
    // Get max valute
    float max = std::numeric_limits<float>::lowest();
    __m512 vmax = _mm512_set1_ps(max);

    for (int off = 0; off < size; off += 16) {
        int remain = size - off;
        __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

        __m512 vx = _mm512_maskz_loadu_ps(mask, input + off);
        vmax = _mm512_mask_max_ps(vmax, mask, vmax, vx);
    }
    max = _mm512_reduce_max_ps(vmax);
    vmax = _mm512_set1_ps(max);

    // Compute vexp(vx - vmax) and sum it
    __m512 vsum = _mm512_set1_ps(0);
    for (int off = 0; off < size; off += 16) {
        int remain = size - off;
        __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

        __m512 vx = _mm512_maskz_loadu_ps(mask, input + off);
        vx = _mm512_mask_sub_ps(vx, mask, vx, vmax);
        vx = BertUtil::vexp(vx);

        vsum = _mm512_mask_add_ps(vsum, mask, vsum, vx);
    }

    float sum = _mm512_reduce_add_ps(vsum);
    float logsum = std::log(sum);
    __m512 vsub = _mm512_set1_ps(max + logsum);

    // Compute vx - max - logsum and store
    for (int off = 0; off < size; off += 16) {
        int remain = size - off;
        __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

        __m512 vx = _mm512_maskz_loadu_ps(mask, input + off);
        vx = _mm512_mask_sub_ps(vx, mask, vx, vsub);
        _mm512_mask_storeu_ps(input + off, mask, vx);
    }
}

bool tupleCompare(const std::tuple<float, int, int> &a, const std::tuple<float, int, int> &b) {
    if (std::get<0>(a) > std::get<0>(b)) {
        return true;
    } else if (std::get<0>(a) == std::get<0>(b) && std::get<2>(a) < std::get<2>(b)) {
        return true;
    } else {
        return false;
    }
}

bool uniqueCompare(const std::tuple<float, int, int> &a, const std::tuple<float, int, int> &b) {
    return (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) == std::get<1>(b));
}

BeamHypotheses::BeamHypotheses(int numBeams, int maxLen, float lenPenalty, bool earlyStopping)
    : maxLen(maxLen - 1), lenPenalty(lenPenalty), earlyStopping(earlyStopping), numBeams(numBeams), worstScore(1e9) {}

void BeamHypotheses::add(std::vector<int32_t> hyp, float sumLogProbs) {
    float score = sumLogProbs / pow(hyp.size(), lenPenalty);

    if (beams.size() < numBeams || score > worstScore) {
        beams.push_back(std::make_pair(score, hyp));
        if (beams.size() > numBeams) {
            // Sort the vector based on the first element (float)
            std::sort(beams.begin(), beams.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
            beams.erase(beams.begin());
            worstScore = beams[0].first;
        } else {
            worstScore = std::min(score, worstScore);
        }
    }
}

bool BeamHypotheses::isDone(float bestSumLogProbs, int curLen) {
    if (beams.size() < numBeams) {
        return false;
    } else if (earlyStopping) {
        return true;
    } else {
        float curScore = bestSumLogProbs / pow(curLen, lenPenalty);
        return worstScore >= curScore;
    }
}

BeamSearchScorer::BeamSearchScorer(
        int batchSize, int maxLen, int numBeams, float lenPenalty, bool doEarlyStopping, int numBeamHypsToKeep)
    : maxLen(maxLen)
    , batchSize(batchSize)
    , numBeams(numBeams)
    , lenPenalty(lenPenalty)
    , doEarlyStopping(doEarlyStopping)
    , numBeamHypsToKeep(numBeamHypsToKeep)
    , beamHyps(batchSize, BeamHypotheses(numBeams, maxLen, lenPenalty, doEarlyStopping))
    , doneBatch(batchSize, false) {
    if (numBeams <= 1) {
        printf("numBeams has to be an integer strictly greater than 1\n");
        exit(-1);
    }
}

bool BeamSearchScorer::isDone() const {
    for (bool flag : doneBatch) {
        if (!flag) { return false; }
    }
    return true;
}

std::tuple<std::vector<float>, std::vector<int32_t>, std::vector<int32_t>> BeamSearchScorer::process(
        std::vector<int32_t> &inputIds, std::vector<float> &nextScores, std::vector<int32_t> &nextTokens,
        std::vector<int32_t> &nextIndices, int padTokenId, int eosTokenId) {
    int shapeSize = batchSize * numBeams;
    // TODO: check inputIds size is batchSize*numBeams*seqLen
    int curLen = inputIds.size() / shapeSize;

    std::vector<float> nextBeamScores(shapeSize, 0.0f);
    std::vector<int32_t> nextBeamTokens(shapeSize, padTokenId);
    std::vector<int32_t> nextBeamIndices(shapeSize, 0);

    for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        if (doneBatch[batchIdx]) {
            assert(beamHyps[batchIdx].size() >= numBeams);
            assert(eosTokenId != -1 && padTokenId != -1);
            // Pad the batch
            continue;
        }

        int beamIdx = 0;

        int rankNum = nextTokens.size() / batchSize; // K in topK
        for (int beamTokenRank = 0; beamTokenRank < rankNum; beamTokenRank++) {
            auto nextToken = nextTokens[batchIdx * rankNum + beamTokenRank];
            auto nextScore = nextScores[batchIdx * rankNum + beamTokenRank];
            auto nextIndex = nextIndices[batchIdx * rankNum + beamTokenRank];
            int batchBeamIdx = batchIdx * numBeams + nextIndex;

            if (eosTokenId != -1 && nextToken == eosTokenId) {
                bool isBeamTokenWorseThanTopNumBeams = beamTokenRank >= numBeams;
                if (isBeamTokenWorseThanTopNumBeams) { continue; }

                beamHyps[batchIdx].add(std::vector<int32_t>(inputIds.begin() + batchBeamIdx * curLen,
                                               inputIds.begin() + (batchBeamIdx + 1) * curLen),
                        nextScore);
            } else {
                nextBeamScores[batchIdx * numBeams + beamIdx] = nextScore;
                nextBeamTokens[batchIdx * numBeams + beamIdx] = nextToken;
                nextBeamIndices[batchIdx * numBeams + beamIdx] = batchBeamIdx;
                beamIdx++;

                if (beamIdx == numBeams) { break; }
            }
        }

        if (beamIdx < numBeams) {
            printf("At most tokens can be equal to `eosTokenId`\n");
            exit(-1);
        }

        doneBatch[batchIdx] = doneBatch[batchIdx]
                || beamHyps[batchIdx].isDone(*std::max_element(nextScores.begin() + batchIdx * numBeams,
                                                     nextScores.begin() + (batchIdx + 1) * numBeams),
                        curLen);
    }

    return std::make_tuple(nextBeamScores, nextBeamTokens, nextBeamIndices);
}

std::vector<int32_t> BeamSearchScorer::finalize(std::vector<int32_t> &inputIds, std::vector<float> &finalBeamScores,
        std::vector<int32_t> &finalBeamTokens, std::vector<int32_t> &finalBeamIndices, int padTokenId, int eosTokenId) {
    int curLen = inputIds.size() / (batchSize * numBeams);

    // Finalize all open beam hypotheses and add them to generated
    // hypotheses
    for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        if (doneBatch[batchIdx]) { continue; }

        // Need to add the best numBeams hypotheses to generated hypotheses
        for (int beamID = 0; beamID < numBeams; beamID++) {
            int batchBeamIdx = batchIdx * numBeams + beamID;
            float final_score = finalBeamScores[batchBeamIdx];
            std::vector<int32_t> final_tokens(
                    inputIds.begin() + batchBeamIdx * curLen, inputIds.begin() + (batchBeamIdx + 1) * curLen);
            beamHyps[batchIdx].add(final_tokens, final_score);
        }
    }

    // Select the best hypotheses
    int shapeSize = batchSize * numBeamHypsToKeep;
    std::vector<int32_t> sentLens(shapeSize);
    std::vector<int32_t> bestScores(shapeSize);
    std::vector<std::vector<int32_t>> best;

    // Retrieve the best hypotheses
    for (int i = 0; i < batchSize; i++) {
        std::vector<std::pair<float, std::vector<int32_t>>> sortedHyps = beamHyps[i].beams;

        std::sort(sortedHyps.begin(), sortedHyps.end(), [](const auto &x, const auto &y) { return x.first < y.first; });

        for (int j = 0; j < numBeamHypsToKeep; ++j) {
            std::pair<float, std::vector<int32_t>> bestHypTuple = sortedHyps.back();
            sortedHyps.pop_back();
            float bestScore = bestHypTuple.first;
            std::vector<int32_t> bestHyp = bestHypTuple.second;

            sentLens[i * numBeamHypsToKeep + j] = bestHyp.size();

            // Append to lists
            best.push_back(bestHyp);
            bestScores[i * numBeamHypsToKeep + j] = bestScore;
        }
    }

    // Prepare for adding eos
    int sentLensMax = *std::max_element(sentLens.begin(), sentLens.end());
    int sentLensMin = *std::min_element(sentLens.begin(), sentLens.end());
    int sentMaxLen = std::min(sentLensMax + 1, maxLen);
    shapeSize = batchSize * numBeamHypsToKeep * sentMaxLen;

    std::vector<int32_t> decoded(shapeSize, padTokenId);
    // Shorter batches are padded if needed
    if (sentLensMin < sentLensMax) {
        assert(padTokenId != -1); // `padTokenId` has to be defined
    }

    // Fill with hypotheses and eosTokenId if the latter fits in
    for (int i = 0; i < best.size(); i++) {
        int hypLen = sentLens[i];
        auto hypo = best[i];
        if (hypLen < maxLen) {
            for (int j = 0; j < hypLen; j++)
                decoded[i * sentMaxLen + j] = hypo[j];
            decoded[i * sentMaxLen + hypLen] = eosTokenId;
        } else {
            for (int j = 0; j < maxLen; j++)
                decoded[i * sentMaxLen + j] = hypo[j];
        }
    }

    return decoded;
}

BeamSearch::BeamSearch(AbstractDecoder &dec, const SearcherConfig &config)
    : decoder(dec)
    , maxLen(config.maxLen)
    , numBeams(config.numBeams)
    , numBeamHypsToKeep(config.numBeamHypsToKeep)
    , lenPenalty(config.lenPenalty)
    , doEarlyStopping(config.doEarlyStopping) {
    vocabSize = decoder.getContext()->vocabSize;
    eosTokenId = config.eosTokenId == -1 ? decoder.getEndId() : config.eosTokenId;
    padTokenId = config.padTokenId == -1 ? eosTokenId : config.padTokenId;
    kVal = 2 * numBeams;
    if (config.repetitionPenalty != 1.0) {
        printf("[Warning] BeamSearch doesn't support repetition penalty now and repetition penalty is %f.\n",
                config.repetitionPenalty);
    }
}

// The first setp to get next tokens accoring to the prompt IDs
std::vector<int> BeamSearch::getNextToken(int *ids, int batchSize, int seqLen) {
    // Assume input has been synced with master in higher level.
    TimeLine t("1st Token");
    this->step = 0;
    this->curLen = seqLen;
    this->batchSize = batchSize;
    this->beamScorer = BeamSearchScorer(batchSize, maxLen, numBeams, lenPenalty, doEarlyStopping, numBeamHypsToKeep);

    nextScores.resize(batchSize * kVal);
    nextTokens.resize(batchSize * kVal);
    nextIndices.resize(batchSize * kVal);

    // expand input from (bs, seq) to (bs, numBeams, seq) for future usage.
    inputIds.resize(batchSize * numBeams * seqLen);
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < numBeams; j++) {
            std::copy(ids + i * seqLen, ids + (i + 1) * seqLen, inputIds.begin() + (i * numBeams + j) * seqLen);
        }
    }

    int64_t dims[3] = {batchSize, numBeams, seqLen};

    // 1st token's input shape is [userSideBS][1][seqLen].
    std::tuple<float *, int, int> result = decoder.forward(ids, dims, this->step++);
    this->curLen++;

    // Initialize -1e9 to all beams except the first one
    beamNextScores = std::vector<float>(batchSize * numBeams, -1e9);
    for (int i = 0; i < batchSize; ++i) {
        beamNextScores[i * numBeams] = 0;
    }

    beam_search(result);
    return beamNextTokens;
}

// Get next tokens according to previous predicted ID
std::vector<int> BeamSearch::getNextToken() {
    TimeLine t("Next Token");
    int64_t dims[3] = {batchSize, numBeams, 1};

    decoder.reorderCache(beamNextIndices.data(), batchSize * numBeams);

    std::tuple<float *, int, int> result = decoder.forward(beamNextTokens.data(), dims, this->step++);
    this->curLen++;

    beam_search(result);
    return beamNextTokens;
}

// Get current output sequence.
std::vector<int32_t> BeamSearch::finalize() {
    auto sequenceOutputs
            = beamScorer.finalize(inputIds, beamNextScores, beamNextTokens, beamNextIndices, padTokenId, eosTokenId);
    TimeLine t("dump_file");
    t.dump_file("timeline.json");
    return sequenceOutputs;
}

bool BeamSearch::isDone() {
    return step != 0 && (beamScorer.isDone() || curLen >= maxLen);
}

void BeamSearch::searchTopK(std::tuple<float *, int, int> &result) {
    TimeLine t("BeamSearch.searchTopK");
    float *outBuf = std::get<0>(result);
    int sampleOffset = std::get<1>(result);
    int sampleSize = std::get<2>(result);

    Messenger &messenger = decoder.getMessenger();
    auto msgerSize = messenger.getSize();

    // 1. adjust_logits_during_generation(next_token_logits, cur_len)
    // adjust tokens for specify model, Marian. For Marian we have to make sure
    // that the `pad_token_id` cannot be generated both before and after the
    // `nn.functional.log_softmax` operation.
    // 2. nn.functional.log_softmax(next_token_logits, dim=-1)
    //    (batch_size * num_beams, vocab_size)
    // 3. use logits_processor(input_ids, next_token_scores)
    // TODO: support logits_processor
    // 4. add beam socre. Initialize -1e9 to all beams except the first one
    if (msgerSize > 1) {
        // Get the maximum value of each beam through all instance
        float maxVal[batchSize * numBeams] = {std::numeric_limits<float>::lowest()};
        float recvMax[batchSize * numBeams * msgerSize];
#pragma omp parallel for
        for (int i = 0; i < batchSize * numBeams; ++i) {
            float *p = outBuf + i * sampleSize;
            __m512 vmax = _mm512_set1_ps(maxVal[i]);
            for (int off = 0; off < sampleSize; off += 16) {
                int remain = sampleSize - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                __m512 vx = _mm512_maskz_loadu_ps(mask, p + off);
                vmax = _mm512_mask_max_ps(vmax, mask, vmax, vx);
            }
            maxVal[i] = _mm512_reduce_max_ps(vmax);
        }

        std::vector<long unsigned int> recvCount(msgerSize, static_cast<long unsigned int>(batchSize * numBeams));
        messenger.allgatherv(maxVal, batchSize * numBeams, recvMax, recvCount);

        float sumVal[batchSize * numBeams] = {0};
        float recvSum[batchSize * numBeams * msgerSize];
        // Get the global sum for each beam
#pragma omp parallel for
        for (int i = 0; i < batchSize * numBeams; ++i) {
            maxVal[i] = recvMax[i];
            for (int j = 1; j < msgerSize; ++j) {
                maxVal[i] = maxVal[i] > recvMax[j * batchSize * numBeams + i] ? maxVal[i]
                                                                              : recvMax[j * batchSize * numBeams + i];
            }
            float *p = outBuf + i * sampleSize;
            __m512 vmax = _mm512_set1_ps(maxVal[i]);
            __m512 vsum = _mm512_set1_ps(0);
            for (int off = 0; off < sampleSize; off += 16) {
                int remain = sampleSize - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                __m512 vx = _mm512_maskz_loadu_ps(mask, p + off);
                vx = _mm512_mask_sub_ps(vx, mask, vx, vmax);
                vx = BertUtil::vexp(vx);

                vsum = _mm512_mask_add_ps(vsum, mask, vsum, vx);
            }
            sumVal[i] = _mm512_reduce_add_ps(vsum);
        }
        messenger.allgatherv(sumVal, batchSize * numBeams, recvSum, recvCount);

        // Complete the LogSoftMax.
#pragma omp parallel for
        for (int i = 0; i < batchSize * numBeams; ++i) {
            float sum = recvSum[i];
            for (int j = 1; j < msgerSize; ++j) {
                sum += recvSum[j * batchSize * numBeams + i];
            }
            float logsum = std::log(sum);
            __m512 vsub = _mm512_set1_ps(maxVal[i] + logsum);
            float *p = outBuf + i * sampleSize;

#pragma omp parallel for
            for (int off = 0; off < sampleSize; off += 16) {
                int remain = sampleSize - off;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

                __m512 vx = _mm512_maskz_loadu_ps(mask, p + off);
                vx = _mm512_mask_sub_ps(vx, mask, vx, vsub);
                _mm512_mask_storeu_ps(p + off, mask, vx);
            }
        }
    } else {
        for (int i = 0; i < batchSize * numBeams; ++i) {
            float *p = outBuf + i * sampleSize;
            computeLogSoftmax(p, sampleSize);
        }
    }

    // Get top K candidates for beam search, use 2 * numBeams for K
    // topK ids and vals for each sample
    int topKIds[batchSize * numBeams * kVal];
    float topKVals[batchSize * numBeams * kVal];

    // Do topK during splited outputs on each beam's sample
    // Each thread handle one sample (one row)
#pragma omp parallel for
    for (int i = 0; i < batchSize * numBeams; ++i) {
        std::vector<std::pair<float, int>> elements(sampleSize);
        for (int j = 0; j < sampleSize; j++) {
            elements[j] = std::make_pair(*(outBuf + i * sampleSize + j), j);
        }
        std::partial_sort(
                elements.begin(), elements.begin() + kVal, elements.end(), std::greater<std::pair<float, int>>());

        for (int j = 0; j < kVal; ++j) {
            // Add beam score here
            topKVals[i * kVal + j] = elements[j].first + beamNextScores[i];
            topKIds[i * kVal + j] = elements[j].second + sampleOffset;
        }
    }

    // Reduce to get the topK index during all batch's beam and sample
    if (msgerSize > 1) {
        float sendBuf[2 * batchSize * numBeams * kVal];
        float recvBuf[2 * batchSize * numBeams * kVal * msgerSize];

        for (int i = 0; i < batchSize * numBeams * kVal; ++i) {
            sendBuf[2 * i] = (float)topKIds[i];
            sendBuf[2 * i + 1] = topKVals[i];
        }

        size_t sendSize = 2 * batchSize * numBeams * kVal;
        std::vector<long unsigned int> recvCount(msgerSize, static_cast<long unsigned int>(sendSize));
        messenger.allgatherv(sendBuf, sendSize, recvBuf, recvCount);

#pragma omp parallel for
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            std::vector<std::tuple<float, int, int>> elements(msgerSize * numBeams * kVal);
            for (int beamIdx = 0; beamIdx < numBeams; ++beamIdx) {
                for (int i = 0; i < msgerSize; ++i) {
                    int idx = ((i * batchSize + batchIdx) * numBeams + beamIdx) * kVal;
                    for (int k = 0; k < kVal; ++k) {
                        elements[(i * numBeams + beamIdx) * kVal + k] = std::make_tuple(
                                recvBuf[2 * (idx + k) + 1], (int)(recvBuf[2 * (idx + k)] + 0.5f), beamIdx);
                    }
                }
            }
            // Remove duplicates and take the top K
            std::sort(elements.begin(), elements.end(), tupleCompare);
            auto newEnd = std::unique(elements.begin(), elements.end(), uniqueCompare);

            for (int i = 0; i < kVal; i++) {
                nextScores[batchIdx * kVal + i] = std::get<0>(elements[i]);
                nextTokens[batchIdx * kVal + i] = std::get<1>(elements[i]);
                nextIndices[batchIdx * kVal + i] = std::get<2>(elements[i]);
            }
        }
    } else {
#pragma omp parallel for
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            std::vector<std::tuple<float, int, int>> elements(numBeams * kVal);
            for (int beamIdx = 0; beamIdx < numBeams; ++beamIdx) {
                int idx = (batchIdx * numBeams + beamIdx) * kVal;
                for (int k = 0; k < kVal; ++k) {
                    elements[beamIdx * kVal + k] = std::make_tuple(topKVals[idx + k], topKIds[idx + k], beamIdx);
                }
            }
            // Remove duplicates and take the top K
            std::sort(elements.begin(), elements.end(), tupleCompare);
            auto newEnd = std::unique(elements.begin(), elements.end(), uniqueCompare);

            for (int i = 0; i < kVal; i++) {
                nextScores[batchIdx * kVal + i] = std::get<0>(elements[i]);
                nextTokens[batchIdx * kVal + i] = std::get<1>(elements[i]);
                nextIndices[batchIdx * kVal + i] = std::get<2>(elements[i]);
            }
        }
    }
}

void BeamSearch::beam_search(std::tuple<float *, int, int> &result) {
    TimeLine t("BeamSearch");
    // Get candidates
    searchTopK(result);

    // Process the beams
    auto beamOutputs = beamScorer.process(inputIds, nextScores, nextTokens, nextIndices, padTokenId, eosTokenId);

    beamNextScores = std::get<0>(beamOutputs);
    beamNextTokens = std::get<1>(beamOutputs);
    beamNextIndices = std::get<2>(beamOutputs);

    std::vector<int32_t> newInputIds(batchSize * numBeams * curLen);
    for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        for (int beamIdx = 0; beamIdx < numBeams; ++beamIdx) {
            int idx = batchIdx * numBeams + beamIdx;
            std::copy(inputIds.begin() + beamNextIndices[idx] * (curLen - 1),
                    inputIds.begin() + (beamNextIndices[idx] + 1) * (curLen - 1), newInputIds.begin() + idx * curLen);
            newInputIds[(idx + 1) * curLen - 1] = beamNextTokens[idx];
        }
    }
    inputIds = newInputIds;
}
