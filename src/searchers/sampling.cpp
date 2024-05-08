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
#include <omp.h>

#include "sampling.h"
#include "timeline.h"

namespace xft {
// Assume all samples have the same sampling params.
std::vector<int> greedySearch(float *logits, int sampleOffset, int sampleSize, int batchSize) {
    TimeLine t("GreedySearch");

    Messenger &messenger = Messenger::getInstance();
    int numThreads = 0;
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid == 0) { numThreads = omp_get_num_threads(); }
    }

    auto msgerSize = messenger.getSize();

    // Max ID and value for each sample
    std::vector<int> maxIds(batchSize);
    float maxVals[batchSize];

    // Small batch size (each sample can have at least 2 threads)
    if (numThreads / batchSize >= 2) {
        int thrPerSample = numThreads / batchSize;
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

    return maxIds;
}

// For greedy search and samlping, not for beam search
void stopCheck(std::vector<int> &generatedIds, std::vector<SequenceGroupMeta *> &seqGroups) {
    int batchSize = generatedIds.size();
#pragma omp parallel for
    for (int b = 0; b < batchSize; b++) {
        // TODO: Deprecate this check, since no need for unequal-length output
        if (seqGroups[b]->getSamplingMeta()->done) {
            generatedIds[b] = seqGroups[b]->getSamplingMeta()->config.eosTokenId;
            continue;
        }

        // If the generated token is EOS, mark the sequence as done
        if (seqGroups[b]->getSamplingMeta()->config.eosTokenId == generatedIds[b]) {
            seqGroups[b]->getSamplingMeta()->done = true;
        }
        // If the sequence meets the max length, mark the sequence as done
        else if (seqGroups[b]->get(0)->getTotalLen() + 1 >= seqGroups[b]->getSamplingMeta()->config.maxLen) {
            seqGroups[b]->getSamplingMeta()->done = true;
        }
        // TODO: stop words check
    }
}
} // namespace xft