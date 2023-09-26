#include "greedy_search.h"

GreedySearch::GreedySearch(AbstractDecoder &dec, const SearcherConfig &config)
    : decoder(dec), maxLen(config.maxLen), step(0) {
    eosTokenId = config.eosTokenId == -1 ? decoder.getEndId() : config.eosTokenId;
    padTokenId = config.padTokenId == -1 ? eosTokenId : config.padTokenId;
}

// Get next tokens accoring to the prompt IDs
std::vector<int> GreedySearch::getNextToken(int *ids, int batchSize, int seqLen) {
    this->step = 0;
    this->batchSize = batchSize;
    this->curLen = seqLen;
    this->doneBatch.resize(batchSize);
    std::fill(doneBatch.begin(), doneBatch.end(), false);

    this->output.resize(batchSize * seqLen);
    std::copy(ids, ids + batchSize * seqLen, output.begin());

    int64_t dims[3] = {batchSize, 1, seqLen};

    std::tuple<float *, int, int> result = decoder.forward(ids, dims, this->step++);

    this->nextTokens = search(result);

    this->curLen++;
    for (int batchId = 0; batchId < batchSize; ++batchId) {
        output.insert(output.begin() + (batchId + 1) * curLen - 1, nextTokens[batchId]);
    }

    return this->nextTokens;
}

// Get next tokens according to previous predicted ID
std::vector<int> GreedySearch::getNextToken() {
    int64_t dims[3] = {batchSize, 1, 1};
    std::tuple<float *, int, int> result = decoder.forward(nextTokens.data(), dims, this->step++);

    this->nextTokens = search(result);

    this->curLen++;
    for (int batchId = 0; batchId < batchSize; ++batchId) {
        output.insert(output.begin() + (batchId + 1) * curLen - 1, nextTokens[batchId]);
    }

    return this->nextTokens;
}

bool GreedySearch::isDone() {
    if (step == 0) {
        return false;
    } else if (curLen >= maxLen) {
        return true;
    } else {
        for (bool flag : doneBatch) {
            if (!flag) { return false; }
        }
    }
    return true;
}

std::vector<int> GreedySearch::search(std::tuple<float *, int, int> &result) {
    DecoderContext *ctx = decoder.getContext();

    float *outBuf = std::get<0>(result);
    int sampleOffset = std::get<1>(result);
    int sampleSize = std::get<2>(result);

    // Max ID and value for each sample
    int maxIds[batchSize];
    float maxVals[batchSize];

    // Small batch size (each sample can have at least 2 threads)
    if (ctx->numThreads / batchSize >= 2) {
        int thrPerSample = ctx->numThreads / batchSize;
        int sizePerThr = (sampleSize + thrPerSample - 1) / thrPerSample;
        int maxIndices[batchSize * thrPerSample];
        float maxValues[batchSize * thrPerSample];

        // TODO: if sampleSize is small, possible to cause out of boundary
#pragma omp parallel for collapse(2)
        for (int b = 0; b < batchSize; ++b) {
            for (int t = 0; t < thrPerSample; ++t) { // thread index inside the sample
                int start = t * sizePerThr;
                int end = (start + sizePerThr) > sampleSize ? sampleSize : (start + sizePerThr);
                float *p = outBuf + b * sampleSize;

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
            float *p = outBuf + i * sampleSize;
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
    Messenger &messenger = decoder.getMessenger();
    if (messenger.getSize() > 1) {
        float sendBuf[2 * batchSize];
        float recvBuf[2 * batchSize * messenger.getSize()];

        for (int i = 0; i < batchSize; ++i) {
            sendBuf[2 * i] = (float)(maxIds[i] + sampleOffset);
            sendBuf[2 * i + 1] = maxVals[i];
        }

        std::vector<long unsigned int> recvCount(messenger.getSize(), static_cast<long unsigned int>(2 * batchSize));
        messenger.allgatherv(sendBuf, 2 * batchSize, recvBuf, recvCount);

        for (int i = 0; i < batchSize; ++i) {
            int maxId = (int)(recvBuf[2 * i] + 0.5f);
            float maxVal = recvBuf[2 * i + 1];
            for (int j = 1; j < messenger.getSize(); ++j) {
                if (recvBuf[2 * j * batchSize + 2 * i + 1] > maxVal) {
                    maxVal = recvBuf[2 * j * batchSize + 2 * i + 1];
                    maxId = (int)(recvBuf[2 * j * batchSize + 2 * i] + 0.5f);
                }
            }
            maxIds[i] = maxId;
        }
    }

    if (eosTokenId != -1) {
        for (int batchId = 0; batchId < batchSize; ++batchId) {
            if (!doneBatch[batchId]) {
                if (maxIds[batchId] == eosTokenId) { doneBatch[batchId] = true; }
            } else {
                // Padding finished seq with padTokenId;
                maxIds[batchId] = padTokenId;
            }
        }
    }
    return std::vector<int>(maxIds, maxIds + batchSize);
}