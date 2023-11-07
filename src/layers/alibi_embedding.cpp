#include "alibi_embedding.h"
#include <cmath>
#include "compile_util.h"

bool AlibiEmbedding::initialized = false;

AlibiEmbedding::AlibiEmbedding(const int headNum, const int seqLen) {
    maxLen = seqLen;
    maxHeadNums = headNum;
    alibiGetRelativePos(maxLen);
    alibiGetSlope(maxHeadNums);
    initialized = true;
}

void AlibiEmbedding::alibiGetBias(const int headIdx, const int seqLen, float *biasMatrx) {
    REQUIRES(initialized == true, "Alibi Embedding ERROR, Alibi is not initialized.");
    REQUIRES(headIdx < maxHeadNums, "Alibi Embedding ERROR, headIdx is exceeds max head nums.");
    if (seqLen > maxLen) {
        maxLen = seqLen;
        alibiGetRelativePos(maxLen);
    }
    for (size_t i = 0; i < seqLen; i++) {
        for (size_t j = 0; j < seqLen; j++) {
            int index = i * seqLen + j;
            biasMatrx[index] = posMatrix[index] * slopeM[headIdx];
        }
    }
}

void AlibiEmbedding::alibiGetRelativePos(const int seqLen) {
    posMatrix = (int *)aligned_alloc(64, seqLen * seqLen * sizeof(int));
    for (int i = 0; i < seqLen; i++) {
        for (int j = 0; j < seqLen; j++) {
            posMatrix[i * seqLen + j] = j - i;
        }
    }
}

void AlibiEmbedding::alibiGetSlope(const int headNum) {
    slopeM = (float *)aligned_alloc(64, headNum * sizeof(float));
    float x = std::pow(2, 8);
    x = std::pow(x, 1.0 / headNum);
    for (int i = 0; i < headNum; i++) {
        slopeM[i] = 1 / std::pow(x, i + 1);
    }
}