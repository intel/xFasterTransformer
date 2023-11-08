#pragma once
#include <iostream>

class AlibiEmbedding {
public:
    AlibiEmbedding(const int headNum, const int seqLen);

    ~AlibiEmbedding() {
        maxLen = 0;
        maxHeadNums = 0;
        if (posMatrix != nullptr) free(posMatrix);
        if (slopeM != nullptr) free(slopeM);
    }

    void alibiGetRelativePos(const int seqLen);

    void alibiGetSlope(const int headNum);

    // headIdx is [0,n]
    void alibiGetBias(const int headIdx, const int seqLen, float *bias_matrx);

private:
    int maxLen = 0;
    int maxHeadNums = 0;
    int *posMatrix;
    float *slopeM;
};
