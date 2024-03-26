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
#include "alibi_embedding.h"
#include <cmath>
#include "allocator.h"
#include "compile_util.h"

AlibiEmbedding::AlibiEmbedding(const int headNum, const int seqLen) {
    maxLen = seqLen;
    maxHeadNums = headNum;
    alibiGetRelativePos(maxLen);
    alibiGetSlope(maxHeadNums);
}

void AlibiEmbedding::alibiGetBias(const int headIdx, const int seqLen, float *biasMatrx) {
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
    posMatrix = (int *)xft::alloc(seqLen * seqLen * sizeof(int));
    for (int i = 0; i < seqLen; i++) {
        for (int j = 0; j < seqLen; j++) {
            posMatrix[i * seqLen + j] = j - i;
        }
    }
}

void AlibiEmbedding::alibiGetSlope(const int headNum) {
    slopeM = (float *)xft::alloc(headNum * sizeof(float));
    float x = std::pow(2, 8);
    x = std::pow(x, 1.0 / headNum);
    for (int i = 0; i < headNum; i++) {
        slopeM[i] = 1 / std::pow(x, i + 1);
    }
}