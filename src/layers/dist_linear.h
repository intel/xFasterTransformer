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
#pragma once
#include "float16.h"
#include "matmul_helper.h"
#include "timeline.h"

/**
 * Distributed linear impl. by vertically spliting the weight
 */
template <typename WeiT>
class DistLinear {
public:
    DistLinear(int inDim, int outDim, int splitIdx, int splits) {
        this->inputSize = inDim;
        this->outputSize = outDim;
        this->splitIdx = splitIdx;
        this->splits = splits;

        this->bias = nullptr;
    }

    ~DistLinear() {
        if (bias) free(bias);
    }

    // Note: the weight passed in is transposed
    //
    //  _______________inputSize(K)______________
    // |                                         |
    // |                                         | splitSize(N)
    // |_________________________________________|
    // |                                         |
    // |                                         | splitSize(N)
    // |_________________________________________|
    void setWeight(const float *w, const float *b) {
        this->splitSize = outputSize / splits;
        this->splitOffset = this->splitSize * splitIdx;

        if (splitIdx < outputSize % splits) {
            this->splitSize += 1;
            this->splitOffset += splitIdx;
        } else {
            this->splitOffset += outputSize % splits;
        }

        int K = inputSize;
        int N = this->splitSize;
        weight.Resize(K, N);
        scaleWeight.Resize(N);
        zeroWeight.Resize(N);

        hpj::Matrix<WeiT> quantizedWeight;
        MMHelper::convertWeight(true, K, N, w + splitOffset * K, quantizedWeight, scaleWeight, zeroWeight, sumWeight);
        MMHelper::packWeight(true, quantizedWeight, weight);

        // Copy Bias
        if (b) {
            bias = (float *)aligned_alloc(64, N * sizeof(float));
            memcpy(bias, b + splitOffset, N * sizeof(float));
        }
    }

    // input is in the shape of (batchSize, inputSize)
    void forward(const float *input, float *output, int batchSize) {
        TimeLine t("DistLinear.forward");
        if (bias) {
            MMHelper::compute_bias(false, batchSize, splitSize, inputSize, 1.0f, input, inputSize, weight.Data(),
                    scaleWeight.Data(), zeroWeight.Data(), sumWeight.Data(), 0.0f, output, splitSize, bias);

        } else {
            MMHelper::compute(false, batchSize, splitSize, inputSize, 1.0f, input, inputSize, weight.Data(),
                    scaleWeight.Data(), zeroWeight.Data(), sumWeight.Data(), 0.0f, output, splitSize);
        }
    }

    int getInputSize() { return inputSize; }

    int getOutputSize() { return outputSize; }

    int getSplitSize() { return splitSize; }

    int getSplitOffset() { return splitOffset; }

private:
    int inputSize;
    int outputSize;

    int splitIdx;
    int splits;

    // = outputSize/splits, but need to consider the case of not divisible
    int splitSize;
    int splitOffset;

    hpj::Matrix<WeiT> weight;
    hpj::Vector<float> scaleWeight; // if weight is int8
    hpj::Vector<float> zeroWeight; // if weight is int8
    hpj::Vector<float> sumWeight; // if weight is int8
    float *bias;
};
