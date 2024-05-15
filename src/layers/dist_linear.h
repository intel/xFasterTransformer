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
#include "allocator.h"
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
    void setWeight(DecoderContext *ctx, const float *w, const float *b = nullptr) {
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

        xft::Matrix<WeiT> quantizedWeight;
        ctx->mmHelper->convertWeight(
                true, K, N, w + splitOffset * K, nullptr, nullptr, quantizedWeight, scaleWeight, zeroWeight, sumWeight);
        ctx->mmHelper->packWeight(true, quantizedWeight, weight);

        // Copy Bias
        if (b) {
            bias = (float *)xft::alloc(N * sizeof(float));
            memcpy(bias, b + splitOffset, N * sizeof(float));
        }
    }

    // input is in the shape of (batchSize, inputSize)
    template <typename T1, typename T2>
    void forward(DecoderContext *ctx, const T1 *input, T2 *output, int batchSize) {
        TimeLine t("DistLinear.forward");
        if (bias) {
            ctx->mmHelper->compute_bias(false, batchSize, splitSize, inputSize, 1.0f, input, inputSize, weight.Data(),
                    scaleWeight.Data(), zeroWeight.Data(), sumWeight.Data(), 0.0f, output, splitSize, bias);

        } else {
            ctx->mmHelper->compute(false, batchSize, splitSize, inputSize, 1.0f, input, inputSize, weight.Data(),
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

    xft::Matrix<WeiT> weight;
    xft::Vector<float> scaleWeight; // if weight is int8
    xft::Vector<float> zeroWeight; // if weight is int8
    xft::Vector<float> sumWeight; // if weight is int8
    float *bias = nullptr;
};
