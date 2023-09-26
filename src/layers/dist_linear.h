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
        MMHelper::convertWeight(true, K, N, w + splitOffset * K, quantizedWeight, scaleWeight, zeroWeight);
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
                    scaleWeight.Data(), zeroWeight.Data(), 0.0f, output, splitSize, bias);

        } else {
            MMHelper::compute(false, batchSize, splitSize, inputSize, 1.0f, input, inputSize, weight.Data(),
                    scaleWeight.Data(), zeroWeight.Data(), 0.0f, output, splitSize);
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
    hpj::Vector<float> scaleWeight; // if weighs is int8
    hpj::Vector<float> zeroWeight; // if weighs is int8
    float *bias;
};