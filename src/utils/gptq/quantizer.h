#pragma once

#include "utils.h"

class Quantizer {
public:
    Quantizer(int wbits, bool perchannel, bool sym, bool mse)
        : wbits(wbits), perchannel(perchannel), sym(sym), mse(mse) {
        maxq = pow(2, wbits) - 1;
    }

    void find_params(const Tensor<float> &weight, Tensor<float> &scale, Tensor<int> &zero);

    int get_maxq() {
        return maxq;
    }

private:
    int wbits;
    int maxq;
    bool perchannel;
    bool sym;
    bool mse;
    float norm = 2.4;
    int grid = 100;
    float maxshrink = 0.8;
};