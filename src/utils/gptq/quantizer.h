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
