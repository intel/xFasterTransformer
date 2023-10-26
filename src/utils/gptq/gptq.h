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

#include "quantizer.h"

class GPTQ {
public:
    GPTQ(const Tensor<float> &weight, int wbits) : wbits(wbits) {
        W.set(std::move(weight));
        quantizer = new Quantizer(wbits, perchannel, sym, mse);
        H.set(W.rows, W.rows, true);
    }

    void add_batch(const Tensor<float> &input);

    void fasterquant(Tensor<int> &Q_int, Tensor<float> &Q_float, Tensor<float> &scale, Tensor<int> &zero,
            int blocksize = 128, float percdamp = 0.01, int groupsize = -1, bool actorder = false);

private:
    int wbits;
    bool perchannel = true;
    bool sym = false;
    bool mse = false;
    Tensor<float> W;
    Tensor<float> H;
    int nsamples = 0;
    Quantizer *quantizer;
};
