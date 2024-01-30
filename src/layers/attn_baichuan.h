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
#include <cmath>
#include <iostream>
#include <cstring>

#include "common_decoder.h"
#include "rms_norm.h"
#include "attention.h"

template <typename WeiT, typename QKPO_CLS = QKPO_Dummy, typename NORM_CLS = RmsNorm>
class BaichuanAttention : public Attention<WeiT, QKPO_CLS, NORM_CLS> {
public:
    BaichuanAttention(int layerId, DecoderContext *ctx) : Attention<WeiT, QKPO_CLS, NORM_CLS>(layerId, ctx) {
        alibiSlopes = nullptr;
    }

    virtual ~BaichuanAttention() {
        if (alibiSlopes != nullptr)
            delete[] alibiSlopes;
    }

    const float* getAlibiSlopes(int attHeadNum) {
        if (alibiSlopes == nullptr) {
            int responsibleHeads = this->endQHead - this->startQHead;
            alibiSlopes = new float[responsibleHeads];
            // alibi mask element 
            float ratio = std::pow(2, 8);
            int closestPowerOf2 = std::pow(2, int(std::log2(attHeadNum)));
            float x0 = std::pow(ratio, 1.0 / closestPowerOf2);
            float x1 = std::pow(ratio, 1.0 / (closestPowerOf2 * 2));
            for (int i = 0, h = this->startQHead; i < responsibleHeads; ++i, ++h) {
                if (h < closestPowerOf2)
                    alibiSlopes[i] = 1 / std::pow(x0, h + 1);
                else
                    alibiSlopes[i] = 1 / std::pow(x1, 2 * (h - closestPowerOf2) + 1);
            }
        }
        return alibiSlopes;
    }

    const int getResponsibleHeads() {
        return this->endQHead - this->startQHead;
    }

protected:

    const float* getMask(const float* attnMask, int bId, int hId, int srcLen, int tgtLen) override {
        return attnMask + hId * srcLen * tgtLen;
    }

private:
    float* alibiSlopes;
};
