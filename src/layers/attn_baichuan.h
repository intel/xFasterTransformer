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
#include <cstring>
#include <iostream>

#include "attention.h"
#include "common_decoder.h"
#include "rms_norm.h"

static int respBaichuanHeads = 0;
static float *alibiSlopes = nullptr;

template <typename WeiT, typename QKPO_CLS = QKPO_Dummy, typename NORM_CLS = RmsNorm, typename InT = float,
        typename ImT = float, typename OutT = float, bool INPUT_AS_RESID = true>
class BaichuanAttention : public Attention<WeiT, QKPO_CLS, NORM_CLS, InT, ImT, OutT, INPUT_AS_RESID> {
public:
    BaichuanAttention(int layerId, DecoderContext *ctx)
        : Attention<WeiT, QKPO_CLS, NORM_CLS, InT, ImT, OutT, INPUT_AS_RESID>(layerId, ctx) {
        if (ctx->maxPosEmbed <= 0 && this->alibiSlopes == nullptr) {
            respBaichuanHeads = this->endQHead - this->startQHead;
            this->alibiSlopes = new float[respBaichuanHeads];
            // alibi mask element
            float ratio = std::pow(2, 8);
            int closestPowerOf2 = std::pow(2, int(std::log2(ctx->attHeadNum)));
            float x0 = std::pow(ratio, 1.0 / closestPowerOf2);
            float x1 = std::pow(ratio, 1.0 / (closestPowerOf2 * 2));
            for (int i = 0, h = this->startQHead; i < respBaichuanHeads; ++i, ++h) {
                if (h < closestPowerOf2)
                    this->alibiSlopes[i] = 1 / std::pow(x0, h + 1);
                else
                    this->alibiSlopes[i] = 1 / std::pow(x1, 2 * (h - closestPowerOf2) + 1);
            }
            alibiSlopes = this->alibiSlopes;
        }
    }

    const static float *getAlibiSlopes() { return alibiSlopes; }

    const static int getResponsibleHeads() { return respBaichuanHeads; }

    virtual ~BaichuanAttention() {
        if (this->alibiSlopes != nullptr) {
            delete[] this->alibiSlopes;
            this->alibiSlopes = nullptr;
        }
    }

protected:
    const float *getMask(const float *attnMask, int bId, int hId, int srcLen, int tgtLen) override {
        if (this->alibiSlopes != nullptr)
            return attnMask + hId * srcLen * tgtLen;
        else
            return attnMask + bId * srcLen * tgtLen;
    }

private:
};

template <typename WeiT, typename QKPO_CLS, typename NORM_CLS, typename InT, typename ImT, typename OutT>
struct AttnTypeExtractor<BaichuanAttention<WeiT, QKPO_CLS, NORM_CLS, InT, ImT, OutT, true>> {
    using Tin = InT;
    using Tim = ImT;
    using Tout = OutT;
};
